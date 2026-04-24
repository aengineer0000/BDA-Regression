import os
import json
import cv2
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from tqdm import tqdm
from shapely import wkt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = './data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_LBL_DIR = os.path.join(DATA_DIR, 'train', 'labels')

PATCH_SIZE = 224
LIMIT_IMAGES = 1200 
BATCH_SIZE = 16    
EPOCHS = 12  

if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using Intel Arc GPU acceleration: {torch.xpu.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("XPU not detected, falling back to CPU.")

DAMAGE_MAP = {
    "no-damage": 0.0,
    "minor-damage": 0.33,
    "major-damage": 0.66,
    "destroyed": 1.0,
    "un-classified": None
}

class SiameseLazyDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
        self._last_path = None
        self._last_imgs = (None, None)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        
        if self._last_path == m['pre_path']:
            img_pre, img_post = self._last_imgs
        else:
            img_pre = cv2.imread(m['pre_path'])
            img_post = cv2.imread(m['post_path'])
            if img_pre is None or img_post is None:
                return torch.zeros((3, PATCH_SIZE, PATCH_SIZE)), torch.zeros((3, PATCH_SIZE, PATCH_SIZE)), torch.tensor([0.0])
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            img_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB)
            self._last_path = m['pre_path']
            self._last_imgs = (img_pre, img_post)
        
        cx, cy = m['cx'], m['cy']
        half = PATCH_SIZE // 2
        p_pre = img_pre[cy-half:cy+half, cx-half:cx+half]
        p_post = img_post[cy-half:cy+half, cx-half:cx+half]
        
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            p_pre_t = self.transform(p_pre)
            torch.manual_seed(seed)
            p_post_t = self.transform(p_post)
        else:
            to_tensor = transforms.ToTensor()
            p_pre_t = to_tensor(p_pre)
            p_post_t = to_tensor(p_post)
            
        label = torch.tensor([m['label']], dtype=torch.float32)
        return p_pre_t, p_post_t, label

class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        
        for param in list(self.backbone.parameters())[:5]:
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(512 * 4, 512), 
            nn.BatchNorm1d(512),
            nn.SiLU(), 
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, pre, post):
        f_pre = self.backbone(pre).view(pre.size(0), -1)
        f_post = self.backbone(post).view(post.size(0), -1)
        
        f_diff = torch.abs(f_post - f_pre)
        f_prod = f_pre * f_post 
        
        combined = torch.cat((f_pre, f_post, f_diff, f_prod), dim=1)
        return self.regressor(combined)

def get_metadata(limit):
    meta = []
    label_files = [f for f in os.listdir(TRAIN_LBL_DIR) if '_post_disaster.json' in f][:limit]
    
    class_counts = {0.0: 0, 0.33: 0, 0.66: 0, 1.0: 0}

    for label_file in tqdm(label_files, desc="Scanning Metadata"):
        with open(os.path.join(TRAIN_LBL_DIR, label_file)) as f:
            data = json.load(f)
        
        base = label_file.replace('_post_disaster.json', '')
        pre_p = os.path.join(TRAIN_IMG_DIR, f"{base}_pre_disaster.png")
        post_p = os.path.join(TRAIN_IMG_DIR, f"{base}_post_disaster.png")
        
        if not os.path.exists(pre_p): continue

        for feat in data['features']['xy']:
            val = DAMAGE_MAP.get(feat['properties'].get('subtype'))
            if val is None: continue
            
            poly = wkt.loads(feat['wkt'])
            cx, cy = int(poly.centroid.x), int(poly.centroid.y)
            h = PATCH_SIZE // 2
            
            if (cy-h > 0 and cy+h < 1024 and cx-h > 0 and cx+h < 1024):
                meta.append({'pre_path': pre_p, 'post_path': post_p, 'cx': cx, 'cy': cy, 'label': val})
                class_counts[val] += 1
    
    meta.sort(key=lambda x: x['pre_path'])
    print(f"Dataset Imbalance: {class_counts}")
    return meta, class_counts

if __name__ == "__main__":
    metadata, counts = get_metadata(LIMIT_IMAGES)
    train_meta, val_meta = train_test_split(metadata, test_size=0.15, random_state=42)

    sample_weights = []
    for m in train_meta:
        weight = 1.0 / np.sqrt(max(counts[m['label']], 1))
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_meta), replacement=True)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(SiameseLazyDataset(train_meta, train_transform), 
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    
    val_loader = DataLoader(SiameseLazyDataset(val_meta, val_transform), 
                            batch_size=BATCH_SIZE, num_workers=4)

    model = SiameseResNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.00008, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.HuberLoss()

    print(f"Smooth-Balanced Training on {len(train_meta)} patches...")
    best_r2 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for pre, post, target in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            pre, post, target = pre.to(DEVICE), post.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(pre, post)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for vpre, vpost, vtarget in val_loader:
                vpre, vpost = vpre.to(DEVICE), vpost.to(DEVICE)
                preds = model(vpre, vpost).cpu().numpy()
                all_p.extend(preds)
                all_y.extend(vtarget.numpy())
        
        cur_r2 = r2_score(all_y, all_p)
        avg_loss = t_loss/len(train_loader)
        scheduler.step(avg_loss)
        
        print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} | Val R2: {cur_r2:.4f}")
        if cur_r2 > best_r2:
            best_r2 = cur_r2
            torch.save(model.state_dict(), 'best_siamese_resnet.pth')

    print(f"BEST R2: {best_r2:.4f}")