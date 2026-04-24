import os
import json
import cv2
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel Scikit-learn extensions active.")
except ImportError:
    print("Consider installing 'scikit-learn-intelex' for faster CPU training.")

DATA_DIR = './data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_LBL_DIR = os.path.join(DATA_DIR, 'train', 'labels')

PATCH_SIZE = 64  
LIMIT_IMAGES = 400 

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

class DeepDamageCNN(nn.Module):
    """
    Improved CNN architecture with Batch Normalization and deeper layers
    to push toward the R2 target of 0.65.
    """
    def __init__(self):
        super(DeepDamageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten()
        )
        
        flat_size = 128 * (PATCH_SIZE // 8) * (PATCH_SIZE // 8)
        
        self.regressor = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

def extract_tabular_features(patch_pre, patch_post):
    pre_f = patch_pre.astype(float)
    post_f = patch_post.astype(float)
    diff = np.abs(post_f - pre_f)
    
    return np.array([
        np.mean(diff),
        np.std(diff),
        np.max(diff),
        np.mean(post_f) - np.mean(pre_f),
        np.std(post_f),
        np.mean(pre_f) 
    ])

def process_dataset(limit=None):
    X_tabular, X_spatial, y_labels = [], [], []
    label_files = [f for f in os.listdir(TRAIN_LBL_DIR) if '_post_disaster.json' in f]
    if limit: label_files = label_files[:limit]

    print(f"Step 1: Extracting patches from {len(label_files)} images...")
    for label_file in tqdm(label_files, desc="Patch Extraction"):
        with open(os.path.join(TRAIN_LBL_DIR, label_file)) as f:
            label_data = json.load(f)
            
        base_id = label_file.replace('_post_disaster.json', '')
        pre_path = os.path.join(TRAIN_IMG_DIR, f"{base_id}_pre_disaster.png")
        post_path = os.path.join(TRAIN_IMG_DIR, f"{base_id}_post_disaster.png")
        
        if not (os.path.exists(pre_path) and os.path.exists(post_path)): continue
            
        img_pre = cv2.imread(pre_path)
        img_post = cv2.imread(post_path)

        for feature in label_data['features']['xy']:
            damage_str = feature['properties'].get('subtype', 'un-classified')
            target_val = DAMAGE_MAP.get(damage_str)
            if target_val is None: continue
            
            poly = wkt.loads(feature['wkt'])
            cx, cy = int(poly.centroid.x), int(poly.centroid.y)
            half = PATCH_SIZE // 2
            
            if (cy - half > 0 and cy + half < 1024 and cx - half > 0 and cx + half < 1024):
                p_pre = img_pre[cy-half:cy+half, cx-half:cx+half]
                p_post = img_post[cy-half:cy+half, cx-half:cx+half]
                
                X_tabular.append(extract_tabular_features(p_pre, p_post))
                spatial = np.concatenate([p_pre, p_post], axis=-1).transpose(2, 0, 1).astype(np.float32) / 255.0
                X_spatial.append(spatial)
                y_labels.append(target_val)
        
        del img_pre, img_post
        if len(X_tabular) % 1000 == 0: gc.collect()

    print(f"Step 2: Converting lists to arrays ({len(y_labels)} samples)...")
    X_tab = np.array(X_tabular)
    X_spa = np.array(X_spatial)
    y_arr = np.array(y_labels)
    return X_tab, X_spa, y_arr

def train_cnn(X_train, y_train, X_val, y_val):
    print("🚀 Step 3: Training DeepDamageCNN...")
    dataset = TensorDataset(torch.from_numpy(X_train).float(), 
                            torch.from_numpy(y_train).float().view(-1, 1))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = DeepDamageCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0003) 
    criterion = nn.HuberLoss() 

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 4 == 0:
            model.eval()
            with torch.no_grad():
                v_x = torch.from_numpy(X_val).float().to(DEVICE)
                v_preds = model(v_x).cpu().numpy()
                v_r2 = r2_score(y_val, v_preds)
                print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.4f} | Val R2: {v_r2:.4f}")
            
    model.eval()
    with torch.no_grad():
        final_preds = model(torch.from_numpy(X_val).float().to(DEVICE)).cpu().numpy()
    return final_preds

if __name__ == "__main__":
    try:
        X_tab, X_spa, y = process_dataset(limit=LIMIT_IMAGES)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        print("🌲 Tuning Random Forest...")
        rf_grid = {'n_estimators': [100], 'max_depth': [None, 20]}
        rf = GridSearchCV(RandomForestRegressor(n_jobs=-1), rf_grid, cv=3, scoring='r2').fit(X_tab[train_idx], y[train_idx])
        rf_preds = rf.best_estimator_.predict(X_tab[test_idx])

        print("🔥 Tuning XGBoost...")
        xgb_grid = {'learning_rate': [0.05, 0.1], 'max_depth': [6, 8]}
        xgb = GridSearchCV(XGBRegressor(n_jobs=-1), xgb_grid, cv=3, scoring='r2').fit(X_tab[train_idx], y[train_idx])
        xgb_preds = xgb.best_estimator_.predict(X_tab[test_idx])

        print("⚡ Training High-Capacity XGBoost...")
        h_xgb = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.03, n_jobs=-1)
        h_xgb.fit(X_tab[train_idx], y[train_idx])
        h_xgb_preds = h_xgb.predict(X_tab[test_idx])

        print("👥 Training KNN...")
        knn = KNeighborsRegressor(n_neighbors=9, n_jobs=-1).fit(X_tab[train_idx], y[train_idx])
        knn_preds = knn.predict(X_tab[test_idx])

        cnn_preds = train_cnn(X_spa[train_idx], y[train_idx], X_spa[test_idx], y[test_idx])

        results = {
            "Random Forest": rf_preds,
            "XGBoost": xgb_preds,
            "HC-XGBoost": h_xgb_preds,
            "KNN": knn_preds,
            "DeepDamageCNN": cnn_preds.flatten()
        }

        print("\n" + "="*50)
        print(f"{'Model':<20} | {'MAE':<8} | {'R2':<8}")
        print("-" * 50)
        for name, p in results.items():
            m = mean_absolute_error(y[test_idx], p)
            r = r2_score(y[test_idx], p)
            print(f"{name:<20} | {m:.4f}   | {r:.4f}")
        print("="*50)
    
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")