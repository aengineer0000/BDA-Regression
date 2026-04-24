import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = './data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_LBL_DIR = os.path.join(DATA_DIR, 'train', 'labels')

PATCH_SIZE = 128  
LIMIT_IMAGES = 400

DAMAGE_MAP = {
    "no-damage": 0.0,
    "minor-damage": 0.33,
    "major-damage": 0.66,
    "destroyed": 1.0,
    "un-classified": None
}

def extract_features(patch_pre, patch_post):
    pre_f = patch_pre.astype(float)
    post_f = patch_post.astype(float)
    diff = np.abs(post_f - pre_f)
    
    features = [
        np.mean(diff),
        np.std(diff),
        np.max(diff),
        np.mean(post_f) - np.mean(pre_f),
        np.std(post_f),
        np.mean(pre_f) 
    ]
    return np.array(features)

def process_dataset(limit=None):
    X_features, y_labels = [], []
    label_files = [f for f in os.listdir(TRAIN_LBL_DIR) if '_post_disaster.json' in f]
    if limit: label_files = label_files[:limit]

    print(f"Processing {len(label_files)} images...")
    for label_file in tqdm(label_files):
        with open(os.path.join(TRAIN_LBL_DIR, label_file)) as f:
            label_data = json.load(f)
            
        base_id = label_file.replace('_post_disaster.json', '')
        pre_path = os.path.join(TRAIN_IMG_DIR, f"{base_id}_pre_disaster.png")
        post_path = os.path.join(TRAIN_IMG_DIR, f"{base_id}_post_disaster.png")
        
        if not os.path.exists(pre_path) or not os.path.exists(post_path): continue
            
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
                feat_vec = extract_features(img_pre[cy-half:cy+half, cx-half:cx+half], 
                                           img_post[cy-half:cy+half, cx-half:cx+half])
                X_features.append(feat_vec)
                y_labels.append(target_val)

    return np.array(X_features), np.array(y_labels)

if __name__ == "__main__":
    X, y = process_dataset(limit=LIMIT_IMAGES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔍 Starting Hyperparameter Tuning (Grid Search)...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    predictions = best_model.predict(X_test)
    print("\n" + "="*30)
    print("📊 TUNED RESULTS")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"R2:  {r2_score(y_test, predictions):.4f}")
    print("="*30)