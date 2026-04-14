"""
load_cleaned_data.py - Load the preprocessed Cleaned SWAN-SF dataset
SUPPORTS BOTH 2D AND 3D DATA FORMATS
"""
import pandas as pd
import numpy as np
import pickle
import os
import glob

def flatten_3d_to_2d(X_3d, method='flatten'):
    """
    Convert 3D time-series data to 2D for traditional ML/FL models
    
    Args:
        X_3d: 3D array of shape (n_samples, n_timesteps, n_features)
        method: 'flatten', 'mean', 'max', 'min', 'std', 'last'
    
    Returns:
        X_2d: 2D array of shape (n_samples, n_features_transformed)
    """
    if X_3d.ndim == 2:
        return X_3d  # Already 2D, no change needed
    
    print(f"      [Transform] Converting 3D {X_3d.shape} → 2D using '{method}' method")
    
    if method == 'flatten':
        # Flatten all timesteps: (N, T, F) → (N, T*F)
        X_2d = X_3d.reshape(X_3d.shape[0], -1)
        print(f"      [Transform] Flattened to: {X_2d.shape} ({X_3d.shape[1]*X_3d.shape[2]} features)")
        
    elif method == 'mean':
        # Average across time: (N, T, F) → (N, F)
        X_2d = np.mean(X_3d, axis=1)
        print(f"      [Transform] Temporal mean: {X_2d.shape} ({X_3d.shape[2]} features)")
        
    elif method == 'max':
        # Max across time: (N, T, F) → (N, F)
        X_2d = np.max(X_3d, axis=1)
        print(f"      [Transform] Temporal max: {X_2d.shape} ({X_3d.shape[2]} features)")
        
    elif method == 'last':
        # Use last timestep: (N, T, F) → (N, F)
        X_2d = X_3d[:, -1, :]
        print(f"      [Transform] Last timestep: {X_2d.shape} ({X_3d.shape[2]} features)")
        
    elif method == 'concat_stats':
        # Concatenate [mean, std, max, min] across time: (N, T, F) → (N, 4*F)
        mean_feat = np.mean(X_3d, axis=1)
        std_feat = np.std(X_3d, axis=1)
        max_feat = np.max(X_3d, axis=1)
        min_feat = np.min(X_3d, axis=1)
        X_2d = np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=1)
        print(f"      [Transform] Statistical features: {X_2d.shape} ({4*X_3d.shape[2]} features)")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_2d.astype(np.float32)


def load_cleaned_partition(partition_num=1, 
                          data_dir='data/cleaned',
                          combine_all_partitions=True,
                          flatten_method='mean'):  # ← NEW PARAMETER
    """
    Load cleaned SWAN-SF dataset with automatic 3D→2D conversion
    
    Args:
        partition_num: Which partition to load (1-5), or 'all'
        data_dir: Base directory containing cleaned data
        combine_all_partitions: If True, combine all 5 partitions
        flatten_method: How to convert 3D to 2D ('flatten', 'mean', 'max', 'last', 'concat_stats')
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names (ALL 2D ARRAYS)
    """
    
    print("=" * 70)
    print("  SF-9: Loading Cleaned SWAN-SF Dataset")
    print("  Mode: Auto-detect 2D/3D + Convert to 2D")
    print("=" * 70)
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # ── DETERMINE PARTITIONS TO LOAD ──
    if combine_all_partitions:
        partitions = [1, 2, 3, 4, 5]
        print(f"\n[Loading] Combining all 5 partitions...")
    else:
        partitions = [partition_num]
        print(f"\n[Loading] Partition {partition_num} only...")
    
    # ── LOAD TRAIN DATA ──
    print("\n[1/4] Loading training data (preprocessed + balanced)...")
    
    X_train_list = []
    y_train_list = []
    
    for part_num in partitions:
        train_file = f"Partition{part_num}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
        label_file = f"Partition{part_num}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
        
        train_path = os.path.join(train_dir, train_file)
        label_path = os.path.join(train_dir, label_file)
        
        if not os.path.exists(train_path):
            print(f"      ⚠ Warning: {train_file} not found, skipping")
            continue
        
        try:
            with open(train_path, 'rb') as f:
                X_part = pickle.load(f)
            
            with open(label_path, 'rb') as f:
                y_part = pickle.load(f)
            
            print(f"      ✓ Partition {part_num}: X={X_part.shape}, y={y_part.shape}")
            
            # AUTO-DETECT 3D AND FLATTEN
            if X_part.ndim == 3:
                print(f"      ⚠ Detected 3D data! Applying '{flatten_method}' transformation...")
                X_part = flatten_3d_to_2d(X_part, method=flatten_method)
            
            X_train_list.append(X_part)
            y_train_list.append(y_part)
            
        except Exception as e:
            print(f"      ✗ Error loading partition {part_num}: {e}")
    
    # Combine all partitions
    if len(X_train_list) > 0:
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
    else:
        raise ValueError("No training partitions could be loaded!")
    
    print(f"      ✓ Total training: X={X_train.shape}, y={y_train.shape}")
    print(f"      ✓ Train flare rate: {y_train.mean():.2%}")
    
    # ── LOAD TEST DATA ──
    print("\n[2/4] Loading test data (cleaned, not resampled)...")
    
    X_test_list = []
    y_test_list = []
    
    for part_num in partitions:
        test_file = f"Partition{part_num}_LSBZM-Norm_FPCKNN-impute.pkl"
        label_file = f"Partition{part_num}_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
        
        test_path = os.path.join(test_dir, test_file)
        label_path = os.path.join(test_dir, label_file)
        
        if not os.path.exists(test_path):
            print(f"      ⚠ Warning: {test_file} not found, skipping")
            continue
        
        try:
            with open(test_path, 'rb') as f:
                X_part = pickle.load(f)
            
            with open(label_path, 'rb') as f:
                y_part = pickle.load(f)
            
            print(f"      ✓ Partition {part_num}: X={X_part.shape}, y={y_part.shape}")
            
            # AUTO-DETECT 3D AND FLATTEN (SAME METHOD AS TRAIN!)
            if X_part.ndim == 3:
                print(f"      ⚠ Detected 3D data! Applying '{flatten_method}' transformation...")
                X_part = flatten_3d_to_2d(X_part, method=flatten_method)
            
            X_test_list.append(X_part)
            y_test_list.append(y_part)
            
        except Exception as e:
            print(f"      ✗ Error loading partition {part_num}: {e}")
    
    if len(X_test_list) > 0:
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        raise ValueError("No test partitions could be loaded!")
    
    print(f"      ✓ Total test: X={X_test.shape}, y={y_test.shape}")
    print(f"      ✓ Test flare rate: {y_test.mean():.2%} (real-world distribution)")
    
    # ── GET FEATURE NAMES ──
    print("\n[3/4] Extracting feature information...")
    
    # Standard SWAN-SF base feature names (24 magnetic parameters)
    base_features = [
        'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
        'SAVNCPP', 'USFLUX', 'AREA_ACR', 'MEANPOT',
        'SHRGT45', 'MEANSHR', 'MEANGAM', 'MEANGBT',
        'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTBSQ',
        'MEANJZD', 'MEANALP', 'R_VALUE', 'EPSY',
        'EPSX', 'EPSZ', 'HARPNUM_MOD', 'TIME_SINCE_LAST_FLARE'
    ]
    
    # Generate appropriate feature names based on flattening method
    n_features = X_train.shape[1]
    
    if flatten_method == 'flatten':
        # Features are timestep_feature format
        n_timesteps = 60  # From your data shape
        n_base = len(base_features)
        feature_names = []
        for t in range(n_timesteps):
            for feat in base_features[:min(n_base, n_features // n_timesteps)]:
                feature_names.append(f"{feat}_t{t}")
        # Trim or pad to match actual size
        feature_names = feature_names[:n_features]
        if len(feature_names) < n_features:
            feature_names += [f'feature_{i}' for i in range(len(feature_names), n_features)]
            
    elif flatten_method == 'concat_stats':
        # Features are stat_feature format
        stats = ['mean', 'std', 'max', 'min']
        feature_names = []
        for stat in stats:
            for feat in base_features:
                feature_names.append(f"{stat}_{feat}")
        feature_names = feature_names[:n_features]
        
    else:
        # mean, max, min, last methods preserve original 24 features
        if n_features == len(base_features):
            feature_names = base_features
        elif n_features == 1440:  # 60*24 flattened
            feature_names = [f'ts{t}_{base_features[i % 24]}' 
                           for t in range(60) for i in range(24)]
        else:
            print(f"      ⚠ Feature count mismatch: data has {n_features}, expected {len(base_features)}")
            feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"      ✓ Generated {len(feature_names)} feature names (method: {flatten_method})")
    
    # ── SUMMARY ──
    print("\n[4/4] Summary:")
    print("=" * 70)
    print(f"  Training Set:")
    print(f"    Samples:     {X_train.shape[0]:>10,}")
    print(f"    Features:    {X_train.shape[1]:>10}")
    print(f"    Flares:      {y_train.sum():>10,} ({y_train.mean():>6.1%})")
    print(f"    Non-flares:  {(len(y_train)-y_train.sum()):>10,} ({(1-y_train.mean()):>6.1%})")
    print()
    print(f"  Test Set:")
    print(f"    Samples:     {X_test.shape[0]:>10,}")
    print(f"    Features:    {X_test.shape[1]:>10}")
    print(f"    Flares:      {y_test.sum():>10,} ({y_test.mean():>6.1%})")
    print(f"    Non-flares:  {(len(y_test)-y_test.sum()):>10,} ({(1-y_test.mean()):>6.1%})")
    print("=" * 70)
    print(f"\n  ✅ Data loaded successfully! Ready for federated learning.")
    print(f"  ▶ Transformation applied: {flatten_method}")
    print(f"  ▶ Next step: python main.py --use-cleaned-data")
    print("=" * 70)
    
    return X_train, y_train, X_test, y_test, feature_names


if __name__ == '__main__':
    # Test loading with different methods
    print("\n" + "="*70)
    print("TESTING DIFFERENT 3D→2D TRANSFORMATION METHODS")
    print("="*70)
    
    for method in ['mean', 'max', 'last', 'flatten', 'concat_stats']:
        try:
            print(f"\n{'─'*70}")
            print(f"Testing method: {method}")
            print(f"{'─'*70}")
            X_train, y_train, X_test, y_test, features = load_cleaned_partition(
                combine_all_partitions=False,  # Just test partition 1 for speed
                partition_num=1,
                flatten_method=method
            )
            print(f"\n✅ SUCCESS with '{method}': Final shape {X_train.shape}")
        except Exception as e:
            print(f"\n❌ FAILED with '{method}': {e}")