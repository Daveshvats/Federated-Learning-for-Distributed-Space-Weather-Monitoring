"""
load_cleaned_data.py - Load the preprocessed Cleaned SWAN-SF dataset
"""
import pandas as pd
import numpy as np
import pickle
import os
import glob

def load_cleaned_partition(partition_num=1, 
                          data_dir='data/cleaned',
                          combine_all_partitions=True):
    """
    Load cleaned SWAN-SF dataset
    
    Args:
        partition_num: Which partition to load (1-5), or 'all'
        data_dir: Base directory containing cleaned data
        combine_all_partitions: If True, combine all 5 partitions
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    
    print("=" * 70)
    print("  SF-9: Loading Cleaned SWAN-SF Dataset")
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
        # Construct filenames
        train_file = f"Partition{part_num}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
        label_file = f"Partition{part_num}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl"
        
        train_path = os.path.join(train_dir, train_file)
        label_path = os.path.join(train_dir, label_file)
        
        if not os.path.exists(train_path):
            print(f"      ⚠ Warning: {train_file} not found, skipping")
            continue
        
        try:
            # Load features
            with open(train_path, 'rb') as f:
                X_part = pickle.load(f)
            
            # Load labels
            with open(label_path, 'rb') as f:
                y_part = pickle.load(f)
            
            print(f"      ✓ Partition {part_num}: X={X_part.shape}, y={y_part.shape}")
            
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
    
    # Standard SWAN-SF feature names (24 magnetic parameters)
    feature_names = [
        'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
        'SAVNCPP', 'USFLUX', 'AREA_ACR', 'MEANPOT',
        'SHRGT45', 'MEANSHR', 'MEANGAM', 'MEANGBT',
        'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTBSQ',
        'MEANJZD', 'MEANALP', 'R_VALUE', 'EPSY',
        'EPSX', 'EPSZ', 'HARPNUM_MOD', 'TIME_SINCE_LAST_FLARE'
    ]
    
    # Adjust if actual dimensions differ
    if X_train.shape[1] != len(feature_names):
        print(f"      ⚠ Feature count mismatch: data has {X_train.shape[1]}, expected {len(feature_names)}")
        print(f"      Using generic feature names: feature_0 to feature_{X_train.shape[1]-1}")
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    else:
        print(f"      ✓ Loaded {len(feature_names)} standard SWAN-SF features")
    
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
    print(f"  ▶ Next step: python main.py --use-cleaned-data")
    print("=" * 70)
    
    return X_train, y_train, X_test, y_test, feature_names


if __name__ == '__main__':
    # Test loading
    X_train, y_train, X_test, y_test, features = load_cleaned_partition(
        partition_num=1,
        combine_all_partitions=True
    )
    
    print("\n📊 Quick verification:")
    print(f"  X_train type: {type(X_train)}, dtype: {X_train.dtype}")
    print(f"  y_train type: {type(y_train)}, dtype: {y_train.dtype}")
    print(f"  Any NaN in X_train? {np.isnan(X_train).sum()}")
    print(f"  Any NaN in y_train? {np.isnan(y_train).sum()}")