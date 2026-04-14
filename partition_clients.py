"""
partition_clients.py - Partition data into regional client shards
FIXED VERSION: Handles both HARPNUM-based and index-based partitioning
"""
import numpy as np
from collections import Counter

def partition_data(X_train, y_train, harpnum_mod=None, n_clients=6, 
                   min_samples_per_client=100, verbose=True):
    """
    Partition training data into non-IID client shards based on HARPNUM_MOD
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        harpnum_mod: HARP numbers for each sample (n_samples,) or None
        n_clients: Number of regional clients
        min_samples_per_client: Minimum samples per client (warn if less)
        verbose: Print partition statistics
    
    Returns:
        shards: List of (X_client, y_client) tuples
    """
    
    if verbose:
        print("[Partition] Splitting training data into regional client shards ...\n")
    
    n_samples = len(y_train)
    
    # ── CASE 1: No HARPNUM provided → Use balanced random partition ──
    if harpnum_mod is None or len(harpnum_mod) == 0:
        if verbose:
            print("      ⚠ No HARPNUM_MOD provided. Using balanced random partition.")
        
        return _partition_balanced_random(X_train, y_train, n_clients, verbose)
    
    # ── CASE 2: HARPNUM provided but might be dummy/synthetic ──
    harpnum_mod = np.array(harpnum_mod).flatten()
    
    # Check if HARPNUM looks like real data or just sequential indices
    unique_harps = np.unique(harpnum_mod)
    n_unique = len(unique_harps)
    
    if verbose:
        print(f"      [Debug] HARPNUM_MOD stats:")
        print(f"              Unique values: {n_unique}")
        print(f"              Range: [{harpnum_mod.min()}, {harpnum_mod.max()}]")
        print(f"              Value counts: {Counter(harpnum_mod).most_common(5)}")
    
    # Detect if HARPNUM is just sequential dummy data (0,1,2,3,4,5,0,1,2,...)
    is_dummy_sequential = (
        n_unique <= n_clients * 2 and 
        np.all(np.sort(unique_harps) == np.arange(n_unique)) and
        n_samples > n_unique * 10  # Many repeats of same values
    )
    
    if is_dummy_sequential:
        if verbose:
            print(f"      ⚠ Detected sequential dummy HARPNUM. Using balanced partition instead.")
        return _partition_balanced_random(X_train, y_train, n_clients, verbose)
    
    # ── CASE 3: Real HARPNUM data ──
    try:
        # Map HARPNUMs to clients (modulo n_clients)
        client_ids = harpnum_mod % n_clients
        
        shards = []
        client_names = [
            "Americas (NASA/NOAA)",
            "Europe (ESA/PROBA-2)", 
            "Asia-Pacific (JAXA)",
            "South Asia (ISRO)",
            "East Asia (KASI)",
            "Oceania (BoM)"
        ]
        
        for client_id in range(n_clients):
            mask = (client_ids == client_id)
            X_client = X_train[mask]
            y_client = y_train[mask]
            
            if len(X_client) < min_samples_per_client:
                if verbose:
                    print(f"      ⚠ Client {client_id} ({client_names[client_id]}) has "
                          f"{len(X_client)} samples (< {min_samples_per_client}). Skipping.")
                continue
            
            shards.append((X_client, y_client))
            
            flare_rate = y_client.mean() * 100
            if verbose:
                print(f"[Partition] {client_names[client_id]:30s} | "
                      f"n={len(X_client):>6,} | flare rate: {flare_rate:.1f}%")
        
        if len(shards) == 0:
            if verbose:
                print("      ✗ ERROR: All clients have insufficient samples!")
                print("      → Falling back to balanced random partition...")
            return _partition_balanced_random(X_train, y_train, n_clients, verbose)
        
        return shards
        
    except Exception as e:
        if verbose:
            print(f"      ✗ Error in HARPNUM partitioning: {e}")
            print("      → Falling back to balanced random partition...")
        return _partition_balanced_random(X_train, y_train, n_clients, verbose)


def _partition_balanced_random(X_train, y_train, n_clients=6, verbose=True):
    """
    Fallback: Create balanced partitions using stratified sampling
    Ensures every client gets samples with realistic flare rates
    """
    
    if verbose:
        print("\n      [Fallback] Using STRATIFIED BALANCED partition...")
    
    np.random.seed(42)  # Reproducibility
    
    # Separate by class
    flare_mask = (y_train == 1)
    noflare_mask = (y_train == 0)
    
    X_flare = X_train[flare_mask]
    y_flare = y_train[flare_mask]
    X_noflare = X_train[noflare_mask]
    y_noflare = y_train[noflare_mask]
    
    n_flares = len(y_flare)
    n_noflares = len(y_noflare)
    
    if verbose:
        print(f"      Total flares: {n_flares:,}, Non-flares: {n_noflares:,}")
    
    # Shuffle both classes
    flare_perm = np.random.permutation(n_flares)
    noflare_perm = np.random.permutation(n_noflares)
    
    X_flare_shuffled = X_flare[flare_perm]
    y_flare_shuffled = y_flare[flare_perm]
    X_noflare_shuffled = X_noflare[noflare_perm]
    y_noflare_shuffled = y_noflare[noflare_perm]
    
    # Distribute flares roughly equally among clients
    shards = []
    client_names = [
        "Americas (NASA/NOAA)",
        "Europe (ESA/PROBA-2)", 
        "Asia-Pacific (JAXA)",
        "South Asia (ISRO)",
        "East Asia (KASI)",
        "Oceania (BoM)"
    ]
    
    flare_per_client = max(1, n_flares // n_clients)
    noflare_per_client = max(1, n_noflares // n_clients)
    
    for client_id in range(n_clients):
        start_f = client_id * flare_per_client
        end_f = min((client_id + 1) * flare_per_client, n_flares)
        
        start_nf = client_id * noflare_per_client
        end_nf = min((client_id + 1) * noflare_per_client, n_noflares)
        
        # Handle last client getting remainder
        if client_id == n_clients - 1:
            end_f = n_flares
            end_nf = n_noflares
        
        X_client_flare = X_flare_shuffled[start_f:end_f]
        y_client_flare = y_flare_shuffled[start_f:end_f]
        
        X_client_noflare = X_noflare_shuffled[start_nf:end_nf]
        y_client_noflare = y_noflare_shuffled[start_nf:end_nf]
        
        # Combine flare + noflare for this client
        X_client = np.vstack([X_client_flare, X_client_noflare]) if \
                   (len(X_client_flare) > 0 and len(X_client_noflare) > 0) else \
                   (X_client_flare if len(X_client_flare) > 0 else X_client_noflare)
                   
        y_client = np.concatenate([y_client_flare, y_client_noflare]) if \
                   (len(y_client_flare) > 0 and len(y_client_noflare) > 0) else \
                   (y_client_flare if len(y_client_flare) > 0 else y_client_noflare)
        
        # Shuffle client data
        perm = np.random.permutation(len(X_client))
        X_client = X_client[perm]
        y_client = y_client[perm]
        
        shards.append((X_client, y_client))
        
        flare_rate = y_client.mean() * 100
        if verbose:
            print(f"[Partition] {client_names[client_id]:30s} | "
                  f"n={len(X_client):>6,} | flare rate: {flare_rate:.1f}%")
    
    if verbose:
        total_in_shards = sum(len(s[0]) for s in shards)
        print(f"\n      ✓ Distributed {total_in_shards:,} samples across {len(shards)} clients")
    
    return shards


def partition_data_dirichlet(X_train, y_train, alpha=0.5, n_clients=6):
    """
    Alternative: Dirichlet-based non-IID partitioning
    Creates more realistic data heterogeneity
    """
    
    print(f"\n[Partition] Using Dirichlet partitioning (α={alpha})...")
    
    np.random.seed(42)
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    
    # Generate Dirichlet proportions for each client-class combination
    proportions = np.random.dirichlet([alpha] * n_clients, size=n_classes)
    
    shards = []
    client_names = [
        "Americas (NASA/NOAA)",
        "Europe (ESA/PROBA-2)", 
        "Asia-Pacific (JAXA)",
        "South Asia (ISRO)",
        "East Asia (KASI)",
        "Oceania (BoM)"
    ]
    
    for client_id in range(n_clients):
        X_client_list = []
        y_client_list = []
        
        for c in range(n_classes):
            class_mask = (y_train == c)
            X_class = X_train[class_mask]
            y_class = y_train[class_mask]
            
            # Number of samples for this client from this class
            n_for_client = int(proportions[c][client_id] * len(y_class))
            
            if n_for_client > 0:
                indices = np.random.choice(len(y_class), size=n_for_client, replace=False)
                X_client_list.append(X_class[indices])
                y_client_list.append(y_class[indices])
        
        if len(X_client_list) > 0:
            X_client = np.vstack(X_client_list)
            y_client = np.concatenate(y_client_list)
            
            # Shuffle
            perm = np.random.permutation(len(X_client))
            X_client = X_client[perm]
            y_client = y_client[perm]
            
            shards.append((X_client, y_client))
            
            flare_rate = y_client.mean() * 100
            print(f"[Partition] {client_names[client_id]:30s} | "
                  f"n={len(X_client):>6,} | flare rate: {flare_rate:.1f}%")
        else:
            print(f"[Partition] Warning: {client_names[client_id]} has 0 samples")
    
    return shards


if __name__ == '__main__':
    # Test partitioning
    from load_cleaned_data import load_cleaned_partition
    
    print("="*70)
    print("TESTING PARTITIONING WITH CLEANED DATASET")
    print("="*70)
    
    X_train, y_train, X_test, y_test, features = load_cleaned_partition(
        combine_all_partitions=False,
        partition_num=1,
        flatten_method='mean'
    )
    
    print("\n" + "="*70)
    print("Testing HARPNUM-based partition:")
    print("="*70)
    shards = partition_data(X_train, y_train, harpnum_mod=None)  # Test fallback
    
    print(f"\n✅ Created {len(shards)} client shards")
    for i, (X_c, y_c) in enumerate(shards):
        print(f"   Client {i}: {X_c.shape}, flare rate: {y_c.mean():.2%}")