import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

def get_subset(df, subset_name):
    if subset_name == 'pooled':
        return df
    elif subset_name == 'big-3':
        allowed = ['STK-L22_4600x3500_8be98bf1', 'MGA-LX9_4000x3000_8be98bf1', 'STK-L22_3300x2400_8be98bf1']
        return df[df['cluster_id'].isin(allowed)]
    elif subset_name == 'huawei':
        prefixes = ('STK-L22', 'MGA-LX9', 'INE-LX2', 'JKM-LX2')
        return df[df['cluster_id'].apply(lambda x: str(x).startswith(prefixes))]
    else:
        raise ValueError("Unknown subset")

def compute_auc(df):
    if len(df['label'].unique()) < 2:
        return np.nan
    return roc_auc_score(df['label'], df['prob'])

def paired_stratified_bootstrap(df1, df2, n_iterations=2000, seed=42):
    np.random.seed(seed)
    
    # Must have exact same index aligning
    assert len(df1) == len(df2), "Dataframes must be same length"
    
    cancer_idx = np.where(df1['label'].values == 1)[0]
    healthy_idx = np.where(df1['label'].values == 0)[0]
    
    diffs = []
    auc1_list = []
    auc2_list = []
    
    for _ in range(n_iterations):
        c_boot = np.random.choice(cancer_idx, size=len(cancer_idx), replace=True)
        h_boot = np.random.choice(healthy_idx, size=len(healthy_idx), replace=True)
        boot_idx = np.concatenate([c_boot, h_boot])
        
        y_true = df1['label'].values[boot_idx]
        y_pred1 = df1['prob'].values[boot_idx]
        y_pred2 = df2['prob'].values[boot_idx]
        
        try:
            auc1 = roc_auc_score(y_true, y_pred1)
            auc2 = roc_auc_score(y_true, y_pred2)
            auc1_list.append(auc1)
            auc2_list.append(auc2)
            diffs.append(auc1 - auc2)
        except ValueError:
            pass # Only 1 class present, highly unlikely with stratified sampling
            
    return auc1_list, auc2_list, diffs

def analyze_subset(df1, df2, subset_name):
    s1 = get_subset(df1, subset_name)
    s2 = get_subset(df2, subset_name)
    
    n_total = len(s1)
    if n_total == 0:
        return None
        
    n_cancer = s1['label'].sum()
    n_healthy = n_total - n_cancer
    
    auc1_base = compute_auc(s1)
    auc2_base = compute_auc(s2)
    
    a1_list, a2_list, diffs = paired_stratified_bootstrap(s1, s2)
    
    ci1 = np.percentile(a1_list, [2.5, 97.5])
    ci2 = np.percentile(a2_list, [2.5, 97.5])
    ci_diff = np.percentile(diffs, [2.5, 97.5])
    
    print(f"--- Subset: {subset_name.upper()} ---")
    print(f"Images: {n_total} (Cancer: {n_cancer}, Healthy: {n_healthy})")
    print(f"Model A AUC: {auc1_base:.4f} [95% CI: {ci1[0]:.4f}, {ci1[1]:.4f}]")
    print(f"Model B AUC: {auc2_base:.4f} [95% CI: {ci2[0]:.4f}, {ci2[1]:.4f}]")
    print(f"Difference (A - B): {np.mean(diffs):.4f} [95% CI: {ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")
    print()

def main():
    file1 = sys.argv[1] if len(sys.argv) > 1 else 'oof_ROI.csv'
    file2 = sys.argv[2] if len(sys.argv) > 2 else 'oof_CENTER.csv'
    
    print(f"Comparing: {file1} (Model A) vs {file2} (Model B)\n")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Sort both to ensure identical alignment
    df1 = df1.sort_values('path').reset_index(drop=True)
    df2 = df2.sort_values('path').reset_index(drop=True)
    
    analyze_subset(df1, df2, 'big-3')
    analyze_subset(df1, df2, 'huawei')
    analyze_subset(df1, df2, 'pooled')

if __name__ == "__main__":
    main()
