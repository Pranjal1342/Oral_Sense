import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

def compute_aucs(df, clusters):
    try:
        pooled = roc_auc_score(df['label'], df['prob'])
    except ValueError:
        pooled = np.nan
    within = []
    for c in clusters:
        cdf = df[df['cluster_id'] == c]
        if len(cdf['label'].unique()) > 1:
            try: within.append(roc_auc_score(cdf['label'], cdf['prob']))
            except ValueError: pass
    return pooled, np.mean(within) if within else np.nan

def analyze_oof(csv_path):
    print(f"\n{'='*50}\nAnalyzing: {csv_path}\n{'='*50}")
    try: df = pd.read_csv(csv_path)
    except FileNotFoundError: return None, None
        
    counts = df.groupby(['cluster_id', 'label']).size().unstack(fill_value=0)
    valid_clusters = counts[(counts[0] >= 5) & (counts[1] >= 5)].index
    df_eval = df[df['cluster_id'].isin(valid_clusters)].copy()
    
    print(f"Evaluable images: {len(df_eval)} (Healthy: {sum(df_eval['label'] == 0)}, Cancer: {sum(df_eval['label'] == 1)})")
    p, w = compute_aucs(df_eval, valid_clusters)
    print(f"Pooled AUC: {p:.4f} | Mean Within-Cluster: {w:.4f}")
    return df_eval, valid_clusters

def paired_group_bootstrap(df_roi, df_center, valid_clusters, n_bootstraps=1000, seeds=[42, 43, 44]):
    print("\n--- Paired Group-Bootstrap (ROI vs CENTER) ---")
    all_diffs = []
    
    for s in seeds:
        np.random.seed(s)
        seed_diffs = []
        for _ in range(n_bootstraps):
            boot_clusters = np.random.choice(valid_clusters, size=len(valid_clusters), replace=True)
            
            p_roi_list = []
            l_roi_list = []
            p_cen_list = []
            
            for c in boot_clusters:
                c_roi = df_roi[df_roi['cluster_id'] == c]
                c_cen = df_center[df_center['cluster_id'] == c]
                p_roi_list.extend(c_roi['prob'].values)
                p_cen_list.extend(c_cen['prob'].values)
                l_roi_list.extend(c_roi['label'].values)
                
            try:
                auc_roi = roc_auc_score(l_roi_list, p_roi_list)
                auc_cen = roc_auc_score(l_roi_list, p_cen_list)
                seed_diffs.append(auc_roi - auc_cen)
                all_diffs.append(auc_roi - auc_cen)
            except ValueError:
                pass
                
        lower = np.percentile(seed_diffs, 2.5)
        upper = np.percentile(seed_diffs, 97.5)
        print(f"Seed {s}: Mean Diff = {np.mean(seed_diffs):.4f} [95% CI: {lower:.4f}, {upper:.4f}]")
        
    lower_all = np.percentile(all_diffs, 2.5)
    upper_all = np.percentile(all_diffs, 97.5)
    mean_all = np.mean(all_diffs)
    print(f"\nOverall (N={len(all_diffs)}): Mean Diff = {mean_all:.4f} [95% CI: {lower_all:.4f}, {upper_all:.4f}]")
    
    if lower_all > 0:
        print("CONCLUSION: ROI is statistically significantly better than CENTER (CI > 0). The WLI crop is likely leaking cues.")
    elif upper_all < 0:
        print("CONCLUSION: CENTER is statistically significantly better than ROI (CI < 0). Unexpected.")
    else:
        print("CONCLUSION: No statistically significant difference between ROI and CENTER. Crop cue is likely not driving the AUC.")

if __name__ == "__main__":
    r_res = analyze_oof("oof_ROI.csv")
    c_res = analyze_oof("oof_CENTER.csv")
    
    if r_res[0] is not None and c_res[0] is not None:
        # Check alignment
        df_roi, vc = r_res
        df_cen, _ = c_res
        
        # Sort both by path to ensure exact pairing
        df_roi = df_roi.sort_values('path').reset_index(drop=True)
        df_cen = df_cen.sort_values('path').reset_index(drop=True)
        
        assert (df_roi['path'] == df_cen['path']).all(), "Paths do not align!"
        assert (df_roi['label'] == df_cen['label']).all(), "Labels do not align!"
        
        paired_group_bootstrap(df_roi, df_cen, vc)
