import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.json':
        print(f"Loading JSON diagnostics data from {file_path}...")
        df = pd.read_json(file_path)
    elif ext == '.csv':
        print(f"Loading CSV diagnostics data from {file_path}...")
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Must be .csv or .json")
    return df

def preprocess_data(df):
    df_clean = df.copy()
    
    # Convert booleans to floats (1.0/0.0), keeping NaNs intact
    for col in df_clean.columns:
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(float)
        elif df_clean[col].dtype == object:
            # check if it contains boolean-like values
            unique_vals = df_clean[col].dropna().unique()
            if set(unique_vals).issubset({True, False, 'True', 'False', 1.0, 0.0, 1, 0}):
                df_clean[col] = df_clean[col].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0, 1.0: 1.0, 0.0: 0.0, 1: 1.0, 0: 0.0})
                
    return df_clean

def main():
    parser = argparse.ArgumentParser(description="RAM-VISLAM Diagnostics & Metrics Analysis Tool")
    parser.add_argument("--file", type=str, default="/home/rv/RAM_VI_SLAM/output/diagnostics_20260626_094637.csv",
                        help="Path to diagnostics CSV or JSON file")
    parser.add_argument("--outdir", type=str, default="/home/rv/RAM_VI_SLAM/output/analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    # If the default CSV doesn't exist, try the JSON version
    file_path = args.file
    if not os.path.exists(file_path):
        if file_path.endswith('.csv'):
            json_fallback = file_path.replace('.csv', '.json')
            if os.path.exists(json_fallback):
                file_path = json_fallback
        elif file_path.endswith('.json'):
            csv_fallback = file_path.replace('.json', '.csv')
            if os.path.exists(csv_fallback):
                file_path = csv_fallback
                
    if not os.path.exists(file_path):
        print(f"Error: Diagnostics file not found at {args.file} (or fallback path).")
        return
        
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load and preprocess
    df = load_data(file_path)
    df_clean = preprocess_data(df)
    
    # Identify numeric columns (including converted booleans)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out columns with zero standard deviation (constant columns)
    valid_cols = [c for c in numeric_cols if df_clean[c].dropna().std() > 0 or c in ["frame_id"]]
    
    # 1. Plot every metric over time
    print("Generating plots for every metric over time...")
    groups = {
        "Tracking & Odometry": [
            "total_frame_time", "fps", "tracking_success", "rgbd_odom_error", "rgbd_odom_time"
        ],
        "ICP Alignment": [
            "icp_fitness", "icp_rmse", "icp_correspondences", "icp_time"
        ],
        "IMU & ESKF Propagation": [
            "imu_propagation_time", "eskf_innovation_norm", 
            "accel_var_x", "accel_var_y", "accel_var_z",
            "gyro_var_x", "gyro_var_y", "gyro_var_z"
        ],
        "Depth & Image Info": [
            "valid_depth_pct", "num_features", "image_width", "image_height"
        ],
        "Surfel Map Stats": [
            "active_surfels", "spawned_surfels", "fused_surfels", "pruned_surfels", "mapping_time"
        ],
        "Keyframes & Loop Closures": [
            "kf_inserted", "kf_id", "loop_candidate_found", 
            "loop_accepted", "loop_similarity", "pgo_time"
        ],
        "System Diagnostics": [
            "cpu_usage", "ram_usage_mb", "gpu_memory_used_mb"
        ]
    }
    
    # Add any remaining numeric/boolean columns not in the predefined groups to an "Other Metrics" group
    all_grouped_cols = []
    for g_cols in groups.values():
        all_grouped_cols.extend(g_cols)
    
    other_cols = [
        c for c in df_clean.columns 
        if c not in all_grouped_cols and c not in ["frame_id", "timestamp"] 
        and c in numeric_cols
    ]
    if other_cols:
        groups["Other Metrics"] = other_cols

    # Setup plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    for group_name, cols in groups.items():
        # Filter for columns that exist in the dataframe and are not completely null
        cols_to_plot = [c for c in cols if c in df_clean.columns and df_clean[c].notna().any()]
        if not cols_to_plot:
            continue
            
        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(14, 2.5 * len(cols_to_plot)), sharex=True)
        if len(cols_to_plot) == 1:
            axes = [axes]
            
        fig.suptitle(f"RAM-VISLAM Diagnostics: {group_name} Over Time", fontsize=16, fontweight='bold', y=0.98)
        
        for ax, col in zip(axes, cols_to_plot):
            # Plot line
            x_data = df_clean["frame_id"].to_numpy()
            y_data = df_clean[col].to_numpy()
            
            # Use step style for binary/boolean variables
            unique_non_nan = df_clean[col].dropna().unique()
            is_binary = len(unique_non_nan) <= 2 and set(unique_non_nan).issubset({0.0, 1.0})
            
            if is_binary:
                ax.step(x_data, y_data, label=col, color='tab:orange', where='mid', alpha=0.8, linewidth=1.5)
                # Add scatter dots on event triggered
                ax.scatter(x_data[y_data == 1.0], y_data[y_data == 1.0], color='tab:red', s=15, zorder=5, label='Event (True)')
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['False', 'True'])
            else:
                ax.plot(x_data, y_data, label=col, color='tab:blue', alpha=0.8, linewidth=1.2)
                
            ax.set_ylabel(col, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
            
            # Highlight max/min values if not boolean
            if not is_binary and len(y_data[~np.isnan(y_data)]) > 0:
                max_idx = np.nanargmax(y_data)
                min_idx = np.nanargmin(y_data)
                ax.scatter(x_data[max_idx], y_data[max_idx], color='red', s=25, zorder=10)
                ax.annotate(f"Max: {y_data[max_idx]:.3g}", (x_data[max_idx], y_data[max_idx]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', fontweight='bold')
                
        axes[-1].set_xlabel("Frame ID", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plot_name = group_name.lower().replace(" & ", "_").replace(" ", "_")
        plot_path = os.path.join(args.outdir, f"{plot_name}_over_time.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved temporal plot: {plot_path}")

    # 2. Correlation Matrix
    print("Computing correlation matrix...")
    valid_numeric_cols = [c for c in valid_cols if c not in ["frame_id", "timestamp"]]
    corr_matrix = df_clean[valid_numeric_cols].corr()
    
    # Save correlation matrix to CSV
    corr_csv_path = os.path.join(args.outdir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)
    print(f"Saved correlation matrix CSV: {corr_csv_path}")
    
    # Save correlation heatmap
    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
                mask=mask, annot_kws={"size": 6, "weight": "bold"}, 
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
    plt.title("RAM-VISLAM Metric Correlation Heatmap (Lower Triangle)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(args.outdir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"Saved correlation heatmap: {heatmap_path}")

    # 3 & 5. Identify variables predicting performance and rank metrics importance
    print("Running performance predictors and metrics ranking...")
    report_lines = []
    report_lines.append("# RAM-VISLAM Diagnostics & Metrics Analysis Report")
    report_lines.append(f"- **Source File**: `{file_path}`")
    report_lines.append(f"- **Total Frames Processed**: {len(df_clean)}")
    if 'kf_inserted' in df_clean.columns:
        report_lines.append(f"- **Keyframes Triggered**: {int(df_clean['kf_inserted'].dropna().sum())}")
    if 'active_surfels' in df_clean.columns:
        valid_surfels = df_clean['active_surfels'].dropna()
        if not valid_surfels.empty:
            report_lines.append(f"- **Final Active Surfels**: {int(valid_surfels.iloc[-1]):,}")
    
    target_col = "total_frame_time"
    
    report_lines.append("\n## Performance Predictors Analysis")
    
    if target_col in corr_matrix.columns:
        corr_with_target = corr_matrix[target_col].sort_values(ascending=False)
        report_lines.append(f"\n### Linear Correlation with `{target_col}` (Processing Bottlenecks)")
        report_lines.append("Positive values indicate that as the metric increases, the frame processing time increases (bottleneck).")
        report_lines.append("\n| Metric | Pearson Correlation | Description |")
        report_lines.append("|---|---|---|")
        for col, val in corr_with_target.items():
            if col != target_col:
                desc = "Strong positive correlation (bottleneck)" if val > 0.5 else \
                       "Moderate positive correlation" if val > 0.2 else \
                       "Strong negative correlation (increases speed)" if val < -0.5 else \
                       "Moderate negative correlation" if val < -0.2 else "Weak correlation"
                report_lines.append(f"| `{col}` | {val:.4f} | {desc} |")
                
        # Random Forest Regressor for metric importance ranking
        exclude_features = ["frame_id", "timestamp", "total_frame_time", "fps", "kf_id"]
        feature_cols = [c for c in valid_numeric_cols if c not in exclude_features]
        # Filter out feature columns with more than 50% NaNs
        feature_cols = [c for c in feature_cols if df_clean[c].isna().sum() < 0.5 * len(df_clean)]
        
        # Clean rows with NaNs in feature cols or target
        clean_df = df_clean[feature_cols + [target_col]].dropna()
        
        if len(clean_df) > 10:
            X = clean_df[feature_cols]
            y = clean_df[target_col]
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            report_lines.append("\n### Metric Importance Ranking (Random Forest Regressor)")
            report_lines.append(f"Features ranked by their predictive power for `{target_col}` (explaining execution time variance):")
            report_lines.append("\n| Rank | Metric | Feature Importance | Cumulative Importance |")
            report_lines.append("|---|---|---|---|")
            cum_importance = 0.0
            for rank, idx in enumerate(indices):
                cum_importance += importances[idx]
                report_lines.append(f"| {rank+1} | `{feature_cols[idx]}` | {importances[idx]:.4f} | {cum_importance:.4f} |")
                
            # Plot Feature Importances
            plt.figure(figsize=(12, 6))
            sns.barplot(x=importances[indices][:15], y=[feature_cols[i] for i in indices[:15]], palette="viridis")
            plt.title("Top 15 Metrics Predicting Frame Processing Time", fontsize=14, fontweight='bold')
            plt.xlabel("Random Forest Feature Importance")
            plt.tight_layout()
            importance_plot_path = os.path.join(args.outdir, "performance_predictors_ranking.png")
            plt.savefig(importance_plot_path, dpi=200)
            plt.close()
            print(f"Saved feature importance plot: {importance_plot_path}")
        else:
            report_lines.append("\n### [Warning] Too few clean data points for Random Forest training.")

    # 4. Outliers and Anomalies Detection
    print("Detecting outliers and anomalies...")
    report_lines.append("\n## Outliers & Anomalies Detection")
    
    # 1. Processing Bottlenecks (total_frame_time > mean + 3 * std)
    mean_time = df_clean["total_frame_time"].mean()
    std_time = df_clean["total_frame_time"].std()
    bottleneck_thresh = mean_time + 3 * std_time
    bottleneck_frames = df_clean[df_clean["total_frame_time"] > bottleneck_thresh]
    
    report_lines.append(f"\n### 1. Temporal Processing Bottlenecks")
    report_lines.append(f"- **Baseline Statistics**: Mean = `{mean_time:.3f}s`, StdDev = `{std_time:.3f}s`")
    report_lines.append(f"- **Outlier Threshold** (Mean + 3*StdDev): `{bottleneck_thresh:.3f}s`")
    report_lines.append(f"- **Detected Anomalous Frames**: `{len(bottleneck_frames)}` ({len(bottleneck_frames)/len(df_clean)*100:.2f}%)")
    
    if not bottleneck_frames.empty:
        report_lines.append("\n| Frame ID | Processing Time | ICP Time | Mapping Time | PGO Time (Loop) | Loop Similarity |")
        report_lines.append("|---|---|---|---|---|---|")
        top_bottlenecks = bottleneck_frames.sort_values(by="total_frame_time", ascending=False).head(15)
        for _, row in top_bottlenecks.iterrows():
            pgo_val = f"{row['pgo_time']:.3f}s" if pd.notna(row.get('pgo_time')) else "N/A"
            loop_sim = f"{row['loop_similarity']:.4f}" if pd.notna(row.get('loop_similarity')) else "N/A"
            report_lines.append(f"| {int(row['frame_id'])} | {row['total_frame_time']:.3f}s | {row.get('icp_time', 0.0):.3f}s | {row.get('mapping_time', 0.0):.3f}s | {pgo_val} | {loop_sim} |")

    # 2. Tracking Failures
    if "tracking_success" in df_clean.columns:
        tracking_failures = df_clean[df_clean["tracking_success"] == 0.0]
        report_lines.append(f"\n### 2. Tracking Failures")
        report_lines.append(f"- **Total Failures**: `{len(tracking_failures)}` ({len(tracking_failures)/len(df_clean)*100:.2f}%)")
        if not tracking_failures.empty:
            report_lines.append("\n| Frame ID | Active Surfels | Valid Depth % | Accel Var Norm | Gyro Var Norm |")
            report_lines.append("|---|---|---|---|---|")
            for _, row in tracking_failures.head(15).iterrows():
                acc_var = np.sqrt(row.get('accel_var_x', 0)**2 + row.get('accel_var_y', 0)**2 + row.get('accel_var_z', 0)**2)
                gyro_var = np.sqrt(row.get('gyro_var_x', 0)**2 + row.get('gyro_var_y', 0)**2 + row.get('gyro_var_z', 0)**2)
                report_lines.append(f"| {int(row['frame_id'])} | {row.get('active_surfels', 0):,} | {row.get('valid_depth_pct', 0):.2f}% | {acc_var:.4f} | {gyro_var:.6f} |")

    # 3. ICP Registration Anomalies
    if "icp_rmse" in df_clean.columns:
        icp_rmse_valid = df_clean["icp_rmse"].dropna()
        if not icp_rmse_valid.empty:
            mean_rmse = icp_rmse_valid.mean()
            std_rmse = icp_rmse_valid.std()
            rmse_thresh = mean_rmse + 3 * std_rmse
            rmse_outliers = df_clean[df_clean["icp_rmse"] > rmse_thresh]
            
            report_lines.append(f"\n### 3. ICP Alignment Quality Outliers")
            report_lines.append(f"- **Baseline Statistics**: Mean RMSE = `{mean_rmse:.4f}m`, StdDev = `{std_rmse:.4f}m`")
            report_lines.append(f"- **Outlier Threshold** (Mean + 3*StdDev): `{rmse_thresh:.4f}m`")
            report_lines.append(f"- **Poor ICP Alignments Detected**: `{len(rmse_outliers)}` ({len(rmse_outliers)/len(df_clean)*100:.2f}%)")
            
            if not rmse_outliers.empty:
                report_lines.append("\n| Frame ID | ICP RMSE | ICP Fitness | Correspondences | Spawned Surfels |")
                report_lines.append("|---|---|---|---|---|")
                top_rmse = rmse_outliers.sort_values(by="icp_rmse", ascending=False).head(15)
                for _, row in top_rmse.iterrows():
                    report_lines.append(f"| {int(row['frame_id'])} | {row['icp_rmse']:.4f}m | {row.get('icp_fitness', 0.0):.4f} | {int(row.get('icp_correspondences', 0)):,} | {row.get('spawned_surfels', 0):,} |")

    # 4. Multivariate Machine Learning Anomaly Detection (Isolation Forest)
    clean_df_anom = df_clean[valid_numeric_cols].dropna()
    if len(clean_df_anom) > 50:
        contamination_rate = 0.005 # 0.5% outliers
        clf = IsolationForest(contamination=contamination_rate, random_state=42)
        preds = clf.fit_predict(clean_df_anom)
        anomaly_df = df_clean.loc[clean_df_anom.index[preds == -1]]
        
        report_lines.append(f"\n### 4. Multivariate Anomalies (Isolation Forest)")
        report_lines.append(f"Detected anomalies using a multivariate **Isolation Forest** model (contamination rate: `{contamination_rate*100}%`):")
        report_lines.append(f"- **Multivariate Anomalies Detected**: `{len(anomaly_df)}` frames")
        
        if not anomaly_df.empty:
            report_lines.append("\n| Frame ID | Total Frame Time | ICP RMSE | ESKF Innovation | Active Surfels | CPU Usage | RAM Usage |")
            report_lines.append("|---|---|---|---|---|---|---|")
            for _, row in anomaly_df.head(15).iterrows():
                eskf_val = f"{row['eskf_innovation_norm']:.4f}" if pd.notna(row.get('eskf_innovation_norm')) else "N/A"
                rmse_val = f"{row['icp_rmse']:.4f}m" if pd.notna(row.get('icp_rmse')) else "N/A"
                report_lines.append(f"| {int(row['frame_id'])} | {row['total_frame_time']:.3f}s | {rmse_val} | {eskf_val} | {row.get('active_surfels', 0):,} | {row.get('cpu_usage', 0):.1f}% | {row.get('ram_usage_mb', 0):.1f}MB |")

    # Save Markdown report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.outdir, "diagnostics_summary_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved Markdown Summary Report: {report_path}")
    
    # Also save a text report version for compatibility
    txt_report_path = os.path.join(args.outdir, "diagnostics_summary_report.txt")
    with open(txt_report_path, "w") as f:
        # Strip markdown syntax for the .txt version
        txt_content = report_text.replace('# ', '').replace('## ', '\n').replace('### ', '\n').replace('**', '').replace('`', '')
        f.write(txt_content)
    print(f"Saved Text Summary Report: {txt_report_path}")

    # Generate a brief console summary
    print("\n" + "="*50)
    print("       RAM-VISLAM ANALYSIS CONSOLE SUMMARY")
    print("="*50)
    print(f"Loaded: {file_path}")
    print(f"Total Frames: {len(df_clean)}")
    print(f"Mean frame time: {mean_time:.3f}s ({1.0/mean_time:.1f} FPS)")
    print(f"Bottleneck Threshold: {bottleneck_thresh:.3f}s (Mean + 3*Std)")
    print(f"Bottlenecks Detected: {len(bottleneck_frames)} frames")
    if not bottleneck_frames.empty:
        print(f"Max frame time: {df_clean['total_frame_time'].max():.3f}s")
    print(f"Analysis plots and reports saved to: {args.outdir}")
    print("="*50)

if __name__ == "__main__":
    main()
