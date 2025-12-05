import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy  

try:
    import torch
except Exception:
    torch = None

# [V6 Import]
from stage2_v6 import DayConfig, simulate_day_with_stage1

# ==========================================
# 1. 評估與計算核心
# ==========================================

def calculate_clinical_metrics(logs, patients):
    y_true = np.array([p.y for p in patients])
    admitted_ids = set(logs.admitted_ids)
    
    pos_indices = [i for i, p in enumerate(patients) if p.y == 1]
    n_pos = len(pos_indices)
    neg_indices = [i for i, p in enumerate(patients) if p.y == 0]
    n_neg = len(neg_indices)
    
    tp = sum(1 for i in pos_indices if i in admitted_ids)
    fp = sum(1 for i in neg_indices if i in admitted_ids)
    
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    
    if len(logs.b_t) > 0:
        avg_occ = np.mean(logs.b_t)
    else:
        avg_occ = 0.0
        
    avg_wait = logs.wait_hours_y1 / n_pos if n_pos > 0 else 0.0
    avg_overflow = logs.overflow_bed_hours / 24.0
    
    return {
        "sensitivity": sensitivity,
        "fpr": fpr,
        "avg_occupancy": avg_occ,
        "avg_wait_hours": avg_wait,
        "avg_overflow_beds": avg_overflow,
        "n_pos": n_pos,
        "n_neg": n_neg
    }

def eval_policy_detailed(w, seeds, cfg, pred_csv=None, sampleN=None):
    agg_metrics = {
        "L_total": [], "L_occ": [], "L_wait": [], 
        "L_miss": [], "L_overflow": [], "L_fp": [] 
    }
    for s in seeds:
        cfg.seed = int(s)
        L, metrics, _, _ = simulate_day_with_stage1(
            cfg, w, pred_csv_path=pred_csv, sample_N=sampleN
        )
        for k in agg_metrics:
            agg_metrics[k].append(metrics.get(k, 0.0))
            
    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in agg_metrics.items()}

def project_w(w):
    w = np.array(w, dtype=float)
    w[1] = max(0.0, w[1]) 
    return w

# ==========================================
# 2. 繪圖功能
# ==========================================

def plot_training_details(history, save_path="loss_breakdown_v6.png"):
    if not history: return
    iters = [h['iter'] for h in history]
    l_total = [h['train_metrics']['L_total'] for h in history]
    l_occ = [h['train_metrics']['L_occ'] for h in history]
    l_wait = [h['train_metrics']['L_wait'] for h in history]
    l_miss = [h['train_metrics']['L_miss'] for h in history]
    l_ovf = [h['train_metrics']['L_overflow'] for h in history]
    l_fp = [h['train_metrics']['L_fp'] for h in history]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1); plt.plot(iters, l_total, 'k-', lw=2); plt.title('Total Loss'); plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 2); plt.plot(iters, l_occ, 'b-'); plt.title('Occupancy Loss (Ratio)'); plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3); plt.plot(iters, l_ovf, 'm-'); plt.title('Overflow Loss (Peak)'); plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 4); plt.plot(iters, l_wait, 'g-'); plt.title('Wait Loss'); plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 5); plt.plot(iters, l_miss, 'r-'); plt.title('Miss Loss'); plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6); plt.plot(iters, l_fp, 'c-'); plt.title('FP Loss'); plt.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"詳細 Loss 圖表已儲存至: {save_path}")

def plot_hourly_avg_theta(hourly_data, save_path, title):
    hours = list(range(24))
    means, stds = [], []
    for h in hours:
        vals = hourly_data[h]
        if len(vals) > 0: means.append(np.mean(vals)); stds.append(np.std(vals))
        else: means.append(0.0); stds.append(0.0)
    means = np.array(means); stds = np.array(stds)

    plt.figure(figsize=(10, 6))
    plt.plot(hours, means, 'b-o', linewidth=2, label='Average Threshold')
    plt.fill_between(hours, means - stds, means + stds, color='b', alpha=0.15, label='Std Dev')
    plt.title(title, fontsize=14)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Threshold (Theta)', fontsize=12)
    plt.ylim(-0.05, 2.05) 
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)
    plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"門檻趨勢圖已儲存至: {save_path}")

def plot_combined_hourly_thetas(theta_c25, theta_c40, theta_c60, save_path):
    hours = list(range(24))
    def process_data(theta_dict):
        means, stds = [], []
        for h in hours:
            vals = theta_dict[h]
            if len(vals) > 0: means.append(np.mean(vals)); stds.append(np.std(vals))
            else: means.append(0.0); stds.append(0.0)
        return np.array(means), np.array(stds)

    m25, s25 = process_data(theta_c25)
    m40, s40 = process_data(theta_c40)
    m60, s60 = process_data(theta_c60)

    plt.figure(figsize=(12, 7))
    plt.plot(hours, m25, 'b-o', linewidth=2, label='C=25 (Standard)')
    plt.fill_between(hours, m25 - s25, m25 + s25, color='b', alpha=0.1)
    plt.plot(hours, m40, 'r-s', linewidth=2, label='C=40 (Loose)')
    plt.fill_between(hours, m40 - s40, m40 + s40, color='r', alpha=0.1)
    plt.plot(hours, m60, 'g-^', linewidth=2, label='C=60 (Super Loose)')
    plt.fill_between(hours, m60 - s60, m60 + s60, color='g', alpha=0.1)
    
    plt.title("Comparison of Hourly Thresholds (C=25 vs 40 vs 60) - V6", fontsize=16)
    plt.xlabel('Hour of Day (0-23)', fontsize=14)
    plt.ylabel('Threshold (Theta)', fontsize=14)
    plt.ylim(-0.05, 2.05) 
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)
    plt.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"合併趨勢圖已儲存至: {save_path}")

# ==========================================
# 3. 測試功能
# ==========================================

def test_model(csv_path, w, cfg, days=30, sampleN=None, plot_suffix=""):
    print(f"\n--- 開始模型臨床測試 (模擬 {days} 天) [床位數 C={cfg.C}] ---")
    rng = np.random.default_rng(999)
    test_seeds = rng.integers(0, 100000, size=days)
    
    results = {
        "sensitivity": [], "fpr": [], "avg_occupancy": [], 
        "avg_wait_hours": [], "avg_overflow_beds": []
    }
    hourly_thetas = {h: [] for h in range(24)}
    
    for s in test_seeds:
        cfg.seed = int(s)
        _, _, logs, patients = simulate_day_with_stage1(
            cfg, w, pred_csv_path=csv_path, sample_N=sampleN
        )
        day_metrics = calculate_clinical_metrics(logs, patients)
        for k in results: results[k].append(day_metrics[k])
        if logs.theta and len(logs.theta) == 24:
            for h in range(24): hourly_thetas[h].append(logs.theta[h])
            
    print(f"{'指標 (Metric)':<25} | {'平均值 (Mean)':<10} | {'標準差 (Std)':<10}")
    print("-" * 55)
    final_report = {}
    for k, v in results.items():
        if len(v) > 0: mean_val = np.mean(v); std_val = np.std(v)
        else: mean_val = 0.0; std_val = 0.0
        final_report[k] = mean_val
        print(f"{k:<25} | {mean_val:<10.4f} | {std_val:<10.4f}")
    print("-" * 55)
    
    save_name = f"avg_theta_C{cfg.C}{plot_suffix}.png"
    plot_title = f"Average Hourly Threshold (C={cfg.C}) - V6"
    plot_hourly_avg_theta(hourly_thetas, save_name, plot_title)
    return final_report, hourly_thetas

# ==========================================
# 4. 訓練主程式
# ==========================================

def train(
    source_csv: str,          
    out_pth: str = "policy_weights_v6.pth",
    loss_fig_path: str = "loss_breakdown_v6.png",
    test_size: float = 0.2,   
    C=25,                     
    iters=800,                
    step=1.0,                 
    step_decay=0.999,         
    batch_size=10,            
    sampleN_train=None,       
    sampleN_val=None,         
    
    # --- V6 Loss Params ---
    occ_hi_frac=0.75,   
    alpha_occ=1.0,      # [Modified] 1.0
    gamma_overflow=2.1, # [Modified] 2.1
    beta_wait=1.0,      # [Modified] 1.0
    c_miss=4.0,         # [Modified] 4.0
    lambda_smooth=0.1, 
    c_fp=3.0,           # [Modified] 3.0
):
    print(f"--- 載入資料: {source_csv} ---")
    if not os.path.exists(source_csv): raise FileNotFoundError(f"找不到檔案: {source_csv}")

    df = pd.read_csv(source_csv)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    
    train_csv_temp = "temp_train_split.csv"
    val_csv_temp = "temp_val_split.csv"
    train_df.to_csv(train_csv_temp, index=False)
    val_df.to_csv(val_csv_temp, index=False)
    print(f"資料已切分: Train ({len(train_df)}), Val ({len(val_df)})")

    rng = np.random.default_rng(2025)
    
    cfg = DayConfig(
        C=C, 
        occ_hi_frac=occ_hi_frac, 
        alpha_occ=alpha_occ,
        gamma_overflow=gamma_overflow,
        beta_wait=beta_wait,
        c_miss=c_miss,
        c_fp=c_fp,
        lambda_smooth=lambda_smooth, 
        q_norm_m=1.0, tail_frac=0.2,
    )

    init_w = np.zeros(9)
    init_w[0] = 0.0
    init_w[1] = 2.0 
    
    w = project_w(init_w)
    best_w = w.copy()

    train_days = 10
    train_seeds = rng.integers(0, 100_000, size=train_days)
    best_metrics = eval_policy_detailed(best_w, train_seeds, cfg, pred_csv=train_csv_temp, sampleN=sampleN_train)
    best_loss = best_metrics['L_total']
    
    print(f"Iter 0: Total Loss={best_loss:.2f}")
    history = [{"iter": 0, "train_metrics": best_metrics, "w": best_w.tolist()}]

    for t in range(1, iters+1):
        candidates = []
        for _ in range(batch_size):
            noise = rng.normal(0, step, size=9)
            cand_w = project_w(best_w + noise)
            current_seeds = rng.integers(0, 1_000_000, size=train_days)
            cand_metrics = eval_policy_detailed(cand_w, current_seeds, cfg, pred_csv=train_csv_temp, sampleN=sampleN_train)
            cand_loss = cand_metrics['L_total']
            candidates.append((cand_loss, cand_w, cand_metrics))
        
        candidates.sort(key=lambda x: x[0])
        best_cand_loss, best_cand_w, best_cand_metrics = candidates[0]
        
        if best_cand_loss < best_loss:
            best_w = best_cand_w
            best_loss = best_cand_loss
            best_metrics = best_cand_metrics
        
        history.append({"iter": t, "train_metrics": best_metrics, "w": best_w.tolist()})
        step *= step_decay 

        if t % 20 == 0:
             print(f"Iter {t}/{iters} | Loss: {best_loss:.2f} | "
                   f"Occ: {best_metrics['L_occ']:.1f}, "
                   f"Ovf: {best_metrics['L_overflow']:.1f}, "
                   f"Wait: {best_metrics['L_wait']:.1f}, "
                   f"Miss: {best_metrics['L_miss']:.1f}, "
                   f"FP: {best_metrics['L_fp']:.1f}")

    print("\n正在繪製詳細 Loss 曲線...")
    plot_training_details(history, save_path=loss_fig_path)

    state = {"w": best_w.tolist(), "cfg": str(cfg), "history": history}
    if torch: torch.save(state, out_pth)
    else:
        import pickle
        with open(out_pth, "wb") as f: pickle.dump(state, f)
            
    print(f"\n最佳權重 (V6): {best_w}")
    print("各項 Loss 最終值 (Training Config):")
    print(f"  - Occupancy: {best_metrics['L_occ']:.2f}")
    print(f"  - Overflow:  {best_metrics['L_overflow']:.2f}")
    print(f"  - Wait:      {best_metrics['L_wait']:.2f}")
    print(f"  - Miss:      {best_metrics['L_miss']:.2f}")
    print(f"  - FP:        {best_metrics['L_fp']:.2f}")

    # ==========================================
    # 三重測試
    # ==========================================

    print(f"\n" + "="*55)
    print(f"【測試報告 A】 標準床位 (C={C}) - 訓練設定")
    print(f"="*55)
    cfg.C = C 
    _, thetas_c25 = test_model(val_csv_temp, best_w, cfg, days=50, sampleN=sampleN_val, plot_suffix="_C25")

    print(f"\n" + "="*55)
    print(f"【測試報告 B】 寬裕床位 (C=40) - 泛化測試")
    print(f"="*55)
    cfg_40 = copy.copy(cfg) 
    cfg_40.C = 40           
    _, thetas_c40 = test_model(val_csv_temp, best_w, cfg_40, days=50, sampleN=sampleN_val, plot_suffix="_C40")
    
    print(f"\n" + "="*55)
    print(f"【測試報告 C】 超寬裕床位 (C=60) - 泛化測試")
    print(f"="*55)
    cfg_60 = copy.copy(cfg) 
    cfg_60.C = 60           
    _, thetas_c60 = test_model(val_csv_temp, best_w, cfg_60, days=50, sampleN=sampleN_val, plot_suffix="_C60")
    
    plot_combined_hourly_thetas(thetas_c25, thetas_c40, thetas_c60, "combined_theta_trend_v6.png")

    if os.path.exists(train_csv_temp): os.remove(train_csv_temp)
    if os.path.exists(val_csv_temp): os.remove(val_csv_temp)
    return out_pth, state

if __name__ == "__main__":
    INPUT_CSV = "stage2_input_val.csv"
    if os.path.exists(INPUT_CSV):
        train(INPUT_CSV, iters=800, C=25)
    else:
        print(f"請先準備 {INPUT_CSV}")