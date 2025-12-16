import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


try:
    import torch
except Exception:
    torch = None

# [V6 Import] 請確保您的 stage2_v6.py 在同一個資料夾
from stage2_v6 import DayConfig, simulate_day_with_stage1

# ==========================================
# 1. 核心評估與繪圖函式 (保持不變)
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
    
    if len(logs.b_t) > 0: avg_occ = np.mean(logs.b_t)
    else: avg_occ = 0.0
        
    avg_wait = logs.wait_hours_y1 / n_pos if n_pos > 0 else 0.0
    avg_overflow = logs.overflow_bed_hours / 24.0
    
    return {
        "sensitivity": sensitivity, "fpr": fpr, "avg_occupancy": avg_occ,
        "avg_wait_hours": avg_wait, "avg_overflow_beds": avg_overflow
    }

def eval_policy_detailed(w, seeds, cfg, pred_csv=None, sampleN=None):
    agg_metrics = {
        "L_total": [], "L_occ": [], "L_wait": [], 
        "L_miss": [], "L_overflow": [], "L_fp": [] 
    }
    for s in seeds:
        cfg.seed = int(s)
        L, metrics, _, _ = simulate_day_with_stage1(cfg, w, pred_csv_path=pred_csv, sample_N=sampleN)
        for k in agg_metrics:
            agg_metrics[k].append(metrics.get(k, 0.0))
    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in agg_metrics.items()}

def project_w(w):
    w = np.array(w, dtype=float)
    w[1] = max(0.0, w[1]) 
    return w

def plot_training_details(history, save_path="loss_breakdown_v6.png"):
    if not history: return
    iters = [h['iter'] for h in history]
    keys = ['L_total', 'L_occ', 'L_overflow', 'L_wait', 'L_miss', 'L_fp']
    titles = ['Total Loss', 'Occ Loss', 'Overflow Loss', 'Wait Loss', 'Miss Loss', 'FP Loss']
    colors = ['k-', 'b-', 'm-', 'g-', 'r-', 'c-']
    
    plt.figure(figsize=(15, 10))
    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        vals = [h['train_metrics'][key] for h in history]
        plt.subplot(2, 3, i+1)
        plt.plot(iters, vals, color, lw=1.5)
        plt.title(title); plt.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Loss 圖表已儲存至: {save_path}")

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
    plt.fill_between(hours, means - stds, means + stds, color='b', alpha=0.15)
    plt.title(title, fontsize=14); plt.ylim(-0.05, 2.05)
    plt.grid(True, alpha=0.3); plt.xticks(hours); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

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
    plt.ylim(-0.05, 2.05); plt.grid(True, alpha=0.3); plt.xticks(hours); plt.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"合併趨勢圖已儲存至: {save_path}")

def test_model(csv_path, w, cfg, days=30, sampleN=None, plot_suffix=""):
    print(f"\n--- [測試] C={cfg.C} (模擬 {days} 天) ---")
    rng = np.random.default_rng(999)
    test_seeds = rng.integers(0, 100000, size=days)
    
    results = {k: [] for k in ["sensitivity", "fpr", "avg_occupancy", "avg_wait_hours", "avg_overflow_beds"]}
    hourly_thetas = {h: [] for h in range(24)}
    
    for s in test_seeds:
        cfg.seed = int(s)
        _, _, logs, patients = simulate_day_with_stage1(cfg, w, pred_csv_path=csv_path, sample_N=sampleN)
        day_metrics = calculate_clinical_metrics(logs, patients)
        for k in results: results[k].append(day_metrics[k])
        if logs.theta and len(logs.theta) == 24:
            for h in range(24): hourly_thetas[h].append(logs.theta[h])
            
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 45)
    for k, v in results.items():
        if len(v) > 0: print(f"{k:<20} | {np.mean(v):<10.4f} | {np.std(v):<10.4f}")
    
    save_name = f"avg_theta_C{cfg.C}{plot_suffix}.png"
    plot_hourly_avg_theta(hourly_thetas, save_name, f"Average Hourly Threshold (C={cfg.C})")
    return hourly_thetas

# ==========================================
# 2. 訓練邏輯 (Train Mode)
# ==========================================

def run_train(args):
    # 讀取檔案
    input_files = args.inputs
    print(f"--- [Train Mode] 讀取資料: {input_files} ---")
    
    df_list = []
    for fpath in input_files:
        if os.path.exists(fpath):
            df_list.append(pd.read_csv(fpath))
        else:
            print(f"警告: 找不到檔案 {fpath}，跳過。")
    
    if not df_list: raise ValueError("無有效資料檔")
    df = pd.concat(df_list, ignore_index=True)
    
    # 這裡會做 80/20 切分，並將資料存為暫存檔供模擬器使用
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # 暫存 CSV 供模擬器讀取
    temp_train = "temp_train_v6.csv"
    temp_val = "temp_val_v6.csv"
    train_df.to_csv(temp_train, index=False)
    val_df.to_csv(temp_val, index=False)
    
    # V6 設定
    cfg = DayConfig(
        C=25, occ_hi_frac=0.75, alpha_occ=1.0, gamma_overflow=2.1,
        beta_wait=1.0, c_miss=4.0, c_fp=3.0, lambda_smooth=0.1,
        q_norm_m=1.0, tail_frac=0.2
    )

    # 初始化權重
    rng = np.random.default_rng(2025)
    init_w = np.zeros(9); init_w[1] = 2.0
    w = project_w(init_w)
    best_w = w.copy()
    
    # 初始 Loss
    train_seeds = rng.integers(0, 100_000, size=10)
    best_metrics = eval_policy_detailed(best_w, train_seeds, cfg, pred_csv=temp_train)
    best_loss = best_metrics['L_total']
    
    print(f"Iter 0: Loss={best_loss:.4f}")
    history = [{"iter": 0, "train_metrics": best_metrics, "w": best_w.tolist()}]

    # 訓練迴圈
    step = 1.0
    for t in range(1, args.iters + 1):
        candidates = []
        for _ in range(10): # batch size = 10
            noise = rng.normal(0, step, size=9)
            cand_w = project_w(best_w + noise)
            curr_seeds = rng.integers(0, 1_000_000, size=10)
            m = eval_policy_detailed(cand_w, curr_seeds, cfg, pred_csv=temp_train)
            candidates.append((m['L_total'], cand_w, m))
        
        candidates.sort(key=lambda x: x[0])
        cand_loss, cand_w, cand_metrics = candidates[0]
        
        if cand_loss < best_loss:
            best_w = cand_w
            best_loss = cand_loss
            best_metrics = cand_metrics
        
        history.append({"iter": t, "train_metrics": best_metrics, "w": best_w.tolist()})
        step *= 0.999 # decay

        if t % 50 == 0:
            print(f"Iter {t}/{args.iters} | Loss: {best_loss:.4f} | Miss: {best_metrics['L_miss']:.2f}")

    # 儲存結果
    plot_training_details(history, save_path="loss_breakdown_v6.png")
    state = {"w": best_w.tolist(), "cfg": str(cfg), "history": history}
    
    if torch: torch.save(state, args.output)
    else:
        import pickle
        with open(args.output, "wb") as f: pickle.dump(state, f)
            
    print(f"\n訓練完成! 權重已儲存至: {args.output}")
    print(f"最佳權重: {best_w}")

    # 順便跑一次驗證 (C=25)
    test_model(temp_val, best_w, cfg, days=30, plot_suffix="_C25")

    # 清理
    if os.path.exists(temp_train): os.remove(temp_train)
    if os.path.exists(temp_val): os.remove(temp_val)

# ==========================================
# 3. 評估邏輯 (Eval Mode)
# ==========================================

def run_eval(args):
    # 檢查權重檔
    if not os.path.exists(args.model):
        print(f"錯誤: 找不到權重檔 {args.model}，請先執行 train。")
        return

    # 載入權重
    print(f"--- [Eval Mode] 載入權重: {args.model} ---")
    if torch: checkpoint = torch.load(args.model)
    else:
        import pickle
        with open(args.model, "rb") as f: checkpoint = pickle.load(f)
    
    best_w = np.array(checkpoint["w"])
    print(f"權重: {best_w}")

    # 準備資料
    if not os.path.exists(args.input):
        print(f"錯誤: 找不到資料檔 {args.input}")
        return
        
    df = pd.read_csv(args.input)
    # 使用與 Training 相同的 Random State 做分割，確保 Eval 是用 Validation Set
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    temp_val = "temp_val_eval.csv"
    val_df.to_csv(temp_val, index=False)

    # V6 標準設定
    def get_cfg(c):
        return DayConfig(
            C=c, occ_hi_frac=0.75, alpha_occ=1.0, gamma_overflow=2.1,
            beta_wait=1.0, c_miss=4.0, c_fp=3.0, lambda_smooth=0.1,
            q_norm_m=1.0, tail_frac=0.2
        )

    # 執行三重測試
    print(f"\n>>> 執行 C=25/40/60 測試 <<<")
    t25 = test_model(temp_val, best_w, get_cfg(25), days=50, plot_suffix="_C25")
    t40 = test_model(temp_val, best_w, get_cfg(40), days=50, plot_suffix="_C40")
    t60 = test_model(temp_val, best_w, get_cfg(60), days=50, plot_suffix="_C60")
    
    # 繪製合併圖
    plot_combined_hourly_thetas(t25, t40, t60, "combined_theta_trend_v6.png")
    
    if os.path.exists(temp_val): os.remove(temp_val)

# ==========================================
# 4. CLI 主程式
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Policy Training & Evaluation (V6)")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="操作模式")

    # --- Train Command ---
    # 預設輸入改為 stage2_input_val.csv
    p_train = subparsers.add_parser("train", help="訓練模型")
    p_train.add_argument("--inputs", nargs="+", default=["stage2_input_val.csv"], help="輸入的 CSV 檔案 (預設: stage2_input_val.csv)")
    p_train.add_argument("--output", type=str, default="stage2_train_v6.pth", help="輸出的權重檔名 (預設: stage2_train_v6.pth)")
    p_train.add_argument("--iters", type=int, default=800, help="訓練迭代次數")

    # --- Eval Command ---
    # 預設輸入改為 stage2_input_val.csv
    p_eval = subparsers.add_parser("eval", help="評估模型 (生成圖表)")
    p_eval.add_argument("--model", type=str, default="stage2_train_v6.pth", help="要載入的權重檔")
    p_eval.add_argument("--input", type=str, default="stage2_input_val.csv", help="用來測試的完整資料 CSV")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
