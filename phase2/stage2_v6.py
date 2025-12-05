from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import os
import pandas as pd

# ==========================================
# 1. 基礎數學與輔助函式
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_risk_stats(values: List[float], tail_frac: float = 0.2) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    max_val = float(np.max(arr))
    k = max(1, int(math.ceil(tail_frac * len(arr))))
    top_k_vals = np.sort(arr)[-k:]
    top_k_mean = float(np.mean(top_k_vals))
    return top_k_mean, max_val

# ==========================================
# 2. 資料結構 (Config & State)
# ==========================================

@dataclass
class DayConfig:
    """
    DayConfig V6 (Final Weights Adjusted):
    - Occupancy: 1.0 * Diff^2
    - Overflow: 2.1
    - Wait: 1.0 * (1 + 0.04*t)
    - Miss: 4.0
    - FP: 3.0
    """
    C: int = 25           
    
    min_arrivals: int = 235
    max_arrivals: int = 349
    
    # 風險變動參數
    hourly_risk_step: float = 0.05 
    hourly_mortality_inc: float = 0.015 
    
    # --- 損失函數權重 ---
    occ_hi_frac: float = 0.75  
    alpha_occ: float = 1.0      # [修改] 1.0
    
    gamma_overflow: float = 2.1 # [修改] 2.1
    
    beta_wait: float = 1.0      # [修改] 1.0
    wait_risk_slope: float = 0.04 
    
    c_miss: float = 4.0         # [修改] 4.0
    c_fp: float = 3.0           # [修改] 3.0

    lambda_smooth: float = 0.1  

    # --- 初始狀態 ---
    init_occ_lo: float = 0.20
    init_occ_hi: float = 0.80
    
    # --- 隨機性控制 ---
    p_init_alpha: int = 6 
    p_init_beta: int = 3  
    seed: Optional[int] = 42

    p_split: float = 0.6  
    tau_sig: float = 0.05 
    eta_step: float = 0.1 
    
    q_norm_m: float = 1.0 
    use_q_avg: bool = True 
    tail_frac: float = 0.2 

@dataclass
class Patient:
    pid: int
    hour: int
    r: float
    y: int
    r_eff: float
    wait_time: int = 0 

@dataclass
class SimLogs:
    theta: List[float] = field(default_factory=list)
    b_t: List[float] = field(default_factory=list) 
    B_t: List[int] = field(default_factory=list)   
    
    features: List[List[float]] = field(default_factory=list)
    
    wait_hours_y1: int = 0             
    miss_count: int = 0    
    fp_count: int = 0 
    wait_cost_accumulated: float = 0.0 
    
    overflow_bed_hours: int = 0 
    max_overflow_beds: int = 0 
    
    admitted_ids: List[int] = field(default_factory=list)
    missed_ids: List[int] = field(default_factory=list)

# ==========================================
# 3. 核心邏輯
# ==========================================

def policy_theta_v6(features: List[float], w: np.ndarray) -> float:
    z = np.dot(w, np.array(features))
    return float(2.0 * sigmoid(np.array([z]))[0])

def smoothing_loss(thetas: List[float], lam: float) -> float:
    if len(thetas) <= 1 or lam <= 0: return 0.0
    diffs = np.diff(np.array(thetas))
    return float(lam * np.sum(diffs**2))

# ==========================================
# 4. 模擬主流程 V6
# ==========================================

def simulate_day(
    cfg: DayConfig,
    w: np.ndarray,
    patients: List[Patient],
):
    assert w.shape == (9,), f"weights w must be shape (9,), got {w.shape}"
    
    rng = np.random.default_rng(cfg.seed)
    
    mid = (cfg.init_occ_lo + cfg.init_occ_hi) / 2
    sigma = (cfg.init_occ_hi - cfg.init_occ_lo) / 6
    b0_raw = rng.normal(mid, sigma)
    b0 = float(np.clip(b0_raw, cfg.init_occ_lo, cfg.init_occ_hi))
    
    B_t = int(np.floor(b0 * cfg.C))

    queue: List[int] = [] 
    admitted: set = set()
    logs = SimLogs()
    
    idx_by_hour: Dict[int, List[int]] = {h: [] for h in range(24)}
    for i, p in enumerate(patients):
        if 0 <= p.hour < 24:
            idx_by_hour[p.hour].append(i)

    current_max_overflow = 0

    for t in range(24):
        
        # 1. 更新佇列
        for idx in queue:
            p_obj = patients[idx]
            p_obj.wait_time += 1 
            
            if p_obj.y == 1: p_obj.r_eff += cfg.hourly_risk_step 
            else: p_obj.r_eff -= cfg.hourly_risk_step 
            p_obj.r_eff = float(np.clip(p_obj.r_eff, 0.0, 1.0))

        # 2. 計算特徵
        b_ratio = B_t / cfg.C if cfg.C > 0 else 0.0
        
        new_idxs = idx_by_hour[t]
        new_risks = [patients[i].r for i in new_idxs]
        A_top20, A_max = get_risk_stats(new_risks, tail_frac=cfg.tail_frac)
        
        queue_risks = [patients[i].r_eff for i in queue]
        Q_top20, Q_max = get_risk_stats(queue_risks, tail_frac=cfg.tail_frac)
        
        N_arr_norm = len(new_idxs) / 20.0
        N_que_norm = len(queue) / 50.0
        T_sin = math.sin(2 * math.pi * t / 24.0)
        
        feature_vec = [1.0, b_ratio, A_top20, A_max, Q_top20, Q_max, N_arr_norm, N_que_norm, T_sin]
        
        # 3. 計算門檻
        theta_t = policy_theta_v6(feature_vec, w)

        logs.theta.append(theta_t)
        logs.b_t.append(b_ratio)
        logs.B_t.append(B_t)
        logs.features.append(feature_vec)

        # 4. 決策
        new_queue = []
        for idx in queue:
            if patients[idx].r_eff >= theta_t:
                admitted.add(idx); B_t += 1
            else:
                new_queue.append(idx)
        queue = new_queue

        for idx in new_idxs:
            if patients[idx].r >= theta_t:
                admitted.add(idx); B_t += 1
            else:
                queue.append(idx)

        # 5. 超收
        if B_t > cfg.C:
            ovf = B_t - cfg.C
            logs.overflow_bed_hours += ovf
            if ovf > current_max_overflow: current_max_overflow = ovf

    logs.max_overflow_beds = current_max_overflow

    # --- 結算 ---
    missed = [i for i, p in enumerate(patients) if (p.y == 1 and i not in admitted)]
    logs.miss_count = len(missed)
    logs.admitted_ids = sorted(list(admitted))
    logs.missed_ids = missed
    fp_count = sum(1 for i in admitted if patients[i].y == 0)
    logs.fp_count = fp_count

    # --- Loss ---
    loss_occ_accum = 0.0
    for ratio in logs.b_t:
        diff = max(0.0, ratio - cfg.occ_hi_frac)
        loss_occ_accum += (diff ** 2)
    L_occ = cfg.alpha_occ * loss_occ_accum
    
    L_ovf = cfg.gamma_overflow * logs.max_overflow_beds
    
    # Wait Logic: (1 + 0.04*t) * weight
    total_wait_cost = 0.0
    for i, p in enumerate(patients):
        if p.y == 1 and i in admitted:
            if p.wait_time > 0:
                cost = 1.0 + (p.wait_time * cfg.wait_risk_slope)
                total_wait_cost += cost
                logs.wait_hours_y1 += p.wait_time 
    
    logs.wait_cost_accumulated = total_wait_cost
    L_wait = cfg.beta_wait * total_wait_cost
    
    L_miss = cfg.c_miss * logs.miss_count
    L_fp = cfg.c_fp * logs.fp_count
    L_smooth = smoothing_loss(logs.theta, cfg.lambda_smooth)
    
    total_L = float(L_occ + L_ovf + L_wait + L_miss + L_fp + L_smooth)

    metrics = {
        "L_total": total_L,
        "L_occ": float(L_occ),
        "L_wait": float(L_wait),
        "L_overflow": float(L_ovf),
        "L_miss": float(L_miss),
        "L_fp": float(L_fp),
        "L_smooth": float(L_smooth),
        "miss_count": float(logs.miss_count),
        "fp_count": float(logs.fp_count),
        "wait_hours_y1": float(logs.wait_hours_y1),
        "overflow_bed_hours": float(logs.overflow_bed_hours),
        "max_overflow_beds": float(logs.max_overflow_beds),
        "peak_b": float(max(logs.b_t) if logs.b_t else 0.0),
        "final_B": float(B_t),
        "N_patients": float(len(patients)),
    }
    return total_L, metrics, logs, patients

# ==========================================
# 5. 資料讀取
# ==========================================
def load_patients_from_csv(csv_path: str, target_N: int, seed: Optional[int] = 2025) -> List[Patient]:
    if not os.path.exists(csv_path): raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'r' in df.columns and 'y' in df.columns:
        r_vals = df['r'].values.astype(float); y_vals = df['y'].values.astype(int)
    else:
        pos_cols = ['class_0_prob', 'class_1_prob', 'class_3_prob']
        available_cols = [c for c in pos_cols if c in df.columns]
        if not available_cols:
            pos_cols_alt = ['prob_0', 'prob_1', 'prob_3']
            available_cols = [c for c in pos_cols_alt if c in df.columns]
        if not available_cols: raise ValueError("CSV columns error")
        r_vals = df[available_cols].sum(axis=1).clip(0, 1).values.astype(float)
        if 'true_label' in df.columns:
            target_cls = [0, 1, 3]
            y_vals = df['true_label'].isin(target_cls).astype(int).values
        else: y_vals = np.zeros(len(df), dtype=int)

    if 'hour' in df.columns: hours = df['hour'].fillna(0).astype(int).values
    else:
        rng = np.random.default_rng(seed)
        hours = rng.integers(0, 24, size=len(df))

    N_total = len(df)
    rng = np.random.default_rng(seed)
    replace_flag = target_N > N_total
    indices = rng.choice(N_total, size=target_N, replace=replace_flag)

    patients = []
    for i, idx in enumerate(indices):
        patients.append(Patient(pid=i, hour=int(hours[idx]), r=float(r_vals[idx]), y=int(y_vals[idx]), r_eff=float(r_vals[idx])))
    return patients

def simulate_day_with_stage1(cfg: DayConfig, w: np.ndarray, pred_csv_path: str, sample_N: Optional[int] = None, **kwargs):
    rng = np.random.default_rng(cfg.seed)
    if sample_N is None:
        N = int(rng.integers(cfg.min_arrivals, cfg.max_arrivals + 1))
    else:
        N = int(sample_N)
    patients = load_patients_from_csv(pred_csv_path, target_N=N, seed=cfg.seed)
    return simulate_day(cfg, w, patients)