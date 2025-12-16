import pandas as pd
import numpy as np
import os

def preprocess_and_merge(pred_path, transfers_path, output_path="stage2_input.csv"):
    print(f"--- 處理中: {pred_path} + {transfers_path} ---")
    
    # 1. 讀取 Stage 1 預測結果
    df_pred = pd.read_csv(pred_path)
    print(f"預測檔筆數: {len(df_pred)}")
    
    # 2. 讀取 MIMIC Transfers (只讀取需要的欄位)
    # careunit: 用來找 ED
    # intime: 用來找時間
    df_transfers = pd.read_csv(transfers_path, usecols=['hadm_id', 'careunit', 'intime'])
    
    # 3. 篩選 ED 資料
    # 只要 careunit 名字裡有 "Emergency"，就視為 ED
    df_ed = df_transfers[df_transfers['careunit'].str.contains('Emergency', case=False, na=False)].copy()
    
    # 轉換時間格式
    df_ed['intime'] = pd.to_datetime(df_ed['intime'])
    df_ed['hour'] = df_ed['intime'].dt.hour  # 提取 0-23 的小時
    
    # 去重：如果一個病人有多筆 ED 紀錄，取最早的那一筆 (First interaction)
    df_ed = df_ed.sort_values('intime').groupby('hadm_id').first().reset_index()
    
    # 4. 合併 (Merge)
    # 使用 inner join，只保留兩邊都有的 hadm_id
    df_merged = pd.merge(df_pred, df_ed[['hadm_id', 'hour']], on='hadm_id', how='inner')
    print(f"合併後筆數 (有 ED 時間): {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("警告：合併後沒有資料！請檢查 transfers.csv 的 hadm_id 是否與 predictions.csv 匹配。")
        return

    # 5. 確保欄位名稱對齊 Stage 2 需求
    # 我們不需要改類別 ID，保留 class_0_prob, class_1_prob 等
    # 但 Stage 2 的 simulate_day 可能需要直接讀取 r 和 y，我們可以在這裡先算好，
    # 也可以留給 stage2.py 算。
    # 為了最簡化流程，我們這裡直接算好 'r' 和 'y'，存成 stage2 直接能吃的格式。

    # 定義陽性類別 (根據您的要求：0, 1, 3 是陽性，2 是陰性)
    POSITIVE_CLASSES = [0, 1, 3]
    
    # 計算 Risk r = P(0) + P(1) + P(3)
    df_merged['r'] = 0.0
    for k in POSITIVE_CLASSES:
        col = f'class_{k}_prob'
        if col in df_merged.columns:
            df_merged['r'] += df_merged[col]
            
    df_merged['r'] = df_merged['r'].clip(0, 1) # 確保在 0-1 之間
    
    # 計算 Ground Truth y
    # 如果 true_label 在 [0, 1, 3] 裡面，y=1，否則 y=0
    df_merged['y'] = df_merged['true_label'].apply(lambda x: 1 if x in POSITIVE_CLASSES else 0)
    
    # 6. 輸出
    # 保留 hadm_id 方便 debug，留下 hour, r, y 供訓練用
    out_cols = ['hadm_id', 'hour', 'r', 'y']
    
    # 如果您還想保留原始機率供參考，可以把下面這行註解掉
    final_df = df_merged[out_cols]
    
    final_df.to_csv(output_path, index=False)
    print(f"--- 完成！已儲存至 {output_path} ---")
    print(final_df.head())

if __name__ == "__main__":
    # 1. 設定檔案路徑
    # predictions.csv 就在專案根目錄，保持不變
    PRED_CSV = "../model_output/val/predictions.csv"
    
    # transfers.csv 在深層目錄，請使用這個相對路徑 (注意要用斜線 /)
    TRANSFERS_CSV = "physionet.org/files/mimiciv/3.1/hosp/transfers.csv/transfers.csv"
    
    # 2. 檢查檔案是否存在並執行
    import os
    if os.path.exists(PRED_CSV) and os.path.exists(TRANSFERS_CSV):
        print(f"找到檔案！\n預測檔: {PRED_CSV}\n轉診檔: {TRANSFERS_CSV}")
        preprocess_and_merge(PRED_CSV, TRANSFERS_CSV)
    else:
        print("錯誤：仍然找不到檔案，請檢查下列路徑是否正確：")
        print(f"1. {os.path.abspath(PRED_CSV)}")
        print(f"2. {os.path.abspath(TRANSFERS_CSV)}")