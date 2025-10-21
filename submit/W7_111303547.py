# === 任務 1️⃣：載入資料 (load_data) ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # 雖然測試不需要，但原始腳本有，保留

def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    print('缺失值總數:', missing_count)
    return df, missing_count

# === 任務 2️⃣：處理缺失值 (handle_missing) ===
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # TODO 2.2: 以 Embarked 眾數填補
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

# === 任務 3️⃣：移除異常值 (remove_outliers) ===
# ***** 修改後的函數 *****
def remove_outliers(df):
    original_len = len(df)
    print(f'移除 Fare 異常值前筆數: {original_len}')

    while True:
        if df.empty or 'Fare' not in df.columns: # 增加檢查
            print("DataFrame 為空或缺少 'Fare' 欄位，停止移除。")
            break

        len_before = len(df)

        # 在迴圈內重新計算平均值和標準差
        fare_mean = df['Fare'].mean()
        fare_std = df['Fare'].std()

        # 處理標準差為0或NaN的情況，避免無限迴圈或錯誤
        if pd.isna(fare_std) or fare_std == 0:
            print("Fare 標準差為 0 或 NaN，停止移除。")
            break

        upper_bound = fare_mean + 3 * fare_std

        # 篩選掉超出上限的值
        df = df[df['Fare'] <= upper_bound].copy() # 使用 .copy() 避免警告

        len_after = len(df)

        # 如果此次迴圈沒有移除任何資料，則跳出
        if len_after == len_before:
            print("沒有更多 Fare 異常值可移除，停止迭代。")
            break

    print(f'移除 Fare 異常值後筆數: {len(df)}')
    return df
# ***** 修改結束 *****

# === 任務 4️⃣：類別變數編碼 (encode_features) ===
def encode_features(df):
    # 使用 pd.get_dummies 對 Sex、Embarked 進行編碼，不刪除第一類
    df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=False)
    return df

# === 任務 5️⃣：數值標準化 (scale_features) ===
def scale_features(df):
    scaler = StandardScaler()
    # 確保欄位存在才進行標準化
    if 'Age' in df.columns and 'Fare' in df.columns:
        df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
    return df

# === 任務 6️⃣：資料切割 (split_data) ===
def split_data(df):
    if 'Survived' not in df.columns:
        print("錯誤：'Survived' 欄位不存在，無法切割資料。")
        return pd.DataFrame(), pd.DataFrame(), [], []

    y = df['Survived'].astype(int)
    # 移除目標變數及非數值欄位
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    X = X.select_dtypes(include=np.number) # 確保只包含數值

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('訓練集筆數:', len(X_train))
    print('測試集筆數:', len(X_test))
    return X_train, X_test, y_train, y_test

# === 任務 7️⃣：輸出結果 (save_data) ===
def save_data(df, output_path):
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'✅ 資料處理完成並已輸出至 {output_path}')
    except Exception as e:
        print(f"❌ 儲存檔案時發生錯誤：{e}")

# === 主流程 ===
if __name__ == "__main__":
    DATA_PATH = 'data/titanic.csv'
    OUTPUT_PATH = 'data/titanic_processed.csv'

    df, missing_count = load_data(DATA_PATH)
    if not df.empty:
        df = handle_missing(df)
        df = remove_outliers(df) # 使用修改後的函數
        df = encode_features(df)
        df = scale_features(df)

        if 'Survived' in df.columns:
             processed_df_full = df.copy()
             X_train, X_test, y_train, y_test = split_data(processed_df_full)
             save_data(processed_df_full, OUTPUT_PATH)
        else:
             print("錯誤：'Survived' 欄位不存在於處理後的 DataFrame，無法完成切割與儲存。")
    else:
        print("❌ 資料載入失敗，無法繼續處理。")