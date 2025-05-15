import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def high_low_mean_diff(x):
    mean_all = x.mean()
    high = x[x > mean_all].mean()
    low = x[x <= mean_all].mean()
    if pd.isna(high) or pd.isna(low) or low == 0:
        return 0
    return abs(high - low) / (low + 1e-5)

def create_features(df):
    df = df.sort_values(['ca', 'period'])
    df['month_counts'] = df.groupby('ca')['period'].transform('count')

    df['kwh_mean_6months'] = df.groupby('ca')['kwh_total'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )
    # ใช้ย้อนหลัง 6 เดือน รวมเดือนปัจจุบัน
    df['violation_count_6months'] = df.groupby('ca')['inspected'].transform(
        lambda x: x.rolling(window=6, min_periods=1).sum()
    )

    df['violation_count'] = df.groupby('ca')['inspected'].transform('sum')

    df['kwh_max'] = df.groupby('ca')['kwh_total'].transform('max')
    df['kwh_min'] = df.groupby('ca')['kwh_total'].transform('min')
    df['kwh_max_min_ratio'] = (df['kwh_max'] - df['kwh_min']) / (df['kwh_max'] + 1e-5)

    df['kwh_mean_high_vs_low_ratio'] = df.groupby('ca')['kwh_total'].transform(high_low_mean_diff)

    df['kwh_prev'] = df.groupby('ca')['kwh_total'].shift(1)
    df['kwh_next'] = df.groupby('ca')['kwh_total'].shift(-1)
    df['kwh_prev_next_diff_ratio'] = np.abs(df['kwh_prev'] - df['kwh_next']) / (df['kwh_prev'] + 1e-5)

    df.fillna(0, inplace=True)
    return df

def main():
    # 1. Load data ทั้งหมด
    file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data.csv'
    data = pd.read_csv(file_path, sep=';')

    # แปลง period เป็น datetime
    data['period'] = pd.to_datetime(data['period'], format='%Y%m', errors='coerce')

    # 2. กรองข้อมูลฝึก 2020-01-01 ถึง 2024-12-31
    train_data = data[(data['period'] >= '2020-01-01') & (data['period'] <= '2024-12-31')].copy()

    # 3. สร้างฟีเจอร์
    train_data = create_features(train_data)

    # 4. เตรียมข้อมูล
    features = [
        'kwh_total', 'kwh_mean_6months', 'violation_count_6months',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ]
    X_train = train_data[features]
    y_train = train_data['inspected']

    # 5. สร้างโมเดล
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 6. ประเมินผลในข้อมูลฝึก (เพื่อดู performance เบื้องต้น)
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    print(f"Training Accuracy: {acc_train:.3f}")
    print(f"Training F1 Score: {f1_train:.3f}")

    # 7. บันทึกโมเดล
    model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
