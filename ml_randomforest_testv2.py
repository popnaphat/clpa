import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

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

    # ไม่ใช้ violation_count total ใน test เพื่อเลี่ยงใช้ข้อมูลอนาคต

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
    # 1. โหลดข้อมูลทดสอบ
    file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data for testing03.csv'
    data = pd.read_csv(file_path, sep=';')
    data['period'] = pd.to_datetime(data['period'], format='%Y%m', errors='coerce')

    # 2. กรองข้อมูลช่วง 2024-09-01 ถึง 2025-03-31
    test_data = data[(data['period'] >= '2024-09-01') & (data['period'] <= '2025-03-31')].copy()

    # 3. สร้างฟีเจอร์เหมือนตอน train
    test_data = create_features(test_data)

    # 4. เตรียมข้อมูล
    features = [
        'kwh_total', 'kwh_mean_6months', 'violation_count_6months',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ]
    X_test = test_data[features]
    y_test = test_data['inspected']

    # 5. โหลดโมเดล
    model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.pkl'
    model = joblib.load(model_path)

    # 6. ทำนายผล
    y_pred = model.predict(X_test)

    # 7. ประเมินผล
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test F1 Score: {f1:.3f}")

    # Accuracy เฉพาะกรณี inspected=1
    if (y_test == 1).sum() > 0:
        acc_detected = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
        print(f"Accuracy (detected=1 only): {acc_detected:.3f}")

    # 8. Export ผลลัพธ์ prediction
    results_df = test_data[['trsg','ca','installation','mru','kwh_total']].copy()
    results_df['period'] = test_data['period'].dt.strftime('%Y%m')
    results_df['Actual'] = y_test
    results_df['Predicted'] = y_pred

    output_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_predictions_test.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Predictions exported to: {output_path}")

if __name__ == "__main__":
    main()
