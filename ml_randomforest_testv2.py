import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

# 1. Load ข้อมูล
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data for testing01.csv'
data = pd.read_csv(file_path, sep=';')

# 2. Preprocess & Feature Engineering (ต้องเหมือนตอนฝึก)
data['period'] = pd.to_datetime(data['period'], format='%Y%m', errors='coerce')
data = data.sort_values(['ca', 'period'])
data['month_counts'] = data.groupby('ca')['period'].transform('count')

data['kwh_mean_6months'] = data.groupby('ca')['kwh_total'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
data['violation_count_6months'] = data.groupby('ca')['inspected'].transform(
    lambda x: x.shift().rolling(window=6, min_periods=1).sum()
)
data['violation_count'] = data.groupby('ca')['inspected'].transform('sum')

data['kwh_max'] = data.groupby('ca')['kwh_total'].transform('max')
data['kwh_min'] = data.groupby('ca')['kwh_total'].transform('min')
data['kwh_max_min_ratio'] = (data['kwh_max'] - data['kwh_min']) / (data['kwh_max'] + 1e-5)

def high_low_mean_diff(x):
    mean_all = x.mean()
    high = x[x > mean_all].mean()
    low = x[x <= mean_all].mean()
    if pd.isna(high) or pd.isna(low) or low == 0:
        return 0
    return abs(high - low) / (low + 1e-5)

data['kwh_mean_high_vs_low_ratio'] = data.groupby('ca')['kwh_total'].transform(high_low_mean_diff)

data['kwh_prev'] = data.groupby('ca')['kwh_total'].shift(1)
data['kwh_next'] = data.groupby('ca')['kwh_total'].shift(-1)
data['kwh_prev_next_diff_ratio'] = np.abs(data['kwh_prev'] - data['kwh_next']) / (data['kwh_prev'] + 1e-5)

data.fillna(0, inplace=True)

# 3. เลือกชุดฟีเจอร์ที่ใช้ (อิงตามตอนฝึก)
features = [
    'kwh_total', 'kwh_mean_6months', 'violation_count',
    'month_counts', 'kwh_max_min_ratio',
    'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
]
X = data[features]
y = data['inspected']

# 4. Load โมเดลที่ฝึกไว้
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.pkl'
model = joblib.load(model_path)

# 5. ทำนายผล
y_pred = model.predict(X)

# 6. ประเมินผล
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
print(f"\n📊 Evaluation on full data:")
print(f"- Accuracy: {acc:.3f}")
print(f"- F1 Score: {f1:.3f}")

# Accuracy เฉพาะกรณีตรวจพบ (inspected=1)
if (y == 1).sum() > 0:
    acc_detected = accuracy_score(y[y == 1], y_pred[y == 1])
    print(f"- Accuracy (detected=1 only): {acc_detected:.3f}")

# 7. Export prediction
results_df = data[['trsg','ca','installation','mru','kwh_total']].copy()
results_df['period'] = data['period'].dt.strftime('%Y%m')
results_df['Actual'] = y
results_df['Predicted'] = y_pred

output_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_predictions01.csv'
results_df.to_csv(output_path, index=False)
print(f"\n📄 Predictions exported to: {output_path}")
