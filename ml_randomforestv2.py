import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# 1. Load CSV
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data.csv'
data = pd.read_csv(file_path, sep=';')

# 2. Preprocess & Feature Engineering
data['period'] = pd.to_datetime(data['period'], format='%Y%m', errors='coerce')
data = data.sort_values(['ca', 'period'])
data['month_counts'] = data.groupby('ca')['period'].transform('count')

# ฟีเจอร์พื้นฐาน
data['kwh_mean_6months'] = data.groupby('ca')['KWH_TOT'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
data['violation_count_6months'] = data.groupby('ca')['inspected'].transform(
    lambda x: x.shift().rolling(window=6, min_periods=1).sum()
)
data['violation_count'] = data.groupby('ca')['inspected'].transform('sum')

data['kwh_max'] = data.groupby('ca')['KWH_TOT'].transform('max')
data['kwh_min'] = data.groupby('ca')['KWH_TOT'].transform('min')
data['kwh_max_min_ratio'] = (data['kwh_max'] - data['kwh_min']) / (data['kwh_max'] + 1e-5)

def high_low_mean_diff(x):
    mean_all = x.mean()
    high = x[x > mean_all].mean()
    low = x[x <= mean_all].mean()
    if pd.isna(high) or pd.isna(low) or low == 0:
        return 0
    return abs(high - low) / (low + 1e-5)

data['kwh_mean_high_vs_low_ratio'] = data.groupby('ca')['KWH_TOT'].transform(high_low_mean_diff)

data['kwh_prev'] = data.groupby('ca')['KWH_TOT'].shift(1)
data['kwh_next'] = data.groupby('ca')['KWH_TOT'].shift(-1)
data['kwh_prev_next_diff_ratio'] = np.abs(data['kwh_prev'] - data['kwh_next']) / (data['kwh_prev'] + 1e-5)

data.fillna(0, inplace=True)

# 3. สร้างชุด feature 2 แบบ
feature_sets = {
    "with_violation_count_6months": [
        'KWH_TOT', 'kwh_mean_6months', 'violation_count_6months',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ],
    "with_violation_count_total": [
        'KWH_TOT', 'kwh_mean_6months', 'violation_count',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ]
}

# 4. ฝึกและประเมินโมเดลทั้ง 2 แบบ
best_model = None
best_score = 0
best_model_name = ""
best_y_pred = None
best_X_test = None
best_y_test = None

for name, feature_list in feature_sets.items():
    X = data[feature_list]
    y = data['inspected']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if (y_test == 1).sum() > 0:
        acc_detected = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
    else:
        acc_detected = float('nan')  # หรือ 0.0

    print(f"\nModel: {name}")
    print(f"- Accuracy: {acc:.3f}")
    print(f"- F1 Score: {f1:.3f}")
    print(f"- Accuracy (only for detected=1): {acc_detected:.3f}")

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name
        best_y_pred = y_pred
        best_X_test = X_test
        best_y_test = y_test

# 5. บันทึกโมเดลที่ดีที่สุด
best_model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\best_model_rdf.pkl'
joblib.dump(best_model, best_model_path)
print(f"\n✅ Best model saved: {best_model_name} → {best_model_path}")

# 6. Export ผลลัพธ์ทำนาย
results_df = best_X_test.copy()
results_df['peacode'] = data['peacode']
results_df['ca'] = data.loc[best_X_test.index, 'ca']
results_df['period'] = data.loc[best_X_test.index, 'period']
results_df['Actual'] = best_y_test
results_df['Predicted'] = best_y_pred

output_file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\best_model_predictions.csv'
results_df.to_csv(output_file_path, index=False)
print(f"📄 Prediction results saved to: {output_file_path}")
