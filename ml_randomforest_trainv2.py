import pandas as pd
import numpy as np
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

# 3. Define feature sets
feature_sets = {
    "with_violation_count_6months": [
        'kwh_total', 'kwh_mean_6months', 'violation_count_6months',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ],
    "with_violation_count_total": [
        'kwh_total', 'kwh_mean_6months', 'violation_count',
        'month_counts', 'kwh_max_min_ratio',
        'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
    ]
}

# 4. Train models on full data
best_model = None
best_score = 0
best_model_name = ""

for name, feature_list in feature_sets.items():
    X = data[feature_list]
    y = data['inspected']

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\nModel: {name}")
    print(f"- Accuracy (full data): {acc:.3f}")
    print(f"- F1 Score (full data): {f1:.3f}")

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

# 5. Save best model
best_model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.pkl'
joblib.dump(best_model, best_model_path)
# Save also as .joblib (same model, different extension)
best_model_joblib_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.joblib'
joblib.dump(best_model, best_model_joblib_path)

print(f"✅ Best model also saved as: {best_model_joblib_path}")