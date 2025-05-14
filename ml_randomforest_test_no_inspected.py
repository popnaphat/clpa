import pandas as pd
import joblib

# 1. โหลดข้อมูลทดสอบจาก CSV
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\test_data.csv'
data = pd.read_csv(file_path, sep=';')

# 2. Feature Engineering
data['period2'] = pd.to_datetime(data['period2'], format='%Y%m', errors='coerce')
data['kwh_mean_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
# data['violation_count_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
#     lambda x: x.shift().rolling(window=6, min_periods=1).sum()
# )
# data['violation_count_6months'] = 0
data['violation_count_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
    lambda x: ((x - x.mean()) > 1.5 * x.std()).rolling(window=6, min_periods=1).sum()
)
data.fillna(0, inplace=True)

# 3. Features
features = ['KWH_TOT', 'kwh_mean_6months', 'violation_count_6months']
X_test = data[features]

# 4. โหลดโมเดล
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_rdf.pkl'
model = joblib.load(model_path)
print(f"✅ Model loaded from: {model_path}")

# 5. ทำนายผล
y_pred = model.predict(X_test)

# 👉 6. แสดงจำนวนและสัดส่วนที่ทำนายว่าเป็น 1
num_detected = (y_pred == 1).sum()
total = len(y_pred)
percent_detected = (num_detected / total) * 100
print(f"🔍 Predicted = 1: {num_detected} records ({percent_detected:.2f}%) from total {total}")

# 7. บันทึกผลลัพธ์
results_df = data[['trsg', 'INSkey', 'period2', 'KWH_TOT']].copy()
results_df['Predicted'] = y_pred
results_df['INSkey'] = results_df['INSkey']
results_df['period2'] = results_df['period2'].dt.strftime('%Y%m')
results_df['mru'] = data['mru']

output_file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\prediction_results_rdf_test.csv'
results_df.to_csv(output_file_path, index=False)

print(f"✅ Prediction results saved to: {output_file_path}")
