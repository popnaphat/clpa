import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. อ่านข้อมูลจากไฟล์ CSV
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data.csv'
try:
    data = pd.read_csv(file_path, sep=';')
    print(f"Data loaded successfully from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    raise

# 2. เตรียมข้อมูล (Feature Engineering)
# แปลงวันที่ในคอลัมน์ 'period2' ให้อยู่ในรูปแบบ datetime
data['period2'] = pd.to_datetime(data['period2'], format='%Y%m', errors='coerce')

# สร้างฟีเจอร์เพิ่มเติม เช่น ค่าเฉลี่ยการใช้ไฟฟ้าใน 6 เดือนก่อนหน้า
data['kwh_mean_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)

data['violation_count_6months'] = data.groupby('INSkey')['inspected'].transform(
    lambda x: x.shift().rolling(window=6, min_periods=1).sum()
)

# เติมค่า Missing Value (ถ้ามี) ด้วย 0
data.fillna(0, inplace=True)

# ตรวจสอบว่า INSkey แต่ละตัวมีการละเมิด 1-7 เดือน
inskey_violation_counts = data.groupby('INSkey')['inspected'].sum()
valid_inskeys = inskey_violation_counts[(inskey_violation_counts >= 1) & (inskey_violation_counts <= 7)].index
data = data[data['INSkey'].isin(valid_inskeys)]

# 3. กำหนด Features และ Target สำหรับการทดสอบ
features = ['KWH_TOT', 'kwh_mean_6months', 'violation_count_6months']
X_test = data[features]
y_test = data['inspected']

# 4. โหลดโมเดลที่บันทึกไว้
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_xgb.pkl'
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"Error: Trained model not found at {model_path}")
    raise

# 5. ทำนายผลลัพธ์ด้วยโมเดลที่โหลดมา
y_pred = model.predict(X_test)

# 6. คำนวณผลลัพธ์
# คำนวณความแม่นยำเฉพาะกรณี 'detected' หรือ 1
detected_indices = y_test == 1
if detected_indices.any():
    accuracy_detected = accuracy_score(y_test[detected_indices], y_pred[detected_indices])
    print(f"Accuracy for 'detected' (1) cases: {accuracy_detected:.2f}")
else:
    print("No detected cases (1) found in test data.")

# รายงานผลลัพธ์
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. บันทึกผลลัพธ์การทำนายเป็นไฟล์ CSV
results_df = X_test.copy()
results_df['INSkey'] = data.loc[X_test.index, 'INSkey']
results_df['period2'] = data.loc[X_test.index, 'period2']
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred

output_file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\prediction_results_xgb_test.csv'
results_df.to_csv(output_file_path, index=False)
print(f"Prediction results saved to: {output_file_path}")
