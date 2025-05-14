import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. อ่านข้อมูลจากไฟล์ CSV
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\clp_pea_pilot+predicted.csv'
data = pd.read_csv(file_path, sep=';')

# 2. เตรียมข้อมูล (Feature Engineering)
# แปลงวันที่ในคอลัมน์ 'period2' ให้อยู่ในรูปแบบ datetime
data['period2'] = pd.to_datetime(data['period2'], format='%Y%m', errors='coerce').dt.strftime('%Y%m')

# สร้างฟีเจอร์เพิ่มเติม เช่น ค่าเฉลี่ยการใช้ไฟฟ้าใน 6 เดือนก่อนหน้า
data['kwh_mean_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)

# สร้างฟีเจอร์จำนวนการละเมิดใน 6 เดือนก่อนหน้า
data['violation_count_6months'] = data.groupby('INSkey')['inspected'].transform(
    lambda x: x.shift().rolling(window=6, min_periods=1).sum()
)

# เติมค่า Missing Value (ถ้ามี) ด้วย 0
data.fillna(0, inplace=True)

# 3. แยกข้อมูล Features และ Target
features = ['KWH_TOT', 'kwh_mean_6months', 'violation_count_6months']
X = data[features]
y = data['inspected']

# 4. โหลดโมเดลที่ฝึกแล้ว
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_rdf.pkl'
model = joblib.load(model_path)

# 5. ทำนายผลในข้อมูลทั้งหมด
y_pred = model.predict(X)

# 6. คำนวณความแม่นยำเฉพาะกรณีที่จริงเป็น 'detected' หรือ 1 เท่านั้น
detected_indices = y == 1
if detected_indices.sum() > 0:
    accuracy_detected = accuracy_score(y[detected_indices], y_pred[detected_indices])
    print(f"Accuracy for detected (1): {accuracy_detected:.2f}")
else:
    print("No detected (1) cases found in data.")


# 7. รายงานผลลัพธ์
print("\nClassification Report:")
print(classification_report(y, y_pred))

# 8. Export Prediction Results to CSV
results_df = X.copy()

# Add additional columns (INSkey and period2)
results_df['trsg'] = data['id']
results_df['INSkey'] = data['INSkey'].astype(int)
results_df['period2'] = data['period2'].dt.strftime('%Y%m')
results_df['Actual'] = y
results_df['Predicted'] = y_pred

# Save the results to a CSV file
output_file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\prediction_results_rdf.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Prediction results saved to: {output_file_path}")
