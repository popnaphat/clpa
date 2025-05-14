import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. อ่านข้อมูลจากไฟล์ CSV
file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\data.csv'
data = pd.read_csv(file_path, sep=';')

# 2. เตรียมข้อมูล (Feature Engineering)
# แปลงวันที่ในคอลัมน์ 'period2' ให้อยู่ในรูปแบบ datetime
data['period2'] = pd.to_datetime(data['period2'], format='%Y%m', errors='coerce')

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

# 4. สร้างและฝึกโมเดล Random Forest ด้วยข้อมูลทั้งหมด 100%
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# 5. บันทึกโมเดลเพื่อใช้งานในอนาคต
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_rdf.pkl'
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")
