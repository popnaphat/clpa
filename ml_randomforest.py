import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# 4. แบ่งข้อมูลเป็น Training (90%) และ Testing (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 5. สร้างและฝึกโมเดล Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. ทำนายผลใน Testing Set
y_pred = model.predict(X_test)

# 7. คำนวณความแม่นยำของโมเดลโดยรวม
accuracy_overall = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy_overall:.2f}")

# 8. คำนวณความแม่นยำเฉพาะกรณีที่ทำนาย 'detected' หรือ 1 เท่านั้น
if (y_test == 1).sum() > 0:
    accuracy_detected = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
    print(f"Accuracy for detected (1): {accuracy_detected:.2f}")
else:
    print("No detected (1) cases found in test data.")

# รายงานผลลัพธ์
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. คำนวณคะแนน Cross-Validation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean():.2f}")
print(f"Standard Deviation of Cross-Validation Scores: {cross_val_scores.std():.2f}")

# 10. บันทึกโมเดลเพื่อใช้งานในอนาคต
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_rdf.pkl'
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

# 11. Export Prediction Results to CSV
results_df = X_test.copy()

# Add additional columns (INSkey and period2)
results_df['INSkey'] = data.loc[X_test.index, 'INSkey']
results_df['period2'] = data.loc[X_test.index, 'period2']
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred

# Save the results to a CSV file
output_file_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\prediction_results_rdf.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Prediction results saved to: {output_file_path}")
