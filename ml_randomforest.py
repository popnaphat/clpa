import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. อ่านข้อมูลจากไฟล์ Excel
file_path = r'D:\abnormalmeterProject\data.xlsx'
data = pd.read_excel(file_path)

# 2. เตรียมข้อมูล (Feature Engineering)
# แปลงวันที่ในคอลัมน์ 'period2' ให้อยู่ในรูปแบบ datetime
data['period2'] = pd.to_datetime(data['period2'], format='%Y%m')

# แปลงค่าใน inspected ให้เป็นตัวเลข
data['inspected_numeric'] = data['inspected'].map({'detected': 1, 'unknown': 0})

# สร้างฟีเจอร์เพิ่มเติม เช่น ค่าเฉลี่ยการใช้ไฟฟ้าใน 6 เดือนก่อนหน้า
data['kwh_mean_6months'] = data.groupby('INSkey')['KWH_TOT'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)

# สร้างฟีเจอร์จำนวนการละเมิดใน 6 เดือนก่อนหน้า
data['violation_count_6months'] = data.groupby('INSkey')['inspected_numeric'].transform(
    lambda x: x.shift().rolling(window=6, min_periods=1).sum()
)

# เติมค่า Missing Value (ถ้ามี) ด้วย 0
data.fillna(0, inplace=True)

# แปลงฟีเจอร์ 'inspected' ให้เป็นค่าเลข (detected=1, Unknown=0)
data['inspected'] = data['inspected'].apply(lambda x: 1 if x == 'detected' else 0)

# ตรวจสอบว่าข้อมูลแต่ละ INSkey มีการละเมิดอย่างน้อย 1-7 เดือน
inskey_violation_counts = data.groupby('INSkey')['inspected'].sum()
valid_inskeys = inskey_violation_counts[(inskey_violation_counts >= 1) & (inskey_violation_counts <= 7)].index
data = data[data['INSkey'].isin(valid_inskeys)]

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

# 7. คำนวณความแม่นยำเฉพาะกรณีที่ทำนาย 'detected' หรือ 1 เท่านั้น
# การคำนวณความแม่นยำเฉพาะกรณี 'detected' หรือ 1
detected_indices = y_test == 1
accuracy_detected = accuracy_score(y_test[detected_indices], y_pred[detected_indices]) if detected_indices.any() else 0.0
print(f"Accuracy for detected (1): {accuracy_detected:.2f}")

# รายงานผลลัพธ์
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. คำนวณคะแนน Cross-Validation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean():.2f}")
print(f"Standard Deviation of Cross-Validation Scores: {cross_val_scores.std():.2f}")

# 9. บันทึกโมเดลเพื่อใช้งานในอนาคต
model_path = r'D:\abnormalmeterProject\trained_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

# 10. Export Prediction Results to CSV
# Create a DataFrame for the results
results_df = X_test.copy()

# Add the additional columns (INSkey and period2)
results_df['INSkey'] = data.loc[X_test.index, 'INSkey']
results_df['period2'] = data.loc[X_test.index, 'period2']
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred

# Save the results to a CSV file
output_file_path = r'D:\abnormalmeterProject\prediction_results.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Prediction results saved to: {output_file_path}")
