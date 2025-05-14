import joblib
from sklearn.tree import export_text
from collections import Counter

# 1. โหลดโมเดล
model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\trained_model_rdf_v2.pkl'
model = joblib.load(model_path)
print(f"Model loaded from: {model_path}")

# 2. รายชื่อฟีเจอร์ (ตามที่ใช้เทรนจริง)
feature_names = [
    'KWH_TOT',
    'kwh_mean_6months',
    'violation_count_6months',
    'month_counts',
    'kwh_max_min_ratio',
    'kwh_mean_high_vs_low_ratio',
    'kwh_prev_next_diff_ratio'
]

# 3. ดู Feature Importance รวม
print("\nFeature Importances:")
importances = model.feature_importances_

for idx, importance in enumerate(importances):
    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
    print(f"- {feature_name}: {importance:.4f}")

# 4. วิเคราะห์เงื่อนไขต้นไม้ทั้งหมด
split_features_counter = Counter()

for i, estimator in enumerate(model.estimators_):
    tree = estimator.tree_
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] != tree.children_right[node_id]:  # ถ้าเป็น node split
            feature_index = tree.feature[node_id]
            if feature_index != -2:  # ไม่ใช่ leaf
                feature_name = feature_names[feature_index] if feature_index < len(feature_names) else f"Feature_{feature_index}"
                split_features_counter[feature_name] += 1

# 5. สรุปว่า feature ไหนถูกใช้ split บ่อยที่สุด
print("\nTop Features used in Splits:")
for feature, count in split_features_counter.most_common():
    print(f"- {feature}: {count} times")

# 6. แสดงตัวอย่างกฎจากต้นไม้ต้นที่ 0
print("\nSample Tree Rules (Tree 0):")
tree_rules = export_text(
    model.estimators_[0],
    feature_names=[name if idx < len(feature_names) else f"Feature_{idx}" for idx, name in enumerate(feature_names)]
)
# Export tree rules
tree_rules_file = r'D:\PEA JOB\M-BA Phase2\code\clpa\tree_rules_sample.txt'
with open(tree_rules_file, 'w', encoding='utf-8') as f:
    f.write(tree_rules)

print(f"Sample tree rules saved to: {tree_rules_file}")
