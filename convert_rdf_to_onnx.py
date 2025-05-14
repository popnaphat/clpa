import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import os

# ======== CONFIG =========
# Path to your .pkl model
pkl_model_path = r'D:\PEA JOB\M-BA Phase2\code\clpa\fraud_detection_trained_rdf_model.pkl'

# List of features used to train the model (must match exactly)
feature_list = [
    'KWH_TOT', 'kwh_mean_6months', 'violation_count_6months',
    'month_counts', 'kwh_max_min_ratio',
    'kwh_mean_high_vs_low_ratio', 'kwh_prev_next_diff_ratio'
]
# ==========================

# Load trained model
model = joblib.load(pkl_model_path)
print("✅ Loaded model from:", pkl_model_path)

# Define input type
initial_type = [('float_input', FloatTensorType([None, len(feature_list)]))]

# Convert to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type, options={'zipmap': False})

# Save ONNX model
onnx_path = os.path.splitext(pkl_model_path)[0] + '.onnx'
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"📦 ONNX model saved to: {onnx_path}")
