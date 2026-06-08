import json
from model.predictor import predict_raw

# Load input JSON
with open("demo/sample_input.json") as f:
    payload = json.load(f)

# Run full pipeline
result = predict_raw(payload)

# Pretty print
print("\n=== FINAL RESULT ===")
for k, v in result.items():
    print(f"{k}: {v}")