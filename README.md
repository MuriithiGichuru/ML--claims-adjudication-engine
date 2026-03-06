# 1. - Requirements

pandas>=2.0.0
numpy>=1.24.0
pymupdf>=1.23.0
scikit-learn>=1.3.0

---

# 2. - README.md

# ML Claims Adjudication Engine

**AI-Powered Claims Adjudication System**  
A lightweight hybrid rule-based + machine learning solution for automated medical and insurance claims processing.

Built for accuracy, speed, and explainability — tailored for the Kenyan insurance and healthcare sector.

---

# 3. Project video

Watch a short video explain the Approach, Methodology and improvements of the ML model.

(https://drive.google.com/file/d/1FgtA5dDyJfyzKWNpx7xUW19kDBJUOdua/view)

---

# 4. Features

- Supports both **CSV** and **PDF** input
- Hybrid ML model: Random Forest (70%) + Isolation Forest (30%)
- Exact probability thresholds:
  - `0.0 – 0.3` → ✅ Pass
  - `0.3 – 0.7` → ⚠️ Flag
  - `0.7 – 1.0` → ❌ Fail
- Reason for each decision is easily readable
- Clean JSON output
- Works with or without historical training data

---

# 5. Quick Start

pip install -r requirements.txt

python claims_adjudication_engine.py --input sample_claims.csv

---

# 6. Output: 

{
  "status": "completed",
  "total_claims": 8,
  "adjudicated_claims": [
    {
      "claim_id": 10001,
      "risk_score": 0.2076,
      "decision": "✅ Pass",
      "confidence": 20.8,
      "reason": "Low risk profile"
    }
  ],
  "generated_at": "2026-03-06T12:48:01.266608",
  "model_version": "1.0"
}

---

# 7. Model Overview

Primary Model: RandomForestClassifier (120 trees, balanced class weight)

Anomaly Detector: IsolationForest (contamination = 0.06)

Risk Score Formula: 0.7 × Fraud Probability + 0.3 × Anomaly Score

