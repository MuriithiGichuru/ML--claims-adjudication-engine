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

# 4. Architecture

## Architecture

**System Flow:**

a. **Input Layer**  
   - Accepts claims from CSV files or PDF documents

b. **Ingestion Layer**  
   - Extracts structured data using PyMuPDF and regex parsing

c. **Feature Engineering Layer**  
   - Creates 9 key features (amount_ratio, over_tariff, hist_frequency, etc.)

d. **Hybrid ML Engine**  
   - RandomForestClassifier (70% weight) – learns complex patterns  
   - IsolationForest (30% weight) – detects anomalies

e. **Decision Layer**  
   - Calculates final Risk Score  
   - Applies exact thresholds:  
     - 0.0 – 0.3 → ✅ Pass  
     - 0.3 – 0.7 → ⚠️ Flag  
     - 0.7 – 1.0 → ❌ Fail

f. **Output Layer**  
   - Generates clean JSON with risk_score, decision, confidence, and reason
---

# 5. Model Overview

Primary Model: RandomForestClassifier (120 trees, balanced class weight)

Anomaly Detector: IsolationForest (contamination = 0.06)

Risk Score Formula: 0.7 × Fraud Probability + 0.3 × Anomaly Score

---

# 6. Assumptions & Trade-offs
## Assumptions:

1. Claims data follows reasonably structured text format
2. Fraud rate is approximately 7%
3. Synthetic data was used for training due to unavailability of real labeled claims

## Trade-offs:

1. Prioritized explainability and speed over maximum accuracy (chose Random Forest over deep learning)
2. Used regex-based PDF extraction (fast but less robust for highly unstructured PDFs)
3. Sacrificed some sophistication for maintainability and ease of deployment

---

# 7. Quick Start

pip install -r requirements.txt

python claims_adjudication_engine.py --input sample_claims.csv

---

# 8. Output: 

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


