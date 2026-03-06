# claims_adjudication_engine.py


import argparse
import pandas as pd
import numpy as np
import fitz
import re
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

class ClaimsAdjudicationEngine:
    def __init__(self):
        self.predictive_model = None
        self.anomaly_detector = None
        self.label_encoders = {}
        
        self.high_amount_threshold = 150000
        self.very_high_amount_threshold = 500000
        self.max_frequency_per_month = 8

    def ingest_data(self, input_path: str) -> pd.DataFrame:
        if input_path.lower().endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.lower().endswith('.pdf'):
            df = self._extract_from_pdf(input_path)
        else:
            raise ValueError("Only .csv or .pdf files are supported")

        rename_map = {
            'Claim ID': 'claim_id', 'Member ID': 'member_id', 'Provider ID': 'provider_id',
            'Diagnosis Code (ICD-10)': 'diagnosis_code', 'Procedure Code (CPT or equivalent)': 'procedure_code',
            'Claimed Amount': 'claimed_amount', 'Approved Tariff Amount': 'approved_tariff',
            'Date of Service': 'date_of_service', 'Provider Type': 'provider_type',
            'Historical Claim Frequency': 'hist_frequency', 'Location': 'location'
        }
        df = df.rename(columns=rename_map)

        df['claimed_amount'] = pd.to_numeric(df['claimed_amount'], errors='coerce')
        df['approved_tariff'] = pd.to_numeric(df['approved_tariff'], errors='coerce')
        df['hist_frequency'] = pd.to_numeric(df['hist_frequency'], errors='coerce').fillna(0)
        return df

    def _extract_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()

        pattern = (
            r"Claim ID[:\s]*(\d+).*?"
            r"Member ID[:\s]*(\d+).*?"
            r"Provider ID[:\s]*(\d+).*?"
            r"Diagnosis Code.*?[:\s]*([A-Z0-9.]+).*?"
            r"Procedure Code.*?[:\s]*([A-Z0-9.]+).*?"
            r"Claimed Amount[:\s]*K?ES?\s*([0-9,]+).*?"
            r"Approved Tariff Amount[:\s]*K?ES?\s*([0-9,]+).*?"
            r"Date of Service[:\s]*(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}).*?"
            r"Provider Type[:\s]*([A-Za-z0-9 ]+).*?"
            r"Historical Claim Frequency[:\s]*(\d+).*?"
            r"Location[:\s]*([A-Za-z ]+)"
        )

        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        claims = []
        for m in matches:
            try:
                claims.append({
                    'claim_id': int(m[0]), 'member_id': int(m[1]), 'provider_id': int(m[2]),
                    'diagnosis_code': m[3].strip(), 'procedure_code': m[4].strip(),
                    'claimed_amount': float(m[5].replace(',', '')),
                    'approved_tariff': float(m[6].replace(',', '')),
                    'date_of_service': m[7], 'provider_type': m[8].strip(),
                    'hist_frequency': int(m[9]), 'location': m[10].strip()
                })
            except:
                continue

        if not claims:
            raise ValueError("No claims could be extracted from the PDF.")
        return pd.DataFrame(claims)

    def preprocess(self, df: pd.DataFrame, training: bool = False):
        df = df.copy()
        df['amount_ratio'] = df['claimed_amount'] / (df['approved_tariff'] + 1)
        df['over_tariff'] = (df['claimed_amount'] > df['approved_tariff']).astype(int)

        cat_cols = ['diagnosis_code', 'procedure_code', 'provider_type', 'location']
        for col in cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if training:
                    self.label_encoders[col].fit(df[col].astype(str))
            df[col] = self.label_encoders[col].transform(df[col].astype(str))

        feature_cols = ['claimed_amount', 'approved_tariff', 'hist_frequency', 'amount_ratio',
                        'over_tariff', 'provider_type', 'diagnosis_code', 'procedure_code', 'location']
        return df[feature_cols], df

    def train_models(self, historical_path: str = None):
        if historical_path and historical_path.endswith('.csv'):
            hist_df = pd.read_csv(historical_path)
        else:
            n = 5000
            np.random.seed(42)
            hist_df = pd.DataFrame({
                'claimed_amount': np.random.lognormal(9, 1.2, n).astype(int),
                'approved_tariff': np.random.lognormal(8.8, 1.1, n).astype(int),
                'hist_frequency': np.random.poisson(3, n),
                'provider_type': np.random.choice(['Hospital', 'Clinic', 'Lab', 'Pharmacy', 'Specialist'], n),
                'diagnosis_code': np.random.choice(['J45.9', 'M54.5', 'E11.9', 'I10', 'B54'], n),
                'procedure_code': np.random.choice(['99213', '85025', '36415', '99214'], n),
                'location': np.random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Eldoret'], n),
                'fraud_label': np.random.choice([0, 1], n, p=[0.93, 0.07])
            })

        X, _ = self.preprocess(hist_df, training=True)
        y = hist_df['fraud_label']

        self.predictive_model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, class_weight='balanced')
        self.predictive_model.fit(X, y)

        self.anomaly_detector = IsolationForest(contamination=0.06, random_state=42)
        self.anomaly_detector.fit(X)

    def adjudicate(self, df: pd.DataFrame) -> list:
        if self.predictive_model is None or self.anomaly_detector is None:
            raise ValueError("Models not trained. Call train_models() first!")

        X, _ = self.preprocess(df)
        
        fraud_prob = self.predictive_model.predict_proba(X)[:, 1]
        anomaly_score = -self.anomaly_detector.score_samples(X)
        anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
        
        risk_score = np.clip(fraud_prob * 0.7 + anomaly_score * 0.3, 0.0, 1.0)

        decisions = np.where(risk_score <= 0.3, "✅ Pass",
                    np.where(risk_score <= 0.7, "⚠️ Flag", "❌ Fail"))

        results = []
        for i, row in df.iterrows():
            prob = float(risk_score[i])
            reasons = []
            if row['hist_frequency'] > self.max_frequency_per_month:
                reasons.append("High claim frequency")
            if row['claimed_amount'] > self.very_high_amount_threshold:
                reasons.append("Very high claim amount")
            elif row['claimed_amount'] > self.high_amount_threshold:
                reasons.append("High claim amount")
            if row['claimed_amount'] > row['approved_tariff'] * 1.5:
                reasons.append("Claim significantly exceeds approved tariff")
            if prob > 0.6:
                reasons.append("ML anomaly detected")
            
            reason = ", ".join(reasons) if reasons else "Low risk profile"

            results.append({
                "claim_id": int(row['claim_id']),
                "risk_score": round(prob, 4),
                "decision": decisions[i],
                "confidence": round(prob * 100, 1),
                "reason": reason
            })
        return results


# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Claims Adjudication Engine")
    parser.add_argument("--input", required=True, help="claims.csv or claims.pdf")
    parser.add_argument("--historical", default=None, help="Optional historical CSV")
    args = parser.parse_args()

    engine = ClaimsAdjudicationEngine()
    engine.train_models(args.historical)

    claims_df = engine.ingest_data(args.input)
    results = engine.adjudicate(claims_df)

    output = {
        "status": "completed",
        "total_claims": len(results),
        "adjudicated_claims": results,
        "generated_at": pd.Timestamp.now().isoformat(),
        "model_version": "1.0"
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    with open("adjudicated_claims.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ JSON saved to: adjudicated_claims.json")