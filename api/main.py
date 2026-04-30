from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import time
import os
import requests
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

load_dotenv(dotenv_path=r'C:\Users\Spoorthy\Desktop\Project\.env')
VT_API_KEY = os.getenv("VT_API_KEY")

app = FastAPI(
    title="Phishing Detection API",
    description="Detects phishing URLs using GBC, LSTM, SHAP and VirusTotal",
    version="1.0.0"
)

# Load models
gbc_model        = joblib.load('../models/gbc_new_model.pkl')
feature_names    = joblib.load('../models/feature_names_new.pkl')
char2idx         = joblib.load('../models/char2idx.pkl')
max_len          = joblib.load('../models/max_len.pkl')
lstm_model       = load_model('../models/lstm_model.keras')
explainer        = shap.TreeExplainer(gbc_model)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def url_to_sequence(url, char2idx, max_len=200):
    return [char2idx.get(c, 0) for c in url[:max_len]]

def check_virustotal(url):
    headers = {"x-apikey": VT_API_KEY}
    try:
        response = requests.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data={"url": url}
        )
        if response.status_code != 200:
            return {"error": f"Submission failed: {response.status_code}"}

        analysis_id = response.json()["data"]["id"]
        time.sleep(3)

        result = requests.get(
            f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
            headers=headers
        )
        if result.status_code != 200:
            return {"error": f"Analysis fetch failed: {result.status_code}"}

        stats      = result.json()["data"]["attributes"]["stats"]
        malicious  = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless   = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)
        total      = malicious + suspicious + harmless + undetected

        return {
            "malicious"     : malicious,
            "suspicious"    : suspicious,
            "harmless"      : harmless,
            "undetected"    : undetected,
            "total_engines" : total,
            "vt_score"      : round((malicious + suspicious) / total * 100, 2) if total > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str

class AnalysisResponse(BaseModel):
    url            : str
    classification : str
    risk_score     : float
    gbc_confidence : float
    lstm_confidence: float
    vt_malicious   : int
    vt_engines     : int
    top_features   : list
    message        : str

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Phishing Detection API is running"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: URLRequest):
    url = request.url

    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http or https")

    try:
        # LSTM prediction
        seq        = url_to_sequence(url, char2idx)
        seq_padded = pad_sequences([seq], maxlen=max_len, padding='post')
        lstm_prob  = float(lstm_model.predict(seq_padded, verbose=0)[0][0])

        # VirusTotal
        vt         = check_virustotal(url)
        vt_score   = vt.get("vt_score", 0)
        vt_mal     = vt.get("malicious", 0)
        vt_engines = vt.get("total_engines", 0)

        # GBC prediction — use average feature values as fallback
        gbc_input = np.zeros((1, len(feature_names)))
        gbc_prob  = float(gbc_model.predict_proba(gbc_input)[0][1])

        # SHAP explanation
        shap_vals = explainer.shap_values(gbc_input)[0]
        impact    = sorted(zip(feature_names, shap_vals.tolist()),
                           key=lambda x: abs(x[1]), reverse=True)
        top_5     = [{"feature": f, "impact": round(v, 4)} for f, v in impact[:5]]

        # Combined risk score
        if vt_engines > 0:
            combined = (0.5 * gbc_prob) + (0.3 * lstm_prob) + (0.2 * (vt_score / 100))
        else:
            combined = (0.6 * gbc_prob) + (0.4 * lstm_prob)

        risk           = round(combined * 100, 2)
        classification = "PHISHING" if combined > 0.5 else "LEGITIMATE"
        message        = f"URL classified as {classification} with {risk}% risk score"

        return AnalysisResponse(
            url             = url,
            classification  = classification,
            risk_score      = risk,
            gbc_confidence  = round(gbc_prob * 100, 2),
            lstm_confidence = round(lstm_prob * 100, 2),
            vt_malicious    = vt_mal,
            vt_engines      = vt_engines,
            top_features    = top_5,
            message         = message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "models"  : ["GBC", "LSTM", "Ensemble"],
        "version" : "1.0.0"
    }
#_________________________________________________________________________________________________
#_________________________________________________________________________________________________

@app.post("/report")
def report(request: URLRequest):
    url = request.url

    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http or https")

    try:
        # LSTM prediction
        seq        = url_to_sequence(url, char2idx)
        seq_padded = pad_sequences([seq], maxlen=max_len, padding='post')
        lstm_prob  = float(lstm_model.predict(seq_padded, verbose=0)[0][0])

        # VirusTotal
        vt         = check_virustotal(url)
        vt_score   = vt.get("vt_score", 0)
        vt_mal     = vt.get("malicious", 0)
        vt_sus     = vt.get("suspicious", 0)
        vt_harm    = vt.get("harmless", 0)
        vt_engines = vt.get("total_engines", 0)

        # GBC prediction
        gbc_input = np.zeros((1, len(feature_names)))
        gbc_prob  = float(gbc_model.predict_proba(gbc_input)[0][1])

        # SHAP explanation
        shap_vals = explainer.shap_values(gbc_input)[0]
        impact    = sorted(zip(feature_names, shap_vals.tolist()),
                           key=lambda x: abs(x[1]), reverse=True)
        top_5     = [{"feature": f, "impact": round(v, 4), 
                      "direction": "phishing" if v > 0 else "legitimate"} 
                     for f, v in impact[:5]]

        # Combined risk score
        if vt_engines > 0:
            combined = (0.5 * gbc_prob) + (0.3 * lstm_prob) + (0.2 * (vt_score / 100))
        else:
            combined = (0.6 * gbc_prob) + (0.4 * lstm_prob)

        risk           = round(combined * 100, 2)
        classification = "PHISHING" if combined > 0.5 else "LEGITIMATE"

        # Risk level label
        if risk >= 75:
            risk_level = "HIGH"
        elif risk >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "url"            : url,
            "classification" : classification,
            "risk_level"     : risk_level,
            "risk_score"     : risk,
            "model_scores": {
                "gbc_confidence" : round(gbc_prob * 100, 2),
                "lstm_confidence": round(lstm_prob * 100, 2),
                "ensemble_score" : risk
            },
            "virustotal": {
                "malicious"     : vt_mal,
                "suspicious"    : vt_sus,
                "harmless"      : vt_harm,
                "total_engines" : vt_engines,
                "vt_risk_score" : vt_score
            },
            "top_5_features"  : top_5,
            "recommendation"  : "Do not visit this URL" if classification == "PHISHING" else "URL appears safe",
            "summary"         : f"URL analyzed by GBC, LSTM and VirusTotal. Risk level is {risk_level} at {risk}%."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))