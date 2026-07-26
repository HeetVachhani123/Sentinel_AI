<div align="center">
  <h1>🛡️ Sentinel-AI</h1>
  <p><b>Next-Generation Behavioral Anomaly Detection for Modern SOCs</b></p>
  <p><i>Developed for the <b>Honeywell Hackathon</b></i></p>
</div>

---

## 💡 The Problem
Traditional signature-based security relies on static rules and known malware hashes. The problem? It completely fails against novel, low-and-slow, or credential-based intrusions (like insider threats or session hijacking). 

## 🚀 The Solution: Sentinel-AI
I built **Sentinel-AI** to tackle this exactly. It is a dual-layered, AI-powered behavioral anomaly detection system. Instead of looking for known bad signatures, it actively learns the "normal" behavioral patterns (timing, location, access sequences) for users, service accounts, and edge devices. 

By combining statistical baselines with deep sequence detection, it catches deviations in near real-time. Even better, it categorizes the exact threat type and uses an **Explainable AI (XAI)** layer (via SHAP) to tell SOC analysts exactly *why* it flagged an event.

## 🏆 Key Hackathon Achievements

* **Robust Data Pipeline:** Wrote a custom data generator to handle extreme class imbalance, processing **50,182** highly-realistic access logs across 500 distinct entities with a controlled 3.35% anomaly rate.
* **Smart Alert Budgeting:** Achieved an ensemble **ROC-AUC of 0.84**. At a realistic 5% SOC Alert Budget, analysts receive high-confidence threats with a tiny 4% False Positive Rate.
* **Granular Threat Categorization:** The supervised classifier achieved an impressive **0.82 Macro F1-score**, correctly disambiguating complex attacks with massive recall:
  * **94%+ Recall** on *Device Spoofing* and *Lateral Movement*.
  * **100% Recall** on *Impossible Travel*, *Insider Drift*, and *Low & Slow Exfiltration*.

## 🏗️ How It Works (Architecture Flow)

1. **Data Ingestion:** Raw access logs are fed into the system.
2. **Statistical Profiler:** Checks basic metadata (location, time, auth failures) against historical entity baselines.
3. **MLP Sequence Detector:** Analyzes the chronological sequence of commands to catch "low & slow" attacks.
4. **Ensemble Fusion:** Combines scores from both models to create a master Risk Score.
5. **Random Forest Classifier:** If the risk breaches the threshold, this model kicks in to categorize the specific attack vector.
6. **SOC Dashboard:** Everything is visualized in a live Streamlit app with SHAP feature attributions.

## ⚙️ Quick Start

**1. Create & Activate Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate Data & Train Models**
```bash
# Generate the synthetic dataset
python data_gen/generate.py

# Train the Baseline, Sequence Detector, and Classifier
python models/train_evaluate.py
```

**4. Launch the Dashboard**
```bash
cd dashboard
streamlit run app.py
```

---
*Built from scratch by Heet Vachhani for the Honeywell Hackathon.*
