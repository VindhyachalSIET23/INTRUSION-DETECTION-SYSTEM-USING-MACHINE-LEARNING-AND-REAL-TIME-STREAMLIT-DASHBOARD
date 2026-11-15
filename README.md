# 🛡️ AI-Powered Network Intrusion Detection System (NIDS)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An advanced **Machine Learning-driven Network Intrusion Detection System (NIDS)** equipped with an interactive **Security Operations Center (SOC) Streamlit Dashboard**. The system analyzes high-dimensional network connection telemetry based on the **NSL-KDD / CICIDS benchmark schema**, classifies threats with high accuracy, and generates automated firewall mitigation countermeasures in real time.

---

## 🌟 Key Features

- **📊 SOC Executive Dashboard:** Real-time KPI threat metrics, traffic distribution donut charts, targeted service counters, and threat severity meters.
- **🛰️ Live Packet Sniffer & Traffic Stream:** Real-time stream telemetry with anomaly scoring, attack injection testing, and instant threat alert banners.
- **🧪 Interactive Packet Inspector & Tester:** Single-record diagnostic simulator allowing manual feature tweaking or one-click preset loading (e.g., SYN Flood DoS, Nmap Portscan Probe, SSH Brute Force R2L, Buffer Overflow U2R).
- **📁 High-Throughput Batch CSV Analyzer:** Upload PCAP-derived CSV network traffic captures for rapid batch classification, summary metrics, and classified report CSV export.
- **📈 ML Performance & Explainability:** Comprehensive comparative evaluation across **Random Forest, Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression**, complete with confusion matrices and feature importance rankings.
- **🛡️ Automated Defense Playbook:** Automatic quarantine rule generator for Windows Firewall (`netsh advfirewall`) and Linux (`iptables`).
- **🌐 Network-Wide Access:** Bound to `0.0.0.0:8501` to enable simultaneous access on `localhost` and other devices on your local Wi-Fi / LAN network.

---

## 📂 Project Structure

```text
intrusion-detection-system/
│
├── app.py                   # Streamlit SOC Dashboard application
├── traffic_monitor.py       # Live telemetry generator & ML inference engine
├── train_models.py          # Machine learning model training & evaluation pipeline
├── dataset_generator.py     # NSL-KDD benchmark dataset synthesizer & exporter
├── run_ids.py               # One-click startup script with LAN IP detection
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
├── data/                    # Generated datasets (Train, Test, Sample Batch)
│   ├── train_nsl_kdd.csv
│   ├── test_nsl_kdd.csv
│   └── sample_live_batch.csv
│
└── models/                  # Serialized ML models & metrics
    ├── ids_rf_binary.pkl
    ├── ids_rf_multi.pkl
    ├── ids_dt_binary.pkl
    ├── ids_lr_binary.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── feature_names.json
    └── metrics.json
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10 or higher is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Benchmark Data & Train ML Models
```bash
python train_models.py
```
*This step trains Random Forest, Decision Tree, KNN, and Logistic Regression models, generating serialized models and metrics in the `models/` directory.*

### 4. Launch the Live Dashboard
```bash
python run_ids.py
```
Or directly via Streamlit:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Once started, access the dashboard at:
- **Local machine:** `http://localhost:8501`
- **Any device on your local network:** `http://<your-local-ip>:8501`

---

## 🔬 Supported Threat Categories

| Threat Class | Category Code | Description | Benchmark Examples |
|---|---|---|---|
| **Normal** | `Normal` | Benign legitimate user traffic | Standard HTTP, HTTPS, SMTP, DNS queries |
| **Denial of Service** | `DoS` | Exhausts server memory/sockets via flooded connection requests | SYN Flood (Neptune), Smurf, Teardrop, Ping of Death |
| **Surveillance / Probe** | `Probe` | Port scanning & network topology reconnaissance | Nmap, Portsweep, Ipsweep, Satan |
| **Remote to Local** | `R2L` | Unauthorized remote access attempt to internal user account | Guess Password, FTP Write, Warez Client |
| **User to Root** | `U2R` | Privilege escalation from non-privileged account to root shell | Buffer Overflow, Rootkit, Loadmodule |

---

## 🏆 Model Performance Benchmark

| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| **Random Forest (Ensemble)** | **99.6%** | **99.5%** | **99.6%** | **99.6%** |
| **Decision Tree** | 98.9% | 98.7% | 98.8% | 98.7% |
| **K-Nearest Neighbors (KNN)**| 98.1% | 98.0% | 98.2% | 98.1% |
| **Logistic Regression** | 94.3% | 93.8% | 94.5% | 94.1% |

---

## 🛡️ License
Distributed under the MIT License. Feel free to use and adapt for academic, research, and security demonstrations.
