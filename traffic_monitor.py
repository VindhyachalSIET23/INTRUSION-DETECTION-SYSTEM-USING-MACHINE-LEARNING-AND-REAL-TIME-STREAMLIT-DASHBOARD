"""
traffic_monitor.py
Real-Time Network Traffic Monitor and Anomaly Detection Engine.
Extracts NSL-KDD features, evaluates them against trained ML models,
and produces real-time intrusion detection alerts.
"""

import os
import time
import json
import random
import socket
import pickle
import numpy as np
from datetime import datetime

class IntrusionDetector:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.binary_model = None
        self.multi_model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self.num_cols = None
        self.cat_cols = None
        self.load_models()

    def is_ready(self):
        return (self.binary_model is not None and 
                self.scaler is not None and 
                self.encoders is not None)

    def load_models(self):
        rf_bin_path = os.path.join(self.models_dir, "ids_rf_binary.pkl")
        rf_multi_path = os.path.join(self.models_dir, "ids_rf_multi.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        encoders_path = os.path.join(self.models_dir, "encoders.pkl")
        features_path = os.path.join(self.models_dir, "feature_names.json")

        if os.path.exists(rf_bin_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path):
            try:
                with open(rf_bin_path, "rb") as f:
                    self.binary_model = pickle.load(f)
                with open(rf_multi_path, "rb") as f:
                    self.multi_model = pickle.load(f) if os.path.exists(rf_multi_path) else None
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(encoders_path, "rb") as f:
                    self.encoders = pickle.load(f)
                if os.path.exists(features_path):
                    with open(features_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.feature_names = meta.get("features", [])
                        self.num_cols = meta.get("numerical", [])
                        self.cat_cols = meta.get("categorical", [])
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                return False
        return False

    def predict_record(self, raw_record: dict):
        """
        Takes a single record dictionary of network features,
        normalizes and encodes it, then predicts Normal vs Intrusion
        and specific Attack Category.
        """
        if not self.is_ready():
            return {
                "is_intrusion": 0,
                "confidence": 0.0,
                "category": "Normal",
                "severity": "Low",
                "color": "#10b981"
            }

        # Build feature vector in exact order
        feat_vector = []
        for col in self.feature_names:
            val = raw_record.get(col, 0)
            if col in self.cat_cols:
                le = self.encoders[col]
                encoded_val = le.mapping.get(str(val), 0)
                feat_vector.append(encoded_val)
            else:
                feat_vector.append(float(val))

        X = np.array([feat_vector], dtype=np.float64)

        # Scale numerical columns
        num_indices = [self.feature_names.index(c) for c in self.num_cols]
        X[:, num_indices] = self.scaler.transform(X[:, num_indices])

        # Predict binary
        bin_pred = int(self.binary_model.predict(X)[0])
        probas = self.binary_model.predict_proba(X)[0]
        confidence = float(probas[bin_pred])

        category = "Normal"
        if bin_pred == 1 and self.multi_model is not None:
            multi_pred = int(self.multi_model.predict(X)[0])
            category = self.encoders['target_multi'].classes_[multi_pred]
            if category == 'Normal':
                category = 'DoS'
        elif bin_pred == 0:
            category = "Normal"

        severity_map = {
            'Normal': ('Low', '#10b981'),
            'Probe': ('Medium', '#f59e0b'),
            'R2L': ('High', '#ef4444'),
            'U2R': ('Critical', '#dc2626'),
            'DoS': ('Critical', '#9333ea')
        }

        sev, color = severity_map.get(category, ('High', '#ef4444'))

        return {
            "is_intrusion": bin_pred,
            "confidence": round(confidence * 100, 1),
            "category": category,
            "severity": sev,
            "color": color
        }

    def generate_simulated_packet(self, inject_attack: str = None):
        """
        Generates a realistic network telemetry packet record with source/dest IPs,
        ports, and NSL-KDD features.
        """
        if inject_attack is None:
            mode = random.choices(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], weights=[0.70, 0.15, 0.09, 0.04, 0.02])[0]
        else:
            mode = inject_attack

        timestamp = datetime.now().strftime("%H:%M:%S")

        internal_ips = [f"192.168.1.{i}" for i in [10, 15, 22, 45, 102, 105, 120]]
        external_legit_ips = ["142.250.190.46", "151.101.65.140", "104.244.42.1", "13.107.42.16"]
        external_suspicious_ips = ["185.220.101.5", "45.154.255.88", "194.26.29.112", "91.240.118.232"]

        if mode == 'Normal':
            src_ip = random.choice(internal_ips)
            dst_ip = random.choice(external_legit_ips)
            src_port = random.randint(49152, 65535)
            dst_port = random.choice([80, 443, 53, 22, 21])
            proto = 'tcp' if dst_port in [80, 443, 22, 21] else 'udp'
            service = 'http' if dst_port in [80, 443] else ('domain_u' if dst_port == 53 else 'ftp')
            flag = 'SF'
            src_bytes = random.randint(120, 2500)
            dst_bytes = random.randint(400, 15000)
            duration = random.randint(0, 5)
            logged_in = 1
            count = random.randint(1, 15)
            srv_count = random.randint(1, 15)
            serror = 0.0
            rerror = 0.0
            same_srv = round(random.uniform(0.85, 1.0), 2)
            diff_srv = round(random.uniform(0.0, 0.15), 2)
            failed_logins = 0
            compromised = 0
            root_shell = 0

        elif mode == 'DoS':
            src_ip = random.choice(external_suspicious_ips)
            dst_ip = random.choice(internal_ips)
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 8080])
            proto = 'tcp'
            service = 'http'
            flag = random.choice(['S0', 'REJ'])
            src_bytes = 0
            dst_bytes = 0
            duration = 0
            logged_in = 0
            count = random.randint(250, 510)
            srv_count = random.randint(200, 510)
            serror = round(random.uniform(0.85, 1.0), 2) if flag == 'S0' else 0.0
            rerror = round(random.uniform(0.85, 1.0), 2) if flag == 'REJ' else 0.0
            same_srv = 1.0
            diff_srv = 0.0
            failed_logins = 0
            compromised = 0
            root_shell = 0

        elif mode == 'Probe':
            src_ip = random.choice(external_suspicious_ips)
            dst_ip = random.choice(internal_ips)
            src_port = random.randint(30000, 40000)
            dst_port = random.choice([21, 22, 23, 25, 80, 110, 139, 445, 1433, 3306, 3389, 8080])
            proto = random.choice(['tcp', 'udp', 'icmp'])
            service = 'private'
            flag = random.choice(['REJ', 'RSTR', 'S0'])
            src_bytes = random.randint(0, 40)
            dst_bytes = random.randint(0, 40)
            duration = 0
            logged_in = 0
            count = random.randint(20, 90)
            srv_count = random.randint(1, 5)
            serror = round(random.uniform(0.3, 0.7), 2)
            rerror = round(random.uniform(0.4, 0.9), 2)
            same_srv = round(random.uniform(0.05, 0.25), 2)
            diff_srv = round(random.uniform(0.75, 1.0), 2)
            failed_logins = 0
            compromised = 0
            root_shell = 0

        elif mode == 'R2L':
            src_ip = random.choice(external_suspicious_ips)
            dst_ip = random.choice(internal_ips)
            src_port = random.randint(45000, 55000)
            dst_port = random.choice([21, 22, 23])
            proto = 'tcp'
            service = 'ssh' if dst_port == 22 else ('ftp' if dst_port == 21 else 'telnet')
            flag = 'SF'
            src_bytes = random.randint(400, 2000)
            dst_bytes = random.randint(600, 4000)
            duration = random.randint(15, 120)
            logged_in = 0
            count = random.randint(3, 10)
            srv_count = random.randint(3, 10)
            serror = 0.0
            rerror = 0.0
            same_srv = 1.0
            diff_srv = 0.0
            failed_logins = random.randint(2, 6)
            compromised = 0
            root_shell = 0

        else: # U2R
            src_ip = random.choice(internal_ips)
            dst_ip = random.choice(internal_ips)
            src_port = random.randint(40000, 50000)
            dst_port = 22
            proto = 'tcp'
            service = 'ssh'
            flag = 'SF'
            src_bytes = random.randint(2500, 6000)
            dst_bytes = random.randint(4000, 10000)
            duration = random.randint(45, 200)
            logged_in = 1
            count = 2
            srv_count = 2
            serror = 0.0
            rerror = 0.0
            same_srv = 1.0
            diff_srv = 0.0
            failed_logins = 0
            compromised = random.randint(3, 8)
            root_shell = 1

        raw_record = {
            'duration': duration,
            'protocol_type': proto,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 2 if mode == 'U2R' else (1 if mode == 'R2L' else 0),
            'num_failed_logins': failed_logins,
            'logged_in': logged_in,
            'num_compromised': compromised,
            'root_shell': root_shell,
            'su_attempted': 1 if mode == 'U2R' else 0,
            'num_root': 4 if mode == 'U2R' else 0,
            'num_file_creations': 2 if mode == 'U2R' else 0,
            'num_shells': 1 if mode == 'U2R' else 0,
            'num_access_files': 1 if mode == 'U2R' else 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror,
            'srv_serror_rate': serror,
            'rerror_rate': rerror,
            'srv_rerror_rate': rerror,
            'same_srv_rate': same_srv,
            'diff_srv_rate': diff_srv,
            'srv_diff_host_rate': 0.05,
            'dst_host_count': 255 if mode in ['DoS', 'Probe'] else 30,
            'dst_host_srv_count': 10 if mode in ['DoS', 'Probe'] else 30,
            'dst_host_same_srv_rate': 0.05 if mode in ['DoS', 'Probe'] else 0.95,
            'dst_host_diff_srv_rate': 0.85 if mode in ['DoS', 'Probe'] else 0.05,
            'dst_host_same_src_port_rate': 0.80 if mode == 'DoS' else 0.05,
            'dst_host_srv_diff_host_rate': 0.05,
            'dst_host_serror_rate': serror,
            'dst_host_srv_serror_rate': serror,
            'dst_host_rerror_rate': rerror,
            'dst_host_srv_rerror_rate': rerror,
        }

        eval_res = self.predict_record(raw_record)

        packet_summary = {
            'timestamp': timestamp,
            'src_ip': src_ip,
            'src_port': src_port,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'proto': proto.upper(),
            'service': service,
            'flag': flag,
            'length': src_bytes + dst_bytes,
            'prediction': 'ATTACK' if eval_res['is_intrusion'] == 1 else 'NORMAL',
            'category': eval_res['category'],
            'confidence': eval_res['confidence'],
            'severity': eval_res['severity'],
            'color': eval_res['color'],
            'raw_features': raw_record
        }

        if eval_res['is_intrusion'] == 1:
            packet_summary['firewall_rule'] = f'netsh advfirewall firewall add rule name="Block-{src_ip}" dir=in action=block remoteip={src_ip}'
            packet_summary['mitigation'] = f"Isolate Source IP {src_ip} via firewall rule and terminate socket connection."
        else:
            packet_summary['firewall_rule'] = "None required (Traffic verified benign)"
            packet_summary['mitigation'] = "Allow packet through normal gateway."

        return packet_summary

def generate_simulated_packet(inject_attack=None):
    """Module-level convenience wrapper around IntrusionDetector.generate_simulated_packet."""
    det = IntrusionDetector()
    return det.generate_simulated_packet(inject_attack=inject_attack)

def get_local_ip():
    """Returns local LAN IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
