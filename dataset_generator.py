"""
dataset_generator.py
Generates realistic Network Intrusion Detection benchmark datasets
based on the standard NSL-KDD benchmark schema with 41 features.
Produces:
  - data/train_nsl_kdd.csv
  - data/test_nsl_kdd.csv
  - data/sample_live_batch.csv
Uses pure Python + NumPy with standard csv module.
"""

import os
import csv
import random
import numpy as np

FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'attack_category', 'is_intrusion'
]

def generate_records(num_samples=8000, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    cat_weights = [0.55, 0.25, 0.12, 0.06, 0.02]
    assigned_cats = random.choices(categories, weights=cat_weights, k=num_samples)

    records = []

    for cat in assigned_cats:
        if cat == 'Normal':
            proto = random.choices(['tcp', 'udp', 'icmp'], weights=[0.75, 0.20, 0.05])[0]
            if proto == 'tcp':
                service = random.choices(['http', 'smtp', 'ftp_data', 'ssh', 'domain_u'], weights=[0.55, 0.15, 0.10, 0.10, 0.10])[0]
                flag = random.choices(['SF', 'S1'], weights=[0.92, 0.08])[0]
                logged_in = 1 if service in ['http', 'smtp', 'ftp_data', 'ssh'] else 0
            elif proto == 'udp':
                service = random.choices(['domain_u', 'other'], weights=[0.8, 0.2])[0]
                flag = 'SF'
                logged_in = 0
            else:
                service = 'eco_i'
                flag = 'SF'
                logged_in = 0

            duration = int(np.random.exponential(scale=3))
            src_bytes = int(np.random.lognormal(mean=6.5, sigma=1.2)) + 40
            dst_bytes = int(np.random.lognormal(mean=7.5, sigma=1.5)) + 40
            land = 0
            wrong_fragment = 0
            urgent = 0
            hot = 0
            num_failed_logins = 0
            num_compromised = 0
            root_shell = 0
            su_attempted = 0
            num_root = 0
            num_file_creations = 0
            num_shells = 0
            num_access_files = 0
            is_host_login = 0
            is_guest_login = 0

            count = random.randint(1, 20)
            srv_count = random.randint(1, 20)
            serror_rate = round(random.uniform(0.0, 0.05), 2)
            srv_serror_rate = round(random.uniform(0.0, 0.05), 2)
            rerror_rate = round(random.uniform(0.0, 0.05), 2)
            srv_rerror_rate = round(random.uniform(0.0, 0.05), 2)
            same_srv_rate = round(random.uniform(0.85, 1.0), 2)
            diff_srv_rate = round(random.uniform(0.0, 0.15), 2)
            srv_diff_host_rate = round(random.uniform(0.0, 0.20), 2)

            dst_host_count = random.randint(10, 255)
            dst_host_srv_count = random.randint(50, 255)
            dst_host_same_srv_rate = round(random.uniform(0.8, 1.0), 2)
            dst_host_diff_srv_rate = round(random.uniform(0.0, 0.15), 2)
            dst_host_same_src_port_rate = round(random.uniform(0.0, 0.2), 2)
            dst_host_srv_diff_host_rate = round(random.uniform(0.0, 0.15), 2)
            dst_host_serror_rate = serror_rate
            dst_host_srv_serror_rate = srv_serror_rate
            dst_host_rerror_rate = rerror_rate
            dst_host_srv_rerror_rate = srv_rerror_rate

            attack_type = 'normal'
            is_intrusion = 0

        elif cat == 'DoS':
            attack_type = random.choice(['neptune', 'smurf', 'teardrop', 'pod'])
            if attack_type == 'smurf':
                proto = 'icmp'
                service = 'eco_i'
                flag = 'SF'
                src_bytes = random.randint(500, 1050)
                dst_bytes = 0
            elif attack_type == 'neptune':
                proto = 'tcp'
                service = random.choice(['private', 'http', 'other', 'telnet'])
                flag = 'S0'
                src_bytes = 0
                dst_bytes = 0
            else:
                proto = 'tcp'
                service = 'http'
                flag = random.choice(['S0', 'REJ'])
                src_bytes = random.randint(0, 100)
                dst_bytes = 0

            duration = 0
            land = 1 if random.random() < 0.05 else 0
            wrong_fragment = 3 if attack_type == 'teardrop' else 0
            urgent = 0
            hot = 0
            num_failed_logins = 0
            logged_in = 0
            num_compromised = 0
            root_shell = 0
            su_attempted = 0
            num_root = 0
            num_file_creations = 0
            num_shells = 0
            num_access_files = 0
            is_host_login = 0
            is_guest_login = 0

            count = random.randint(150, 510)
            srv_count = random.randint(120, 510)
            serror_rate = round(random.uniform(0.85, 1.0), 2) if flag == 'S0' else 0.0
            srv_serror_rate = round(random.uniform(0.85, 1.0), 2) if flag == 'S0' else 0.0
            rerror_rate = round(random.uniform(0.85, 1.0), 2) if flag == 'REJ' else 0.0
            srv_rerror_rate = round(random.uniform(0.85, 1.0), 2) if flag == 'REJ' else 0.0
            same_srv_rate = round(random.uniform(0.90, 1.0), 2)
            diff_srv_rate = round(random.uniform(0.0, 0.08), 2)
            srv_diff_host_rate = round(random.uniform(0.0, 0.05), 2)

            dst_host_count = 255
            dst_host_srv_count = random.randint(1, 30)
            dst_host_same_srv_rate = round(random.uniform(0.01, 0.15), 2)
            dst_host_diff_srv_rate = round(random.uniform(0.5, 0.95), 2)
            dst_host_same_src_port_rate = round(random.uniform(0.7, 1.0), 2)
            dst_host_srv_diff_host_rate = round(random.uniform(0.0, 0.05), 2)
            dst_host_serror_rate = serror_rate
            dst_host_srv_serror_rate = srv_serror_rate
            dst_host_rerror_rate = rerror_rate
            dst_host_srv_rerror_rate = srv_rerror_rate

            is_intrusion = 1

        elif cat == 'Probe':
            attack_type = random.choice(['satan', 'ipsweep', 'portsweep', 'nmap'])
            proto = random.choices(['tcp', 'icmp', 'udp'], weights=[0.7, 0.2, 0.1])[0]
            service = random.choice(['private', 'other', 'http', 'smtp', 'eco_i'])
            flag = random.choice(['REJ', 'RSTR', 'SH', 'S0'])

            duration = int(np.random.exponential(scale=1))
            src_bytes = random.randint(0, 50)
            dst_bytes = random.randint(0, 50)
            land = 0
            wrong_fragment = 0
            urgent = 0
            hot = 0
            num_failed_logins = 0
            logged_in = 0
            num_compromised = 0
            root_shell = 0
            su_attempted = 0
            num_root = 0
            num_file_creations = 0
            num_shells = 0
            num_access_files = 0
            is_host_login = 0
            is_guest_login = 0

            count = random.randint(20, 90)
            srv_count = random.randint(1, 10)
            serror_rate = round(random.uniform(0.2, 0.7), 2)
            srv_serror_rate = round(random.uniform(0.2, 0.7), 2)
            rerror_rate = round(random.uniform(0.3, 0.8), 2)
            srv_rerror_rate = round(random.uniform(0.3, 0.8), 2)
            same_srv_rate = round(random.uniform(0.05, 0.35), 2)
            diff_srv_rate = round(random.uniform(0.65, 1.0), 2)
            srv_diff_host_rate = round(random.uniform(0.5, 1.0), 2)

            dst_host_count = 255
            dst_host_srv_count = random.randint(1, 15)
            dst_host_same_srv_rate = round(random.uniform(0.01, 0.1), 2)
            dst_host_diff_srv_rate = round(random.uniform(0.6, 1.0), 2)
            dst_host_same_src_port_rate = round(random.uniform(0.3, 0.9), 2)
            dst_host_srv_diff_host_rate = round(random.uniform(0.3, 0.9), 2)
            dst_host_serror_rate = serror_rate
            dst_host_srv_serror_rate = srv_serror_rate
            dst_host_rerror_rate = rerror_rate
            dst_host_srv_rerror_rate = srv_rerror_rate

            is_intrusion = 1

        elif cat == 'R2L':
            attack_type = random.choice(['guess_passwd', 'warezclient', 'ftp_write'])
            proto = 'tcp'
            service = random.choice(['telnet', 'ftp', 'ssh', 'http'])
            flag = random.choice(['SF', 'RSTO'])

            duration = random.randint(15, 180)
            src_bytes = random.randint(300, 3000)
            dst_bytes = random.randint(400, 6000)
            land = 0
            wrong_fragment = 0
            urgent = 0
            hot = random.randint(1, 5)
            num_failed_logins = random.randint(2, 6)
            logged_in = 0
            num_compromised = 0
            root_shell = 0
            su_attempted = 0
            num_root = 0
            num_file_creations = 0
            num_shells = 0
            num_access_files = 0
            is_host_login = 0
            is_guest_login = 1 if attack_type == 'warezclient' else 0

            count = random.randint(2, 10)
            srv_count = random.randint(2, 10)
            serror_rate = 0.0
            srv_serror_rate = 0.0
            rerror_rate = round(random.uniform(0.0, 0.2), 2)
            srv_rerror_rate = round(random.uniform(0.0, 0.2), 2)
            same_srv_rate = 1.0
            diff_srv_rate = 0.0
            srv_diff_host_rate = 0.0

            dst_host_count = random.randint(1, 50)
            dst_host_srv_count = random.randint(1, 50)
            dst_host_same_srv_rate = 1.0
            dst_host_diff_srv_rate = 0.0
            dst_host_same_src_port_rate = round(random.uniform(0.0, 0.1), 2)
            dst_host_srv_diff_host_rate = 0.0
            dst_host_serror_rate = 0.0
            dst_host_srv_serror_rate = 0.0
            dst_host_rerror_rate = rerror_rate
            dst_host_srv_rerror_rate = srv_rerror_rate

            is_intrusion = 1

        else: # U2R
            attack_type = random.choice(['buffer_overflow', 'rootkit', 'loadmodule'])
            proto = 'tcp'
            service = random.choice(['telnet', 'ssh'])
            flag = 'SF'

            duration = random.randint(30, 250)
            src_bytes = random.randint(2000, 7000)
            dst_bytes = random.randint(3000, 12000)
            land = 0
            wrong_fragment = 0
            urgent = 0
            hot = random.randint(3, 8)
            num_failed_logins = 0
            logged_in = 1
            num_compromised = random.randint(3, 8)
            root_shell = 1
            su_attempted = 1
            num_root = random.randint(2, 8)
            num_file_creations = random.randint(1, 4)
            num_shells = 1
            num_access_files = random.randint(1, 3)
            is_host_login = 0
            is_guest_login = 0

            count = 2
            srv_count = 2
            serror_rate = 0.0
            srv_serror_rate = 0.0
            rerror_rate = 0.0
            srv_rerror_rate = 0.0
            same_srv_rate = 1.0
            diff_srv_rate = 0.0
            srv_diff_host_rate = 0.0

            dst_host_count = random.randint(1, 20)
            dst_host_srv_count = random.randint(1, 20)
            dst_host_same_srv_rate = 1.0
            dst_host_diff_srv_rate = 0.0
            dst_host_same_src_port_rate = 0.05
            dst_host_srv_diff_host_rate = 0.0
            dst_host_serror_rate = 0.0
            dst_host_srv_serror_rate = 0.0
            dst_host_rerror_rate = 0.0
            dst_host_srv_rerror_rate = 0.0

            is_intrusion = 1

        rec = [
            duration, proto, service, flag, src_bytes, dst_bytes,
            land, wrong_fragment, urgent, hot, num_failed_logins, logged_in,
            num_compromised, root_shell, su_attempted, num_root, num_file_creations,
            num_shells, num_access_files, is_host_login, is_guest_login,
            count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
            srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate,
            dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
            dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate,
            dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate,
            dst_host_srv_rerror_rate, attack_type, cat, is_intrusion
        ]
        records.append(rec)

    return records

def save_benchmark_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating comprehensive NSL-KDD benchmark datasets (Pure-Python)...")
    all_records = generate_records(num_samples=10000, random_seed=42)

    train_size = int(len(all_records) * 0.8)
    train_records = all_records[:train_size]
    test_records = all_records[train_size:]
    sample_records = random.sample(test_records, 150)

    train_path = os.path.join(output_dir, "train_nsl_kdd.csv")
    test_path = os.path.join(output_dir, "test_nsl_kdd.csv")
    batch_path = os.path.join(output_dir, "sample_live_batch.csv")

    for p, recs in [(train_path, train_records), (test_path, test_records), (batch_path, sample_records)]:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES)
            writer.writerows(recs)

    print(f" Saved Train: {train_path} ({len(train_records)} rows)")
    print(f" Saved Test:  {test_path} ({len(test_records)} rows)")
    print(f" Saved Batch: {batch_path} ({len(sample_records)} rows)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_benchmark_datasets(os.path.join(base_dir, "data"))
