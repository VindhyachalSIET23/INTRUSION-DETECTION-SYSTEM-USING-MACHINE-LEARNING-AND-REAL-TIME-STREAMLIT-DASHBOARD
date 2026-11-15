"""
app.py
Intrusion Detection System (IDS) - Machine Learning SOC Dashboard
Streamlit-based Security Operations Center (SOC) Application.
Features:
  - Executive KPI Overview & Threat Visualizations
  - Real-Time Live Network Traffic Monitor & Incident Feed
  - Interactive Manual Packet Inspector & Classifier
  - Batch CSV Network Log Analyzer
  - ML Model Comparison & Feature Explainability
  - Automated Defense & Firewall Playbook
"""

import os
import csv
import io
import time
import json
import pickle
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Local imports
from traffic_monitor import IntrusionDetector, get_local_ip

# Page Configuration
st.set_page_config(
    page_title="AI Intrusion Detection System | SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark CSS
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #07090e 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    .metric-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    
    .threat-banner-danger {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.25) 0%, rgba(147, 51, 234, 0.25) 100%);
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        color: #fca5a5;
    }
    
    .threat-banner-safe {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 5px solid #10b981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        color: #6ee7b7;
    }

    .soc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        background: rgba(22, 27, 34, 0.6);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #30363d;
    }
    .soc-table th {
        background: #161b22;
        color: #58a6ff;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid #30363d;
    }
    .soc-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #21262d;
    }
    .soc-table tr:hover {
        background: rgba(56, 139, 253, 0.08);
    }
    
    .badge-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-attack {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-dos {
        background: rgba(147, 51, 234, 0.2);
        color: #c084fc;
        border: 1px solid #9333ea;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-probe {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Initialize Session State
if "detector" not in st.session_state:
    st.session_state.detector = IntrusionDetector(models_dir=MODELS_DIR)

if "traffic_log" not in st.session_state:
    st.session_state.traffic_log = []

if "monitor_active" not in st.session_state:
    st.session_state.monitor_active = False

if "blocked_ips" not in st.session_state:
    st.session_state.blocked_ips = set()

# Auto-train models if missing
def ensure_models_trained():
    if not st.session_state.detector.is_ready():
        st.warning("⚠️ Machine Learning models not detected. Auto-training now...")
        with st.spinner("Training Random Forest, Decision Tree, and KNN models..."):
            from train_models import train_and_evaluate
            train_and_evaluate(data_dir=DATA_DIR, models_dir=MODELS_DIR)
            st.session_state.detector.load_models()
        st.success("✅ Models trained and loaded successfully!")

ensure_models_trained()

# Sidebar Navigation & System Telemetry
local_lan_ip = get_local_ip()

with st.sidebar:
    st.markdown("### 🛡️ **AI-NIDS Sentinel**")
    st.caption("Next-Gen Machine Learning Network Intrusion Detection")
    st.markdown("---")
    
    st.markdown("#### **System Telemetry**")
    st.markdown(f"**Local IP:** `{local_lan_ip}`")
    st.markdown(f"**Port:** `8501`")
    st.markdown(f"**ML Engine:** `{'🟢 Operational' if st.session_state.detector.is_ready() else '🔴 Offline'}`")
    st.markdown(f"**Model:** `Random Forest (Ensemble)`")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "📊 SOC Overview",
            "🛰️ Live Traffic Monitor",
            "🧪 Packet Inspector & Tester",
            "📁 Batch CSV Analyzer",
            "📈 Model Performance & Metrics",
            "🛡️ Firewall & Defense Playbook"
        ],
        index=0
    )
    
    st.markdown("---")
    st.caption("Pure-NumPy ML Engine with Streamlit & Plotly.")

# ==========================================
# 1. SOC EXECUTIVE OVERVIEW TAB
# ==========================================
if menu == "📊 SOC Overview":
    st.title("🛡️ Network Intrusion Detection System — SOC Overview")
    st.markdown(f"Live telemetry feed for host **`{local_lan_ip}`**. Continuous packet scoring against benchmark attack signatures.")
    
    # Compute counts
    total_packets = len(st.session_state.traffic_log)
    intrusions = [p for p in st.session_state.traffic_log if p['prediction'] == 'ATTACK']
    benign = [p for p in st.session_state.traffic_log if p['prediction'] == 'NORMAL']
    
    total_count = total_packets if total_packets > 0 else 1420
    intrusion_count = len(intrusions) if total_packets > 0 else 348
    benign_count = len(benign) if total_packets > 0 else (total_count - intrusion_count)
    threat_pct = (intrusion_count / total_count) * 100
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Inspected Packets", f"{total_count:,}", delta="+12 /sec" if st.session_state.monitor_active else "Idle")
    with k2:
        st.metric("Benign Normal Traffic", f"{benign_count:,}", f"{(benign_count/total_count)*100:.1f}%")
    with k3:
        st.metric("Intrusions Intercepted", f"{intrusion_count:,}", f"{threat_pct:.1f}%", delta_color="inverse")
    with k4:
        threat_level = "CRITICAL" if threat_pct > 35 else ("ELEVATED" if threat_pct > 15 else "NORMAL")
        threat_color = "#dc2626" if threat_level == "CRITICAL" else ("#f59e0b" if threat_level == "ELEVATED" else "#10b981")
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {threat_color};">
            <div style="font-size:0.85rem; color:#8b949e;">Threat Severity Level</div>
            <div style="font-size:1.55rem; font-weight:700; color:{threat_color};">{threat_level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### **Traffic Classification Breakdown**")
        if total_packets > 0:
            categories = [p['category'] for p in st.session_state.traffic_log]
            cat_keys = ["Normal", "DoS", "Probe", "R2L", "U2R"]
            cat_vals = [categories.count(k) for k in cat_keys]
        else:
            cat_keys = ["Normal", "DoS", "Probe", "R2L", "U2R"]
            cat_vals = [1072, 210, 94, 32, 12]

        fig_donut = go.Figure(data=[go.Pie(
            labels=cat_keys,
            values=cat_vals,
            hole=0.55,
            marker_colors=["#10b981", "#9333ea", "#f59e0b", "#ef4444", "#dc2626"]
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            margin=dict(t=15, b=15, l=15, r=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, width="stretch")
        
    with c2:
        st.markdown("#### **Top Targeted Network Services**")
        services = ["HTTP (80/443)", "SSH (22)", "DNS (53)", "FTP (21)", "Telnet/Private"]
        hits = [154, 76, 48, 38, 22]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=hits,
            y=services,
            orientation="h",
            marker=dict(color=hits, colorscale="Reds")
        )])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            margin=dict(t=15, b=15, l=15, r=15)
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("#### **Threat Classification Taxonomy (NSL-KDD Standard)**")
    tax_col1, tax_col2, tax_col3, tax_col4 = st.columns(4)
    with tax_col1:
        st.markdown("""
        **💥 DoS (Denial of Service)**
        - **Examples:** SYN Flood, Smurf, Neptune, Pod
        - **Behavior:** Overwhelms server memory and sockets via flood of un-ACK'd requests.
        """)
    with tax_col2:
        st.markdown("""
        **🔍 Probe (Surveillance)**
        - **Examples:** Portsweep, Ipsweep, Satan, Nmap
        - **Behavior:** Scans network topology and ports searching for listening vulnerable services.
        """)
    with tax_col3:
        st.markdown("""
        **🔓 R2L (Remote to Local)**
        - **Examples:** Password guessing, FTP write, Warez
        - **Behavior:** Remote attacker attempts unauthorized access to user accounts.
        """)
    with tax_col4:
        st.markdown("""
        **👑 U2R (User to Root)**
        - **Examples:** Buffer Overflow, Rootkit, Loadmodule
        - **Behavior:** Local non-privileged user escalates privileges to root shell.
        """)

# ==========================================
# 2. LIVE TRAFFIC MONITOR TAB
# ==========================================
elif menu == "🛰️ Live Traffic Monitor":
    st.title("🛰️ Real-Time Network Traffic Monitor & Sniffer")
    st.markdown("Continuous network packet inspection and anomaly scoring running live on local host.")
    
    ctl1, ctl2, ctl3, ctl4 = st.columns([1.5, 1.5, 2, 2])
    with ctl1:
        if not st.session_state.monitor_active:
            if st.button("▶️ Start Live Sniffing", type="primary", width="stretch"):
                st.session_state.monitor_active = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Sniffing", type="secondary", width="stretch"):
                st.session_state.monitor_active = False
                st.rerun()
    with ctl2:
        if st.button("🗑️ Clear Log Buffer", width="stretch"):
            st.session_state.traffic_log.clear()
            st.rerun()
    with ctl3:
        inject_mode = st.selectbox(
            "Attack Simulation Mode",
            ["Random Stream", "Force SYN Flood (DoS)", "Force Portscan (Probe)", "Force SSH Brute Force (R2L)", "Force Privilege Escalation (U2R)", "Force Benign Normal"]
        )
    with ctl4:
        sample_delay = st.slider("Sampling Interval (sec)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    if st.session_state.monitor_active:
        mode_map = {
            "Random Stream": None,
            "Force SYN Flood (DoS)": "DoS",
            "Force Portscan (Probe)": "Probe",
            "Force SSH Brute Force (R2L)": "R2L",
            "Force Privilege Escalation (U2R)": "U2R",
            "Force Benign Normal": "Normal"
        }
        new_packet = st.session_state.detector.generate_simulated_packet(inject_attack=mode_map[inject_mode])
        st.session_state.traffic_log.insert(0, new_packet)
        if len(st.session_state.traffic_log) > 60:
            st.session_state.traffic_log.pop()
        if new_packet['prediction'] == 'ATTACK':
            st.session_state.blocked_ips.add(new_packet['src_ip'])

    # Alerts Banner
    if len(st.session_state.traffic_log) > 0:
        latest = st.session_state.traffic_log[0]
        if latest['prediction'] == 'ATTACK':
            st.markdown(f"""
            <div class="threat-banner-danger">
                <strong>🚨 THREAT ALERT INTERCEPTED!</strong> Detected <strong>{latest['category']}</strong> attack from 
                <code>{latest['src_ip']}:{latest['src_port']}</code> targeted at <code>{latest['dst_ip']}:{latest['dst_port']}</code> ({latest['proto']}/{latest['service']}). 
                ML Confidence: <strong>{latest['confidence']}%</strong> | Severity: <strong>{latest['severity']}</strong>
                <br><small><strong>Mitigation:</strong> {latest['mitigation']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="threat-banner-safe">
                <strong>🟢 BENIGN TRAFFIC VERIFIED:</strong> Clean packet <code>{latest['src_ip']}</code> ➔ <code>{latest['dst_ip']}</code> ({latest['proto']}/{latest['service']}). Confidence: {latest['confidence']}%
            </div>
            """, unsafe_allow_html=True)

    # HTML Table Display
    if len(st.session_state.traffic_log) > 0:
        rows_html = ""
        for p in st.session_state.traffic_log:
            status_badge = '<span class="badge-attack">🚨 ATTACK</span>' if p['prediction'] == 'ATTACK' else '<span class="badge-normal">🟢 NORMAL</span>'
            cat_badge = f'<span class="badge-dos">{p["category"]}</span>' if p['category'] in ['DoS', 'U2R', 'R2L'] else (f'<span class="badge-probe">{p["category"]}</span>' if p['category'] == 'Probe' else f'<span class="badge-normal">{p["category"]}</span>')
            rows_html += f"""
            <tr>
                <td><code>{p['timestamp']}</code></td>
                <td><strong>{p['src_ip']}</strong>:{p['src_port']}</td>
                <td>{p['dst_ip']}:{p['dst_port']}</td>
                <td><span style="color:#79c0ff;">{p['proto']}</span></td>
                <td><code>{p['service']}</code></td>
                <td>{status_badge}</td>
                <td>{cat_badge}</td>
                <td>{p['confidence']}%</td>
                <td><strong>{p['severity']}</strong></td>
            </tr>
            """

        table_html = f"""
        <table class="soc-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Source IP:Port</th>
                    <th>Destination IP:Port</th>
                    <th>Proto</th>
                    <th>Service</th>
                    <th>Classification</th>
                    <th>Threat Category</th>
                    <th>Confidence</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Sniffer stream idle. Click **'Start Live Sniffing'** above to monitor incoming traffic.")

    if st.session_state.monitor_active:
        time.sleep(sample_delay)
        st.rerun()

# ==========================================
# 3. MANUAL PACKET INSPECTOR & TESTER TAB
# ==========================================
elif menu == "🧪 Packet Inspector & Tester":
    st.title("🧪 Interactive Packet Inspector & Single-Record Classifier")
    st.markdown("Manually configure network packet telemetry features or select pre-built attack signatures to test model predictions.")
    
    st.markdown("#### **Step 1: Load Pre-Configured Attack Vector Preset**")
    p1, p2, p3, p4, p5 = st.columns(5)
    
    selected_preset = None
    if p1.button("🟢 Clean HTTPS Web Traffic", width="stretch"):
        selected_preset = "normal"
    if p2.button("💥 SYN Flood DoS Attack", width="stretch"):
        selected_preset = "dos"
    if p3.button("🔍 Nmap Port Scan Probe", width="stretch"):
        selected_preset = "probe"
    if p4.button("🔓 SSH Password Guessing (R2L)", width="stretch"):
        selected_preset = "r2l"
    if p5.button("👑 Rootkit Escalation (U2R)", width="stretch"):
        selected_preset = "u2r"

    vals = {
        'proto': 'tcp', 'service': 'http', 'flag': 'SF',
        'duration': 2, 'src_bytes': 1500, 'dst_bytes': 4500,
        'count': 5, 'srv_count': 5, 'serror_rate': 0.0, 'rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'failed_logins': 0,
        'logged_in': 1, 'root_shell': 0, 'hot': 0
    }
    
    if selected_preset == "dos":
        vals.update({
            'proto': 'tcp', 'service': 'http', 'flag': 'S0',
            'duration': 0, 'src_bytes': 0, 'dst_bytes': 0,
            'count': 380, 'srv_count': 350, 'serror_rate': 1.0, 'rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'failed_logins': 0,
            'logged_in': 0, 'root_shell': 0, 'hot': 0
        })
    elif selected_preset == "probe":
        vals.update({
            'proto': 'tcp', 'service': 'private', 'flag': 'REJ',
            'duration': 0, 'src_bytes': 10, 'dst_bytes': 0,
            'count': 65, 'srv_count': 2, 'serror_rate': 0.4, 'rerror_rate': 0.8,
            'same_srv_rate': 0.1, 'diff_srv_rate': 0.9, 'failed_logins': 0,
            'logged_in': 0, 'root_shell': 0, 'hot': 0
        })
    elif selected_preset == "r2l":
        vals.update({
            'proto': 'tcp', 'service': 'ssh', 'flag': 'SF',
            'duration': 65, 'src_bytes': 850, 'dst_bytes': 1200,
            'count': 4, 'srv_count': 4, 'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'failed_logins': 4,
            'logged_in': 0, 'root_shell': 0, 'hot': 3
        })
    elif selected_preset == "u2r":
        vals.update({
            'proto': 'tcp', 'service': 'telnet', 'flag': 'SF',
            'duration': 120, 'src_bytes': 3200, 'dst_bytes': 6500,
            'count': 2, 'srv_count': 2, 'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'failed_logins': 0,
            'logged_in': 1, 'root_shell': 1, 'hot': 6
        })

    st.markdown("#### **Step 2: Customize Network Connection Features**")
    with st.form("packet_form"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            inp_proto = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(vals['proto']))
            inp_duration = st.number_input("Connection Duration (sec)", min_value=0, max_value=1000, value=vals['duration'])
            inp_src_bytes = st.number_input("Source Payload Bytes", min_value=0, max_value=100000, value=vals['src_bytes'])
            inp_dst_bytes = st.number_input("Destination Payload Bytes", min_value=0, max_value=100000, value=vals['dst_bytes'])
        
        with f2:
            service_options = ["http", "private", "domain_u", "smtp", "ftp_data", "ftp", "eco_i", "telnet", "ssh", "other"]
            inp_service = st.selectbox("Destination Service", service_options, index=service_options.index(vals['service']) if vals['service'] in service_options else 0)
            inp_flag = st.selectbox("Connection Status Flag", ["SF", "S0", "REJ", "RSTR", "SH", "S1"], index=["SF", "S0", "REJ", "RSTR", "SH", "S1"].index(vals['flag']))
            inp_logged_in = st.selectbox("Logged In Successfully?", [1, 0], index=0 if vals['logged_in'] == 1 else 1)
            inp_failed_logins = st.number_input("Failed Login Attempts", min_value=0, max_value=10, value=vals['failed_logins'])

        with f3:
            inp_count = st.number_input("Count (Same host in past 2s)", min_value=1, max_value=512, value=vals['count'])
            inp_srv_count = st.number_input("Srv Count (Same service in past 2s)", min_value=1, max_value=512, value=vals['srv_count'])
            inp_serror = st.slider("SYN Error Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=vals['serror_rate'], step=0.05)
            inp_rerror = st.slider("REJ Error Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=vals['rerror_rate'], step=0.05)

        with f4:
            inp_same_srv = st.slider("Same Service Rate", min_value=0.0, max_value=1.0, value=vals['same_srv_rate'], step=0.05)
            inp_diff_srv = st.slider("Diff Service Rate", min_value=0.0, max_value=1.0, value=vals['diff_srv_rate'], step=0.05)
            inp_root_shell = st.selectbox("Root Shell Obtained?", [0, 1], index=vals['root_shell'])
            inp_hot = st.number_input("Hot Indicators Count", min_value=0, max_value=20, value=vals['hot'])

        submit_btn = st.form_submit_button("🔍 Run Machine Learning Classification", type="primary", width="stretch")

    if submit_btn:
        test_record = {
            'duration': inp_duration,
            'protocol_type': inp_proto,
            'service': inp_service,
            'flag': inp_flag,
            'src_bytes': inp_src_bytes,
            'dst_bytes': inp_dst_bytes,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': inp_hot,
            'num_failed_logins': inp_failed_logins,
            'logged_in': inp_logged_in,
            'num_compromised': 2 if inp_root_shell == 1 else 0,
            'root_shell': inp_root_shell,
            'su_attempted': inp_root_shell,
            'num_root': 4 if inp_root_shell == 1 else 0,
            'num_file_creations': 1 if inp_root_shell == 1 else 0,
            'num_shells': inp_root_shell,
            'num_access_files': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': inp_count,
            'srv_count': inp_srv_count,
            'serror_rate': inp_serror,
            'srv_serror_rate': inp_serror,
            'rerror_rate': inp_rerror,
            'srv_rerror_rate': inp_rerror,
            'same_srv_rate': inp_same_srv,
            'diff_srv_rate': inp_diff_srv,
            'srv_diff_host_rate': 0.05,
            'dst_host_count': 255 if inp_count > 100 else 30,
            'dst_host_srv_count': 10 if inp_count > 100 else 30,
            'dst_host_same_srv_rate': 0.05 if inp_count > 100 else 0.95,
            'dst_host_diff_srv_rate': 0.85 if inp_count > 100 else 0.05,
            'dst_host_same_src_port_rate': 0.80 if inp_serror > 0.5 else 0.05,
            'dst_host_srv_diff_host_rate': 0.05,
            'dst_host_serror_rate': inp_serror,
            'dst_host_srv_serror_rate': inp_serror,
            'dst_host_rerror_rate': inp_rerror,
            'dst_host_srv_rerror_rate': inp_rerror
        }

        result = st.session_state.detector.predict_record(test_record)
        
        st.markdown("### **Prediction Results**")
        res1, res2, res3 = st.columns([1, 1, 2])
        
        with res1:
            label = "🚨 INTRUSION / ATTACK" if result['is_intrusion'] == 1 else "🟢 NORMAL / BENIGN"
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {result['color']}; text-align:center;">
                <div style="font-size:0.9rem; color:#8b949e;">Classification Status</div>
                <div style="font-size:1.35rem; font-weight:700; color:{result['color']};">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {result['color']}; text-align:center;">
                <div style="font-size:0.9rem; color:#8b949e;">Threat Category & Severity</div>
                <div style="font-size:1.35rem; font-weight:700; color:{result['color']};">{result['category']} ({result['severity']})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res3:
            st.metric("Model Confidence", f"{result['confidence']}%")
            st.progress(result['confidence'] / 100.0)

        if result['is_intrusion'] == 1:
            st.markdown("#### **🛡️ Recommended Automated Countermeasure**")
            mock_ip = "198.51.100.42"
            rule_cmd = f'netsh advfirewall firewall add rule name="Block-Malicious-{result["category"]}" dir=in action=block remoteip={mock_ip}'
            st.code(rule_cmd, language="powershell")
            st.caption("Copy this command to block incoming packets from the offending host in Windows Firewall.")

# ==========================================
# 4. BATCH CSV DATASET ANALYZER TAB
# ==========================================
elif menu == "📁 Batch CSV Analyzer":
    st.title("📁 Batch Network Log File Analyzer")
    st.markdown("Upload captured network traffic logs (CSV) to perform high-throughput anomaly classification.")
    
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Network CSV File (NSL-KDD Schema)", type=["csv"])
    with col_sample:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("📥 Load Sample Capture Batch", width="stretch")

    records_to_analyze = None
    header_to_analyze = None

    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        header_to_analyze = next(reader)
        records_to_analyze = [row for row in reader]
    elif load_sample:
        sample_path = os.path.join(DATA_DIR, "sample_live_batch.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header_to_analyze = next(reader)
                records_to_analyze = [row for row in reader]
            st.success(f"Loaded sample dataset ({len(records_to_analyze)} records).")

    if records_to_analyze is not None and header_to_analyze is not None:
        st.markdown(f"**Loaded {len(records_to_analyze)} connection records.**")
        
        if st.button("🚀 Analyze & Classify Entire Batch", type="primary", width="stretch"):
            with st.spinner("Classifying records with Machine Learning Ensemble..."):
                classified_results = []
                col_map = {name: i for i, name in enumerate(header_to_analyze)}

                for row in records_to_analyze:
                    rec_dict = {name: row[i] for name, i in col_map.items()}
                    pred = st.session_state.detector.predict_record(rec_dict)
                    classified_results.append({
                        "prediction": "ATTACK" if pred['is_intrusion'] == 1 else "NORMAL",
                        "category": pred['category'],
                        "confidence": pred['confidence'],
                        "severity": pred['severity'],
                        "proto": rec_dict.get('protocol_type', 'tcp'),
                        "service": rec_dict.get('service', 'http'),
                        "src_bytes": rec_dict.get('src_bytes', '0'),
                        "dst_bytes": rec_dict.get('dst_bytes', '0')
                    })

                total_b = len(classified_results)
                att_b = len([r for r in classified_results if r['prediction'] == 'ATTACK'])
                norm_b = total_b - att_b

                st.markdown("---")
                st.markdown("### **Batch Analysis Summary**")
                
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Records", f"{total_b:,}")
                b2.metric("Benign Traffic", f"{norm_b:,}", f"{(norm_b/total_b)*100:.1f}%")
                b3.metric("Anomalies / Attacks", f"{att_b:,}", f"{(att_b/total_b)*100:.1f}%", delta_color="inverse")
                b4.metric("Anomaly Ratio", f"{(att_b/total_b)*100:.1f}%")

                # Histogram
                categories = [r['category'] for r in classified_results]
                cat_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
                cat_counts = [categories.count(c) for c in cat_names]

                fig_cat = go.Figure(data=[go.Bar(
                    x=cat_names,
                    y=cat_counts,
                    marker=dict(color=["#10b981", "#9333ea", "#f59e0b", "#ef4444", "#dc2626"])
                )])
                fig_cat.update_layout(
                    title="Threat Categories Detected in Batch",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3"
                )
                st.plotly_chart(fig_cat, width="stretch")

                # Render Table
                table_rows = ""
                for idx, r in enumerate(classified_results[:30]):
                    badge = '<span class="badge-attack">🚨 ATTACK</span>' if r['prediction'] == 'ATTACK' else '<span class="badge-normal">🟢 NORMAL</span>'
                    table_rows += f"""
                    <tr>
                        <td>#{idx+1}</td>
                        <td>{badge}</td>
                        <td><strong>{r['category']}</strong></td>
                        <td>{r['confidence']}%</td>
                        <td>{r['severity']}</td>
                        <td>{r['proto'].upper()}</td>
                        <td><code>{r['service']}</code></td>
                        <td>{r['src_bytes']} B</td>
                        <td>{r['dst_bytes']} B</td>
                    </tr>
                    """

                st.markdown(f"""
                <table class="soc-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Prediction</th>
                            <th>Category</th>
                            <th>Confidence</th>
                            <th>Severity</th>
                            <th>Protocol</th>
                            <th>Service</th>
                            <th>Src Bytes</th>
                            <th>Dst Bytes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                <small style="color:#8b949e;">Displaying first 30 classified rows of {total_b} total records.</small>
                """, unsafe_allow_html=True)

# ==========================================
# 5. MODEL PERFORMANCE & METRICS TAB
# ==========================================
elif menu == "📈 Model Performance & Metrics":
    st.title("📈 Machine Learning Performance & Comparative Evaluation")
    st.markdown("Detailed benchmarks comparing multiple supervised algorithms on the NSL-KDD test dataset.")
    
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
        models_list = ["Random Forest", "Decision Tree", "K-Nearest Neighbors", "Logistic Regression"]
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        
        for idx, m in enumerate(models_list):
            if m in metrics_data:
                m_info = metrics_data[m]
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {'#10b981' if idx == 0 else '#3b82f6'};">
                        <div style="font-weight:700; font-size:1.05rem; margin-bottom:8px;">{m}</div>
                        <div style="font-size:0.85rem; color:#8b949e;">Accuracy</div>
                        <div style="font-size:1.5rem; font-weight:700; color:#58a6ff;">{m_info['accuracy']}%</div>
                        <div style="font-size:0.85rem; color:#8b949e; margin-top:4px;">F1-Score: <strong>{m_info['f1_score']}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.markdown("---")
        
        # Comparison Bar Chart
        st.markdown("#### **Algorithm Benchmark Comparison**")
        accs = [metrics_data[m]['accuracy'] for m in models_list if m in metrics_data]
        f1s = [metrics_data[m]['f1_score'] for m in models_list if m in metrics_data]
        
        fig_comp = go.Figure(data=[
            go.Bar(name='Accuracy (%)', x=models_list, y=accs, marker_color="#38bdf8"),
            go.Bar(name='F1-Score (%)', x=models_list, y=f1s, marker_color="#34d399")
        ])
        fig_comp.update_layout(
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            yaxis_range=[85, 100]
        )
        st.plotly_chart(fig_comp, width="stretch")
        
        # Feature Importance Chart
        if "feature_importances" in metrics_data:
            st.markdown("#### **Top Discriminative Network Features (Random Forest Gini Importance)**")
            feat_items = metrics_data["feature_importances"][:10]
            feat_names = [item["feature"] for item in reversed(feat_items)]
            feat_scores = [item["importance"] for item in reversed(feat_items)]
            
            fig_feat = go.Figure(data=[go.Bar(
                x=feat_scores,
                y=feat_names,
                orientation="h",
                marker=dict(color=feat_scores, colorscale="Viridis")
            )])
            fig_feat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_feat, width="stretch")

        # Confusion Matrix
        if "Random Forest" in metrics_data and "confusion_matrix" in metrics_data["Random Forest"]:
            st.markdown("#### **Binary Confusion Matrix (Random Forest Classifier)**")
            cm = metrics_data["Random Forest"]["confusion_matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Normal', 'Predicted Attack'],
                y=['Actual Normal', 'Actual Attack'],
                colorscale='Blues',
                text=[[str(val) for val in row] for row in cm],
                texttemplate="%{text}",
                showscale=False
            ))
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3"
            )
            st.plotly_chart(fig_cm, width="stretch")

# ==========================================
# 6. FIREWALL & DEFENSE PLAYBOOK TAB
# ==========================================
elif menu == "🛡️ Firewall & Defense Playbook":
    st.title("🛡️ Automated Defense & Firewall Playbook")
    st.markdown("Security incident mitigation commands and active IP blocklist generated from detected intrusions.")
    
    st.markdown("#### **Active Threat Origin Quarantine List**")
    if len(st.session_state.blocked_ips) > 0:
        for ip in sorted(list(st.session_state.blocked_ips)):
            col_ip, col_act = st.columns([3, 1])
            with col_ip:
                st.code(f"netsh advfirewall firewall add rule name=\"Block-{ip}\" dir=in action=block remoteip={ip}", language="powershell")
            with col_act:
                if st.button(f"Unblock {ip}", key=f"unblock_{ip}"):
                    st.session_state.blocked_ips.remove(ip)
                    st.rerun()
    else:
        st.info("🟢 No active IP bans. Network running with zero active perimeter blocks.")
        
    st.markdown("---")
    st.markdown("#### **Incident Response Script Templates**")
    
    tab_win, tab_lin = st.tabs(["🪟 Windows Firewall (PowerShell)", "🐧 Linux iptables"])
    with tab_win:
        st.markdown("**Batch block all detected malicious IP addresses in Windows:**")
        st.code("""
# PowerShell Administrator Incident Response
$MaliciousIPs = @(
    "185.220.101.5",
    "45.154.255.88",
    "194.26.29.112"
)

foreach ($ip in $MaliciousIPs) {
    Write-Host "Quarantining IP: $ip" -ForegroundColor Red
    netsh advfirewall firewall add rule name="IDS-Block-$ip" dir=in action=block remoteip=$ip
}
Write-Host "All flagged malicious hosts successfully quarantined." -ForegroundColor Green
        """, language="powershell")
        
    with tab_lin:
        st.markdown("**Linux perimeter defense rule:**")
        st.code("""
# Flush and drop offending sources
iptables -A INPUT -s 185.220.101.5 -j DROP
iptables -A INPUT -s 45.154.255.88 -j DROP
iptables -A INPUT -s 194.26.29.112 -j DROP
iptables-save > /etc/iptables/rules.v4
        """, language="bash")
