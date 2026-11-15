"""
run_ids.py
Unified startup script for AI Intrusion Detection System.
Launches the Django SOC Dashboard (Port 8000) and/or the Streamlit Analytics Dashboard (Port 8501),
both bound to 0.0.0.0 for localhost & local network LAN access.
"""

import os
import sys
import socket
import subprocess
import argparse
import time

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def ensure_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    rf_model_path = os.path.join(models_dir, "ids_rf_binary.pkl")
    if not os.path.exists(rf_model_path):
        print("=" * 60)
        print("  Initializing Machine Learning Pipeline & Training Models...")
        print("=" * 60)
        from train_models import train_and_evaluate
        train_and_evaluate(data_dir=data_dir, models_dir=models_dir)

def main():
    parser = argparse.ArgumentParser(description="AI Network Intrusion Detection System Launcher")
    parser.add_argument("--mode", choices=["django", "streamlit", "both"], default="both",
                        help="Dashboard to run: django (port 8000), streamlit (port 8501), or both")
    args = parser.parse_args()

    ensure_models()

    lan_ip = get_lan_ip()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 65)
    print("  🛡️  AI-POWERED INTRUSION DETECTION SYSTEM (NIDS)")
    print("=" * 65)

    processes = []

    if args.mode in ["django", "both"]:
        print(f"  [Django SOC Dashboard]")
        print(f"    Localhost:       http://localhost:8000")
        print(f"    LAN Access:      http://{lan_ip}:8000")
        django_cmd = [sys.executable, os.path.join(base_dir, "manage.py"), "runserver", "0.0.0.0:8000"]
        p_django = subprocess.Popen(django_cmd)
        processes.append(p_django)

    if args.mode in ["streamlit", "both"]:
        print(f"  [Streamlit Dashboard]")
        print(f"    Localhost:       http://localhost:8501")
        print(f"    LAN Access:      http://{lan_ip}:8501")
        st_cmd = [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(base_dir, "app.py"),
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        p_st = subprocess.Popen(st_cmd)
        processes.append(p_st)

    print("=" * 65)
    print("  Press Ctrl+C to terminate running services.\n")

    try:
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    break
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
