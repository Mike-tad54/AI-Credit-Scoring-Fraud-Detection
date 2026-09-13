"""
run.py — Application Entry Point
Unity University — AI Credit Scoring & Fraud Detection — 2025/2026

SETUP (run once):
    pip install -r requirements.txt
    python ml/train_models.py

START:
    python run.py
    → Open http://localhost:5000

Demo credentials:
    admin    / Admin@2025    (Administrator)
    officer1 / Officer@2025  (Loan Officer)
    selam    / Selam@2025    (Loan Applicant)
    bereket  / Bereket@2025  (Loan Applicant)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Unity University — AI Credit Scoring & Fraud Detection  ║")
    print(f"║  http://localhost:{port}  — Press CTRL+C to stop           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=port, debug=True)
