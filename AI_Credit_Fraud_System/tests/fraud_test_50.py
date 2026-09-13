"""tests/fraud_test_50.py — Fraud Detection Functional Test on 50 Transactions.

Unity University — AI Credit Scoring & Fraud Detection.

Run::

    python tests/fraud_test_50.py

This script runs the fraud detection model on 50 realistic transactions
covering a mix of legitimate, suspicious, and fraudulent patterns. It
prints a per-transaction breakdown and a final summary showing how many
were classified LEGITIMATE / SUSPICIOUS / FRAUD.

The 50 transactions are designed to mirror real-world Ethiopian banking
patterns across 5 categories:
    * 25 clearly legitimate (small business-hour payments, weekdays)
    * 10 legitimate large payments (salary, rent, B2B — up to 35,000 ETB)
    * 5 suspicious (borderline amounts / odd hours / weekend)
    * 5 clearly fraudulent (large round amounts at 2-4 AM on weekends)
    * 5 clearly fraudulent (huge transfers, night, weekend)

Expected outcome (BR-02 thresholds: >0.50 FRAUD, 0.30-0.50 SUSPICIOUS,
<0.30 LEGITIMATE): roughly 35 LEGITIMATE, 5-8 SUSPICIOUS, 7-10 FRAUD.
"""
import os
import sys
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
os.environ["RATELIMIT_ENABLED"] = "False"

from app import create_app, db  # noqa: E402
from app.services.fraud_service import FraudDetectionService  # noqa: E402

CFG = {
    "TESTING": True,
    "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    "JWT_SECRET_KEY": "fraud-test-secret",
    "JWT_TOKEN_LOCATION": ["headers"],
    "JWT_COOKIE_CSRF_PROTECT": False,
    "RATELIMIT_ENABLED": False,
}

# ── 50 Test Transactions ─────────────────────────────────────────────────
TEST_TRANSACTIONS = [
    # ── 25 Clearly legitimate (small, business hours, weekdays) ──
    {"id": 1,  "amount": 350.00,   "txn_type": "PAYMENT",  "hour_of_day": 10, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small payment, weekday morning"},
    {"id": 2,  "amount": 1200.50,  "txn_type": "PAYMENT",  "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, weekday afternoon"},
    {"id": 3,  "amount": 85.00,    "txn_type": "DEBIT",    "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, weekday"},
    {"id": 4,  "amount": 500.00,   "txn_type": "PAYMENT",  "hour_of_day": 9,  "is_weekend": 0, "expected": "LEGITIMATE", "note": "Round small payment, business hours"},
    {"id": 5,  "amount": 2300.00,  "txn_type": "PAYMENT",  "hour_of_day": 15, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, weekday"},
    {"id": 6,  "amount": 75.50,    "txn_type": "CASH_IN",  "hour_of_day": 12, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, midday"},
    {"id": 7,  "amount": 1100.00,  "txn_type": "PAYMENT",  "hour_of_day": 16, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, late afternoon"},
    {"id": 8,  "amount": 450.00,   "txn_type": "DEBIT",    "hour_of_day": 13, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, afternoon"},
    {"id": 9,  "amount": 1900.00,  "txn_type": "PAYMENT",  "hour_of_day": 10, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, morning"},
    {"id": 10, "amount": 60.00,    "txn_type": "CASH_IN",  "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, afternoon"},
    {"id": 11, "amount": 800.00,   "txn_type": "PAYMENT",  "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, morning"},
    {"id": 12, "amount": 320.00,   "txn_type": "DEBIT",    "hour_of_day": 15, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, afternoon"},
    {"id": 13, "amount": 1500.00,  "txn_type": "PAYMENT",  "hour_of_day": 9,  "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, morning"},
    {"id": 14, "amount": 275.50,   "txn_type": "CASH_IN",  "hour_of_day": 12, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, midday"},
    {"id": 15, "amount": 950.00,   "txn_type": "PAYMENT",  "hour_of_day": 13, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, afternoon"},
    {"id": 16, "amount": 175.25,   "txn_type": "DEBIT",    "hour_of_day": 10, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, morning"},
    {"id": 17, "amount": 2200.00,  "txn_type": "PAYMENT",  "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, afternoon"},
    {"id": 18, "amount": 90.00,    "txn_type": "CASH_IN",  "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, morning"},
    {"id": 19, "amount": 670.00,   "txn_type": "PAYMENT",  "hour_of_day": 16, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, late afternoon"},
    {"id": 20, "amount": 410.50,   "txn_type": "DEBIT",    "hour_of_day": 12, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, midday"},
    {"id": 21, "amount": 1850.00,  "txn_type": "PAYMENT",  "hour_of_day": 9,  "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, morning"},
    {"id": 22, "amount": 240.00,   "txn_type": "CASH_IN",  "hour_of_day": 15, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, afternoon"},
    {"id": 23, "amount": 1380.00,  "txn_type": "PAYMENT",  "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Medium payment, morning"},
    {"id": 24, "amount": 525.75,   "txn_type": "DEBIT",    "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small debit, afternoon"},
    {"id": 25, "amount": 75.00,    "txn_type": "CASH_IN",  "hour_of_day": 10, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Small cash-in, morning"},

    # ── 10 Legitimate large payments (salary, rent, B2B, car — up to 100,000 ETB) ──
    # Per v2.2 spec: transactions below 100,000 ETB should be LEGITIMATE if
    # other features are good (business hours, weekday, non-round amount).
    {"id": 26, "amount": 25000.00, "txn_type": "TRANSFER", "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Salary transfer (25k), weekday morning"},
    {"id": 27, "amount": 45000.00, "txn_type": "PAYMENT",  "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "B2B payment (45k), weekday afternoon"},
    {"id": 28, "amount": 8000.00,  "txn_type": "TRANSFER", "hour_of_day": 10, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Rent payment (8k), weekday morning"},
    {"id": 29, "amount": 75000.00, "txn_type": "TRANSFER", "hour_of_day": 13, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Equipment purchase (75k), weekday afternoon"},
    {"id": 30, "amount": 12000.00, "txn_type": "PAYMENT",  "hour_of_day": 15, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Wholesale purchase (12k), weekday"},
    {"id": 31, "amount": 60000.00, "txn_type": "TRANSFER", "hour_of_day": 9,  "is_weekend": 0, "expected": "LEGITIMATE", "note": "Car downpayment (60k), weekday morning"},
    {"id": 32, "amount": 35000.00, "txn_type": "PAYMENT",  "hour_of_day": 12, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Bulk inventory (35k), weekday midday"},
    {"id": 33, "amount": 95000.00, "txn_type": "TRANSFER", "hour_of_day": 16, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Property payment (95k), weekday afternoon"},
    {"id": 34, "amount": 50000.00, "txn_type": "TRANSFER", "hour_of_day": 11, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Vehicle purchase (50k), weekday morning"},
    {"id": 35, "amount": 18000.00, "txn_type": "PAYMENT",  "hour_of_day": 14, "is_weekend": 0, "expected": "LEGITIMATE", "note": "Annual service contract (18k), weekday"},

    # ── 5 Suspicious (borderline patterns) ──
    {"id": 36, "amount": 8000.00,  "txn_type": "TRANSFER", "hour_of_day": 19, "is_weekend": 0, "expected": "SUSPICIOUS", "note": "Large transfer, evening weekday"},
    {"id": 37, "amount": 12000.00, "txn_type": "CASH_OUT", "hour_of_day": 20, "is_weekend": 1, "expected": "SUSPICIOUS", "note": "Large cash-out, evening weekend"},
    {"id": 38, "amount": 15000.00, "txn_type": "TRANSFER", "hour_of_day": 8,  "is_weekend": 1, "expected": "SUSPICIOUS", "note": "Large transfer, early morning weekend"},
    {"id": 39, "amount": 10000.00, "txn_type": "CASH_OUT", "hour_of_day": 21, "is_weekend": 0, "expected": "SUSPICIOUS", "note": "Large cash-out, late evening"},
    {"id": 40, "amount": 18000.00, "txn_type": "TRANSFER", "hour_of_day": 22, "is_weekend": 1, "expected": "SUSPICIOUS", "note": "Large transfer, late evening weekend"},

    # ── 5 Clearly fraudulent (large round, night, weekend) ──
    {"id": 41, "amount": 50000.00, "txn_type": "TRANSFER", "hour_of_day": 3,  "is_weekend": 1, "expected": "FRAUD", "note": "Large round transfer, 3 AM weekend"},
    {"id": 42, "amount": 100000.00,"txn_type": "CASH_OUT", "hour_of_day": 2,  "is_weekend": 1, "expected": "FRAUD", "note": "Huge round cash-out, 2 AM weekend"},
    {"id": 43, "amount": 75000.00, "txn_type": "TRANSFER", "hour_of_day": 23, "is_weekend": 1, "expected": "FRAUD", "note": "Large round transfer, 11 PM weekend"},
    {"id": 44, "amount": 60000.00, "txn_type": "CASH_OUT", "hour_of_day": 4,  "is_weekend": 1, "expected": "FRAUD", "note": "Large round cash-out, 4 AM weekend"},
    {"id": 45, "amount": 90000.00, "txn_type": "TRANSFER", "hour_of_day": 1,  "is_weekend": 1, "expected": "FRAUD", "note": "Huge round transfer, 1 AM weekend"},

    # ── 5 More clearly fraudulent (huge transfers, night, weekend) ──
    {"id": 46, "amount": 120000.00,"txn_type": "TRANSFER", "hour_of_day": 0,  "is_weekend": 1, "expected": "FRAUD", "note": "Massive transfer, midnight weekend"},
    {"id": 47, "amount": 80000.00, "txn_type": "CASH_OUT", "hour_of_day": 5,  "is_weekend": 1, "expected": "FRAUD", "note": "Huge cash-out, 5 AM weekend"},
    {"id": 48, "amount": 110000.00,"txn_type": "TRANSFER", "hour_of_day": 22, "is_weekend": 1, "expected": "FRAUD", "note": "Massive transfer, 10 PM weekend"},
    {"id": 49, "amount": 95000.00, "txn_type": "CASH_OUT", "hour_of_day": 3,  "is_weekend": 1, "expected": "FRAUD", "note": "Huge cash-out, 3 AM weekend"},
    {"id": 50, "amount": 150000.00,"txn_type": "TRANSFER", "hour_of_day": 2,  "is_weekend": 1, "expected": "FRAUD", "note": "Maximum transfer, 2 AM weekend"},
]


def build_fv(row):
    """Build the 10-dim feature vector expected by FraudDetectionService.

    Must match FRAUD_FEATURES in ml/train_models.py:
        [amount, hour_of_day, is_weekend,
         txn_type_TRANSFER, txn_type_PAYMENT, txn_type_CASH_OUT,
         txn_type_DEBIT, txn_type_CASH_IN,
         is_round_amount, log_amount]
    """
    TXN_TYPES = ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]
    amt = float(row["amount"])
    tt = str(row["txn_type"]).strip().upper()
    one_hot = [1 if tt == t else 0 for t in TXN_TYPES]
    is_round = 1 if (amt == int(amt) and int(amt) % 100 == 0) else 0
    return [
        amt,
        int(row["hour_of_day"]),
        int(row["is_weekend"]),
        *one_hot,
        is_round,
        float(math.log1p(amt)),
    ]


def main():
    """Run fraud detection on 50 test transactions and print results."""
    app = create_app(CFG)
    with app.app_context():
        print("=" * 100)
        print("  AI Fraud Detection — Functional Test on 50 Transactions")
        print("  Unity University — AI Credit Scoring & Fraud Detection")
        print("=" * 100)
        print(f"\nBR-02 Thresholds (per documentation Table 3.3):")
        print(f"  FRAUD       — probability > 0.50")
        print(f"  SUSPICIOUS  — 0.30 <= probability <= 0.50")
        print(f"  LEGITIMATE  — probability < 0.30")
        print(f"\nModel: Ensemble (XGBoost + RandomForest, SMOTE 25% + isotonic calib.)")
        print()

        # Load model info for context
        with open("ml/models/fraud_model_info.json") as f:
            info = json.load(f)
        print(f"Model Performance (NFR-02):")
        print(f"  ROC-AUC:   {info['roc_auc']:.4f}  (target >= 0.95)  "
              f"{'PASS' if info['roc_auc'] >= 0.95 else 'FAIL'}")
        print(f"  F1-Score:  {info['f1_score']:.4f}  (target >= 0.80)  "
              f"{'PASS' if info['f1_score'] >= 0.80 else 'FAIL'}")
        print(f"  Recall:    {info['recall']:.4f}  (target >= 0.85 for BR-07)")
        print(f"  Precision: {info['precision']:.4f}")
        print()

        # ── Run detection on all 50 transactions ──────────────────────
        results = []
        for tx in TEST_TRANSACTIONS:
            fv = build_fv(tx)
            det = FraudDetectionService.detect(fv)
            results.append({**tx, **det})

        # ── Per-transaction table ─────────────────────────────────────
        print("-" * 100)
        print(f"{'#':>3}  {'Amount (ETB)':>14}  {'Type':<10}  {'Hour':>4}  {'Wknd':>4}  "
              f"{'Probability':>11}  {'Label':<12}  {'Expected':<12}  {'Match':<6}")
        print("-" * 100)
        for r in results:
            match = "✓" if r["label"] == r["expected"] else "✗"
            print(f"{r['id']:>3}  {r['amount']:>14,.2f}  {r['txn_type']:<10}  "
                  f"{r['hour_of_day']:>4}  {r['is_weekend']:>4}  "
                  f"{r['fraud_probability']*100:>10.2f}%  "
                  f"{r['label']:<12}  {r['expected']:<12}  {match:<6}")
        print("-" * 100)

        # ── Summary ────────────────────────────────────────────────────
        legit = sum(1 for r in results if r["label"] == "LEGITIMATE")
        susp = sum(1 for r in results if r["label"] == "SUSPICIOUS")
        fraud = sum(1 for r in results if r["label"] == "FRAUD")
        total = len(results)
        print(f"\nSUMMARY — {total} Transactions Tested")
        print("=" * 70)
        print(f"  LEGITIMATE  : {legit:>3}  ({legit/total*100:.1f}%)  "
              f"probability < 0.30")
        print(f"  SUSPICIOUS  : {susp:>3}  ({susp/total*100:.1f}%)  "
              f"0.30 <= probability <= 0.50")
        print(f"  FRAUD       : {fraud:>3}  ({fraud/total*100:.1f}%)  "
              f"probability > 0.50")
        print("=" * 70)
        print(f"  TOTAL       : {total:>3}")
        print()

        # ── Expected vs Actual breakdown ───────────────────────────────
        exp_legit = sum(1 for r in results if r["expected"] == "LEGITIMATE")
        exp_susp = sum(1 for r in results if r["expected"] == "SUSPICIOUS")
        exp_fraud = sum(1 for r in results if r["expected"] == "FRAUD")
        print("EXPECTED vs ACTUAL:")
        print("-" * 70)
        print(f"  {'Category':<12}  {'Expected':>10}  {'Actual':>10}  {'Match Rate':>12}")
        print(f"  {'LEGITIMATE':<12}  {exp_legit:>10}  {legit:>10}  "
              f"{legit/exp_legit*100 if exp_legit else 0:>10.1f}%")
        print(f"  {'SUSPICIOUS':<12}  {exp_susp:>10}  {susp:>10}  "
              f"{susp/exp_susp*100 if exp_susp else 0:>10.1f}%")
        print(f"  {'FRAUD':<12}  {exp_fraud:>10}  {fraud:>10}  "
              f"{fraud/exp_fraud*100 if exp_fraud else 0:>10.1f}%")
        print()

        # ── Detection quality analysis ────────────────────────────────
        # Ground truth: txns 1-35 legitimate, 36-40 suspicious, 41-50 fraud
        gt_legit_ids = {r["id"] for r in results if r["expected"] == "LEGITIMATE"}
        gt_susp_ids = {r["id"] for r in results if r["expected"] == "SUSPICIOUS"}
        gt_fraud_ids = {r["id"] for r in results if r["expected"] == "FRAUD"}

        predicted_fraud = {r["id"] for r in results if r["label"] == "FRAUD"}
        predicted_legit_or_susp = {r["id"] for r in results
                                    if r["label"] in ("LEGITIMATE", "SUSPICIOUS")}

        # For FRAUD detection quality, treat FRAUD as positive class
        # Legitimate + Suspicious are both "not clearly fraud"
        actual_fraud = gt_fraud_ids
        actual_not_fraud = gt_legit_ids | gt_susp_ids

        tp = len(actual_fraud & predicted_fraud)  # fraud caught
        fn = len(actual_fraud - predicted_fraud)  # fraud missed
        fp = len(predicted_fraud - actual_fraud)  # legit/susp flagged as fraud
        tn = len(actual_not_fraud & predicted_legit_or_susp)

        print("DETECTION QUALITY (FRAUD as positive class):")
        print("-" * 70)
        print(f"  True Positives  (fraud caught)        : {tp}/{len(actual_fraud)}")
        print(f"  False Negatives (fraud missed)        : {fn}/{len(actual_fraud)}")
        print(f"  False Positives (legit flagged)       : {fp}/{len(actual_not_fraud)}")
        print(f"  True Negatives  (legit approved)      : {tn}/{len(actual_not_fraud)}")
        if tp + fp > 0:
            print(f"  Precision (fraud predictions correct) : {tp/(tp+fp)*100:.1f}%")
        if tp + fn > 0:
            print(f"  Recall    (fraud cases caught)       : {tp/(tp+fn)*100:.1f}%")
        print()

        # ── Detailed analysis by category ─────────────────────────────
        print("DETAILED ANALYSIS BY CATEGORY:")
        print("-" * 70)
        categories = [
            ("Small Legit (1-25)", range(1, 26)),
            ("Large Legit (26-35)", range(26, 36)),
            ("Suspicious (36-40)", range(36, 41)),
            ("Fraud (41-50)", range(41, 51)),
        ]
        for cat_name, ids in categories:
            cat_results = [r for r in results if r["id"] in ids]
            cat_legit = sum(1 for r in cat_results if r["label"] == "LEGITIMATE")
            cat_susp = sum(1 for r in cat_results if r["label"] == "SUSPICIOUS")
            cat_fraud = sum(1 for r in cat_results if r["label"] == "FRAUD")
            print(f"  {cat_name:<25}  LEGIT={cat_legit:>2}  SUSP={cat_susp:>2}  FRAUD={cat_fraud:>2}  "
                  f"(total {len(cat_results)})")
        print()

        # ── Per-transaction notes for misclassifications ──────────────
        mismatches = [r for r in results if r["label"] != r["expected"]]
        if mismatches:
            print("MISMATCH DETAILS:")
            print("-" * 70)
            for r in mismatches:
                print(f"  Txn #{r['id']:>2} — Expected: {r['expected']}, Got: {r['label']}")
                print(f"      {r['note']}")
                print(f"      ETB {r['amount']:>10,.2f}  |  {r['txn_type']:<10}  |  "
                      f"hour={r['hour_of_day']:>2}  |  weekend={'Y' if r['is_weekend'] else 'N'}")
                print(f"      → Probability: {r['fraud_probability']*100:.2f}%")
                print()
        else:
            print("No mismatches — all 50 transactions classified as expected.")
            print()

        # ── Final verdict ─────────────────────────────────────────────
        print("=" * 70)
        fp_rate = fp / len(actual_not_fraud) * 100 if actual_not_fraud else 0
        if tp >= 9 and fp_rate < 10:
            print("  VERDICT: Fraud detection is functioning correctly.")
            print(f"  Caught {tp}/{len(actual_fraud)} fraud cases.")
            print(f"  False positive rate: {fp_rate:.1f}% (target < 10%)")
        else:
            print("  VERDICT: Fraud detection needs further tuning.")
            print(f"  Caught {tp}/{len(actual_fraud)} fraud cases.")
            print(f"  False positive rate: {fp_rate:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
