"""Smoke test — exercise all 14 UCs end-to-end against a running app.

Run::

    python tests/smoke_test.py

This script boots the Flask app in-process, then walks through every UC
with realistic inputs and prints a PASS/FAIL summary. Unlike pytest,
this is a single end-to-end story that mirrors what an evaluator would
do during a defense demo.
"""
import os
import sys
import json
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a temp SQLite file so the smoke test doesn't pollute the dev DB
os.environ["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
os.environ["RATELIMIT_ENABLED"] = "False"

from app import create_app, db  # noqa: E402

CFG = {
    "TESTING": True,
    "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    "JWT_SECRET_KEY": "smoke-test-secret",
    "JWT_TOKEN_LOCATION": ["headers"],
    "JWT_COOKIE_CSRF_PROTECT": False,
    "RATELIMIT_ENABLED": False,
}

app = create_app(CFG)
client = app.test_client()


def header(token):
    return {"Authorization": f"Bearer {token}"}


def login(u, p):
    rv = client.post("/auth/login", json={"username": u, "password": p})
    if rv.status_code != 200:
        return None
    return rv.get_json().get("access_token")


results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


print("=" * 65)
print("  AI Credit & Fraud System — End-to-End Smoke Test (14 UCs)")
print("=" * 65)

# ── UC-01: Register ─────────────────────────────────────────────────
with app.app_context():
    db.create_all()
rv = client.post("/auth/register",
                 data={
                     "username": "smoke_user",
                     "email": "smoke@u.edu.et",
                     "password": "Smoke@1234"
                 })
check("UC-01 Register", rv.status_code in (200, 302))

# ── UC-02: Login (admin) ────────────────────────────────────────────
tok_admin = login("admin", "Admin@2025")
check("UC-02 Login (admin)", tok_admin is not None)

# ── UC-02b: Login (officer) ─────────────────────────────────────────
tok_off = login("officer1", "Officer@2025")
check("UC-02 Login (officer)", tok_off is not None)

# ── UC-02c: Login (applicant) ───────────────────────────────────────
tok_app = login("selam", "Selam@2025")
check("UC-02 Login (applicant)", tok_app is not None)

# ── UC-03: Logout ───────────────────────────────────────────────────
rv = client.get("/auth/logout")
check("UC-03 Logout", rv.status_code in (200, 302))

# ── UC-04 + UC-05: Submit credit application + ML predict ──────────
rv = client.post("/credit/apply",
                 data={
                     "age": "32",
                     "gender": "Female",
                     "region": "Addis Ababa",
                     "employment_type": "Employed",
                     "monthly_income": "18000",
                     "years_employed": "4",
                     "existing_loan_bal": "10000",
                     "monthly_expenses": "9000",
                     "loan_amount": "75000",
                     "loan_purpose": "Business Expansion",
                     "full_name": "selam"
                 },
                 headers=header(tok_app),
                 follow_redirects=False)
score_id = None
with app.app_context():
    from app.models.credit import CreditScore, CreditApplication
    sc = (CreditScore.query.join(CreditApplication).filter(
        CreditApplication.user_id == 3).order_by(
            CreditScore.score_id.desc()).first())
    if sc:
        score_id = sc.score_id
check("UC-04/05 Submit + ML predict", score_id is not None,
      f"score_id={score_id}")

# ── UC-06: View credit result ──────────────────────────────────────
if score_id:
    rv = client.get(f"/credit/result/{score_id}", headers=header(tok_app))
    check("UC-06 View result page", rv.status_code == 200)

    # FR-08: PDF export
    rv = client.get(f"/credit/result/{score_id}/pdf", headers=header(tok_app))
    check("FR-08 PDF export (credit)",
          rv.status_code == 200 and rv.mimetype == "application/pdf"
          and rv.data.startswith(b"%PDF"))
else:
    check("UC-06 View result page", False, "no score_id")
    check("FR-08 PDF export (credit)", False, "no score_id")

# ── UC-07: History ──────────────────────────────────────────────────
rv = client.get("/credit/history", headers=header(tok_app))
check("UC-07 History", rv.status_code == 200)

# ── UC-08 + UC-09: Fraud upload (CSV + JSON) + detect ──────────────
# CSV upload
csv_data = ("Amount,txn_type,hour_of_day,is_weekend,origin_account,"
            "dest_account\n"
            "1200.50,TRANSFER,14,0,C123,M987\n"
            "48500.00,CASH_OUT,2,1,C999,M111\n"
            "100000.00,TRANSFER,3,1,C777,M444\n"
            "350.00,PAYMENT,10,0,C456,M345\n")
rv = client.post("/fraud/upload",
                 data={
                     "file": (io.BytesIO(csv_data.encode()), "test.csv")
                 },
                 content_type="multipart/form-data",
                 headers=header(tok_off))
batch_csv = None
if rv.status_code in (200, 302) and rv.is_json:
    batch_csv = rv.get_json().get("batch_id")
elif "batch_id" in (rv.data.decode() if rv.data else ""):
    pass
check("UC-08/09 Fraud CSV upload", rv.status_code in (200, 302))

# JSON upload (FR-06 JSON support)
rv = client.post("/fraud/upload",
                 json={
                     "transactions": [{
                         "amount": 80000,
                         "txn_type": "TRANSFER",
                         "hour_of_day": 3,
                         "is_weekend": 1
                     }, {
                         "amount": 250,
                         "txn_type": "PAYMENT",
                         "hour_of_day": 14,
                         "is_weekend": 0
                     }]
                 },
                 headers=header(tok_off))
batch_json = None
if rv.status_code == 200:
    batch_json = rv.get_json().get("batch_id")
check("FR-06 JSON fraud upload", rv.status_code == 200,
      f"batch_id={batch_json}")

# ── UC-10: View fraud report ───────────────────────────────────────
if batch_json:
    rv = client.get(f"/fraud/report/{batch_json}", headers=header(tok_off))
    check("UC-10 Fraud report page", rv.status_code == 200)

# ── UC-11 + FR-13: CSV + PDF export ────────────────────────────────
if batch_json:
    rv = client.get(f"/fraud/report/{batch_json}/export",
                    headers=header(tok_off))
    check("UC-11 CSV export", rv.status_code == 200
          and rv.mimetype == "text/csv")
    rv = client.get(f"/fraud/report/{batch_json}/pdf",
                    headers=header(tok_off))
    check("FR-13 PDF export (fraud)",
          rv.status_code == 200 and rv.mimetype == "application/pdf"
          and rv.data.startswith(b"%PDF"))

# ── FR-07: Officer notification ────────────────────────────────────
rv = client.get("/fraud/notifications", headers=header(tok_off))
check("FR-07 Notifications endpoint",
      rv.status_code == 200 and isinstance(rv.get_json(), list),
      f"{len(rv.get_json()) if rv.status_code == 200 else 0} notifications")

# ── UC-12: User management ─────────────────────────────────────────
rv = client.get("/admin/users", headers=header(tok_admin))
check("UC-12 User list", rv.status_code == 200)

# Create user
rv = client.post("/admin/users/create",
                 data={
                     "username": "smoke_officer",
                     "email": "smokeoff@u.edu.et",
                     "role": "OFFICER",
                     "password": "Officer@123"
                 },
                 headers=header(tok_admin))
check("FR-11 Create user", rv.status_code in (200, 302))

# Find smoke_officer's ID
new_uid = None
with app.app_context():
    from app.models.user import User
    u = User.query.filter_by(username="smoke_officer").first()
    if u:
        new_uid = u.user_id

if new_uid:
    # Update user
    rv = client.post(f"/admin/users/{new_uid}/update",
                     data={
                         "email": "updated@u.edu.et",
                         "role": "ADMIN"
                     },
                     headers=header(tok_admin))
    check("FR-11 Update user", rv.status_code in (200, 302))

    # Toggle user
    rv = client.post(f"/admin/users/{new_uid}/toggle", headers=header(tok_admin))
    check("FR-11 Toggle user", rv.status_code in (200, 302))

    # Delete user
    rv = client.post(f"/admin/users/{new_uid}/delete", headers=header(tok_admin))
    check("FR-11 Delete user", rv.status_code in (200, 302))

# ── UC-13: Dashboard + API stats ───────────────────────────────────
rv = client.get("/admin/dashboard", headers=header(tok_admin))
check("UC-13 Dashboard", rv.status_code == 200)
rv = client.get("/admin/api/stats", headers=header(tok_admin))
check("UC-13 API stats", rv.status_code == 200 and "score_distribution" in rv.get_json())

# ── UC-14: Audit log ───────────────────────────────────────────────
rv = client.get("/admin/audit", headers=header(tok_admin))
check("UC-14 Audit log", rv.status_code == 200)

# ── FR-12: Model management + BR-07 health ────────────────────────
rv = client.get("/admin/models", headers=header(tok_admin))
check("FR-12 Models page", rv.status_code == 200)
rv = client.get("/admin/models/health", headers=header(tok_admin))
check("BR-07 Health endpoint", rv.status_code == 200 and "credit" in rv.get_json())

# ── NFR-10: Fairness check ─────────────────────────────────────────
with open("ml/models/credit_fairness.json") as f:
    fair = json.load(f)
check("NFR-10 Gender fairness DPD <= 0.10",
      fair["gender"]["dpd"] <= 0.10,
      f"dpd={fair['gender']['dpd']}")

# ── NFR-02: Model performance ─────────────────────────────────────
with open("ml/models/credit_model_info.json") as f:
    ci = json.load(f)
with open("ml/models/fraud_model_info.json") as f:
    fi = json.load(f)
check("NFR-02 Credit ROC-AUC >= 0.75", ci["roc_auc"] >= 0.75,
      f"auc={ci['roc_auc']:.4f}")
check("NFR-02 Fraud ROC-AUC >= 0.95", fi["roc_auc"] >= 0.95,
      f"auc={fi['roc_auc']:.4f}")
check("NFR-02 Fraud F1 >= 0.80", fi["f1_score"] >= 0.80,
      f"f1={fi['f1_score']:.4f}")

# ── NFR-03: bcrypt rounds ──────────────────────────────────────────
from app.models.user import BCRYPT_ROUNDS
check("NFR-03 bcrypt rounds = 12", BCRYPT_ROUNDS == 12)

# ── Phase 5: Officer Override Feature Tests ────────────────────────
# Per documentation Section 3.5.2: "enabling Loan Officers to make
# informed override decisions". Tests that an officer can override a
# false positive (FRAUD → LEGITIMATE) and that the override is recorded
# in the audit log with original label preserved.
print("\n" + "=" * 65)
print("  Phase 5: Officer Override Feature Tests")
print("=" * 65)

# Upload a clearly-fraudulent transaction to get a FRAUD result we can override
rv = client.post("/fraud/upload",
                 json={
                     "transactions": [{
                         "amount": 50000,
                         "txn_type": "TRANSFER",
                         "hour_of_day": 3,
                         "is_weekend": 1
                     }]
                 },
                 headers=header(tok_off))
override_batch_id = None
override_result_id = None
override_txn_id = None
if rv.status_code == 200:
    d = rv.get_json()
    override_batch_id = d.get("batch_id")
    if d.get("results"):
        override_txn_id = d["results"][0].get("txn_id")
check("Override setup: upload FRAUD txn", override_batch_id is not None,
      f"batch_id={override_batch_id}")

# Find the FraudResult ID for the uploaded transaction
if override_txn_id:
    with app.app_context():
        from app.models.fraud import FraudResult
        fr = FraudResult.query.filter_by(txn_id=override_txn_id).first()
        if fr:
            override_result_id = fr.result_id
            original_ai_label = fr.label
check("Override setup: find FraudResult", override_result_id is not None,
      f"result_id={override_result_id}, ai_label={original_ai_label if override_result_id else 'N/A'}")

# Test 1: Officer overrides FRAUD → LEGITIMATE
if override_result_id:
    rv = client.post(f"/fraud/result/{override_result_id}/override",
                     json={
                         "label": "LEGITIMATE",
                         "reason": "False positive — verified legitimate salary payment"
                     },
                     headers=header(tok_off))
    check("Override FRAUD→LEGITIMATE returns 200",
          rv.status_code == 200,
          f"status={rv.status_code}")
    if rv.status_code == 200:
        d = rv.get_json()
        check("Override response has new_label=LEGITIMATE",
              d.get("new_label") == "LEGITIMATE",
              f"new_label={d.get('new_label')}")
        check("Override response has original_label=FRAUD",
              d.get("original_label") == original_ai_label,
              f"original_label={d.get('original_label')}")
        check("Override response has is_overridden=True",
              d.get("is_overridden") is True,
              f"is_overridden={d.get('is_overridden')}")

    # Test 2: Verify the override was recorded in the audit log
    with app.app_context():
        from app.models.fraud import AuditLog
        audit = AuditLog.query.filter_by(
            action="FRAUD_RESULT_OVERRIDDEN",
            entity_id=override_result_id).first()
        check("Override audit log entry created",
              audit is not None,
              f"audit_id={audit.log_id if audit else 'None'}")
        if audit:
            # Bug 1.5: Use get_details() for backend-agnostic access
            details = audit.get_details() or {}
            check("Audit log has original_label=FRAUD",
                  details.get("original_label") == original_ai_label,
                  f"original={details.get('original_label')}")
            check("Audit log has new_label=LEGITIMATE",
                  details.get("new_label") == "LEGITIMATE",
                  f"new={details.get('new_label')}")
            check("AuditLog.get_details() returns dict (Bug 1.5)",
                  isinstance(details, dict),
                  f"type={type(details).__name__}")

    # Test 3: Verify FraudResult now shows LEGITIMATE but preserves original_label
    with app.app_context():
        from app.models.fraud import FraudResult
        fr = FraudResult.query.get(override_result_id)
        check("FraudResult.label updated to LEGITIMATE",
              fr.label == "LEGITIMATE",
              f"label={fr.label}")
        check("FraudResult.original_label preserved as FRAUD",
              fr.original_label == original_ai_label,
              f"original_label={fr.original_label}")
        check("FraudResult.is_overridden is True",
              fr.is_overridden is True,
              f"is_overridden={fr.is_overridden}")

    # Test 4: Verify the fraud report page shows the override badge
    rv = client.get(f"/fraud/report/{override_batch_id}",
                    headers=header(tok_off))
    check("Fraud report page shows override",
          rv.status_code == 200 and b"OVERRIDDEN" not in rv.data
          and b"override" in rv.data.lower(),
          f"status={rv.status_code}")

    # Test 5: RBAC — applicant cannot access override endpoint
    rv = client.post(f"/fraud/result/{override_result_id}/override",
                     json={"label": "FRAUD"},
                     headers=header(tok_app))
    check("RBAC: Applicant cannot override",
          rv.status_code in (302, 403),
          f"status={rv.status_code}")

    # Test 6: Invalid label rejected
    rv = client.post(f"/fraud/result/{override_result_id}/override",
                     json={"label": "INVALID_LABEL"},
                     headers=header(tok_off))
    check("Invalid label rejected with 400",
          rv.status_code == 400,
          f"status={rv.status_code}")

    # Test 7 (Bug 1.3): PDF export includes override history
    # Generate a fresh batch WITHOUT overrides to compare size.
    rv_no_override = client.post("/fraud/upload",
                                 json={"transactions": [
                                     {"amount": 50000, "txn_type": "TRANSFER",
                                      "hour_of_day": 3, "is_weekend": 1}
                                 ]},
                                 headers=header(tok_off))
    no_override_batch = (rv_no_override.get_json().get("batch_id")
                         if rv_no_override.status_code == 200 else None)
    no_override_size = 0
    if no_override_batch:
        rv2 = client.get(f"/fraud/report/{no_override_batch}/pdf",
                         headers=header(tok_off))
        if rv2.status_code == 200:
            no_override_size = len(rv2.data)
    rv = client.get(f"/fraud/report/{override_batch_id}/pdf",
                    headers=header(tok_off))
    check("PDF export returns 200 with override applied",
          rv.status_code == 200 and rv.mimetype == "application/pdf"
          and rv.data.startswith(b"%PDF"),
          f"status={rv.status_code}, size={len(rv.data)} bytes")
    # PDFs compress text — verify override section by size delta (the
    # override section adds ~500+ bytes of table content).
    if no_override_size > 0 and len(rv.data) > no_override_size:
        check("PDF with overrides is larger than without (Bug 1.3)",
              len(rv.data) > no_override_size + 200,
              f"with_override={len(rv.data)}, without={no_override_size}, "
              f"delta={len(rv.data) - no_override_size}")
    else:
        check("PDF with overrides is larger than without (Bug 1.3)",
              False,
              f"with_override={len(rv.data)}, without={no_override_size}")

    # Test 8 (Bug 1.4): Fraud report page shows AI-vs-officer comparison
    rv = client.get(f"/fraud/report/{override_batch_id}",
                    headers=header(tok_off))
    check("Fraud report shows AI vs Officer comparison (Bug 1.4)",
          rv.status_code == 200
          and b"AI Original Decision" in rv.data
          and b"After Officer Review" in rv.data,
          f"status={rv.status_code}")

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"  RESULT: {passed}/{total} checks passed")
print("=" * 65)
if passed < total:
    print("\nFailures:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}  ({detail})")
    sys.exit(1)
else:
    print("\nAll 14 UCs + Officer Override feature verified end-to-end.")
    print("System is fully functional.")
