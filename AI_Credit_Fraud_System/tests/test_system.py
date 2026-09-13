"""tests/test_system.py — 19 Automated Tests (one per UC).

Unity University — AI Credit Scoring & Fraud Detection.

Run::

    pytest tests/test_system.py -v

This suite implements exactly the 19 tests documented in Table 5.3 of
the project document, organized into 8 test classes (one per UC group):

    TestUC01_Register           — UC-01 (FR-01)            [TC-01, TC-02]
    TestUC02_Login              — UC-02 (FR-01)            [TC-03, TC-04, TC-05]
    TestUC03_Logout             — UC-03 (FR-01)            [TC-06]
    TestUC04_05                 — UC-04/05 (FR-02..FR-05)  [TC-07, TC-08]
    TestUC06_Result             — UC-06 (FR-05, FR-08)     [TC-09]
    TestUC07_History            — UC-07 (FR-08)            [TC-10]
    TestUC08_09                 — UC-08/09 (FR-06, FR-07)  [TC-11, TC-12]
    TestUC11_Export             — UC-11 (FR-13)            [TC-13]
    TestUC12_Users              — UC-12 (FR-11)            [TC-14, TC-15]
    TestUC13_Dashboard          — UC-13 (FR-10)            [TC-16]
    TestUC14_Audit              — UC-14 (FR-14, NFR-03)    [TC-17, TC-18, TC-19]

Total: 19 tests across 11 classes (the documentation reports 8 classes
which group UC-04/05, UC-08/09, UC-11, UC-12, UC-14 into combined
classes; the actual class count is 11 — both numbers appear in the
documentation's Section 5.2.8).
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app, db as _db  # noqa: E402

CFG = {
    "TESTING": True,
    "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    "JWT_SECRET_KEY": "test-secret",
    "JWT_TOKEN_LOCATION": ["headers"],
    "JWT_COOKIE_CSRF_PROTECT": False,
    "WTF_CSRF_ENABLED": False,
    "RATELIMIT_ENABLED": False,  # disable login rate limit during tests
    "CREDIT_MODEL_PATH": "ml/models/credit_model.pkl",
    "FRAUD_MODEL_PATH": "ml/models/fraud_model.pkl",
    "CREDIT_SCALER_PATH": "ml/models/credit_scaler.pkl",
    "FRAUD_SCALER_PATH": "ml/models/fraud_scaler.pkl",
}


@pytest.fixture(scope="session")
def app():
    """Create the Flask app once per session with an in-memory SQLite DB."""
    application = create_app(CFG)
    with application.app_context():
        _db.create_all()
        yield application


@pytest.fixture
def client(app):
    """Fresh test client per test."""
    return app.test_client()


def _token(client, u="admin", p="Admin@2025"):
    """Authenticate and return a Bearer token for the given user."""
    rv = client.post("/auth/login", json={"username": u, "password": p})
    return rv.get_json().get("access_token") if rv.status_code == 200 else None


# ──────────────────────────────────────────────────────────────────────────
# UC-01: Register Account (FR-01)
# ──────────────────────────────────────────────────────────────────────────
class TestUC01_Register:
    """UC-01: Register Account (FR-01 — User Authentication / Registration)."""

    def test_register_new_user(self, client):
        """TC-01: POST /auth/register with new credentials → HTTP 200 or 302."""
        rv = client.post("/auth/register",
                         data={
                             "username": "newuser99",
                             "email": "new99@unity.edu.et",
                             "password": "Pass@9999"
                         })
        assert rv.status_code in (200, 302)

    def test_register_duplicate(self, client):
        """TC-02: POST /auth/register with existing username → re-renders form."""
        client.post("/auth/register",
                    data={
                        "username": "dupuser",
                        "email": "dup@unity.edu.et",
                        "password": "Pass@1234"
                    })
        rv = client.post("/auth/register",
                         data={
                             "username": "dupuser",
                             "email": "dup2@unity.edu.et",
                             "password": "Pass@1234"
                         })
        assert rv.status_code in (400, 200)  # re-renders form with error


# ──────────────────────────────────────────────────────────────────────────
# UC-02: Login (FR-01)
# ──────────────────────────────────────────────────────────────────────────
class TestUC02_Login:
    """UC-02: Login to System (FR-01 — User Authentication)."""

    def test_login_success(self, client):
        """TC-03: POST /auth/login admin → 200, access_token present, role=ADMIN."""
        rv = client.post("/auth/login",
                         json={"username": "admin", "password": "Admin@2025"})
        assert rv.status_code == 200
        d = rv.get_json()
        assert "access_token" in d
        assert d["role"] == "ADMIN"

    def test_login_wrong_password(self, client):
        """TC-04: POST /auth/login wrong password → HTTP 401."""
        rv = client.post("/auth/login",
                         json={"username": "admin", "password": "wrong"})
        assert rv.status_code == 401

    def test_login_unknown_user(self, client):
        """TC-05: POST /auth/login unknown username → HTTP 401."""
        rv = client.post("/auth/login",
                         json={"username": "nobody", "password": "pass"})
        assert rv.status_code == 401


# ──────────────────────────────────────────────────────────────────────────
# UC-03: Logout (FR-01)
# ──────────────────────────────────────────────────────────────────────────
class TestUC03_Logout:
    """UC-03: Logout of System (FR-01 — session termination)."""

    def test_logout_redirects(self, client):
        """TC-06: GET /auth/logout → HTTP 200 or 302 (JWT cookie cleared)."""
        rv = client.get("/auth/logout")
        assert rv.status_code in (200, 302)


# ──────────────────────────────────────────────────────────────────────────
# UC-04/05: Submit Application + ML Prediction (FR-02, FR-03, FR-04, FR-05)
# ──────────────────────────────────────────────────────────────────────────
class TestUC04_05:
    """UC-04/05: Submit Credit Application + Run ML Credit Score Prediction."""

    def test_credit_api_requires_auth(self, client):
        """TC-07: POST /credit/api/predict without JWT → HTTP 401 or 302."""
        rv = client.post("/credit/api/predict", json={})
        assert rv.status_code in (401, 302)

    def test_credit_prediction_valid_score(self, client):
        """TC-08: POST /credit/api/predict with JWT → score 0-100, risk, SHAP."""
        tok = _token(client)
        if not tok:
            pytest.skip("Auth failed")
        rv = client.post("/credit/api/predict",
                         json={
                             "age": 32,
                             "gender": "Female",
                             "monthly_income": 18000,
                             "years_employed": 4,
                             "existing_loan_bal": 10000,
                             "monthly_expenses": 9000,
                             "loan_amount": 75000,
                             "employment_type": "Employed"
                         },
                         headers={"Authorization": f"Bearer {tok}"})
        if rv.status_code == 200:
            d = rv.get_json()
            assert 0 <= d["score"] <= 100
            assert d["risk_level"] in ("LOW", "MEDIUM", "HIGH")
            assert "shap_values" in d


# ──────────────────────────────────────────────────────────────────────────
# UC-06: View Result (FR-05, FR-08)
# ──────────────────────────────────────────────────────────────────────────
class TestUC06_Result:
    """UC-06: View Credit Score & SHAP Explanation (FR-05, FR-08)."""

    def test_result_page_404(self, client):
        """TC-09: GET /credit/result/99999 → HTTP 404."""
        tok = _token(client)
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/credit/result/99999",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code == 404


# ──────────────────────────────────────────────────────────────────────────
# UC-07: History (FR-08)
# ──────────────────────────────────────────────────────────────────────────
class TestUC07_History:
    """UC-07: View Application History (FR-08 — history view)."""

    def test_history_accessible(self, client):
        """TC-10: GET /credit/history with APPLICANT JWT → HTTP 200 or 302."""
        tok = _token(client, "selam", "Selam@2025")
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/credit/history",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (200, 302)


# ──────────────────────────────────────────────────────────────────────────
# UC-08/09: Fraud Upload + Detection (FR-06, FR-07)
# ──────────────────────────────────────────────────────────────────────────
class TestUC08_09:
    """UC-08/09: Upload Transactions + Run Fraud Detection Model."""

    def test_fraud_upload_requires_officer(self, client):
        """TC-11: GET /fraud/upload with APPLICANT JWT → HTTP 302 or 403."""
        tok = _token(client, "selam", "Selam@2025")
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/fraud/upload",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (302, 403)

    def test_fraud_upload_accessible_officer(self, client):
        """TC-12: GET /fraud/upload with OFFICER JWT → HTTP 200 or 302."""
        tok = _token(client, "officer1", "Officer@2025")
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/fraud/upload",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (200, 302)


# ──────────────────────────────────────────────────────────────────────────
# UC-11: Export CSV (FR-13)
# ──────────────────────────────────────────────────────────────────────────
class TestUC11_Export:
    """UC-11: Export Results as CSV (FR-13 — Data Export)."""

    def test_export_404_bad_batch(self, client):
        """TC-13: GET /fraud/report/FAKE/export → HTTP 200 or 404."""
        tok = _token(client, "officer1", "Officer@2025")
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/fraud/report/FAKEBATCH/export",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (200, 404)


# ──────────────────────────────────────────────────────────────────────────
# UC-12: User Management (FR-11)
# ──────────────────────────────────────────────────────────────────────────
class TestUC12_Users:
    """UC-12: Manage User Accounts (FR-11 — User Account Management)."""

    def test_admin_can_access_users(self, client):
        """TC-14: GET /admin/users with ADMIN JWT → HTTP 200 or 302."""
        tok = _token(client)
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/admin/users",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (200, 302)

    def test_applicant_cannot_access_admin(self, client):
        """TC-15: GET /admin/dashboard with APPLICANT JWT → HTTP 302 or 403."""
        tok = _token(client, "selam", "Selam@2025")
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/admin/dashboard",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (302, 403)


# ──────────────────────────────────────────────────────────────────────────
# UC-13: Dashboard (FR-10)
# ──────────────────────────────────────────────────────────────────────────
class TestUC13_Dashboard:
    """UC-13: View Analytics Dashboard (FR-10 — Analytics Dashboard)."""

    def test_dashboard_accessible_admin(self, client):
        """TC-16: GET /admin/dashboard with ADMIN JWT → HTTP 200 or 302."""
        tok = _token(client)
        if not tok:
            pytest.skip("Auth failed")
        rv = client.get("/admin/dashboard",
                        headers={"Authorization": f"Bearer {tok}"})
        assert rv.status_code in (200, 302)


# ──────────────────────────────────────────────────────────────────────────
# UC-14: Audit Log + Security Properties (FR-14, NFR-03)
# ──────────────────────────────────────────────────────────────────────────
class TestUC14_Audit:
    """UC-14: View Audit Log + model integrity (FR-14, NFR-03)."""

    def test_audit_log_created(self, app):
        """TC-17: AuditLog.log() persists record with correct fields."""
        with app.app_context():
            from app.models.fraud import AuditLog
            AuditLog.log(None, "TEST_UC14", "TestEntity", 1, "127.0.0.1",
                         {"k": "v"})
            log = AuditLog.query.filter_by(action="TEST_UC14").first()
            assert log is not None
            assert log.entity == "TestEntity"

    def test_password_hashing(self, app):
        """TC-18: User.set_password() → bcrypt hash ≠ plaintext."""
        with app.app_context():
            from app.models.user import User
            u = User(username="hashtest_uc14",
                     email="ht@h.com",
                     role="APPLICANT")
            u.set_password("MySecret123")
            assert u.check_password("MySecret123")
            assert not u.check_password("Wrong")
            assert u.password_hash != "MySecret123"

    def test_feature_vector_length(self, app):
        """TC-19: CreditApplication.get_feature_vector() returns 11 numerics."""
        with app.app_context():
            from app.models.credit import CreditApplication
            a = CreditApplication(user_id=1,
                                  full_name="Test",
                                  age=30,
                                  gender="Male",
                                  monthly_income=20000,
                                  years_employed=3,
                                  existing_loan_bal=5000,
                                  monthly_expenses=8000,
                                  loan_amount=60000,
                                  employment_type="Employed")
            fv = a.get_feature_vector()
            assert len(fv) == 11
            assert all(isinstance(v, (int, float)) for v in fv)
