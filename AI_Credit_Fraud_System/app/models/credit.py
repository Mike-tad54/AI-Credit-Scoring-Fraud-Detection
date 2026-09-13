"""Credit application and credit score data models.

Implements:
    * ``CreditApplication`` — the loan application form data (UC-04)
    * ``CreditScore`` — the ML-predicted score + SHAP explanation (UC-05/06)

The ``shap_values`` column is declared as ``db.JSON`` which SQLAlchemy
maps to ``JSONB`` on PostgreSQL (per Table 4.3 — Persistent Model) and to
``TEXT`` on SQLite. This satisfies the design trade-off documented in
Section 4.3 (Normalization vs. Query Performance).
"""
from datetime import datetime
import json
from app import db


class CreditApplication(db.Model):
    """A loan applicant's submission (UC-04, FR-02, FR-03).

    The ``get_feature_vector()`` method produces the 11-dim input expected
    by ``CreditScoringService.predict()`` (per UC-S05 main success flow).
    """
    __tablename__ = "credit_applications"

    app_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,
                        db.ForeignKey("users.user_id"),
                        nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20))
    region = db.Column(db.String(100))
    employment_type = db.Column(db.String(50))
    monthly_income = db.Column(db.Float, nullable=False)
    years_employed = db.Column(db.Float)
    existing_loan_bal = db.Column(db.Float, default=0.0)
    monthly_expenses = db.Column(db.Float)
    loan_amount = db.Column(db.Float, nullable=False)
    loan_purpose = db.Column(db.String(100))
    status = db.Column(db.String(20), default="PENDING")
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 1:1 dependent entity — cascade-delete with parent application.
    credit_score = db.relationship("CreditScore",
                                   backref="application",
                                   uselist=False,
                                   cascade="all, delete-orphan")

    def to_dict(self):
        """Return a JSON-safe dict of public application fields."""
        return {
            "app_id": self.app_id,
            "full_name": self.full_name,
            "age": self.age,
            "monthly_income": self.monthly_income,
            "loan_amount": self.loan_amount,
            "status": self.status
        }

    def get_feature_vector(self):
        """Build the 11-dim ML feature vector expected by the credit model.

        Order MUST match ``credit_feature_names.json`` produced by
        ``ml/train_models.py``:
            [age, gender_enc, monthly_income, years_employed,
             existing_loan_bal, monthly_expenses, loan_amount,
             debt_to_income, expense_ratio, loan_to_income,
             employment_type_enc]

        Returns:
            list[float]: 11 numeric features.
        """
        mi = self.monthly_income or 1
        dti = (self.existing_loan_bal or 0) / mi
        er = (self.monthly_expenses or 0) / mi
        lti = (self.loan_amount or 0) / mi
        ge = 1 if self.gender == "Male" else 0
        ee = {
            "Employed": 2,
            "Self-Employed": 1,
            "Unemployed": 0
        }.get(self.employment_type or "", 1)
        return [
            self.age or 30, ge, mi, self.years_employed or 0,
            self.existing_loan_bal or 0, self.monthly_expenses or 0,
            self.loan_amount or 0, dti, er, lti, ee
        ]


class CreditScore(db.Model):
    """ML-predicted credit score and SHAP explanation (UC-05/06, FR-04/05)."""
    __tablename__ = "credit_scores"

    score_id = db.Column(db.Integer, primary_key=True)
    # 1:1 with CreditApplication — UNIQUE constraint enforces cardinality.
    app_id = db.Column(db.Integer,
                       db.ForeignKey("credit_applications.app_id"),
                       nullable=False,
                       unique=True)
    score = db.Column(db.Float, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)
    # JSONB on PostgreSQL, TEXT on SQLite — stores top-5 SHAP features.
    shap_values = db.Column(db.JSON)
    model_version = db.Column(db.String(50), default="v1.0")
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_shap(self):
        """Return the parsed SHAP dict (or ``{}`` if empty)."""
        if not self.shap_values:
            return {}
        if isinstance(self.shap_values, dict):
            return self.shap_values
        try:
            return json.loads(self.shap_values)
        except (TypeError, ValueError):
            return {}

    def to_dict(self):
        """Return a JSON-safe dict of the score record."""
        return {
            "score_id": self.score_id,
            "app_id": self.app_id,
            "score": round(self.score, 2),
            "probability": round(self.probability, 4),
            "risk_level": self.risk_level,
            "shap_values": self.get_shap()
        }
