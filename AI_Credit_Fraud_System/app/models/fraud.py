"""Fraud detection and audit-log data models.

Implements:
    * ``Transaction`` — a single financial transaction submitted for fraud
      analysis (UC-08).
    * ``FraudResult`` — the ensemble model's prediction + SHAP explanation
      for a transaction (UC-09).
    * ``AuditLog`` — append-only system audit trail (FR-14, BR-05).
    * ``Notification`` — officer-facing fraud alert inbox (FR-07).

``shap_values`` columns are declared as ``db.JSON`` (maps to JSONB on
PostgreSQL per Table 4.3, TEXT on SQLite).
"""
from datetime import datetime
import json
from app import db


class Transaction(db.Model):
    """A single financial transaction analysed for fraud (UC-08, FR-06)."""
    __tablename__ = "transactions"

    txn_id = db.Column(db.Integer, primary_key=True)
    officer_id = db.Column(db.Integer,
                           db.ForeignKey("users.user_id"),
                           nullable=False)
    amount = db.Column(db.Float, nullable=False)
    txn_type = db.Column(db.String(50))
    merchant_id = db.Column(db.String(100))
    txn_time = db.Column(db.DateTime, default=datetime.utcnow)
    origin_account = db.Column(db.String(50))
    dest_account = db.Column(db.String(50))
    batch_id = db.Column(db.String(50))

    # 1:1 dependent entity — cascade-delete with parent transaction.
    fraud_result = db.relationship("FraudResult",
                                   backref="transaction",
                                   uselist=False,
                                   cascade="all, delete-orphan")

    def to_dict(self):
        """Return a JSON-safe dict of public transaction fields."""
        return {
            "txn_id": self.txn_id,
            "amount": self.amount,
            "txn_type": self.txn_type,
            "batch_id": self.batch_id
        }


class FraudResult(db.Model):
    """Ensemble fraud model's prediction for a transaction (UC-09, FR-06/07).

    Supports officer override (per documentation Section 3.5.2 — "enabling
    Loan Officers to make informed override decisions"). When an officer
    overrides the AI's label, the new label is stored in ``label`` (so
    dashboards/reports show the corrected value) and the original AI
    label is preserved in ``original_label`` for audit transparency.
    """
    __tablename__ = "fraud_results"

    result_id = db.Column(db.Integer, primary_key=True)
    txn_id = db.Column(db.Integer,
                       db.ForeignKey("transactions.txn_id"),
                       nullable=False,
                       unique=True)
    fraud_probability = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(15), nullable=False)
    shap_values = db.Column(db.JSON)  # JSONB on PostgreSQL
    model_version = db.Column(db.String(50), default="v1.0")
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    # FR-07: when True, officer has been notified of this fraud.
    alert_sent = db.Column(db.Boolean, default=False)

    # ── Officer override fields (per documentation Section 3.5.2) ──────
    # When an officer overrides the AI's decision, these fields capture
    # the original AI label, who overrode it, when, and why. The ``label``
    # field above is updated to the new (corrected) label so all
    # dashboards and reports show the officer's final decision.
    original_label = db.Column(db.String(15), nullable=True)  # AI's original label
    overridden_by = db.Column(db.Integer,
                              db.ForeignKey("users.user_id"),
                              nullable=True)
    overridden_at = db.Column(db.DateTime, nullable=True)
    override_reason = db.Column(db.String(500), nullable=True)

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

    @property
    def is_overridden(self):
        """Return True if an officer has overridden the AI's label."""
        return self.overridden_by is not None

    def override(self, new_label, overridden_by, reason=None):
        """Override the AI's fraud label with an officer's corrected decision.

        Preserves the original AI label in ``original_label`` (only on the
        first override — subsequent overrides update ``label`` but don't
        overwrite the original). Updates ``label`` to ``new_label`` so all
        dashboards and reports reflect the officer's decision.

        Args:
            new_label (str): the corrected label — FRAUD, SUSPICIOUS, or LEGITIMATE.
            overridden_by (int): user_id of the officer/admin making the override.
            reason (str | None): optional explanation for audit trail.
        """
        # Preserve original AI label on first override only.
        if self.original_label is None:
            self.original_label = self.label
        self.label = new_label
        self.overridden_by = overridden_by
        self.overridden_at = datetime.utcnow()
        self.override_reason = reason

    def to_dict(self):
        """Return a JSON-safe dict of the fraud result."""
        return {
            "result_id": self.result_id,
            "txn_id": self.txn_id,
            "fraud_probability": round(self.fraud_probability, 4),
            "label": self.label,
            "alert_sent": self.alert_sent,
            "is_overridden": self.is_overridden,
            "original_label": self.original_label,
            "overridden_by": self.overridden_by,
            "overridden_at": (self.overridden_at.isoformat()
                              if self.overridden_at else None),
            "override_reason": self.override_reason,
        }


class AuditLog(db.Model):
    """Append-only system audit trail (FR-14, BR-05).

    The table is INSERT-only: no UPDATE or DELETE methods are exposed.
    ``details`` is JSONB for variable-length structured context.
    """
    __tablename__ = "audit_logs"

    log_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,
                        db.ForeignKey("users.user_id"),
                        nullable=True)
    action = db.Column(db.String(100), nullable=False)
    entity = db.Column(db.String(50))
    entity_id = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50))
    details = db.Column(db.JSON)  # JSONB on PostgreSQL

    @staticmethod
    def log(user_id,
            action,
            entity=None,
            entity_id=None,
            ip=None,
            details=None):
        """Insert a new audit log entry (BR-05 — append-only).

        Args:
            user_id (int | None): acting user's ID, or None for system events.
            action (str): short action code, e.g. ``LOGIN``.
            entity (str | None): affected entity name.
            entity_id (int | None): affected entity primary key.
            ip (str | None): request remote address.
            details (dict | None): structured context (stored as JSONB).
        """
        e = AuditLog(user_id=user_id,
                     action=action,
                     entity=entity,
                     entity_id=entity_id,
                     ip_address=ip,
                     details=details if isinstance(details,
                                                   (dict, list)) else
                     (json.dumps(details) if details else None))
        db.session.add(e)
        db.session.commit()

    def get_details(self):
        """Return the ``details`` field as a dict, regardless of backend.

        Bug fix 1.5: On PostgreSQL, ``details`` is stored as JSONB and
        SQLAlchemy returns it as a dict directly. On SQLite, it's stored
        as TEXT and returned as a JSON string. This method normalises
        both cases so calling code can always do ``audit.get_details().get("key")``
        without worrying about the backend type.

        Returns:
            dict | list | None: the parsed details, or ``None`` if empty.
        """
        if not self.details:
            return None
        if isinstance(self.details, (dict, list)):
            return self.details
        try:
            return json.loads(self.details)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None


class Notification(db.Model):
    """Officer-facing alert inbox entry (FR-07).

    When the fraud model labels a transaction FRAUD, a Notification is
    created for the responsible officer and surfaced via a red bell in
    the navbar. Marking a notification read sets ``is_read=True``.
    """
    __tablename__ = "notifications"

    notification_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer,
                        db.ForeignKey("users.user_id"),
                        nullable=False)
    txn_id = db.Column(db.Integer,
                       db.ForeignKey("transactions.txn_id"),
                       nullable=True)
    batch_id = db.Column(db.String(50), nullable=True)
    message = db.Column(db.String(500), nullable=False)
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Return a JSON-safe dict for API responses."""
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "txn_id": self.txn_id,
            "batch_id": self.batch_id,
            "message": self.message,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat()
        }
