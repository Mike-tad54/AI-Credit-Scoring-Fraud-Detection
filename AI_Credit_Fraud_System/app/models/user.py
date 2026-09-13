"""User data model — authentication, RBAC, and account lockout (FR-01).

This module defines the SQLAlchemy ORM entity for application users. It
implements:
    * bcrypt password hashing with **12 rounds** (NFR-03)
    * Role-Based Access Control via the ``role`` field (BR-06)
    * Account lockout after 5 failed login attempts for 15 minutes (FR-01)

The model is consumed by ``app/routes/auth.py`` (login, logout, register)
and ``app/services/rbac.py`` (the ``roles_required`` decorator).
"""
from datetime import datetime, timedelta
from app import db, bcrypt

# NFR-03: bcrypt hashing must use a minimum of 12 rounds.
BCRYPT_ROUNDS = 12


class User(db.Model):
    """Application user — Applicant, Loan Officer, or Administrator.

    Attributes:
        user_id (int): primary key.
        username (str): unique login name.
        email (str): unique contact email.
        password_hash (str): bcrypt hash (12 rounds). Plaintext is never stored.
        role (str): one of APPLICANT / OFFICER / ADMIN.
        is_active (bool): account status; False blocks login.
        failed_login_count (int): consecutive failed authentications.
        locked_until (datetime | None): if set, account is locked until this time.
        created_at (datetime): account creation timestamp.
        last_login (datetime | None): most recent successful login.
    """
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="APPLICANT")
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    # FR-01: Account lockout policy fields.
    failed_login_count = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # ── Relationships ────────────────────────────────────────────────
    applications = db.relationship("CreditApplication",
                                   backref="applicant",
                                   lazy=True,
                                   foreign_keys="CreditApplication.user_id")
    transactions = db.relationship("Transaction",
                                   backref="officer",
                                   lazy=True)
    audit_logs = db.relationship("AuditLog",
                                 backref="user",
                                 lazy=True)
    notifications = db.relationship("Notification",
                                    backref="user",
                                    lazy=True,
                                    foreign_keys="Notification.user_id")

    # ── Password security ────────────────────────────────────────────
    def set_password(self, password):
        """Hash ``password`` with bcrypt (12 rounds) and store the digest."""
        self.password_hash = bcrypt.generate_password_hash(
            password, rounds=BCRYPT_ROUNDS).decode("utf-8")

    def check_password(self, password):
        """Return True if ``password`` matches the stored hash.

        Uses bcrypt's constant-time comparison to prevent timing attacks.
        """
        return bcrypt.check_password_hash(self.password_hash, password)

    # ── FR-01: Account lockout helpers ───────────────────────────────
    def is_locked(self):
        """Return True if the account is currently locked (FR-01)."""
        return (self.locked_until is not None
                and self.locked_until > datetime.utcnow())

    def register_failed_login(self, max_attempts=5, lock_minutes=15):
        """Increment the failed-login counter; lock account if threshold hit.

        Returns:
            bool: True if the account was just locked on this call.
        """
        self.failed_login_count = (self.failed_login_count or 0) + 1
        if self.failed_login_count >= max_attempts:
            self.locked_until = datetime.utcnow() + timedelta(
                minutes=lock_minutes)
            self.failed_login_count = 0  # reset after locking
            return True
        return False

    def reset_login_attempts(self):
        """Reset failed-login counter and lock timestamp (called on success)."""
        self.failed_login_count = 0
        self.locked_until = None

    # ── Serialisation ────────────────────────────────────────────────
    def to_dict(self):
        """Return a JSON-safe dict of public user fields."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active
        }

    def __repr__(self):
        return f"<User {self.username} [{self.role}]>"
