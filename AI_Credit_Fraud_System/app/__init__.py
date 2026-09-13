"""AI Credit Scoring & Fraud Detection — Application Factory.

Unity University — Department of Computer Science — Final Project 2025/2026.

This module implements the Flask application factory pattern (create_app).
It configures all extensions (SQLAlchemy, JWT, Bcrypt, Limiter, CORS),
registers the five Blueprints (auth, credit, fraud, admin, main), creates
the database tables on first run, and seeds four demo users.

Database:
    SQLite is used as the prototype backend (per Code Sample 5.1 — the
    ``USE_SQLITE`` environment variable enables the demo to run on any
    machine without a PostgreSQL install). PostgreSQL 16 is documented
    as the production target (Table 4.3); migration is a config-only
    change via ``SQLALCHEMY_DATABASE_URI``.
"""
import os
import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

# ── Extension singletons (initialised against the app inside create_app) ──
db = SQLAlchemy()
jwt = JWTManager()
bcrypt = Bcrypt()
limiter = Limiter(key_func=get_remote_address,
                  default_limits=["200 per day", "100 per hour"])


def create_app(config_override=None):
    """Create and configure the Flask application instance.

    Args:
        config_override (dict | None): Optional config dict merged last; used
            by the pytest suite to swap in an in-memory SQLite DB and a short
            JWT secret.

    Returns:
        Flask: configured Flask application with all extensions initialised,
        Blueprints registered, tables created, and demo users seeded.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    sqlite_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data",
                     "ai_credit_fraud.db"))
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    app.config.update({
        # ── Security ────────────────────────────────────────────────────
        "SECRET_KEY":
        os.getenv("SECRET_KEY", "dev-secret-unity-2025"),
        "JWT_SECRET_KEY":
        os.getenv("JWT_SECRET_KEY", "jwt-unity-2025"),
        "JWT_TOKEN_LOCATION": ["headers", "cookies"],
        "JWT_COOKIE_SECURE":
        False,  # True in production behind HTTPS reverse proxy
        "JWT_COOKIE_CSRF_PROTECT":
        False,  # Enable in production
        "JWT_ACCESS_TOKEN_EXPIRES":
        3600,  # 1 hour per UC-S02

        # ── Database (SQLite for prototype; PostgreSQL for production) ──
        "SQLALCHEMY_DATABASE_URI":
        os.environ.get("SQLALCHEMY_DATABASE_URI",
                       f"sqlite:///{sqlite_path}"),
        "SQLALCHEMY_TRACK_MODIFICATIONS":
        False,

        # ── ML model paths ──────────────────────────────────────────────
        "CREDIT_MODEL_PATH":
        "ml/models/credit_model.pkl",
        "FRAUD_MODEL_PATH":
        "ml/models/fraud_model.pkl",
        "CREDIT_SCALER_PATH":
        "ml/models/credit_scaler.pkl",
        "FRAUD_SCALER_PATH":
        "ml/models/fraud_scaler.pkl",
    })
    if config_override:
        app.config.update(config_override)

    # Jinja filter: safely parse AuditLog.details JSON for template rendering.
    @app.template_filter("from_json")
    def _from_json(s):
        """Jinja filter that parses a JSON string into a Python object."""
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    # ── Initialise extensions ──────────────────────────────────────────
    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    limiter.init_app(app)
    CORS(app, resources={r"/api/*": {
        "origins": "*"
    }})

    # ── Register Blueprints ────────────────────────────────────────────
    from app.routes.auth import auth_bp
    from app.routes.credit import credit_bp
    from app.routes.fraud import fraud_bp
    from app.routes.admin import admin_bp
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(credit_bp, url_prefix="/credit")
    app.register_blueprint(fraud_bp, url_prefix="/fraud")
    app.register_blueprint(admin_bp, url_prefix="/admin")

    with app.app_context():
        db.create_all()
        _seed(app)
    return app


def _seed(app):
    """Seed four demo users on first run.

    Idempotent — exits early if any user already exists.
    """
    from app.models.user import User
    if User.query.count() > 0:
        return
    for u in [("admin", "admin@unity.edu.et", "ADMIN", "Admin@2025"),
              ("officer1", "officer@unity.edu.et", "OFFICER", "Officer@2025"),
              ("selam", "selam@unity.edu.et", "APPLICANT", "Selam@2025"),
              ("bereket", "bereket@unity.edu.et", "APPLICANT",
               "Bereket@2025")]:
        usr = User(username=u[0], email=u[1], role=u[2])
        usr.set_password(u[3])
        db.session.add(usr)
    db.session.commit()
    print("Demo users seeded.")
