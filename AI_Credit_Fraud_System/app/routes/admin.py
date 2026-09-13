"""Admin Blueprint — FR-10 Dashboard, FR-11 Users, FR-12 ML Models, FR-14 Audit.

Routes:
    GET  /admin/dashboard             — analytics dashboard (UC-13, FR-10)
    GET  /admin/users                 — list users (UC-12, FR-11)
    POST /admin/users/create          — create user (FR-11)
    POST /admin/users/<id>/update     — update user role/email (FR-11)
    POST /admin/users/<id>/toggle     — activate/deactivate user (FR-11)
    POST /admin/users/<id>/delete     — delete user (FR-11, last-admin guard)
    GET  /admin/models                — model cards + fairness (FR-12, NFR-10)
    POST /admin/models/retrain        — trigger retraining job (FR-12, BR-07)
    POST /admin/models/<name>/switch  — switch active model version (FR-12)
    GET  /admin/models/health         — model health check (BR-07)
    GET  /admin/audit                 — paginated audit log (UC-14, FR-14)
    GET  /admin/api/stats             — JSON chart data for dashboard
"""
import json
import os
import glob
import subprocess
import threading
from datetime import datetime
from flask import (Blueprint, render_template, request, redirect, url_for,
                   flash, jsonify)
from app import db
from app.models.user import User
from app.models.credit import CreditApplication, CreditScore
from app.models.fraud import Transaction, FraudResult, AuditLog
from app.services.rbac import roles_required, current_user_id

admin_bp = Blueprint("admin", __name__)

# ── FR-12: retrain job status (in-memory; survives across requests) ──
_RETRAIN_STATUS = {"running": False, "started_at": None, "log": ""}


@admin_bp.route("/dashboard")
@roles_required("ADMIN")
def dashboard():
    """UC-13: View Analytics Dashboard (FR-10)."""
    ta = CreditApplication.query.count()
    sa = CreditApplication.query.filter_by(status="SCORED").count()
    tt = Transaction.query.count()
    ft = FraudResult.query.filter_by(label="FRAUD").count()
    tu = User.query.count()
    fr = round((ft / tt * 100) if tt else 0, 2)
    rs = (CreditScore.query.order_by(
        CreditScore.generated_at.desc()).limit(5).all())
    rf = (FraudResult.query.filter_by(label="FRAUD").order_by(
        FraudResult.detected_at.desc()).limit(5).all())
    return render_template("admin/dashboard.html",
                           total_apps=ta,
                           scored_apps=sa,
                           total_txns=tt,
                           fraud_txns=ft,
                           fraud_rate=fr,
                           total_users=tu,
                           recent_scores=rs,
                           recent_fraud=rf)


@admin_bp.route("/users")
@roles_required("ADMIN")
def users():
    """UC-12: List all users (FR-11)."""
    return render_template("admin/users.html",
                           users=User.query.order_by(
                               User.created_at.desc()).all())


@admin_bp.route("/users/create", methods=["POST"])
@roles_required("ADMIN")
def create_user():
    """FR-11: Create a new user."""
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    role = request.form.get("role", "APPLICANT")
    password = request.form.get("password", "")
    if not username or not email or not password:
        flash("Username, email, and password are all required.", "danger")
        return redirect(url_for("admin.users"))
    if User.query.filter_by(username=username).first():
        flash("Username already exists.", "danger")
        return redirect(url_for("admin.users"))
    if User.query.filter_by(email=email).first():
        flash("Email already registered.", "danger")
        return redirect(url_for("admin.users"))
    u = User(username=username, email=email, role=role)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    AuditLog.log(current_user_id(), "USER_CREATED", "User", u.user_id)
    flash(f"User {username} created.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/update", methods=["POST"])
@roles_required("ADMIN")
def update_user(user_id):
    """FR-11: Update an existing user's role and email."""
    user = User.query.get_or_404(user_id)
    new_email = request.form.get("email", "").strip()
    new_role = request.form.get("role", user.role)
    if new_email and new_email != user.email:
        if User.query.filter_by(email=new_email).first():
            flash("Email already in use by another account.", "danger")
            return redirect(url_for("admin.users"))
        user.email = new_email
    if new_role in ("APPLICANT", "OFFICER", "ADMIN"):
        user.role = new_role
    db.session.commit()
    AuditLog.log(current_user_id(), "USER_UPDATED", "User", user.user_id,
                 None, {"username": user.username, "role": user.role})
    flash(f"User {user.username} updated.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@roles_required("ADMIN")
def toggle_user(user_id):
    """FR-11: Activate or deactivate a user."""
    user = User.query.get_or_404(user_id)
    # BR-06: never deactivate the last active admin.
    if (user.role == "ADMIN" and user.is_active
            and User.query.filter_by(role="ADMIN", is_active=True).count() <= 1):
        flash("Cannot deactivate the last admin account.", "danger")
        return redirect(url_for("admin.users"))
    user.is_active = not user.is_active
    db.session.commit()
    AuditLog.log(current_user_id(), "USER_TOGGLED", "User", user_id)
    flash(f"User {user.username} "
          f"{'activated' if user.is_active else 'deactivated'}.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@roles_required("ADMIN")
def delete_user(user_id):
    """FR-11: Permanently delete a user (with last-admin guard)."""
    user = User.query.get_or_404(user_id)
    if user.role == "ADMIN" and User.query.filter_by(
            role="ADMIN", is_active=True).count() <= 1:
        flash("Cannot delete the last admin account.", "danger")
        return redirect(url_for("admin.users"))
    username = user.username
    db.session.delete(user)
    db.session.commit()
    AuditLog.log(current_user_id(), "USER_DELETED", "User", user_id, None,
                 {"username": username})
    flash(f"User {username} deleted.", "success")
    return redirect(url_for("admin.users"))


# ── FR-12: ML Model Management ──────────────────────────────────────────
@admin_bp.route("/models")
@roles_required("ADMIN")
def models():
    """FR-12 + Objective 7 / NFR-10: Model versions, metrics, and fairness."""
    model_info = {}
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    for name, path in [("credit", "ml/models/credit_model_info.json"),
                       ("fraud", "ml/models/fraud_model_info.json")]:
        full = os.path.join(root, path)
        if os.path.exists(full):
            with open(full) as f:
                model_info[name] = json.load(f)
        else:
            model_info[name] = {
                "model_name": "Not trained yet",
                "roc_auc": 0,
                "f1_score": 0,
                "accuracy": 0,
                "version": "N/A",
                "trained_at": "N/A",
            }

    credit_count = CreditScore.query.count()
    fraud_count = FraudResult.query.count()

    # ── Available model versions (FR-12 — version switching) ──────────
    available_versions = {"credit": [], "fraud": []}
    for kind in ("credit", "fraud"):
        pattern = os.path.join(root, "ml", "models", f"{kind}_model_v*.pkl")
        for p in sorted(glob.glob(pattern)):
            available_versions[kind].append(os.path.basename(p))

    # ── Training-time fairness (saved during model training) ──────────
    fairness = {}
    fp = os.path.join(root, "ml/models/credit_fairness.json")
    if os.path.exists(fp):
        with open(fp) as f:
            fairness = json.load(f)

    # ── Live fairness from actual DB records (gender + region) ────────
    from collections import defaultdict
    live_fair = {"gender": {}, "region": {}}
    pairs = (db.session.query(CreditApplication, CreditScore).join(
        CreditScore,
        CreditApplication.app_id == CreditScore.app_id).all())
    if pairs:
        g_data = defaultdict(list)
        r_data = defaultdict(list)
        for app, sc in pairs:
            appr = 1 if sc.score >= 70 else 0
            if app.gender:
                g_data[app.gender].append(appr)
            if app.region:
                r_data[app.region].append(appr)
        live_fair["gender"] = {
            g: {
                "count": len(v),
                "approval_rate": round(sum(v) / len(v) * 100, 1)
            }
            for g, v in g_data.items()
        }
        live_fair["region"] = {
            r: {
                "count": len(v),
                "approval_rate": round(sum(v) / len(v) * 100, 1)
            }
            for r, v in r_data.items() if len(v) >= 2
        }
        if len(live_fair["gender"]) >= 2:
            rates = [v["approval_rate"] for v in live_fair["gender"].values()]
            live_fair["gender_dpd"] = round(
                (max(rates) - min(rates)) / 100, 4)
        if len(live_fair["region"]) >= 2:
            rates = [v["approval_rate"] for v in live_fair["region"].values()]
            live_fair["region_dpd"] = round(
                (max(rates) - min(rates)) / 100, 4)

    # ── BR-07: model health check ─────────────────────────────────────
    health = _compute_model_health(model_info)

    return render_template(
        "admin/models.html",
        model_info=model_info,
        credit_count=credit_count,
        fraud_count=fraud_count,
        fairness=fairness,
        live_fair=live_fair,
        available_versions=available_versions,
        health=health,
        retrain_status=_RETRAIN_STATUS,
    )


def _compute_model_health(model_info):
    """BR-07: Evaluate whether models should be flagged for retraining.

    Per the documented rule:
        * Credit: retrain if accuracy < 70%
        * Fraud:  retrain if false-negative rate > 15% (approximated via
                  recall = 1 - FNR; flag when recall < 0.85)

    Also tracks officer override statistics (per documentation Section
    3.5.2 — "enabling Loan Officers to make informed override decisions").
    A high override rate (FRAUD→LEGITIMATE) signals the model is producing
    too many false positives and should be retrained.
    """
    health = {}
    credit = model_info.get("credit", {})
    fraud = model_info.get("fraud", {})
    credit_acc = credit.get("accuracy", 0)
    fraud_recall = fraud.get("recall", 0)

    # ── Officer override stats (Phase 4 — BR-07 feedback loop) ────────
    override_stats = {
        "total_overrides": 0,
        "fraud_to_legit": 0,
        "legit_to_fraud": 0,
        "other_overrides": 0,
    }
    total_fraud_preds = 0
    fp_override_rate = 0.0
    try:
        from app.models.fraud import FraudResult
        overridden = FraudResult.query.filter(
            FraudResult.original_label.isnot(None)).all()
        override_stats["total_overrides"] = len(overridden)
        for fr in overridden:
            if (fr.original_label == "FRAUD"
                    and fr.label in ("LEGITIMATE", "SUSPICIOUS")):
                override_stats["fraud_to_legit"] += 1
            elif (fr.label == "FRAUD"
                  and fr.original_label in ("LEGITIMATE", "SUSPICIOUS")):
                override_stats["legit_to_fraud"] += 1
            else:
                override_stats["other_overrides"] += 1
        # Total FRAUD predictions = current FRAUD + ones overridden away.
        total_fraud_preds = (
            FraudResult.query.filter_by(label="FRAUD").count()
            + override_stats["fraud_to_legit"])
        fp_override_rate = (
            override_stats["fraud_to_legit"] / total_fraud_preds
            if total_fraud_preds > 0 else 0.0)
    except Exception:
        pass  # DB not ready (e.g., during initial migration)

    health["credit"] = {
        "accuracy":
        credit_acc,
        "status":
        "OK" if credit_acc >= 0.70 else "RETRAIN_RECOMMENDED",
        "threshold":
        ">= 0.70",
    }
    health["fraud"] = {
        "recall":
        fraud_recall,
        "status":
        "OK" if fraud_recall >= 0.85 and fp_override_rate < 0.15
        else "RETRAIN_RECOMMENDED",
        "threshold":
        ">= 0.85 (FNR <= 0.15)",
        "override_stats":
        override_stats,
        "false_positive_override_rate":
        round(fp_override_rate, 4),
    }
    return health


@admin_bp.route("/models/retrain", methods=["POST"])
@roles_required("ADMIN")
def retrain_models():
    """FR-12: Trigger model retraining (runs in background thread).

    Bug fix 1.1: After the subprocess completes, the Flask process still
    has the OLD models loaded in memory (singleton pattern with
    ``_model is not None`` guard). We must clear the singleton caches
    so the next request loads the freshly-trained models from disk.

    Bug fix 1.2: If the subprocess crashes or the server restarts mid-
    retrain, ``_RETRAIN_STATUS["running"]`` stays True forever. We add
    a stale-check: if started_at is more than 10 minutes ago, force-reset
    running to False so the admin can trigger a new retrain.
    """
    # ── Bug 1.2: Stale retrain status check ──────────────────────────
    if _RETRAIN_STATUS["running"] and _RETRAIN_STATUS.get("started_at"):
        try:
            started = datetime.fromisoformat(_RETRAIN_STATUS["started_at"])
            elapsed = (datetime.utcnow() - started).total_seconds()
            if elapsed > 600:  # 10 minutes
                _RETRAIN_STATUS["running"] = False
                _RETRAIN_STATUS["log"] = (
                    "Stale retrain status reset (was running for "
                    f"{elapsed/60:.1f} minutes). You can retrain again.")
        except (ValueError, TypeError):
            _RETRAIN_STATUS["running"] = False

    if _RETRAIN_STATUS["running"]:
        flash("A retraining job is already running. Please wait.", "warning")
        return redirect(url_for("admin.models"))

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    script = os.path.join(root, "ml", "train_models.py")

    def _run():
        _RETRAIN_STATUS["running"] = True
        _RETRAIN_STATUS["started_at"] = datetime.utcnow().isoformat()
        _RETRAIN_STATUS["log"] = ""
        try:
            result = subprocess.run(
                ["python", script],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            _RETRAIN_STATUS["log"] = (result.stdout + "\n" +
                                      result.stderr)[-4000:]
            # ── Bug 1.1: Clear model singletons so new models load ──
            # The subprocess wrote new .pkl files to disk, but the Flask
            # process still has the OLD models in memory. Clearing the
            # singleton guards forces the next request to reload from disk.
            try:
                from app.services.credit_service import CreditScoringService
                from app.services.fraud_service import FraudDetectionService
                CreditScoringService._model = None
                CreditScoringService._scaler = None
                CreditScoringService._features = None
                CreditScoringService._shap_base = None
                FraudDetectionService._xgb = None
                FraudDetectionService._rf = None
                FraudDetectionService._scaler = None
                FraudDetectionService._features = None
                FraudDetectionService._shap_base = None
                FraudDetectionService._explainer = None
                FraudDetectionService._calibrator = None
                _RETRAIN_STATUS["log"] += (
                    "\n[OK] Model singletons cleared. New models will "
                    "load on next request.")
            except Exception as e:
                _RETRAIN_STATUS["log"] += (
                    f"\n[WARN] Could not clear singletons: {e}")
        except subprocess.TimeoutExpired:
            _RETRAIN_STATUS["log"] = "Retraining timed out after 10 minutes."
        except Exception as e:
            _RETRAIN_STATUS["log"] = f"Error: {e}"
        finally:
            _RETRAIN_STATUS["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    AuditLog.log(current_user_id(), "MODEL_RETRAIN_TRIGGERED", "MLModel",
                 None, request.remote_addr, {"script": script})
    flash("Retraining started in the background. Refresh this page in "
          "1-2 minutes to see updated metrics. New models will auto-load "
          "after retraining completes.", "info")
    return redirect(url_for("admin.models"))


@admin_bp.route("/models/<name>/switch", methods=["POST"])
@roles_required("ADMIN")
def switch_model_version(name):
    """FR-12: Switch the active model version (filename-based)."""
    if name not in ("credit", "fraud"):
        flash("Unknown model type.", "danger")
        return redirect(url_for("admin.models"))
    new_version = request.form.get("version", "").strip()
    if not new_version:
        flash("No version specified.", "danger")
        return redirect(url_for("admin.models"))
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    src = os.path.join(root, "ml", "models", new_version)
    dst = os.path.join(root, "ml", "models", f"{name}_model.pkl")
    if not os.path.exists(src):
        flash(f"Model file {new_version} not found.", "danger")
        return redirect(url_for("admin.models"))
    import shutil
    shutil.copyfile(src, dst)
    # Clear the service singleton so the new model loads on next request.
    if name == "credit":
        from app.services.credit_service import CreditScoringService
        CreditScoringService._model = None
    else:
        from app.services.fraud_service import FraudDetectionService
        FraudDetectionService._xgb = None
        FraudDetectionService._rf = None
    AuditLog.log(current_user_id(), "MODEL_VERSION_SWITCHED", "MLModel", None,
                 request.remote_addr,
                 {"model": name, "version": new_version})
    flash(f"Active {name} model switched to {new_version}.", "success")
    return redirect(url_for("admin.models"))


@admin_bp.route("/models/health")
@roles_required("ADMIN")
def model_health():
    """BR-07: JSON endpoint returning the latest model health evaluation."""
    model_info = {}
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    for name, path in [("credit", "ml/models/credit_model_info.json"),
                       ("fraud", "ml/models/fraud_model_info.json")]:
        full = os.path.join(root, path)
        if os.path.exists(full):
            with open(full) as f:
                model_info[name] = json.load(f)
    return jsonify(_compute_model_health(model_info)), 200


@admin_bp.route("/audit")
@roles_required("ADMIN")
def audit():
    """UC-14: View Audit Log (FR-14) — paginated."""
    page = request.args.get("page", 1, type=int)
    per_page = 100
    pagination = (AuditLog.query.order_by(
        AuditLog.timestamp.desc()).paginate(page=page,
                                            per_page=per_page,
                                            error_out=False))
    return render_template("admin/audit.html",
                           logs=pagination.items,
                           pagination=pagination)


@admin_bp.route("/api/stats")
@roles_required("ADMIN")
def api_stats():
    """UC-13: JSON endpoint for dashboard chart data (Chart.js)."""
    from sqlalchemy import func
    sd = (db.session.query(CreditScore.risk_level,
                           func.count(CreditScore.score_id)).group_by(
                               CreditScore.risk_level).all())
    fd = (db.session.query(FraudResult.label,
                           func.count(FraudResult.result_id)).group_by(
                               FraudResult.label).all())
    return jsonify({
        "score_distribution": {
            r: c for r, c in sd
        },
        "fraud_distribution": {
            l: c for l, c in fd
        }
    })
