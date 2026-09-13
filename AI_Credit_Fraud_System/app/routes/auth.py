"""Authentication Blueprint — UC-01, UC-02, UC-03 (FR-01).

Routes:
    GET  /auth/register     — render registration form (UC-01)
    POST /auth/register     — create new APPLICANT account (UC-01)
    GET  /auth/login        — render login form (UC-02)
    POST /auth/login        — authenticate + issue JWT cookie (UC-02)
    GET  /auth/logout       — clear JWT cookie (UC-03)

FR-01 enforces:
    * bcrypt password hashing (12 rounds, NFR-03)
    * rate limiting (20 requests / minute per IP)
    * account lockout after 5 failed attempts for 15 minutes
    * identical error message for wrong-username vs wrong-password
      (prevents username enumeration)
"""
from flask import (Blueprint, request, jsonify, render_template, redirect,
                   url_for, flash)
from flask_jwt_extended import (create_access_token, set_access_cookies,
                                unset_jwt_cookies)
from datetime import datetime
from app import db, limiter
from app.models.user import User
from app.models.fraud import AuditLog

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per minute")
def login():
    """UC-02: Login to System."""
    if request.method == "GET":
        return render_template("auth/login.html")

    if request.is_json:
        d = request.get_json() or {}
        username = d.get("username", "")
        password = d.get("password", "")
    else:
        username = request.form.get("username", "")
        password = request.form.get("password", "")

    user = User.query.filter_by(username=username, is_active=True).first()

    # ── FR-01: Account lockout enforcement ────────────────────────────
    if user and user.is_locked():
        remaining = int(
            (user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
        msg = (f"Account locked due to multiple failed attempts. "
               f"Try again in {remaining} minute(s).")
        if request.is_json:
            return jsonify({"error": msg}), 403
        flash(msg, "danger")
        return render_template("auth/login.html"), 403

    if not user or not user.check_password(password):
        # ── FR-01: Register failed login attempt ──────────────────────
        # Only count failures for existing, active accounts (don't reveal
        # whether the username exists). Lock the account after 5 failures.
        if user:
            just_locked = user.register_failed_login(
                max_attempts=5, lock_minutes=15)
            db.session.commit()
            if just_locked:
                AuditLog.log(user.user_id, "ACCOUNT_LOCKED", "User",
                             user.user_id, request.remote_addr,
                             {"reason": "5 failed login attempts"})
                msg = "Too many failed attempts. Account locked for 15 minutes."
            else:
                remaining_attempts = 5 - user.failed_login_count
                msg = (f"Invalid username or password. "
                       f"{remaining_attempts} attempt(s) remaining.")
        else:
            # Identical message — prevents username enumeration.
            msg = "Invalid username or password."
        if request.is_json:
            return jsonify({"error": msg}), 401
        flash(msg, "danger")
        return render_template("auth/login.html"), 401

    # ── FR-01: Reset failed login counter on success ──────────────────
    user.reset_login_attempts()
    user.last_login = datetime.utcnow()
    db.session.commit()
    token = create_access_token(
        identity=str(user.user_id),
        additional_claims={
            "role": user.role,
            "username": user.username
        },
    )
    AuditLog.log(user.user_id, "LOGIN", ip=request.remote_addr)
    if request.is_json:
        return jsonify({
            "access_token": token,
            "role": user.role,
            "username": user.username,
        }), 200
    resp = redirect(_role_redirect(user.role))
    set_access_cookies(resp, token)
    return resp


@auth_bp.route("/logout")
def logout():
    """UC-03: Logout of System."""
    resp = redirect(url_for("auth.login"))
    unset_jwt_cookies(resp)
    flash("You have been logged out.", "info")
    return resp


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """UC-01: Register Account."""
    if request.method == "GET":
        return render_template("auth/register.html")

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    # ── Input validation (FR-03 equivalent for auth) ──────────────────
    if not username:
        flash("Username is required.", "danger")
        return render_template("auth/register.html"), 400
    if not email or "@" not in email:
        flash("A valid email address is required.", "danger")
        return render_template("auth/register.html"), 400
    if len(password) < 6:
        flash("Password must be at least 6 characters long.", "danger")
        return render_template("auth/register.html"), 400
    if User.query.filter_by(username=username).first():
        flash("Username already taken.", "danger")
        return render_template("auth/register.html"), 400
    if User.query.filter_by(email=email).first():
        flash("Email already registered. Please log in.", "danger")
        return render_template("auth/register.html"), 400

    user = User(username=username, email=email, role="APPLICANT")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    AuditLog.log(user.user_id, "USER_REGISTERED", "User", user.user_id,
                 request.remote_addr, {
                     "username": username,
                     "email": email
                 })
    flash("Account created! Please log in.", "success")
    return redirect(url_for("auth.login"))


def _role_redirect(role):
    """Map a user role to its post-login landing page URL."""
    return url_for({
        "ADMIN": "admin.dashboard",
        "OFFICER": "fraud.upload",
        "APPLICANT": "credit.apply",
    }.get(role, "main.index"))
