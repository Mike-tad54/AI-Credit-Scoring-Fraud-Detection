"""RBAC service — ``roles_required`` decorator and JWT helpers.

The ``roles_required`` decorator wraps any route handler and verifies:
    1. A valid JWT is present (in Authorization header OR HttpOnly cookie).
    2. The JWT's ``role`` claim is in the allowed set.

On failure it returns:
    * HTTP 401 for JSON requests (no/invalid token)
    * HTTP 403 for JSON requests (wrong role)
    * HTTP 302 redirect to /auth/login for browser requests

This implements NFR-03 (RBAC) and BR-06 (role-based access enforcement).
"""
from functools import wraps
from flask import redirect, url_for, flash, request, jsonify
from flask_jwt_extended import (verify_jwt_in_request, get_jwt,
                                get_jwt_identity)


def roles_required(*roles):
    """Decorator factory: require the caller's JWT role to be in ``roles``.

    Usage::

        @credit_bp.route("/apply")
        @roles_required("APPLICANT", "OFFICER", "ADMIN")
        def apply(): ...
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request(locations=["headers", "cookies"])
                if get_jwt().get("role") not in roles:
                    if request.is_json:
                        return jsonify({"error": "Forbidden"}), 403
                    flash("Access denied.", "danger")
                    return redirect(url_for("auth.login"))
            except Exception:
                if request.is_json:
                    return jsonify({"error": "Unauthorized"}), 401
                flash("Please log in.", "warning")
                return redirect(url_for("auth.login"))
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def current_user_id():
    """Return the authenticated user's ID (or ``None`` if unauthenticated)."""
    try:
        verify_jwt_in_request(locations=["headers", "cookies"])
        return int(get_jwt_identity())
    except Exception:
        return None


def current_role():
    """Return the authenticated user's role (or ``None``)."""
    try:
        verify_jwt_in_request(locations=["headers", "cookies"])
        return get_jwt().get("role")
    except Exception:
        return None
