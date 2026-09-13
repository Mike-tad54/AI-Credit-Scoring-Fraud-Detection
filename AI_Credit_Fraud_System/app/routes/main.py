"""Main Blueprint — landing page with role-aware redirect."""
from flask import Blueprint, render_template, redirect, url_for
from flask_jwt_extended import verify_jwt_in_request, get_jwt

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Landing page; redirects authenticated users to their dashboard."""
    try:
        verify_jwt_in_request(locations=["cookies"])
        role = get_jwt().get("role")
        target = {
            "ADMIN": "admin.dashboard",
            "OFFICER": "fraud.upload",
            "APPLICANT": "credit.apply",
        }.get(role, "auth.login")
        return redirect(url_for(target))
    except Exception:
        return render_template("main/index.html")
