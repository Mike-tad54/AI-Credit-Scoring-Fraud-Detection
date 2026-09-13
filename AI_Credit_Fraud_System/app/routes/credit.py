"""Credit Scoring Blueprint — UC-04 to UC-07, FR-02 to FR-05, FR-08.

Routes:
    GET  /credit/apply              — render application form (UC-04)
    POST /credit/apply              — submit application + ML predict (UC-04/05)
    GET  /credit/result/<id>        — view score + SHAP (UC-06)
    GET  /credit/result/<id>/pdf    — download PDF report (FR-08)
    GET  /credit/history            — list own applications (UC-07)
    POST /credit/api/predict        — REST endpoint returning JSON score
"""
import json
from flask import (Blueprint, request, jsonify, render_template, redirect,
                   url_for, flash)
from app import db
from app.models.credit import CreditApplication, CreditScore
from app.models.fraud import AuditLog
from app.services.credit_service import CreditScoringService
from app.services.rbac import roles_required, current_user_id, current_role
from app.services.pdf_export import generate_credit_pdf

credit_bp = Blueprint("credit", __name__)


@credit_bp.route("/apply", methods=["GET", "POST"])
@roles_required("APPLICANT", "OFFICER", "ADMIN")
def apply():
    """UC-04: Submit Credit Application + UC-05: Run ML prediction."""
    uid = current_user_id()
    from app.models.user import User
    auth_user = User.query.get(uid)
    auth_name = auth_user.username if auth_user else "Unknown"

    if request.method == "GET":
        history = (CreditApplication.query.filter_by(user_id=uid).order_by(
            CreditApplication.submitted_at.desc()).all())
        return render_template("credit/apply.html",
                               history=history,
                               auth_username=auth_name)

    form = request.form
    # BR-06: Lock identity to authenticated user — ignore form-supplied name.
    # BR-03: Reject if more than 20% of application fields are missing.
    _chk = [
        "full_name", "age", "gender", "region", "employment_type",
        "monthly_income", "years_employed", "existing_loan_bal",
        "monthly_expenses", "loan_amount", "loan_purpose"
    ]
    _miss = sum(1 for f in _chk if not str(form.get(f, "")).strip())
    if _miss / len(_chk) > 0.20:
        flash(
            f"{_miss} of {len(_chk)} fields are empty "
            f"({_miss * 100 // len(_chk)}% missing). Please complete at "
            f"least 80% of the form before submitting.", "danger")
        history = (CreditApplication.query.filter_by(user_id=uid).order_by(
            CreditApplication.submitted_at.desc()).all())
        return render_template("credit/apply.html",
                               history=history,
                               auth_username=auth_name), 400

    def g(k, t=str, d=None):
        """Coerce form field ``k`` to type ``t`` with default ``d``."""
        v = form.get(k, d)
        try:
            return t(v) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d

    app_obj = CreditApplication(
        user_id=uid,
        full_name=auth_name,
        age=g("age", int, 30),
        gender=g("gender"),
        region=g("region"),
        employment_type=g("employment_type"),
        monthly_income=g("monthly_income", float, 0),
        years_employed=g("years_employed", float, 0),
        existing_loan_bal=g("existing_loan_bal", float, 0),
        monthly_expenses=g("monthly_expenses", float, 0),
        loan_amount=g("loan_amount", float, 0),
        loan_purpose=g("loan_purpose"),
        status="PENDING")
    db.session.add(app_obj)
    db.session.flush()  # get app_id before ML call

    result = CreditScoringService.predict(app_obj.get_feature_vector())
    sc = CreditScore(
        app_id=app_obj.app_id,
        score=result["score"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        shap_values=json.dumps(result["shap_values"]),
        model_version=result["model_version"])
    app_obj.status = "SCORED"
    db.session.add(sc)
    db.session.commit()
    AuditLog.log(uid, "CREDIT_SCORE_GENERATED", "CreditScore", sc.score_id,
                 request.remote_addr, {
                     "score": result["score"],
                     "risk": result["risk_level"]
                 })

    if request.is_json:
        return jsonify({
            **result, "app_id": app_obj.app_id, "score_id": sc.score_id
        }), 201
    flash(f"Score: {result['score']:.0f}/100 — {result['risk_level']} RISK",
          "success")
    return redirect(url_for("credit.result", score_id=sc.score_id))


@credit_bp.route("/result/<int:score_id>")
@roles_required("APPLICANT", "OFFICER", "ADMIN")
def result(score_id):
    """UC-06: View Credit Score & SHAP Explanation."""
    sc = CreditScore.query.get_or_404(score_id)
    # BR-06: applicants may only view their own results.
    if current_role() == "APPLICANT" and sc.application.user_id != current_user_id():
        flash("Access denied. You can only view your own results.", "danger")
        return redirect(url_for("credit.history"))
    return render_template("credit/result.html",
                           score=sc,
                           application=sc.application,
                           shap=sc.get_shap())


@credit_bp.route("/result/<int:score_id>/pdf")
@roles_required("APPLICANT", "OFFICER", "ADMIN")
def result_pdf(score_id):
    """FR-08: Download the credit score report as a PDF."""
    sc = CreditScore.query.get_or_404(score_id)
    if current_role() == "APPLICANT" and sc.application.user_id != current_user_id():
        flash("Access denied.", "danger")
        return redirect(url_for("credit.history"))
    AuditLog.log(current_user_id(), "CREDIT_REPORT_EXPORTED", "CreditScore",
                 sc.score_id, request.remote_addr,
                 {"format": "pdf", "score_id": sc.score_id})
    return generate_credit_pdf(sc, sc.application, sc.get_shap())


@credit_bp.route("/history")
@roles_required("APPLICANT", "OFFICER", "ADMIN")
def history():
    """UC-07: View Application History."""
    uid = current_user_id()
    apps = (CreditApplication.query.filter_by(user_id=uid).order_by(
        CreditApplication.submitted_at.desc()).all())
    return render_template("credit/history.html", applications=apps)


@credit_bp.route("/api/predict", methods=["POST"])
@roles_required("APPLICANT", "OFFICER", "ADMIN")
def api_predict():
    """REST endpoint: score an applicant from JSON payload."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON required"}), 400
    mi = data.get("monthly_income", 1) or 1
    fv = [
        data.get("age", 30),
        1 if data.get("gender") == "Male" else 0,
        mi,
        data.get("years_employed", 0),
        data.get("existing_loan_bal", 0),
        data.get("monthly_expenses", 0),
        data.get("loan_amount", 0),
        data.get("existing_loan_bal", 0) / mi,
        data.get("monthly_expenses", 0) / mi,
        data.get("loan_amount", 0) / mi,
        {
            "Employed": 2,
            "Self-Employed": 1,
            "Unemployed": 0
        }.get(data.get("employment_type", ""), 1),
    ]
    return jsonify(CreditScoringService.predict(fv)), 200
