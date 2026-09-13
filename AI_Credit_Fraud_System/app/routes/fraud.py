"""Fraud Detection Blueprint — UC-08 to UC-11, FR-06, FR-07, FR-09, FR-13.

Routes:
    GET  /fraud/upload                       — render upload form (UC-08)
    POST /fraud/upload                       — accept CSV, JSON, or manual (FR-06)
    GET  /fraud/report/<batch_id>            — view batch report (UC-10, FR-09)
    GET  /fraud/report/<batch_id>/export     — CSV export (UC-11, FR-13)
    GET  /fraud/report/<batch_id>/pdf        — PDF export (FR-13)
"""
import io
import csv
import json
import uuid
from flask import (Blueprint, request, jsonify, render_template, redirect,
                   url_for, flash, Response)
from sqlalchemy.orm import joinedload
from app import db
from app.models.fraud import Transaction, FraudResult, AuditLog, Notification
from app.services.fraud_service import FraudDetectionService
from app.services.rbac import roles_required, current_user_id
from app.services.pdf_export import generate_fraud_pdf
import numpy as np

fraud_bp = Blueprint("fraud", __name__)

# Order MUST match FRAUD_FEATURES in ml/train_models.py
_TXN_TYPES = ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]


def _is_round_amount(amt):
    """Detect amounts that are exact multiples of 100 (e.g. 500, 5000)."""
    try:
        amt = float(amt)
    except (TypeError, ValueError):
        return 0
    if amt != int(amt):
        return 0
    return 1 if int(amt) % 100 == 0 else 0


def _one_hot_txn_type(txn_type_str):
    """One-hot encode transaction type into 5 binary columns."""
    tt = str(txn_type_str).strip().upper()
    return [1 if tt == t else 0 for t in _TXN_TYPES]


def build_fv(row):
    """Build the 10-feature vector from any row dict (manual, CSV, or JSON).

    Features (must match ``FRAUD_FEATURES`` in ``ml/train_models.py``):
        [amount, hour_of_day, is_weekend,
         txn_type_TRANSFER, txn_type_PAYMENT, txn_type_CASH_OUT,
         txn_type_DEBIT, txn_type_CASH_IN,
         is_round_amount, log_amount]
    """
    amt = float(row.get("Amount", row.get("amount", 0)) or 0)
    hour = int(float(row.get("hour_of_day", row.get("hour", 12)) or 12))
    hour = max(0, min(23, hour))  # clamp to valid range
    is_wknd = int(float(row.get("is_weekend", 0) or 0))
    is_wknd = 1 if is_wknd else 0
    txn_type = str(
        row.get("txn_type", row.get("type", "TRANSFER"))).strip().upper()
    return [
        amt,
        hour,
        is_wknd,
        *_one_hot_txn_type(txn_type),
        _is_round_amount(amt),
        float(np.log1p(amt)),  # log(1 + amount)
    ]


def _parse_incoming_transactions():
    """Parse the request body into a list of transaction dicts.

    Supports three input modes (FR-06 — CSV or JSON):
        * JSON body — ``{"transactions": [...]}`` or ``[...]``
        * Multipart CSV file upload
        * Manual HTML form entry (single transaction)

    Returns:
        tuple[list[dict], str | None]: (transactions, error_message)
    """
    # ── JSON request (FR-06 — JSON support) ────────────────────────────
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if isinstance(data, list):
            return data, None
        if isinstance(data, dict) and "transactions" in data:
            return data["transactions"], None
        if isinstance(data, dict) and data:
            # Single transaction as flat JSON object
            return [data], None
        return [], "JSON body must be a list or {transactions: [...]}."

    # ── Manual form entry ──────────────────────────────────────────────
    if request.form.get("manual") == "1":
        try:
            txn_data = [{
                "amount":
                float(request.form.get("amount", 0) or 0),
                "txn_type":
                request.form.get("txn_type", "TRANSFER"),
                "hour_of_day":
                int(float(request.form.get("hour_of_day", 12) or 12)),
                "is_weekend":
                int(float(request.form.get("is_weekend", 0) or 0)),
                "merchant_id":
                request.form.get("merchant_id", "N/A"),
                "origin_account":
                request.form.get("origin_account", ""),
                "dest_account":
                request.form.get("dest_account", ""),
            }]
        except (ValueError, TypeError) as e:
            return [], f"Invalid input: {e}. Please check numeric fields."
        return txn_data, None

    # ── CSV file upload ────────────────────────────────────────────────
    file = request.files.get("file")
    if not file:
        return [], "No file uploaded."
    stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
    txn_data = list(csv.DictReader(stream))
    return txn_data, None


@fraud_bp.route("/upload", methods=["GET", "POST"])
@roles_required("OFFICER", "ADMIN")
def upload():
    """UC-08/09: Upload transactions and run fraud detection."""
    if request.method == "GET":
        return render_template("fraud/upload.html")

    uid = current_user_id()
    batch_id = str(uuid.uuid4())[:8].upper()
    txn_data, err = _parse_incoming_transactions()
    if err:
        if request.is_json:
            return jsonify({"error": err}), 400
        flash(err, "danger")
        return redirect(url_for("fraud.upload"))

    if len(txn_data) > 500:
        flash("Batch limited to 500 transactions.", "warning")
        txn_data = txn_data[:500]

    skipped = 0
    results = []
    fraud_count = 0
    for row in txn_data:
        try:
            amt = float(row.get("Amount", row.get("amount", 0)) or 0)
        except (ValueError, TypeError):
            skipped += 1
            continue
        fv = build_fv(row)
        txn = Transaction(
            officer_id=uid,
            amount=amt,
            txn_type=row.get("txn_type", row.get("type", "TRANSFER")),
            merchant_id=row.get("merchant_id", row.get("nameOrig", "N/A")),
            origin_account=row.get("origin_account",
                                   row.get("nameOrig", "")),
            dest_account=row.get("dest_account", row.get("nameDest", "")),
            batch_id=batch_id)
        db.session.add(txn)
        db.session.flush()  # get txn_id before ML call

        det = FraudDetectionService.detect(fv)
        fr = FraudResult(
            txn_id=txn.txn_id,
            fraud_probability=det["fraud_probability"],
            label=det["label"],
            shap_values=json.dumps(det["shap_values"]),
            model_version=det["model_version"],
            alert_sent=det["label"] == "FRAUD")
        db.session.add(fr)
        results.append({**txn.to_dict(), **det})
        if det["label"] == "FRAUD":
            fraud_count += 1

    db.session.commit()

    # FR-07: Generate officer notifications for every FRAUD transaction.
    for r in results:
        if r.get("label") == "FRAUD":
            n = Notification(
                user_id=uid,
                txn_id=r.get("txn_id"),
                batch_id=batch_id,
                message=(f"FRAUD detected — Txn #{r.get('txn_id')} "
                         f"(ETB {r.get('amount', 0):,.2f}, "
                         f"{r.get('txn_type', 'N/A')}) — "
                         f"probability {r.get('fraud_probability', 0)*100:.1f}%"))
            db.session.add(n)
    db.session.commit()

    AuditLog.log(uid, "FRAUD_BATCH_ANALYZED", "Batch", None,
                 request.remote_addr, {
                     "batch_id": batch_id,
                     "total": len(results),
                     "fraud": fraud_count,
                     "skipped": skipped
                 })

    if request.is_json:
        return jsonify({
            "batch_id": batch_id,
            "results": results,
            "skipped": skipped,
            "fraud_count": fraud_count
        }), 200

    if skipped:
        flash(f"{skipped} row(s) skipped due to invalid amount.", "warning")
    if not results:
        flash("No valid transactions to analyze.", "danger")
        return redirect(url_for("fraud.upload"))
    flash(
        f"Batch {batch_id}: {len(results)} transactions analysed "
        f"({fraud_count} fraud).", "success")
    return redirect(url_for("fraud.batch_report", batch_id=batch_id))


@fraud_bp.route("/report/<batch_id>")
@roles_required("OFFICER", "ADMIN")
def batch_report(batch_id):
    """UC-10: View Fraud Detection Report.

    Bug fix 1.4: Shows both AI-original counts and post-officer-review
    counts so the examiner can see the difference (e.g. "AI detected 6
    fraud, officer corrected 3 to LEGITIMATE, final fraud count = 3").
    """
    txns = (Transaction.query.filter_by(batch_id=batch_id).order_by(
        Transaction.txn_id).all())
    rows = [{
        "txn_id": t.txn_id,
        "amount": t.amount,
        "txn_type": t.txn_type,
        "label": t.fraud_result.label if t.fraud_result else "N/A",
        "probability":
        t.fraud_result.fraud_probability if t.fraud_result else 0,
        "alert_sent":
        t.fraud_result.alert_sent if t.fraud_result else False,
        "top_factors":
        list(json.loads(t.fraud_result.shap_values).keys())[:3]
        if t.fraud_result and t.fraud_result.shap_values else [],
        # Override fields for UI transparency.
        "is_overridden":
        t.fraud_result.is_overridden if t.fraud_result else False,
        "original_label":
        t.fraud_result.original_label if t.fraud_result else None,
        "result_id":
        t.fraud_result.result_id if t.fraud_result else None,
    } for t in txns]

    # ── Current (post-override) counts ───────────────────────────────
    fc = sum(1 for r in rows if r["label"] == "FRAUD")
    sc = sum(1 for r in rows if r["label"] == "SUSPICIOUS")
    lc = len(rows) - fc - sc

    # ── Bug 1.4: AI-original counts (what the AI said before override) ──
    # For each transaction, the "AI label" is original_label if overridden,
    # otherwise the current label (no override = AI label == current label).
    ai_fraud = sum(1 for r in rows
                   if (r["original_label"] or r["label"]) == "FRAUD")
    ai_susp = sum(1 for r in rows
                  if (r["original_label"] or r["label"]) == "SUSPICIOUS")
    ai_legit = len(rows) - ai_fraud - ai_susp
    override_count = sum(1 for r in rows if r["is_overridden"])

    return render_template("fraud/report.html",
                           rows=rows,
                           batch_id=batch_id,
                           total=len(rows),
                           fraud_count=fc,
                           susp_count=sc,
                           legit_count=lc,
                           # AI-original counts (Bug 1.4)
                           ai_fraud_count=ai_fraud,
                           ai_susp_count=ai_susp,
                           ai_legit_count=ai_legit,
                           override_count=override_count)


@fraud_bp.route("/result/<int:result_id>/override", methods=["POST"])
@roles_required("OFFICER", "ADMIN")
def override_result(result_id):
    """Override the AI's fraud label with an officer's corrected decision.

    Per documentation Section 3.5.2: "enabling Loan Officers to make
    informed override decisions". The officer can correct a false positive
    (FRAUD → LEGITIMATE) or flag a missed fraud (LEGITIMATE → FRAUD).

    Updates the ``label`` field (so dashboards show the corrected value),
    preserves the original AI label in ``original_label``, writes an
    audit log entry, and updates the officer's notification bell.
    """
    fr = FraudResult.query.get_or_404(result_id)
    uid = current_user_id()

    # Parse new label and reason from JSON or form.
    if request.is_json:
        data = request.get_json() or {}
        new_label = data.get("label", "").upper()
        reason = data.get("reason", "").strip() or None
    else:
        new_label = request.form.get("label", "").upper()
        reason = request.form.get("reason", "").strip() or None

    if new_label not in ("LEGITIMATE", "SUSPICIOUS", "FRAUD"):
        if request.is_json:
            return jsonify({"error": "Invalid label. Must be LEGITIMATE, "
                                     "SUSPICIOUS, or FRAUD."}), 400
        flash("Invalid label. Must be LEGITIMATE, SUSPICIOUS, or FRAUD.",
              "danger")
        return redirect(url_for("fraud.batch_report",
                                batch_id=fr.transaction.batch_id))

    original_label = fr.label  # capture before override for audit log
    fr.override(new_label, uid, reason)
    db.session.commit()

    # ── Phase 3: Notification cleanup ────────────────────────────────
    # If officer overrides FRAUD → LEGITIMATE/SUSPICIOUS, auto-mark the
    # related FRAUD notification as read (don't leave stale alerts).
    if original_label == "FRAUD" and new_label != "FRAUD":
        Notification.query.filter_by(
            txn_id=fr.txn_id, is_read=False).update({"is_read": True})
        db.session.commit()
    # If officer overrides non-FRAUD → FRAUD, create a new notification
    # (officer confirmed AI missed a fraud — false negative caught).
    elif new_label == "FRAUD" and original_label != "FRAUD":
        n = Notification(
            user_id=uid,
            txn_id=fr.txn_id,
            batch_id=fr.transaction.batch_id,
            message=(f"OFFICER OVERRIDE — Txn #{fr.txn_id} "
                     f"escalated to FRAUD by officer (was {original_label})"))
        db.session.add(n)
        db.session.commit()

    # ── BR-05: Audit log entry (INSERT-only, immutable) ──────────────
    AuditLog.log(
        uid, "FRAUD_RESULT_OVERRIDDEN", "FraudResult", fr.result_id,
        request.remote_addr, {
            "txn_id": fr.txn_id,
            "original_label": original_label,
            "new_label": new_label,
            "reason": reason or "(no reason provided)",
            "ai_probability": fr.fraud_probability,
        })

    if request.is_json:
        return jsonify({
            "ok": True,
            "result_id": fr.result_id,
            "original_label": original_label,
            "new_label": new_label,
            "is_overridden": fr.is_overridden,
        }), 200

    flash(f"Txn #{fr.txn_id} label changed: {original_label} → {new_label}.",
          "success")
    return redirect(url_for("fraud.batch_report",
                            batch_id=fr.transaction.batch_id))


@fraud_bp.route("/report/<batch_id>/export")
@roles_required("OFFICER", "ADMIN")
def export_csv(batch_id):
    """UC-11 / FR-13: Export fraud report as CSV."""
    txns = (Transaction.query.options(
        joinedload(Transaction.fraud_result)).filter_by(batch_id=batch_id).
            order_by(Transaction.txn_id).all())
    AuditLog.log(current_user_id(), "FRAUD_REPORT_EXPORTED", "Batch", None,
                 request.remote_addr,
                 {"batch_id": batch_id, "format": "csv", "count": len(txns)})
    # Pre-render rows as strings while the session is still alive (v1.3 bug fix).
    csv_rows = ["txn_id,amount,txn_type,label,fraud_probability,alert_sent"]
    for t in txns:
        r = t.fraud_result
        csv_rows.append(
            f"{t.txn_id},{t.amount},\"{t.txn_type or ''}\","
            f"{r.label if r else 'N/A'},"
            f"{r.fraud_probability if r else 0},"
            f"{r.alert_sent if r else False}")
    payload = "\n".join(csv_rows) + "\n"
    return Response(
        payload,
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            f"attachment;filename=fraud_report_{batch_id}.csv"
        },
    )


@fraud_bp.route("/report/<batch_id>/pdf")
@roles_required("OFFICER", "ADMIN")
def export_pdf(batch_id):
    """FR-13: Export fraud report as PDF.

    Bug fix 1.3: Includes officer override history in the PDF so the
    audit trail is preserved on export (not just the final labels).
    """
    txns = (Transaction.query.options(
        joinedload(Transaction.fraud_result)).filter_by(batch_id=batch_id).
            order_by(Transaction.txn_id).all())
    rows = [{
        "txn_id": t.txn_id,
        "amount": t.amount,
        "txn_type": t.txn_type,
        "label": t.fraud_result.label if t.fraud_result else "N/A",
        "probability":
        t.fraud_result.fraud_probability if t.fraud_result else 0,
    } for t in txns]
    fc = sum(1 for r in rows if r["label"] == "FRAUD")
    sc = sum(1 for r in rows if r["label"] == "SUSPICIOUS")
    totals = {
        "total": len(rows),
        "fraud": fc,
        "susp": sc,
        "legit": len(rows) - fc - sc,
    }
    # Collect override info for any transactions that were officer-corrected.
    overrides = []
    for t in txns:
        if t.fraud_result and t.fraud_result.is_overridden:
            overrides.append({
                "txn_id": t.txn_id,
                "original_label": t.fraud_result.original_label,
                "new_label": t.fraud_result.label,
                "reason": t.fraud_result.override_reason,
                "overridden_at": (t.fraud_result.overridden_at.isoformat()
                                  if t.fraud_result.overridden_at else None),
            })
    AuditLog.log(current_user_id(), "FRAUD_REPORT_EXPORTED", "Batch", None,
                 request.remote_addr,
                 {"batch_id": batch_id, "format": "pdf",
                  "count": len(txns), "overrides_included": len(overrides)})
    return generate_fraud_pdf(batch_id, rows, totals, overrides=overrides)


@fraud_bp.route("/notifications")
@roles_required("OFFICER", "ADMIN")
def notifications():
    """FR-07: List fraud-alert notifications for the current officer."""
    uid = current_user_id()
    notes = (Notification.query.filter_by(user_id=uid).order_by(
        Notification.created_at.desc()).limit(50).all())
    return jsonify([n.to_dict() for n in notes]), 200


@fraud_bp.route("/notifications/<int:nid>/read", methods=["POST"])
@roles_required("OFFICER", "ADMIN")
def mark_notification_read(nid):
    """FR-07: Mark a single notification as read."""
    n = Notification.query.get_or_404(nid)
    if n.user_id != current_user_id():
        return jsonify({"error": "Forbidden"}), 403
    n.is_read = True
    db.session.commit()
    return jsonify({"ok": True}), 200
