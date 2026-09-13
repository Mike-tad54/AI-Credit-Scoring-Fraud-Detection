"""Fraud detection ML service — ensemble XGBoost + Random Forest.

Singleton-pattern inference service. The ensemble averages the fraud
probability from both models, applies isotonic calibration to flatten
borderline probabilities, and then applies the BR-02 three-tier
classification thresholds documented in Chapter 3:

    * FRAUD       — probability > 0.50
    * SUSPICIOUS  — 0.30 <= probability <= 0.50
    * LEGITIMATE  — probability < 0.30

The service loads model artifacts exactly once per process via a class-
level guard (``_xgb is not None``) to satisfy NFR-01 (response time).
"""
import os
import json
import joblib
import numpy as np
from flask import current_app


class FraudDetectionService:
    """Ensemble fraud detection service (XGBoost + Random Forest)."""

    # Class-level singleton state — loaded lazily on first ``detect()``.
    _xgb = _rf = _scaler = _features = _shap_base = _explainer = _calibrator = None

    # ── BR-02: Documented classification thresholds (Chapter 3, Table 3.3) ──
    FRAUD_THRESHOLD = 0.50
    SUSPICIOUS_THRESHOLD = 0.30

    @classmethod
    def _load(cls):
        """Lazily load model artifacts on first call (singleton guard)."""
        if cls._xgb is not None:
            return
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        xp = os.path.join(root, "ml/models/fraud_xgb_model.pkl")
        rp = os.path.join(root, "ml/models/fraud_rf_model.pkl")
        sp = os.path.join(
            root,
            current_app.config.get("FRAUD_SCALER_PATH",
                                   "ml/models/fraud_scaler.pkl"))
        fp = os.path.join(root, "ml/models/fraud_feature_names.json")
        hp = os.path.join(root, "ml/models/fraud_shap_importance.json")
        cp = os.path.join(root, "ml/models/fraud_calibrator.pkl")

        if not os.path.exists(xp):
            single = os.path.join(
                root,
                current_app.config.get("FRAUD_MODEL_PATH",
                                       "ml/models/fraud_model.pkl"))
            if not os.path.exists(single):
                raise FileNotFoundError("Run: python ml/train_models.py")
            cls._xgb = joblib.load(single)
            cls._rf = None
        else:
            cls._xgb = joblib.load(xp)
            cls._rf = joblib.load(rp) if os.path.exists(rp) else None

        cls._scaler = joblib.load(sp)
        # Load isotonic calibrator if present (v2.1+).
        cls._calibrator = joblib.load(cp) if os.path.exists(cp) else None
        with open(fp) as f:
            cls._features = json.load(f)
        with open(hp) as f:
            cls._shap_base = json.load(f)

        # Build SHAP TreeExplainer for per-prediction explanations (NFR-08).
        try:
            import shap
            cls._explainer = shap.TreeExplainer(cls._xgb)
        except Exception:
            cls._explainer = None

    @classmethod
    def detect(cls, fv):
        """Classify a transaction feature vector.

        Args:
            fv (list[float]): 10-dim feature vector (must match
                ``fraud_feature_names.json``).

        Returns:
            dict: ``{fraud_probability, label, shap_values, model_version}``.

        Threshold logic implements BR-02 verbatim:
            * probability > 0.50       → ``FRAUD``
            * 0.30 <= probability <= 0.50 → ``SUSPICIOUS``
            * probability < 0.30       → ``LEGITIMATE``
        """
        cls._load()
        n = len(cls._features)
        X = np.array(fv, dtype=float).reshape(1, -1)
        # Pad or trim feature vector to match the trained model's dimensionality.
        if X.shape[1] < n:
            X = np.pad(X, ((0, 0), (0, n - X.shape[1])))
        elif X.shape[1] > n:
            X = X[:, :n]
        Xs = cls._scaler.transform(X)

        # Ensemble: average XGBoost + Random Forest raw probabilities.
        xp = float(cls._xgb.predict_proba(Xs)[0][1])
        rp = float(cls._rf.predict_proba(Xs)[0][1]) if cls._rf is not None else xp
        prob_raw = (xp + rp) / 2.0

        # Apply isotonic calibration if available (v2.1+) — flattens
        # borderline probabilities so legitimate salary/rent payments
        # don't get pushed above 0.50 by SMOTE's prior shift.
        if cls._calibrator is not None:
            prob = float(cls._calibrator.predict([prob_raw])[0])
        else:
            prob = prob_raw

        # BR-02: documented three-tier classification.
        if prob > cls.FRAUD_THRESHOLD:
            label = "FRAUD"
        elif prob >= cls.SUSPICIOUS_THRESHOLD:
            label = "SUSPICIOUS"
        else:
            label = "LEGITIMATE"

        # Per-prediction SHAP top-5 (NFR-08, BR-04).
        shap_values = {}
        if cls._explainer is not None:
            try:
                sv = cls._explainer.shap_values(Xs)
                if isinstance(sv, list):
                    sv = sv[0]
                sv_row = np.asarray(sv).reshape(-1)
                contribs = {
                    cls._features[i]: float(sv_row[i])
                    for i in range(min(len(sv_row), n))
                }
                shap_values = dict(
                    sorted(contribs.items(),
                           key=lambda x: abs(x[1]),
                           reverse=True)[:5])
            except Exception:
                shap_values = {}
        if not shap_values:
            # Fallback: global SHAP importance from training.
            shap_values = dict(
                sorted({
                    f: cls._shap_base.get(f, 0.01) for f in cls._features
                }.items(),
                       key=lambda x: abs(x[1]),
                       reverse=True)[:5])

        return {
            "fraud_probability": round(prob, 4),
            "label": label,
            "shap_values": shap_values,
            "model_version": "v2.1" if cls._calibrator is not None else "v2.0"
        }
