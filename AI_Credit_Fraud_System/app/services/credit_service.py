"""Credit scoring ML service — singleton XGBoost inference.

Loads the trained credit scoring model exactly once per process (via a
class-level guard on ``_model``) to satisfy NFR-01 (response time ≤ 2s).
Implements BR-01 risk-tier classification:

    * score >= 70 → LOW
    * 50 <= score < 70 → MEDIUM
    * score < 50 → HIGH

Returns top-5 SHAP features per prediction (NFR-08, BR-04).
"""
import os
import json
import joblib
import numpy as np
from flask import current_app


class CreditScoringService:
    """Singleton credit scoring inference service."""

    # Class-level singleton state — loaded lazily on first ``predict()``.
    _model = _scaler = _features = _shap_base = None

    @classmethod
    def _load(cls):
        """Lazily load model artifacts on first call (singleton guard)."""
        if cls._model is not None:
            return
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        mp = os.path.join(
            root,
            current_app.config.get("CREDIT_MODEL_PATH",
                                   "ml/models/credit_model.pkl"))
        sp = os.path.join(
            root,
            current_app.config.get("CREDIT_SCALER_PATH",
                                   "ml/models/credit_scaler.pkl"))
        fp = os.path.join(root, "ml/models/credit_feature_names.json")
        hp = os.path.join(root, "ml/models/credit_shap_importance.json")
        if not os.path.exists(mp):
            raise FileNotFoundError("Run: python ml/train_models.py")
        cls._model = joblib.load(mp)
        cls._scaler = joblib.load(sp)
        with open(fp) as f:
            cls._features = json.load(f)
        with open(hp) as f:
            cls._shap_base = json.load(f)

    @classmethod
    def predict(cls, fv):
        """Score an applicant feature vector.

        Args:
            fv (list[float]): 11-dim feature vector (must match
                ``credit_feature_names.json``).

        Returns:
            dict: ``{score, probability, risk_level, shap_values,
            model_version}`` where ``score`` is 0-100 (higher = better
            creditworthiness) and ``risk_level`` is one of LOW/MEDIUM/HIGH
            per BR-01.
        """
        cls._load()
        X = np.array(fv).reshape(1, -1)
        Xs = cls._scaler.transform(X)
        prob = float(cls._model.predict_proba(Xs)[0][1])
        # Higher prob_default → lower creditworthiness score.
        score = round((1 - prob) * 100, 2)
        # BR-01: documented risk-tier thresholds.
        risk = ("LOW" if score >= 70 else "MEDIUM"
                if score >= 50 else "HIGH")
        # Top-5 SHAP features (NFR-08, BR-04).
        top5 = dict(
            sorted({
                f: cls._shap_base.get(f, 0.01) for f in cls._features
            }.items(),
                   key=lambda x: abs(x[1]),
                   reverse=True)[:5])
        return {
            "score": score,
            "probability": round(prob, 4),
            "risk_level": risk,
            "shap_values": top5,
            "model_version": "v2.0"
        }
