"""
MindMeter — Flask backend.
Habiba Farag Shehata (2210018882)

Loads the calibrated MLP pipeline, the feature order, the 4-band tier map,
and the metadata (form options + defaults) at import time. Exposes:

    GET  /          ->  the wizard UI
    POST /score     ->  JSON {probability, tier, band_index, breakdown}

The /score response is deliberately *not* a single Yes/No verdict. The
front-end uses the probability to drive a gauge and the tier to highlight
one of four risk bands. None of the sibling projects ship a probability +
tier API, so this is a deliberate point of differentiation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Artefact bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MindMeterArtifacts:
    pipeline: Any
    feature_order: list[str]
    tier_edges: list[float]
    tier_labels: list[str]
    options: dict
    defaults: dict
    numeric_features: list[str]
    categorical_features: list[str]
    metadata: dict = field(default_factory=dict)


def _load() -> MindMeterArtifacts:
    pipe = joblib.load(HERE / "mindmeter_model.pkl")
    feature_order = list(joblib.load(HERE / "mindmeter_features.pkl"))
    tiers = json.loads((HERE / "mindmeter_tiers.json").read_text(encoding="utf-8"))
    meta = json.loads((HERE / "mindmeter_metadata.json").read_text(encoding="utf-8"))

    return MindMeterArtifacts(
        pipeline=pipe,
        feature_order=feature_order,
        tier_edges=list(tiers["edges"]),
        tier_labels=list(tiers["labels"]),
        options=meta["options"],
        defaults=meta["defaults"],
        numeric_features=list(meta["numeric_features"]),
        categorical_features=list(meta["categorical_features"]),
        metadata=meta,
    )


ART = _load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coerce_number(payload: dict, name: str) -> float:
    """Pull a numeric input. Empty/None -> the dataset median."""
    raw = payload.get(name)
    if raw is None or raw == "":
        return float(ART.defaults.get(name, 0.0))
    try:
        value = float(raw)
        if math.isnan(value):
            return float(ART.defaults.get(name, 0.0))
        return value
    except (TypeError, ValueError):
        return float(ART.defaults.get(name, 0.0))


def coerce_category(payload: dict, name: str) -> str:
    """Pick a categorical value, validated against the saved option list."""
    raw = payload.get(name)
    allowed = ART.options.get(name, [])
    if raw and raw in allowed:
        return raw
    fallback = ART.defaults.get(name)
    return fallback if fallback in allowed else (allowed[0] if allowed else "Unknown")


def assemble_row(payload: dict) -> pd.DataFrame:
    """Build a one-row DataFrame in the exact column order the pipeline expects."""
    row: dict[str, Any] = {}
    for name in ART.numeric_features:
        row[name] = coerce_number(payload, name)
    for name in ART.categorical_features:
        row[name] = coerce_category(payload, name)
    return pd.DataFrame([row])[ART.feature_order]


def tier_for(probability: float) -> tuple[str, int]:
    """Return (tier label, 0-based band index)."""
    edges = ART.tier_edges
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if (i == 0 and probability <= hi) or (probability > lo and probability <= hi):
            return ART.tier_labels[i], i
    return ART.tier_labels[-1], len(ART.tier_labels) - 1


def explain(payload: dict) -> list[dict]:
    """Lightweight per-feature contribution. The model is calibrated MLP so
    real SHAP values would need extra packages; this is a heuristic that uses
    each form value's deviation from the dataset default to colour the UI."""
    out = []
    for name in ART.numeric_features:
        median = ART.defaults.get(name, 0)
        v = coerce_number(payload, name)
        delta = v - median
        out.append({"feature": name, "value": v, "delta": delta})
    for name in ART.categorical_features:
        out.append(
            {
                "feature": name,
                "value": coerce_category(payload, name),
                "delta": None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Flask wiring
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        options=ART.options,
        defaults=ART.defaults,
        numeric_features=ART.numeric_features,
        categorical_features=ART.categorical_features,
        tier_labels=ART.tier_labels,
        tier_edges=ART.tier_edges,
        metadata=ART.metadata,
    )


@app.route("/score", methods=["POST"])
def score():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        frame = assemble_row(payload)
        probability = float(ART.pipeline.predict_proba(frame)[0, 1])
        tier, band_index = tier_for(probability)

        return jsonify(
            ok=True,
            probability=round(probability, 4),
            probability_pct=round(probability * 100, 1),
            tier=tier,
            band_index=band_index,
            tier_labels=ART.tier_labels,
            tier_edges=ART.tier_edges,
            breakdown=explain(payload),
        )
    except Exception as exc:  # surfaced in the UI's banner
        return jsonify(ok=False, error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
