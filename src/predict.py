"""
predict.py — Compute bind_score(submission_id, t) for one or all submissions.

Usage:
    # Score a specific submission at all t values
    python src/predict.py --submission-id 42

    # Score a specific submission at a specific t
    python src/predict.py --submission-id 42 --t 7

    # Score all submissions at all t values
    python src/predict.py

    # Score all submissions at t=30, custom data paths
    python src/predict.py --t 30 \
        --submissions data/features_submissions.csv \
        --events data/features_events.csv \
        --model-path output/model/bind_score.json
"""
import argparse
import os
import sys

import pandas as pd
import xgboost as xgb

# Ensure src/ is on the path so train.py can be imported directly
sys.path.insert(0, os.path.dirname(__file__))
from train import T_VALUES, FEATURE_COLS, build_base_features, assign_agent_rates


# ── Core function ──────────────────────────────────────────────────────────

def bind_score(submission_id, t, model, submissions_df, events_df):
    """
    Return the bind score for a single (submission_id, t) pair.

    Score ∈ [0, 1]. Higher means more likely to bind.
    t must be one of {0, 7, 30}.
    """
    if t not in T_VALUES:
        raise ValueError(f"t must be one of {T_VALUES}, got {t}.")

    feat = build_base_features(t, submissions_df, events_df)
    feat = assign_agent_rates(feat, submissions_df, submissions_df)

    if submission_id not in feat.index:
        raise ValueError(f"submission_id {submission_id} not found in submissions data.")

    row = feat.loc[[submission_id], FEATURE_COLS]
    return float(model.predict_proba(row.values)[:, 1][0])


# ── Batch scoring (used by CLI) ────────────────────────────────────────────

def score_batch(submission_ids, t_values, model, submissions_df, events_df):
    """
    Score all (submission_id, t) combinations efficiently.
    Builds the feature matrix once per t rather than once per submission.
    """
    records = []
    for t in t_values:
        feat   = build_base_features(t, submissions_df, events_df)
        feat   = assign_agent_rates(feat, submissions_df, submissions_df)
        sids   = [s for s in submission_ids if s in feat.index]
        scores = model.predict_proba(feat.loc[sids, FEATURE_COLS].values)[:, 1]
        for sid, score in zip(sids, scores):
            records.append({"submissionId": sid, "t": t, "score": round(float(score), 4)})

    return (
        pd.DataFrame(records)
          .sort_values(["t", "score"], ascending=[True, False])
          .reset_index(drop=True)
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute bind_score(submission_id, t).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path",    default="output/model/bind_score.json",
                   help="Path to saved XGBoost model (.json)")
    p.add_argument("--submissions",   default="data/features_submissions.csv",
                   help="Path to submissions CSV")
    p.add_argument("--events",        default="data/features_events.csv",
                   help="Path to events CSV")
    p.add_argument("--submission-id", type=int, default=None,
                   help="Submission to score. If omitted, scores all submissions.")
    p.add_argument("--t", type=int, choices=T_VALUES, default=None,
                   help="Time snapshot in days. If omitted, scores all t values.")
    return p.parse_args()


def main():
    args = parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model_path)

    subs   = pd.read_csv(args.submissions, parse_dates=["createdDate", "resolvedDate"])
    events = pd.read_csv(args.events,      parse_dates=["event_date"])

    t_values       = [args.t] if args.t is not None else T_VALUES
    submission_ids = [args.submission_id] if args.submission_id is not None \
                     else subs["submissionId"].tolist()

    results = score_batch(submission_ids, t_values, model, subs, events)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
