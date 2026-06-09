"""
Bind Score — train.py

Feature computation, LOO-CV training, evaluation, and model serialisation.

Usage:
    python src/train.py
    python src/train.py --submissions data/features_submissions.csv \
                        --events data/features_events.csv \
                        --output_dir output/model_eval_metrics \
                        --model_dir output/model
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

T_VALUES     = [0, 7, 30]
RANDOM_STATE = 42
FEATURE_COLS = [
    "t",
    "agent_hist_conversion_rate",
    "event_count",
    "quote_received",
    "inbound_email_count",
    "outbound_email_count",
    "total_attachments",
    "median_inter_event_gap_hours",
    "days_since_last_event",
]

XGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    verbosity=0,
)


# ── Feature computation ────────────────────────────────────────────────────

def build_base_features(t, submissions_df, events_df):
    """All features except agent_hist_conversion_rate at cutoff day t."""
    ev = events_df.merge(
        submissions_df[["submissionId", "createdDate"]], on="submissionId"
    ).copy()
    ev["days_since_creation"] = (
        (ev["event_date"] - ev["createdDate"]).dt.total_seconds() / 86400
    )
    ev_t = ev[ev["days_since_creation"] <= t]

    feat = submissions_df[["submissionId", "label"]].set_index("submissionId").copy()
    feat["t"] = float(t)

    feat["event_count"] = (
        ev_t.groupby("submissionId").size().reindex(feat.index, fill_value=0)
    )

    if t > 0:
        # quote_received — count of QUOTE_RECEIVED events
        qr = ev_t[ev_t["event_type"] == "QUOTE_RECEIVED"].groupby("submissionId").size()
        feat["quote_received"] = qr.reindex(feat.index, fill_value=0)

        # inbound / outbound email counts
        ib = ev_t[ev_t["event_type"] == "EMAIL_INBOUND"].groupby("submissionId").size()
        feat["inbound_email_count"] = ib.reindex(feat.index, fill_value=0)

        ob = ev_t[ev_t["event_type"] == "EMAIL_OUTBOUND"].groupby("submissionId").size()
        feat["outbound_email_count"] = ob.reindex(feat.index, fill_value=0)

        email_ev = ev_t[ev_t["event_type"].isin(["EMAIL_INBOUND", "EMAIL_OUTBOUND"])]
        ta = email_ev.groupby("submissionId")["email_attachment_count"].sum()
        feat["total_attachments"] = ta.reindex(feat.index, fill_value=0)

        ev_s = ev_t.sort_values(["submissionId", "event_date"]).copy()
        ev_s["gap_hours"] = (
            ev_s.groupby("submissionId")["event_date"]
                .diff().dt.total_seconds() / 3600
        )
        gap_med = ev_s.groupby("submissionId")["gap_hours"].median()
        feat["median_inter_event_gap_hours"] = gap_med.reindex(feat.index)

        # days_since_last_event — t minus the day of the most recent event
        last_day = ev_t.groupby("submissionId")["days_since_creation"].max()
        feat["days_since_last_event"] = (
            t - last_day.reindex(feat.index, fill_value=np.nan)
        )
        # submissions with no events: silent since creation, so value = t
        feat["days_since_last_event"] = feat["days_since_last_event"].fillna(float(t))
    else:
        feat["quote_received"] = 0
        feat["inbound_email_count"] = 0
        feat["outbound_email_count"] = 0
        feat["total_attachments"] = 0
        feat["median_inter_event_gap_hours"] = np.nan
        feat["days_since_last_event"] = np.nan

    return feat


AGENT_SMOOTHING_ALPHA = 10  # pseudo-observations toward global mean


def assign_agent_rates(feat_df, train_subs, all_subs):
    """
    Compute Bayesian-smoothed agent conversion rate from train_subs.
    smoothed = (positives + alpha * global_mean) / (n + alpha)
    Agents absent from train_subs get the global mean.
    """
    global_mean  = train_subs["label"].mean()
    agent_stats  = train_subs.groupby("agentEmail")["label"].agg(["sum", "count"])
    smoothed     = (
        (agent_stats["sum"] + AGENT_SMOOTHING_ALPHA * global_mean)
        / (agent_stats["count"] + AGENT_SMOOTHING_ALPHA)
    )
    rates = (
        all_subs.set_index("submissionId")["agentEmail"]
                .map(smoothed)
                .fillna(global_mean)
    )
    out = feat_df.copy()
    out["agent_hist_conversion_rate"] = rates
    return out


def make_model(scale_pos_weight):
    return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)


# ── LOO-CV ─────────────────────────────────────────────────────────────────

def loo_cv(submissions_df, events_df):
    """
    Leave-one-out CV on submission_id.
    Returns DataFrame with columns: submissionId, t, label, score.
    """
    print("Pre-computing base feature matrices ...")
    base = {t: build_base_features(t, submissions_df, events_df) for t in T_VALUES}

    all_sids = submissions_df["submissionId"].values
    records  = []

    print(f"Running LOO-CV over {len(all_sids)} submissions ...")
    for i, test_sid in enumerate(all_sids):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(all_sids)}")

        train_subs = submissions_df[submissions_df["submissionId"] != test_sid]

        all_feat = pd.concat(
            [assign_agent_rates(base[t], train_subs, submissions_df) for t in T_VALUES]
        )

        is_test = all_feat.index == test_sid
        X_train = all_feat.loc[~is_test, FEATURE_COLS].values
        y_train = all_feat.loc[~is_test, "label"].values
        X_test  = all_feat.loc[ is_test, FEATURE_COLS].values

        spw   = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = make_model(spw)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

        for j, (_, row) in enumerate(all_feat.loc[is_test].iterrows()):
            records.append({
                "submissionId": test_sid,
                "t":            int(row["t"]),
                "label":        int(row["label"]),
                "score":        float(scores[j]),
            })

    return pd.DataFrame(records)


# ── Final model ────────────────────────────────────────────────────────────

def train_final_model(submissions_df, events_df):
    """
    Train on all data with global agent rates.
    Used for feature importance and saved as the deployable model.
    Not used for evaluation — LOO-CV handles that.
    """
    all_feat = pd.concat([
        assign_agent_rates(
            build_base_features(t, submissions_df, events_df),
            submissions_df,
            submissions_df,
        )
        for t in T_VALUES
    ])
    X = all_feat[FEATURE_COLS].values
    y = all_feat["label"].values

    spw   = (y == 0).sum() / max((y == 1).sum(), 1)
    model = make_model(spw)
    model.fit(X, y)
    return model


# ── Plotting ───────────────────────────────────────────────────────────────

COL_T = {0: "#5B8DB8", 7: "#E07B54", 30: "#6BAE75", "all": "#888888"}


def plot_curves(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for t_val, lbl in [(0, "t=0"), (7, "t=7"), (30, "t=30"), ("all", "all t")]:
        sub   = results if t_val == "all" else results[results["t"] == t_val]
        y, s  = sub["label"].values, sub["score"].values
        color = COL_T[t_val]

        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{lbl}  AUC={roc_auc_score(y, s):.3f}",
                     color=color, linewidth=2)

        prec, rec, _ = precision_recall_curve(y, s)
        axes[1].plot(rec, prec, label=f"{lbl}  AP={average_precision_score(y, s):.3f}",
                     color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve (LOO-CV)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("ROC Curve (LOO-CV)", fontweight="bold")

    baseline = results["label"].mean()
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.4,
                    label=f"random  ({baseline:.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall Curve (LOO-CV)")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Precision-Recall Curve (LOO-CV)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    label_colors = {0: "#5B8DB8", 1: "#E07B54"}
    label_names  = {0: "Not Bound", 1: "Bound"}

    for ax, t_val in zip(axes, T_VALUES):
        sub = results[results["t"] == t_val]
        for lbl, grp in sub.groupby("label"):
            ax.hist(grp["score"], bins=30, alpha=0.65,
                    label=label_names[lbl], color=label_colors[lbl], edgecolor="white")
        ax.set_title(f"t = {t_val} days", fontweight="bold")
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("Score Distribution by Label (LOO-CV)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_score_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, output_dir):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp.index, imp.values, color="#E07B54", edgecolor="white")
    ax.set_title("Feature Importance — Final Model (gain)", fontweight="bold")
    ax.set_xlabel("Importance (gain)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── Metrics ────────────────────────────────────────────────────────────────

def print_metrics(results):
    print("\n-- LOO-CV Metrics ------------------------------")
    print(f"{'':>6}  {'ROC-AUC':>9}  {'PR-AUC':>8}")
    print("-" * 32)
    for t_val in T_VALUES + ["all"]:
        sub  = results if t_val == "all" else results[results["t"] == t_val]
        y, s = sub["label"].values, sub["score"].values
        tag  = f"t={t_val}" if t_val != "all" else "all t"
        print(f"{tag:>6}  {roc_auc_score(y, s):>9.4f}  {average_precision_score(y, s):>8.4f}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate the bind score model.")
    p.add_argument("--submissions", default="data/features_submissions.csv")
    p.add_argument("--events",      default="data/features_events.csv")
    p.add_argument("--output_dir",  default="output/model_eval_metrics")
    p.add_argument("--model_dir",   default="output/model")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir,  exist_ok=True)

    print("Loading data ...")
    subs   = pd.read_csv(args.submissions, parse_dates=["createdDate", "resolvedDate"])
    events = pd.read_csv(args.events,      parse_dates=["event_date"])
    print(f"Submissions: {len(subs)} | Events: {len(events)}")

    results = loo_cv(subs, events)
    print_metrics(results)

    print("Saving evaluation plots ...")
    plot_curves(results, args.output_dir)
    plot_score_distributions(results, args.output_dir)

    print("Training final model ...")
    final_model = train_final_model(subs, events)
    plot_feature_importance(final_model, args.output_dir)

    model_path = os.path.join(args.model_dir, "bind_score.json")
    final_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    print(f"Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
