import pandas as pd
import numpy as np


def agent_hist_conversion_rate(
    submission_id: int,
    t: int,
    submissions_df: pd.DataFrame,
) -> float:
    """
    Historical conversion rate of the agent associated with this submission.

    Computed across all submissions in submissions_df for that agent.
    When used in a model pipeline, submissions_df must contain only the
    training fold to avoid label leakage.

    Available at t=0 (no event data required).
    """
    agent_email = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "agentEmail"
    ].iloc[0]
    agent_subs = submissions_df[submissions_df["agentEmail"] == agent_email]
    return float(agent_subs["label"].mean())


def event_count(
    submission_id: int,
    t: int,
    events_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
) -> int:
    """
    Total number of events of any type up to day t.

    Available at t=0 (counts events on the creation day itself).
    """
    created_date = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "createdDate"
    ].iloc[0]
    cutoff = created_date + pd.Timedelta(days=t)
    mask = (events_df["submissionId"] == submission_id) & (events_df["event_date"] <= cutoff)
    return int(mask.sum())


def quote_received(
    submission_id: int,
    t: int,
    events_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
) -> int:
    """
    Binary flag: 1 if a QUOTE_RECEIVED event occurred by day t, else 0.

    Raises ValueError at t=0 — a quote cannot meaningfully arrive
    at the moment of submission creation.
    """
    if t == 0:
        raise ValueError("quote_received is not defined at t=0.")

    created_date = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "createdDate"
    ].iloc[0]
    cutoff = created_date + pd.Timedelta(days=t)
    mask = (
        (events_df["submissionId"] == submission_id)
        & (events_df["event_type"] == "QUOTE_RECEIVED")
        & (events_df["event_date"] <= cutoff)
    )
    return int(mask.any())


def outbound_email_count(
    submission_id: int,
    t: int,
    events_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
) -> int:
    """
    Number of outbound emails sent up to day t.

    Raises ValueError at t=0 — broker outreach takes time to accumulate
    and is uniformly zero at creation, providing no signal.
    """
    if t == 0:
        raise ValueError("outbound_email_count is not defined at t=0.")

    created_date = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "createdDate"
    ].iloc[0]
    cutoff = created_date + pd.Timedelta(days=t)
    mask = (
        (events_df["submissionId"] == submission_id)
        & (events_df["event_type"] == "EMAIL_OUTBOUND")
        & (events_df["event_date"] <= cutoff)
    )
    return int(mask.sum())


def total_attachments(
    submission_id: int,
    t: int,
    events_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
) -> int:
    """
    Total email attachments (inbound + outbound) up to day t.

    Raises ValueError at t=0 — attachment activity is uniformly
    zero at creation and carries no signal.
    """
    if t == 0:
        raise ValueError("total_attachments is not defined at t=0.")

    created_date = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "createdDate"
    ].iloc[0]
    cutoff = created_date + pd.Timedelta(days=t)
    mask = (
        (events_df["submissionId"] == submission_id)
        & (events_df["event_type"].isin(["EMAIL_INBOUND", "EMAIL_OUTBOUND"]))
        & (events_df["event_date"] <= cutoff)
    )
    return int(events_df.loc[mask, "email_attachment_count"].sum())


def median_inter_event_gap_hours(
    submission_id: int,
    t: int,
    events_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
) -> float:
    """
    Median time in hours between consecutive events up to day t.

    Lower values indicate faster engagement cadence, which is positively
    associated with conversion.

    Raises ValueError at t=0 (no gaps possible with fewer than 2 events)
    or when fewer than 2 events exist in the window.
    """
    if t == 0:
        raise ValueError("median_inter_event_gap_hours requires t > 0.")

    created_date = submissions_df.loc[
        submissions_df["submissionId"] == submission_id, "createdDate"
    ].iloc[0]
    cutoff = created_date + pd.Timedelta(days=t)
    sub_events = (
        events_df[
            (events_df["submissionId"] == submission_id)
            & (events_df["event_date"] <= cutoff)
        ]
        .sort_values("event_date")
    )
    if len(sub_events) < 2:
        raise ValueError(
            f"Submission {submission_id} has fewer than 2 events at t={t}."
        )
    gaps = sub_events["event_date"].diff().dropna().dt.total_seconds() / 3600
    return float(gaps.median())
