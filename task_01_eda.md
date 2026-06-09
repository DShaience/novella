# Task 1: Exploratory Data Analysis

Full notebook: [EDA.ipynb](EDA.ipynb)

---

## Class Distribution

The dataset is heavily imbalanced — only ~13% of submissions bind.

![Label distribution](output/eda_figures/01_label_distribution.png)

## Resolution Time

Bound and not-bound submissions resolve in similar timeframes, so resolution speed alone is not a useful signal. The `t=7` and `t=30` windows cover the bulk of the lifecycle.

![Resolution time](output/eda_figures/02_resolution_time.png)

## Data Leakage Audit

All events precede the submission's `resolvedDate` — no post-resolution events exist in the data, so there is no temporal leakage risk.

## Event Activity Over Time

Bound submissions generate significantly more event activity per day than not-bound ones, and this divergence is visible from the first week. **Early prediction is feasible.**

![Event activity over time](output/eda_figures/03_event_activity_over_time.png)

## Event Type Mix

Bound submissions accumulate more events of every type. `QUOTE_RECEIVED` is a particularly strong milestone signal.

![Event type mix](output/eda_figures/04_event_type_mix.png)

## Email Engagement

Email character count does not differentiate bound from not-bound submissions. Attachment counts do — specifically in outbound emails, where bound submissions show a higher and wider spread, consistent with brokers sending more substantive material (quotes, policy documents) on deals they are actively working.

![Email engagement](output/eda_figures/05_email_engagement.png)

## Engagement Cadence

Bound submissions exhibit faster back-and-forth: shorter median inter-event gaps and a more balanced inbound/outbound ratio.

![Cadence](output/eda_figures/06_cadence.png)

## Agent Conversion Rate

Most agents have never converted a submission. A small number are consistent converters. The agent's historical conversion rate is the single strongest predictor of binding.

![Agent bind rate](output/eda_figures/07_agent_bind_rate.png)

## Feature Correlation Summary

At `t=0`, agent history dominates — no event data exists yet. From `t=7` onward, activity-based features (email counts, quotes, attachments) add meaningful signal.

![Feature correlations](output/eda_figures/09_feature_correlations.png)
