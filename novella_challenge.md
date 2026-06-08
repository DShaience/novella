# Challenge

## Context

Novella is an E&S insurance brokerage firm - we receive requests for insurance called submissions, and try to bind (sell) them.

Your task is to develop a "bind score" - given `(submission id, time t)` output a score such that submissions with higher score are more likely to bind than submissions with lower score.

The goal of the score is **effort prioritization**. Brokers will use the score to decide which submissions to work on first, as they are more likely to sell. In addition, early prediction in the submission lifecycle is more valuable than late predictions, as it saves more time for our brokers.

## Data

Data is given in the format:

### Submissions

| submissionId | createdDate | resolvedDate | agentEmail | label (sold / not sold) |
|---|---|---|---|---|
| 1 | 2020-01-01 | 2020-01-11 | .. | 0 |
| 2 | 2020-01-01 | 2020-01-11 | .. | 0 |
| 3 | 2021-01-01 | .. | .. | 1 |

### Events

| submissionId | eventDate | eventType | email_char_count | email_attachment_count |
|---|---|---|---|---|
| 1 | 2020-01-01 | EMAIL_INBOUND | 160 | 2 |
| 1 | 2020-01-01 | EMAIL_OUTBOUND | 40 | 0 |
| 2 | 2021-01-01 | QUOTE_RECIEVED | .. | 0 |

Where:

1. `agentEmail` - email of the agent requesting the insurance.
2. `EMAIL_INBOUND` / `EMAIL_OUTBOUND` - email sent/received for submission.
3. `QUOTE_RECIEVED` - retail agent is offered an insurance proposition.
4. `Email_char_count` - character count in the email.
5. `Email attachment count` - number of attachments in the email.

## Tasks

1. Devise 3-4 predictive features from the dataset.

   `feature(submission_id, t)` - calculate feature for submission at time `t`, where `t` = number of days since submission creation date.

   Use `t ∈ {0, 7, 30}`.

2. Sort the features by their predictive significance.
3. Develop `bind_score(submission_id, t)`.

## Output format

Please serve your code as a git repository, with `README.md` including:

1. Repository layout.
2. How to run.
3. For each task - results and a short design note: features chosen, metric/s, model + evaluation choices.

If you tested multiple model choices, include the comparison in the result table.
