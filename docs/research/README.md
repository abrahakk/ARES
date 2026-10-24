# Research Documentation

This directory is for the capstone research write-up and anything
that supports it directly: the literature review, the formal problem
statement, and the reasoning behind the experimental and evaluation
design.

## Primary document

The full capstone proposal - literature review, Dec-POMDP
formulation, research questions with falsification conditions,
experimental design, statistical methodology, security/ethics
treatment, and project plan - is maintained as the project's
formal deliverable document rather than duplicated here as markdown.
Place the current version of that document (e.g.
`ARES_Capstone_Proposal_Full.docx` or its successor) in this
directory, or link to wherever the team's canonical copy lives, so a
reader has one place to start.

## Model card

The model card described in the proposal's Section 15.4 - what the
released policy was and was not trained/evaluated on, its known
failure modes, and its explicitly out-of-scope uses - belongs here
too, versioned alongside whichever checkpoint it documents (cross-
reference `models/metadata/model_registry.yaml` for the exact
checkpoint a given model card describes).

## What belongs here vs. docs/architecture/ vs. docs/experiments/

- **This directory** - the research argument: why this problem, why
  this formulation, why these research questions, and the model card.
- **`docs/architecture/`** - how the code actually implements that
  argument, kept current against the code rather than the plan.
- **`docs/experiments/`** - what running the code actually showed,
  per experiment, including honest failure cases.

If a claim in this directory and the behavior of the code in
`docs/architecture/` ever disagree, that disagreement is itself worth
writing down here rather than silently resolving it one way - it is
usually either a stale proposal claim or a real bug, and both are
worth a paper trail.
