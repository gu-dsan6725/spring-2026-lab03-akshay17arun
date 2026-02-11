# Part 2: AI-Assisted Coding with Google Antigravity

## Overview

In this lab you will use [Google Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/) to build a complete ML pipeline on the UCI Wine dataset. You will not write Python code yourself. Instead, you will use Antigravity's planning, coding, and quality features to have the agent build the pipeline for you. The lab is designed so that a single task naturally exercises every major Antigravity capability -- GEMINI.md/rules for project instructions, workflows for on-demand commands, skills for reusable capabilities, and the Manager view for multi-agent orchestration.

The `demo/` folder contains a reference implementation built on the California Housing dataset (regression). Your task uses a **different dataset and problem type** (Wine classification), so you will need to rely on Antigravity rather than copying from the demo.

## What You Will Learn

- How **GEMINI.md and rules** enforce coding standards without manual review
- How **workflows** (`/run-eda`, `/train-model`) provide on-demand step-by-step instructions
- How **skills** (`data-profiler`) give the agent reusable capabilities
- How the **Manager view** orchestrates multiple agents working in parallel
- How Antigravity compares to Claude Code for each of these concepts

## Prerequisites

- Google Antigravity IDE installed ([Getting Started Codelab](https://codelabs.developers.google.com/getting-started-google-antigravity))
- Repository cloned and dependencies installed: `uv sync`
- Familiarity with the UCI Wine dataset: 178 samples, 13 features (alcohol, malic acid, ash, etc.), 3 wine classes. Available via `sklearn.datasets.load_wine()`

## How Antigravity Features Work Together

### Rules -- The Rulebook (GEMINI.md + .agent/rules/)

Rules are always-on instructions that guide the agent's behavior. They function as system instructions the agent considers before generating any code or plan.

- **Global rules**: `.gemini/GEMINI.md` -- applies to this project
- **Project rules**: `.agent/rules/*.md` -- applies to this project only

Rules are the equivalent of Claude Code's **CLAUDE.md**. Both provide persistent project instructions, but Antigravity splits them into separate files per concern (one rule file for coding style, another for data quality, etc.).

Reference: [Customize Antigravity with rules and workflows](https://atamel.dev/posts/2025/11-25_customize_antigravity_rules_workflows/)

### Workflows -- On-Demand Commands (.agent/workflows/)

Workflows are saved prompts triggered on demand with `/workflow-name`. They define step-by-step instructions the agent follows when invoked.

Workflows are the equivalent of Claude Code's **slash commands** (`.claude/commands/`). Both let you create reusable, explicitly triggered actions.

This project includes:
- `/run-eda`: Performs exploratory data analysis
- `/train-model`: Trains and evaluates an XGBoost model

Reference: [Customize Antigravity with rules and workflows](https://atamel.dev/posts/2025/11-25_customize_antigravity_rules_workflows/)

### Skills -- Reusable Capabilities (.agent/skills/)

Skills are directory-based packages containing a `SKILL.md` definition and optional supporting files. The agent loads skills on demand when it determines they are relevant to the current task.

Skills in Antigravity work similarly to Claude Code skills. Both:
- Are triggered automatically based on the agent's assessment of relevance
- Use a `SKILL.md` file as the entry point
- Can include supporting files (templates, scripts, references)

The key difference is directory location:
- Antigravity: `.agent/skills/<name>/SKILL.md` or `~/.gemini/antigravity/skills/<name>/SKILL.md`
- Claude Code: `.claude/skills/<name>/SKILL.md` or `~/.claude/skills/<name>/SKILL.md`

Reference: [Custom Skills in Google Antigravity](https://medium.com/google-cloud/tutorial-getting-started-with-antigravity-skills-864041811e0d)

### Manager View -- Multi-Agent Orchestration

Antigravity's Manager view lets you spawn and orchestrate multiple agents working in parallel across workspaces. Each agent operates autonomously and produces artifacts (task lists, plans, screenshots, code) that you can review.

This is Antigravity's approach to the problem Claude Code solves with **subagents**.

| Feature | Claude Code Subagents | Antigravity Manager View |
|---|---|---|
| Spawning | Automatic via Task tool | Manual from Manager UI |
| Monitoring | Results returned to main context | Artifact review in Manager |
| Parallelism | Concurrent subagents | Concurrent agents across workspaces |
| Verification | Hook-based automation | Artifact review + execution policies |

Reference: [Google Developers Blog: Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/)

### Hooks Gap

Google Antigravity does **not** currently have native hook-based triggers equivalent to Claude Code hooks. In Claude Code, hooks guarantee that commands (like ruff or py_compile) run automatically at specific lifecycle points. In Antigravity, the closest alternatives are:

- **Rules**: Instruct the agent to always run quality checks (but not enforced programmatically)
- **Pre-commit framework**: Standard git hooks that run on commit
- **Workflow steps**: Include quality checks as explicit steps

This is a meaningful difference: Claude Code hooks guarantee execution regardless of the agent's behavior, while Antigravity rules are instructions the agent should follow but are not enforced programmatically.

Reference: [InfoWorld: A first look at Google's Antigravity IDE](https://www.infoworld.com/article/4096113/a-first-look-at-googles-new-antigravity-ide.html)

## The Lab Task

### Your Mission

Build a complete ML pipeline for classifying wines into 3 classes using the UCI Wine dataset (`sklearn.datasets.load_wine()`). The pipeline must include:

1. **Exploratory data analysis** with summary statistics, distribution plots, a correlation heatmap, class balance check, and outlier detection
2. **Feature engineering** with at least 3 derived features, standard scaling, and a stratified train/test split
3. **XGBoost classification model training** with 5-fold cross-validation and evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
4. **A comprehensive evaluation report** with metrics, feature importance, and recommendations

You will **not** write any Python code yourself. You will use Antigravity to plan and build the entire pipeline.

## Step-by-Step Walkthrough

### Step 1: Explore the Configuration

Before you start building, understand what Antigravity already knows about this project. Read the following files:

```bash
# Project rules the agent will follow
cat part2_antigravity/.gemini/GEMINI.md
cat part2_antigravity/.agent/rules/code-style-guide.md

# Workflows
cat part2_antigravity/.agent/workflows/run-eda.md
cat part2_antigravity/.agent/workflows/train-model.md

# Skills
cat part2_antigravity/.agent/skills/data-profiler/SKILL.md
```

No Antigravity interaction yet -- this is manual reading to understand the setup.

### Step 2: Set Up Pre-Commit Hooks

Since Antigravity lacks native hooks like Claude Code, set up the `pre-commit` framework to get automated quality checks on commit:

Create `.pre-commit-config.yaml` in the repo root:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Install with:
```bash
uv add --dev pre-commit
uv run pre-commit install
```

**What to Watch For:**
- This replaces what Claude Code hooks do automatically -- in Antigravity you set this up yourself
- Pre-commit runs on `git commit`, not on every file write like Claude Code hooks

### Step 3: Create a Plan

Open Antigravity in this directory and ask the agent to create an implementation plan. Write your own prompt based on the mission above -- be specific about what you want. Here is a minimal example:

```
Create a plan for building a Wine classification pipeline with EDA (summary statistics, distributions plots, correlation heatmap, class balance check, outlier detection), feature engineering (at least 3 derived features + standard scaling + stratified train/test split),
XGBoost with 5-fold cross-validation (and metrics: Accuracy + Precision + Recall + F1 + Confusion Matrix), and an evaluation report (metrics, feature importances, recommendations). Use
load_wine from sklearn. Put scripts in part2_antigravity/src/ and output
in output/. Write the plan to plan.md.
```

The more detail you provide, the better the plan will be. Try adding specifics like the metrics you want, the number of CV folds, or the derived features you have in mind.

**What to Watch For:**
- The agent follows rules from `.gemini/GEMINI.md` and `.agent/rules/` when designing the plan
- Use the "Auto" execution policy so the agent pauses for your review on higher-risk operations
- Compare this to Claude Code's `/plan` slash command -- similar concept, different mechanism

### Step 4: Review and Refine the Plan

Read the plan the agent created. Then provide at least one modification. Some suggestions:

- "Add hyperparameter tuning using RandomizedSearchCV with at least 20 iterations"
- "Add a per-class precision/recall breakdown and a confusion matrix heatmap"
- "Include a --debug CLI flag that sets logging to DEBUG level"

### Step 5: Watch the Agent Build

After you approve the plan, let the agent implement it. Do not interrupt unless something looks clearly wrong.

**What to Watch For:**

| What Happens | Antigravity Feature | How You Can Tell |
|---|---|---|
| Agent writes a `.py` file | Editor view | You see the file content appear |
| Code uses `polars` not `pandas` | Rules enforcement | Check the import statements |
| Logging uses the prescribed format | Rules enforcement | Check the `logging.basicConfig()` call |
| Private functions start with `_` | Rules enforcement | Check function names |
| Constants at top of file | Rules enforcement | Look at the top of each file |
| Agent checks if quality checks ran | Workflow steps / rules | Agent may run ruff and py_compile |

### Step 6: Run the Pipeline

After the agent finishes building, run the scripts in order:

```bash
uv run python part2_antigravity/src/01_eda.py
uv run python part2_antigravity/src/02_feature_engineering.py
uv run python part2_antigravity/src/03_xgboost_model.py
```

Verify that the `output/` directory contains distribution plots, correlation matrix, parquet files, model file, confusion matrix, feature importance chart, and evaluation report.

### Step 7: Use the Workflows

Now use the pre-built workflows to have the agent perform specific tasks:

```
/run-eda
/train-model
```

**What to Watch For:**
- Each workflow loads its instructions from `.agent/workflows/`
- The agent follows the workflow steps (compare against the workflow files you read in Step 1)
- Compare this to Claude Code's `/analyze-data` and `/evaluate-model` skills

### Step 8: Try the Manager View

Use Antigravity's Manager view to orchestrate multiple agents working in parallel. Create separate agent tasks:

- **Agent 1: Data Preparation** -- Load the dataset, run EDA, perform feature engineering
- **Agent 2: Model Training** -- Train XGBoost model using prepared data
- **Agent 3: Evaluation** -- Evaluate the model and generate the report

**What to Watch For:**
- Each agent operates independently in its own workspace
- Artifacts (plans, code, output) are produced by each agent for your review
- Compare this to Claude Code's automatic subagent spawning via the Task tool

### Step 9: Commit and Observe Pre-Commit Hooks

Stage your files and commit to trigger the pre-commit hooks:

```bash
git add part2_antigravity/src/
git commit -m "Add Wine classification pipeline"
```

**What to Watch For:**
- Ruff runs automatically on commit and fixes style issues
- Compare this to Claude Code's PostToolUse hooks that run on every file write (not just commit)

## Feature Summary Checklist

After completing the lab, confirm you observed each feature:

- [ ] **Rules (GEMINI.md)**: Code uses polars, not pandas
- [ ] **Rules**: Logging format matches the prescribed pattern
- [ ] **Rules**: Private functions start with underscore
- [ ] **Rules**: Constants declared at file top, not hardcoded in functions
- [ ] **Rules**: Functions are under 50 lines
- [ ] **Workflow**: `/run-eda` produced EDA analysis
- [ ] **Workflow**: `/train-model` produced model training output
- [ ] **Skill**: `data-profiler` skill was available for dataset profiling
- [ ] **Manager view**: Multiple agents worked in parallel
- [ ] **Pre-commit hooks**: Ruff ran on commit
- [ ] **Comparison**: You understand the difference between Antigravity rules and Claude Code hooks

## Comparison with Claude Code

After completing both labs, reflect on these differences:

| Concept | Claude Code | Antigravity |
|---|---|---|
| Project instructions | `CLAUDE.md` (single file) | `GEMINI.md` + `.agent/rules/` (multiple files) |
| On-demand commands | Slash commands (`.claude/commands/`) | Workflows (`.agent/workflows/`) |
| Reusable capabilities | Skills (`.claude/skills/`) | Skills (`.agent/skills/`) |
| Automated quality checks | Hooks (enforced programmatically) | Rules (advisory) + pre-commit (on commit only) |
| Task decomposition | Subagents (automatic) | Manager view (manual orchestration) |
| Execution control | Plan mode | Terminal execution policies (Off, Auto, Turbo) |

## Reference Material

The `demo/` folder contains reference implementations and the original granular exercises:

- `demo/solved/` -- Pre-built pipeline scripts that show one possible correct implementation (California Housing regression)
- `demo/exercises/` -- The original three separate exercises, now consolidated into the single lab above

You can compare your Antigravity-generated code against `demo/solved/` to see how implementations differ.

| File | Description |
|------|-------------|
| `demo/solved/01_eda.py` | Loads California Housing, computes statistics with polars, generates distribution and correlation plots |
| `demo/solved/02_feature_engineering.py` | Creates derived features, handles infinite values, scales features, splits into train/test |
| `demo/solved/03_xgboost_model.py` | Trains XGBoost regressor, computes metrics (RMSE, MAE, R2), generates residual and importance plots |

## Troubleshooting

- **Agent is not following rules**: Make sure you opened Antigravity in the `part2_antigravity/` directory so it picks up `.gemini/GEMINI.md` and `.agent/rules/`.
- **Workflows not found**: Workflows must be in `.agent/workflows/`. Verify with `ls .agent/workflows/`.
- **Skills not found**: Skills must be in `.agent/skills/<name>/SKILL.md`. Verify with `ls .agent/skills/`.
- **`uv run` fails**: Run `uv sync` from the repo root first.
- **Output directory missing**: The scripts create `output/` automatically, but you can also run `mkdir -p output`.
- **Pre-commit not running**: Make sure you ran `uv run pre-commit install` after adding the config file.

## Further Reading

- [Customize Antigravity with rules and workflows](https://atamel.dev/posts/2025/11-25_customize_antigravity_rules_workflows/)
- [Custom Skills in Google Antigravity](https://medium.com/google-cloud/tutorial-getting-started-with-antigravity-skills-864041811e0d)
- [Google Developers Blog: Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/)
- [InfoWorld: A first look at Google's Antigravity IDE](https://www.infoworld.com/article/4096113/a-first-look-at-googles-new-antigravity-ide.html)
- [Getting Started Codelab](https://codelabs.developers.google.com/getting-started-google-antigravity)
