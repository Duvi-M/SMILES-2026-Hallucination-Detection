# GitHub Profile Setup

## Recommended Repository Name

Recommended portfolio name:

```text
llm-hallucination-detection
```

Other good options:

- `hallucination-detection-probe`
- `llm-hidden-state-hallucination-detector`
- `qwen-hallucination-probe`
- `smiles-hallucination-detection`

The current repository URL can stay as-is for the SMILES-2026 submission. If
renaming after submission, update the GitHub repository name first, then update
the remote URL locally:

```bash
git remote set-url origin https://github.com/Duvi-M/llm-hallucination-detection.git
```

## Suggested GitHub Description

```text
Detecting LLM hallucinations with Qwen hidden states, response-tail pooling,
PCA, and a lightweight probe classifier.
```

## Suggested GitHub Topics

```text
machine-learning
nlp
llm
hallucination-detection
transformers
qwen
scikit-learn
python
reproducible-ml
classification
```

## Pinned Project Summary

Reproducible ML project for detecting hallucinated responses from a small
language model. The pipeline extracts Qwen2.5 hidden states, builds
response-focused activation features, applies PCA, and trains a balanced
logistic regression probe. Includes documented experiments, failed attempts,
metrics, and one-command artifact generation.
