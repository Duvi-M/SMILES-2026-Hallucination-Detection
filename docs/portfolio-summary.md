# Portfolio Summary

## Project Title

LLM Hallucination Detection Probe

## One-Line Summary

Built a reproducible NLP pipeline that detects hallucinated responses from a
small language model using Qwen2.5 hidden states, response-tail pooling, PCA,
and balanced logistic regression.

## Resume Version

LLM Hallucination Detection Probe - Built a reproducible machine learning
pipeline using Qwen2.5-0.5B hidden states to classify generated responses as
truthful or hallucinated. Engineered response-tail activation features, reduced
overfitting with PCA, trained a balanced logistic regression probe, documented
failed experiments, and generated competition-ready predictions with a
one-command workflow.

## LinkedIn / Portfolio Version

This project detects hallucinated answers from a small language model by
training a lightweight probe on the model's internal hidden states. Instead of
fine-tuning the LLM, the pipeline extracts activations from selected transformer
layers, pools the response tail, applies PCA, and trains a balanced logistic
regression classifier.

The project demonstrates reproducible ML engineering, NLP feature design,
experiment comparison, and clear documentation. It includes an official
competition-style report, generated predictions, evaluation metrics, and a
single command for reproducing the artifacts.

## Skills Demonstrated

- Python machine learning workflow design
- Transformer hidden-state feature engineering
- NLP hallucination detection
- scikit-learn modeling and evaluation
- Dimensionality reduction with PCA
- Class imbalance handling
- Reproducible experiment documentation
- Competition-style artifact generation

## Interview Talking Points

- Why hidden states can expose useful signals beyond surface text.
- Why response-tail pooling was a better fit than full-sequence pooling.
- How PCA helped control overfitting with a small dataset and large feature
  vectors.
- Why a simple balanced logistic regression model was a strong baseline.
- Which attempted ideas were rejected and how validation behavior guided the
  final design.
