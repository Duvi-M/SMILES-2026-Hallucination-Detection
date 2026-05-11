"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


SELECTED_LAYERS = (-1, -2, -4, -8)
TAIL_TOKENS = 64


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    real_positions = attention_mask.to(torch.bool).nonzero(as_tuple=False).squeeze(-1)
    real_positions = real_positions.to(hidden_states.device)
    if real_positions.numel() == 0:
        raise ValueError("attention_mask contains no real tokens")

    tail_positions = real_positions[-min(TAIL_TOKENS, real_positions.numel()) :]

    pooled: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        real_tokens = layer.index_select(0, real_positions)
        tail_tokens = layer.index_select(0, tail_positions)

        pooled.extend(
            [
                layer[real_positions[-1]],  # answer-final token
                tail_tokens.mean(dim=0),     # response-tail summary
                real_tokens.mean(dim=0),     # full prompt+response summary
            ]
        )

    return torch.cat(pooled, dim=0).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    real_positions = attention_mask.to(torch.bool).nonzero(as_tuple=False).squeeze(-1)
    real_positions = real_positions.to(hidden_states.device)
    if real_positions.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=hidden_states.device)

    features: list[torch.Tensor] = [
        torch.tensor([float(real_positions.numel())], device=hidden_states.device)
    ]

    last_vectors = []
    for layer_idx in SELECTED_LAYERS:
        tokens = hidden_states[layer_idx].index_select(0, real_positions).float()
        token_norms = torch.linalg.vector_norm(tokens, dim=1)
        last_vector = tokens[-1]
        last_vectors.append(last_vector)
        features.append(
            torch.stack(
                [
                    token_norms.mean(),
                    token_norms.std(unbiased=False),
                    torch.linalg.vector_norm(last_vector),
                ]
            )
        )

    for left, right in zip(last_vectors, last_vectors[1:]):
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
        features.append(cosine.reshape(1))

    return torch.cat(features, dim=0).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
