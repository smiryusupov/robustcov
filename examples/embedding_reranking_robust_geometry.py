"""Example: robust embedding retrieval / RAG-style filtering.

This example uses the real 20 Newsgroups text dataset when available. It builds
dense text embeddings with TF-IDF + SVD, injects a controlled leverage-like
embedding artifact into part of the reference set, and compares:

1. clean cosine retrieval;
2. contaminated cosine retrieval;
3. cosine retrieval with robust leverage filtering.

The goal is not to replace semantic embedding models. The goal is to show where
robust covariance geometry can act as a second-stage filter when embedding
directions contain leverage-like artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import robustcov as rc


def _require_sklearn():
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This example requires scikit-learn. Install the optional sklearn "
            "dependencies before running it."
        ) from exc

    return fetch_20newsgroups, TruncatedSVD, TfidfVectorizer, normalize


def precision_at_k(ranked_labels, query_label, k=5):
    ranked_labels = np.asarray(ranked_labels)
    return float(np.mean(ranked_labels[:k] == query_label))


def evaluate_cosine(X_ref, y_ref, X_query, y_query, top_k=5):
    """Evaluate cosine retrieval from normalized embeddings."""
    scores = X_query @ X_ref.T
    order = np.argsort(-scores, axis=1)

    top1 = []
    p_at_k = []

    for i in range(X_query.shape[0]):
        labels = y_ref[order[i]]
        top1.append(labels[0] == y_query[i])
        p_at_k.append(precision_at_k(labels, y_query[i], k=top_k))

    return float(np.mean(top1)), float(np.mean(p_at_k)), order


def robust_leverage_scores(X_ref_raw, location, precision):
    centered = X_ref_raw - location
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def evaluate_robust_leverage_filter(
    X_ref_cos,
    y_ref,
    X_query_cos,
    y_query,
    first_stage_order,
    leverage,
    keep_mask,
    candidate_k=150,
    top_k=5,
):
    """Filter high-leverage candidates, then rank the rest by cosine.

    This keeps semantic similarity as the ranking score. Robust geometry is used
    only to remove candidates that look like leverage artifacts under the robust
    reference geometry.
    """
    top1 = []
    p_at_k = []
    filtered_orders = []

    for i in range(X_query_cos.shape[0]):
        candidates = first_stage_order[i, :candidate_k]

        kept = candidates[keep_mask[candidates]]
        removed = candidates[~keep_mask[candidates]]

        # Rank kept candidates by cosine similarity.
        if kept.size:
            kept_scores = X_query_cos[i] @ X_ref_cos[kept].T
            kept = kept[np.argsort(-kept_scores)]

        # Fill with removed candidates only if needed, also by cosine.
        if removed.size:
            removed_scores = X_query_cos[i] @ X_ref_cos[removed].T
            removed = removed[np.argsort(-removed_scores)]

        local_order = np.concatenate([kept, removed])
        filtered_orders.append(local_order)

        labels = y_ref[local_order]
        top1.append(labels[0] == y_query[i])
        p_at_k.append(precision_at_k(labels, y_query[i], k=top_k))

    return float(np.mean(top1)), float(np.mean(p_at_k)), np.asarray(filtered_orders)


def leverage_fraction(order, leverage_mask, k=10):
    return float(np.mean(leverage_mask[order[:, :k]]))


def main():
    fetch_20newsgroups, TruncatedSVD, TfidfVectorizer, normalize = _require_sklearn()

    rng = np.random.default_rng(11)
    outdir = Path("results/examples/embedding_reranking_robust_geometry")
    outdir.mkdir(parents=True, exist_ok=True)

    categories = [
        "comp.graphics",
        "rec.autos",
        "sci.space",
        "talk.politics.misc",
    ]

    try:
        data = fetch_20newsgroups(
            subset="train",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Could not load the 20 Newsgroups dataset. This example needs "
            "scikit-learn's dataset cache or network access for the first run."
        ) from exc

    texts = np.asarray(data.data)
    labels = np.asarray(data.target)

    keep = np.asarray([len(t.strip()) > 100 for t in texts])
    texts = texts[keep]
    labels = labels[keep]

    idx = rng.permutation(len(texts))[:1200]
    texts = texts[idx]
    labels = labels[idx]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_features=6000,
    )
    X_sparse = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=32, random_state=0)
    X = svd.fit_transform(X_sparse)

    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-8)

    n_ref = 850
    n_query = 250

    X_ref_clean = X[:n_ref]
    y_ref = labels[:n_ref]

    X_query = X[n_ref : n_ref + n_query]
    y_query = labels[n_ref : n_ref + n_query]

    # ------------------------------------------------------------------
    # Controlled leverage-like embedding artifact.
    #
    # We append a nuisance coordinate. It is present in all queries and in a
    # contaminated subset of reference embeddings. This mimics a batch,
    # boilerplate, sensor, or embedding-pipeline artifact that can influence
    # cosine similarity even though it is not semantic.
    # ------------------------------------------------------------------
    artifact_strength = 3.5
    artifact_noise = 0.03

    X_ref_cont = np.column_stack(
        [
            X_ref_clean,
            rng.normal(0.0, artifact_noise, size=n_ref),
        ]
    )

    X_query_cont = np.column_stack(
        [
            X_query,
            artifact_strength
            + rng.normal(0.0, artifact_noise, size=X_query.shape[0]),
        ]
    )

    leverage_mask = np.zeros(n_ref, dtype=bool)
    n_bad = int(0.10 * n_ref)
    bad_idx = rng.choice(n_ref, size=n_bad, replace=False)
    leverage_mask[bad_idx] = True

    X_ref_cont[bad_idx, -1] = artifact_strength + rng.normal(
        0.0, artifact_noise, size=n_bad
    )

    # Cosine baselines use normalized embeddings.
    X_ref_clean_cos = normalize(X_ref_clean)
    X_query_clean_cos = normalize(X_query)

    X_ref_cont_cos = normalize(X_ref_cont)
    X_query_cont_cos = normalize(X_query_cont)

    clean_top1, clean_p5, _ = evaluate_cosine(
        X_ref_clean_cos,
        y_ref,
        X_query_clean_cos,
        y_query,
    )

    cont_top1, cont_p5, cont_order = evaluate_cosine(
        X_ref_cont_cos,
        y_ref,
        X_query_cont_cos,
        y_query,
    )

    # Robust geometry fitted on contaminated reference embeddings.
    fit = rc.FastMCD(
        contamination=0.20,
        quality="high",
        random_state=0,
    ).fit(X_ref_cont)

    leverage = robust_leverage_scores(
        X_ref_raw=X_ref_cont,
        location=fit.location_,
        precision=fit.precision_,
    )

    # Use robust geometry as a filter: remove the highest-leverage 10% of the
    # reference set. In real workflows this threshold is a validation parameter.
    leverage_cutoff = np.quantile(leverage, 0.80)
    keep_mask = leverage <= leverage_cutoff

    robust_top1, robust_p5, robust_order = evaluate_robust_leverage_filter(
        X_ref_cos=X_ref_cont_cos,
        y_ref=y_ref,
        X_query_cos=X_query_cont_cos,
        y_query=y_query,
        first_stage_order=cont_order,
        leverage=leverage,
        keep_mask=keep_mask,
        candidate_k=150,
        top_k=5,
    )

    cont_lev10 = leverage_fraction(cont_order, leverage_mask, k=10)
    robust_lev10 = leverage_fraction(robust_order, leverage_mask, k=10)

    detected_bad = float(np.mean(~keep_mask[leverage_mask]))
    false_filtered = float(np.mean(~keep_mask[~leverage_mask]))

    print("Robust embedding retrieval / RAG-style filtering")
    print("================================================")
    print()
    print("Dataset: 20 Newsgroups")
    print("Embedding: TF-IDF + TruncatedSVD(32)")
    print("Reference documents:", n_ref)
    print("Query documents:", n_query)
    print("Injected leverage-like reference embeddings:", n_bad)
    print()
    print("Robust leverage filter")
    print("filtered reference fraction:          ", f"{np.mean(~keep_mask):8.3f}")
    print("injected artifacts filtered:          ", f"{detected_bad:8.3f}")
    print("ordinary references filtered:         ", f"{false_filtered:8.3f}")
    print()
    print("Retrieval quality")
    print("method                            top1_label_match   precision_at_5")
    print(f"clean cosine                            {clean_top1:8.3f}        {clean_p5:8.3f}")
    print(f"contaminated cosine                     {cont_top1:8.3f}        {cont_p5:8.3f}")
    print(f"robust leverage filter                  {robust_top1:8.3f}        {robust_p5:8.3f}")
    print()
    print("Leverage-like candidates in top-10")
    print(f"contaminated cosine:                    {cont_lev10:8.3f}")
    print(f"robust leverage filter:                 {robust_lev10:8.3f}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels_plot = ["clean\ncosine", "contaminated\ncosine", "robust\nfilter"]
    top1_values = [clean_top1, cont_top1, robust_top1]
    p5_values = [clean_p5, cont_p5, robust_p5]

    x = np.arange(len(labels_plot))
    width = 0.35

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, top1_values, width, label="top-1 label match")
    ax.bar(x + width / 2, p5_values, width, label="precision@5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Robust filtering under leverage-like embedding artifacts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "embedding_reranking_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(
        ["contaminated\ncosine", "robust\nfilter"],
        [cont_lev10, robust_lev10],
    )
    ax.set_ylim(0.0, max(cont_lev10, robust_lev10, 1e-6) * 1.25)
    ax.set_ylabel("fraction of top-10 candidates")
    ax.set_title("Leverage-like candidates in retrieval results")
    fig.tight_layout()
    fig.savefig(outdir / "embedding_reranking_leverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
