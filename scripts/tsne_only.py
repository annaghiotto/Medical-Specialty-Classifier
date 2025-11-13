import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

CACHE_PATH = "artifacts/tsne_train_embeddings.npz"

def plot_tsne_3d_from_cache(
    cache_path=CACHE_PATH,
    n_samples=1500,
    perplexity=40,
    random_state=42,
):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache file {cache_path} not found. "
            "Run med_spec_fix.py once to generate embeddings."
        )

    data = np.load(cache_path)
    X = data["X"]
    y = data["y"]

    print(f"[INFO] Loaded embeddings: X shape = {X.shape}, labels = {len(y)}")

    df = pd.DataFrame(X)
    df["label"] = y

    if len(df) > n_samples:
        df_sampled = (
            df.groupby("label", group_keys=False)
              .apply(
                  lambda g: g.sample(
                      min(len(g), max(30, n_samples // df["label"].nunique())),
                      random_state=random_state,
                  )
              )
              .reset_index(drop=True)
        )
    else:
        df_sampled = df

    X_s = df_sampled.drop(columns=["label"]).values
    y_s = df_sampled["label"].values

    print(f"[t-SNE] Using {len(X_s)} samples for visualization")

    tsne = TSNE(
        n_components=3,
        perplexity=min(perplexity, len(X_s) - 1),
        max_iter=1000,
        learning_rate="auto",
        init="random",
        random_state=random_state,
        verbose=1,
    )
    X_3d = tsne.fit_transform(X_s)

    vis_df = pd.DataFrame({
        "x": X_3d[:, 0],
        "y": X_3d[:, 1],
        "z": X_3d[:, 2],
        "label": y_s,
    })

    fig = px.scatter_3d(
        vis_df,
        x="x",
        y="y",
        z="z",
        color="label",
        opacity=0.8,
        hover_data=["label"],
        title="t-SNE 3D of Bio_ClinicalBERT embeddings (train cache)",
    )
    fig.update_traces(marker=dict(size=4))

    os.makedirs("plots", exist_ok=True)
    fig.write_html("plots/tsne_3d.html")   # interactive
    fig.write_image("plots/tsne_3d.png", scale=2)   # PNG

    print("[INFO] Saved 3D t-SNE")

    fig.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=CACHE_PATH, help="Path npz with X,y")
    ap.add_argument("--n_samples", type=int, default=1500)
    ap.add_argument("--perplexity", type=float, default=40.0)
    args = ap.parse_args()

    plot_tsne_3d_from_cache(
        cache_path=args.cache,
        n_samples=args.n_samples,
        perplexity=args.perplexity,
    )
