from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def plot_tsne_3d(X, labels, n_samples=1500, perplexity=40, random_state=42):
    """
    X: np.ndarray (n_samples, dim) - embeddings (es: from encode_texts)
    labels: array-like of str, corresponding labels for coloring
    n_samples: undersampling for velocity
    """
    import plotly.express as px

    X = np.asarray(X)
    labels = np.asarray(labels)

    df = pd.DataFrame(X)
    df["label"] = labels

    if len(df) > n_samples:
        df_sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max(50, n_samples // df["label"].nunique())),
                                        random_state=random_state))
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
        title="t-SNE 3D of Bio_ClinicalBERT embeddings",
    )
    fig.update_traces(marker=dict(size=4))
    fig.show()
