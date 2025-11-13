import datetime
import json
import os
import subprocess
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from tsne_vis import plot_tsne_3d

# ==== BEST CONFIG ====
CSV_PATH   = "data/mtsamples_clean.csv"  # must contain: text, label
ENCODER    = "emilyalsentzer/Bio_ClinicalBERT"
C_SVM      = 15
LOSS_SVM   = "squared_hinge"
MIN_PER_CLASS = 40          # k
RARE_STRATEGY = "drop"      # drop | other | train_only
BATCH_SIZE = 32
MAX_LEN    = 384
TRUNCATE_SENT = 8           # truncate to 8 sentences

# =============== Utils ===============

def truncate_text(text, max_sent=8):
    if not isinstance(text, str):
        return ""
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return ". ".join(parts[:max_sent])

def plot_confusion(y_true, y_pred, title="Confusion Matrix", save_path=None):
    import seaborn as sns
    labels = np.unique(np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                annot=True, cmap="Blues", square=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df["label"] = df["label"].astype(str).str.strip()
    df["text"]  = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"]).copy()
    df = df.drop_duplicates(subset=["text"])
    return df

def handle_rare_classes(df, label_col="label", k=40, strategy="drop"):
    counts = df[label_col].value_counts()
    rare = counts[counts < k].index
    if strategy == "drop":
        return df[~df[label_col].isin(rare)].copy(), None, "drop"
    elif strategy == "other":
        df2 = df.copy()
        df2.loc[df2[label_col].isin(rare), label_col] = "Other"
        return df2, None, "other"
    elif strategy == "train_only":
        return df.copy(), rare.tolist(), "train_only"
    else:
        raise ValueError("rare_strategy not valid.")

def split_data(df, test_size=0.2, seed=42,
                     label_col="label", text_col="text", n_len_bins=4):
    lengths = df[text_col].fillna("").str.split().map(len).clip(upper=5000)
    try:
        len_bins = pd.qcut(lengths, q=n_len_bins, labels=False, duplicates="drop")
    except ValueError:
        len_bins = pd.Series(0, index=df.index)
    strat_key = df[label_col].astype(str) + "||" + len_bins.astype(str)

    key_counts = strat_key.value_counts()
    valid_keys = key_counts[key_counts >= 2].index
    df_common = df[strat_key.isin(valid_keys)].copy()
    df_rare_keys = df[~strat_key.isin(valid_keys)].copy()

    if len(df_common) == 0:
        try:
            tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])
        except ValueError:
            tr, te = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        return tr.reset_index(drop=True), te.reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df_common))
    (train_idx, test_idx) = next(sss.split(idx, strat_key[df_common.index]))
    train_df = pd.concat([df_common.iloc[train_idx], df_rare_keys], ignore_index=True)
    test_df  = df_common.iloc[test_idx].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# =============== Encoder ===============

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts).detach().cpu().numpy()

def encode_texts(texts, tok, model, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    import torch
    embs = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            for k in enc:
                enc[k] = enc[k].to(model.device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embs.append(pooled)
    return np.vstack(embs)

# =============== Saving artifact ===============

def save_artifacts(clf, labels, artifacts_dir="artifacts"):
    import joblib
    os.makedirs(artifacts_dir, exist_ok=True)
    # classifier
    joblib.dump(clf, os.path.join(artifacts_dir, "classifier.joblib"))
    # encoder name
    open(os.path.join(artifacts_dir, "encoder_name.txt"), "w", encoding="utf-8").write(ENCODER + "\n")
    # label list
    labels_sorted = sorted(list(set(labels)))
    json.dump(labels_sorted, open(os.path.join(artifacts_dir, "label_list.json"), "w", encoding="utf-8"))
    # config
    cfg = dict(
        max_len=MAX_LEN, batch_size=BATCH_SIZE, truncate_sent=TRUNCATE_SENT,
        min_per_class=MIN_PER_CLASS, rare_strategy=RARE_STRATEGY,
        classifier=dict(kind="LinearSVC", C=C_SVM, loss=LOSS_SVM, class_weight="balanced")
    )
    yaml.safe_dump(cfg, open(os.path.join(artifacts_dir, "config.yaml"), "w", encoding="utf-8"))
    # version
    try:
        sha = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        sha = "nogit"
    ts = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
    open(os.path.join(artifacts_dir, "version.txt"), "w", encoding="utf-8").write(f"v0.1.0+{sha}  {ts}\n")
    print(f"[OK] Artifacts saved in: {artifacts_dir}/")

# =============== Pipeline (BioClinicalBERT + LinearSVC C=15, squared_hinge) ===============

def run_fixed_pipeline(do_tsne=True, do_confusion=True):
    df = load_data(CSV_PATH)

    df["text"] = df["text"].apply(lambda t: truncate_text(t, max_sent=TRUNCATE_SENT))

    df_filtered, rare_list_or_none, used = handle_rare_classes(df, label_col="label", k=MIN_PER_CLASS, strategy=RARE_STRATEGY)
    print("[INFO] Classi dopo filtro:", df_filtered["label"].value_counts().to_dict())

    train_df, test_df = split_data(df_filtered, test_size=0.2, seed=42, label_col="label", text_col="text", n_len_bins=4)
    if used == "train_only" and rare_list_or_none:
        test_df = test_df[~test_df["label"].isin(rare_list_or_none)].reset_index(drop=True)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    mdl = AutoModel.from_pretrained(ENCODER).to("cuda" if torch.cuda.is_available() else "cpu")

    X_train = encode_texts(train_df["text"].tolist(), tok, mdl, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    X_test  = encode_texts(test_df["text"].tolist(),  tok, mdl, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    y_train = train_df["label"].tolist()
    y_test  = test_df["label"].tolist()

    os.makedirs("artifacts", exist_ok=True)
    np.savez(
        "artifacts/tsne_train_embeddings.npz",
        X=X_train,
        y=np.array(y_train, dtype="U"),
    )
    print("[INFO] Saved TSNE cache to artifacts/tsne_train_embeddings.npz")

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    if do_tsne:
        plot_tsne_3d(X_train, y_train, n_samples=1500, perplexity=40)

    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=C_SVM, loss=LOSS_SVM, class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n[FIXED] {ENCODER} + LinearSVC(C={C_SVM}, loss={LOSS_SVM})")
    print(f"Top-1: {acc:.3f} | Macro-F1: {f1m:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    if do_confusion:
        title = f"Confusion Matrix - {ENCODER} + LinearSVC(C={C_SVM}, loss={LOSS_SVM})"
        filename = f"cm_BioClinicalBERT_SVM_C{C_SVM}_{LOSS_SVM}.png"
        outpath = os.path.join("plots", filename)
        plot_confusion(y_test, y_pred, title=title, save_path=outpath)
        print(f"[OK] Confusion matrix salvata in: {outpath}")


    all_labels = list(pd.concat([train_df["label"], test_df["label"]]).unique())
    save_artifacts(clf, all_labels, artifacts_dir="artifacts")

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--tsne_only", action="store_true",
                    help="Only plot t-SNE 3D.")
    ap.add_argument("--no_tsne", action="store_true",
                    help="No t-SNE.")
    ap.add_argument("--no_confusion", action="store_true",
                    help="No confusion matrix.")
    args = ap.parse_args()

    if args.tsne_only:
        do_tsne = True
        do_conf = False
    else:
        do_tsne = not args.no_tsne
        do_conf = not args.no_confusion

    run_fixed_pipeline(do_tsne=do_tsne, do_confusion=do_conf)
