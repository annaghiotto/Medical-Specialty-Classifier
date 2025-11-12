import argparse, os, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =============== Utils: run tag & plotting ===============

def make_run_tag(args):
    enc = args.encoder.split("/")[-1] if hasattr(args, "encoder") else "enc"
    parts = [
        f"mode={getattr(args,'mode','')}",
        f"enc={enc}",
        f"clf={getattr(args,'clf','')}",
        f"k={getattr(args,'min_per_class','')}",
        f"rare={getattr(args,'rare_strategy','')}",
        f"maxlen={getattr(args,'max_len',256)}",
        f"bs={getattr(args,'batch_size',16)}",
        ("notrunc" if getattr(args, "no_truncate", False) else "trunc"),
    ]
    safe = [p.replace("/", "-").replace(" ", "") for p in parts if p]
    return "__".join(safe)

def plot_confusion(y_true, y_pred, title="Confusion Matrix", save_path=None):
    import seaborn as sns
    labels = np.unique(np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                annot=False, cmap="Blues", square=False)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# =============== Data loading & text building ===============

def load_data(path="mtsamples_clean.csv"):
    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in the CSV")

    df["label"] = df["label"].astype(str).str.strip()

    if "text" in df.columns:
        df["text"] = df["text"].astype(str).str.strip()
        df = df.dropna(subset=["text", "label"]).copy()
        df = df.drop_duplicates(subset=["text"])
    else:
        # fallback
        if "transcription" not in df.columns:
            raise ValueError("Missing both 'text' and 'transcription' columns in the CSV")
        df = df.dropna(subset=["transcription", "label"]).copy()
        df["transcription"] = df["transcription"].astype(str)
        df = df.drop_duplicates(subset=["transcription"])
    return df

def truncate_text(text, max_sent=8):
    if not isinstance(text, str):
        return ""
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return ". ".join(parts[:max_sent])

# =============== Rare classes handling & split smart ===============

def handle_rare_classes(df, label_col="label", k=10, strategy="drop"):
    counts = df[label_col].value_counts()
    rare = counts[counts < k].index
    if strategy == "drop":
        df2 = df[~df[label_col].isin(rare)].copy()
        return df2, None, "drop"
    elif strategy == "other":
        df2 = df.copy()
        df2.loc[df2[label_col].isin(rare), label_col] = "Other"
        return df2, None, "other"
    elif strategy == "train_only":
        return df.copy(), rare.tolist(), "train_only"
    else:
        raise ValueError("rare_strategy non valido")

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
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=df[label_col]
            )
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df_common))
    (train_idx, test_idx) = next(sss.split(idx, strat_key[df_common.index]))

    train_df = pd.concat([df_common.iloc[train_idx], df_rare_keys], ignore_index=True)
    test_df  = df_common.iloc[test_idx].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# =============== Frozen encoder ===============

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts).detach().cpu().numpy()

def encode_texts(texts, tok, model, batch_size=16, max_len=256):
    import torch
    embs = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            for k in enc: enc[k] = enc[k].to(model.device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embs.append(pooled)
    return np.vstack(embs)

def eval_frozen_encoder(train_df, test_df,
                        encoder_name="emilyalsentzer/Bio_ClinicalBERT",
                        clf_type="svm", save_tag=None,
                        batch_size=16, max_len=256):
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained(encoder_name)
    mdl = AutoModel.from_pretrained(encoder_name).to("cuda" if torch.cuda.is_available() else "cpu")

    X_train = encode_texts(train_df["text"].tolist(), tok, mdl, batch_size=batch_size, max_len=max_len)
    X_test  = encode_texts(test_df["text"].tolist(),  tok, mdl, batch_size=batch_size, max_len=max_len)
    y_train = train_df["label"].tolist()
    y_test  = test_df["label"].tolist()

    X_train = normalize(X_train); X_test = normalize(X_test)

    # classifier + gridsearch
    if clf_type == "svm":
        from sklearn.svm import LinearSVC
        base_clf = LinearSVC(class_weight="balanced", max_iter=5000)
        param_grid = {
            "C": [7, 10, 15, 100],
            "loss": ["hinge", "squared_hinge"]
        }

    elif clf_type in ["logreg_l2", "logreg_l1"]:
        from sklearn.linear_model import LogisticRegression
        penalty = "l1" if clf_type == "logreg_l1" else "l2"
        base_clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            penalty=penalty,
            n_jobs=-1
        )
        if penalty == "l1":
            param_grid = {"C": [0.01, 0.1, 1, 5, 10]}
        elif penalty == "l2":
            param_grid = {"C": [0.01, 0.1, 1, 5, 10]}
        else:  # elasticnet
            param_grid = {
                "C": [0.01, 0.1, 1, 5, 10],
                "l1_ratio": [0.0, 0.5, 1.0]
            }

    else:
        raise ValueError("clf_type must be 'svm' or 'logreg_l2' or 'logreg_l1'")

    # Grid search
    print("\n[GridSearch] Starting hyperparams search")
    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f"[GridSearch] Best hyperparams: {grid.best_params_}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n[Frozen {encoder_name} + {clf.__class__.__name__}] Top-1: {acc:.3f} | Macro-F1: {f1m:.3f}")

    title = f"Confusion Matrix - Frozen({clf.__class__.__name__})\n{encoder_name}\n{save_tag}" if save_tag else f"Confusion Matrix - Frozen({clf.__class__.__name__})"
    outpath = f"plots/cm_frozen__{save_tag}.png" if save_tag else None
    plot_confusion(y_test, y_pred, title=title, save_path=outpath)

# =============== Main ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="mtsamples_clean.csv")
    ap.add_argument("--encoder", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--clf", type=str, choices=["svm","logreg_l2","logreg_l1"], default="svm")
    ap.add_argument("--min_per_class", type=int, default=40,
                    help="Only keep classes with more than k instances")
    ap.add_argument("--rare_strategy", type=str, choices=["drop","other","train_only"],
                    default="drop", help="How to handle classes with less than k instances: drop/other/train_only")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    ap.add_argument("--max_len", type=int, default=256, help="Max token for encoding")
    ap.add_argument("--no_truncate", action="store_true", help="Do not truncate pre-cleaned texts")

    args = ap.parse_args()
    
    # pipeline
    df = load_data(args.csv)

    if not args.no_truncate:
        df["text"] = df["text"].apply(lambda t: truncate_text(t, max_sent=8))

    # rare class handling
    df_filtered, rare_list_or_none, used = handle_rare_classes(
        df, label_col="label",
        k=args.min_per_class, strategy=args.rare_strategy
    )
    class_counts = df_filtered["label"].value_counts()
    print("[INFO] Classes after rare-class filtering:", class_counts.to_dict())

    # Splitting (label × lunghezza)
    train_df, test_df = split_data(df_filtered, test_size=0.2, seed=42,
                                         label_col="label", text_col="text", n_len_bins=4)
    if used == "train_only" and rare_list_or_none:
        test_df = test_df[~test_df["label"].isin(rare_list_or_none)].reset_index(drop=True)

    # Run tags for saving
    run_tag = make_run_tag(args)

    # eval
    eval_frozen_encoder(train_df, test_df,
                            encoder_name=args.encoder,
                            clf_type=args.clf,
                            save_tag=run_tag,
                            batch_size=args.batch_size,
                            max_len=args.max_len)

if __name__ == "__main__":
    main()
