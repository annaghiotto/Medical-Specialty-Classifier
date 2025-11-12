# src/infer.py
import json, os
import numpy as np
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch

MIN_WORDS = 20

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts)

class MedicalSpecialtyPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):

        enc_path = os.path.join(artifacts_dir, "encoder_name.txt")
        clf_path = os.path.join(artifacts_dir, "classifier.joblib")
        labels_path = os.path.join(artifacts_dir, "label_list.json")
        cfg_path = os.path.join(artifacts_dir, "config.yaml")

        if not (os.path.exists(enc_path) and os.path.exists(clf_path) and os.path.exists(labels_path)):
            raise FileNotFoundError("Missing artifact (encoder_name.txt / classifier.joblib / label_list.json) in 'artifacts/'")

        self.encoder_name = open(enc_path, "r", encoding="utf-8").read().strip()

        import joblib, yaml
        self.clf = joblib.load(clf_path)
        self.labels = json.load(open(labels_path, "r", encoding="utf-8"))
        self.cfg = dict(max_len=384, batch_size=32)
        if os.path.exists(cfg_path):
            self.cfg.update(yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {})

        self.tok = AutoTokenizer.from_pretrained(self.encoder_name)
        self.mdl = AutoModel.from_pretrained(self.encoder_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.eval()

    def _encode(self, texts):
        embs = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.cfg["batch_size"]):
                batch = texts[i:i + self.cfg["batch_size"]]
                enc = self.tok(
                    batch, padding=True, truncation=True,
                    max_length=self.cfg["max_len"], return_tensors="pt"
                )
                for k in enc:
                    enc[k] = enc[k].to(self.mdl.device)
                out = self.mdl(**enc).last_hidden_state
                pooled = _mean_pool(out, enc["attention_mask"])
                embs.append(pooled.detach().cpu().numpy())
        X = np.vstack(embs)
        X = normalize(X)
        return X

    def predict(self, texts, topk: int = 3):
        """
        If topk == 1 -> list of labels (or 'ABSTAIN' if abstain).
        If topk  > 1 -> list of lists (top-k labels ordered, or ['ABSTAIN'] if abstain).
        Policy: text with less than MIN_WORDS words -> ABSTAIN
        """
        if isinstance(texts, str):
            texts = [texts]

        def _too_short(t: str) -> bool:
            return len(str(t).split()) < MIN_WORDS

        n = len(texts)
        outputs = [None] * n

        short_idx = [i for i, t in enumerate(texts) if _too_short(t)]       # text too short
        for i in short_idx:
            outputs[i] = ["ABSTAIN"] if topk > 1 else "ABSTAIN"

        normal_idx = [i for i in range(n) if i not in short_idx]
        if normal_idx:
            X = self._encode([texts[i] for i in normal_idx])

            if hasattr(self.clf, "decision_function"):
                scores = self.clf.decision_function(X)   # (m, n_classes) or (m,) binary
                if scores.ndim == 1:
                    # binary -> shape (m, 2)
                    scores = np.stack([-scores, scores], axis=1)

                max_scores = scores.max(axis=1)
                order = np.argsort(scores, axis=1)[:, ::-1]  # desc

                for pos, i in enumerate(normal_idx):
                    if topk == 1:
                        j = order[pos, 0]
                        outputs[i] = self.clf.classes_[j]
                    else:
                        js = order[pos, :topk]
                        outputs[i] = [self.clf.classes_[j] for j in js]
            else:
                # fallback
                preds = self.clf.predict(X)
                for pos, i in enumerate(normal_idx):
                    outputs[i] = [preds[pos]] if topk > 1 else preds[pos]

        return outputs if topk > 1 else outputs

