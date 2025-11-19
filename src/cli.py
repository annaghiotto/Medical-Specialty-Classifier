import argparse
import json

from .infer import MedicalSpecialtyPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Text to classify")
    ap.add_argument(
        "--file", type=str, help="File .txt with one report per line (UTF-8)"
    )
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    pred = MedicalSpecialtyPredictor(args.artifacts)

    if args.text:
        out = pred.predict(args.text, topk=args.topk)
        print(json.dumps({"input": args.text, "pred": out}, ensure_ascii=False))
        return

    if args.file:
        texts = [line.strip() for line in open(args.file, encoding="utf-8")]
        outs = pred.predict(texts, topk=args.topk)
        for t, o in zip(texts, outs, strict=True):
            print(
                json.dumps(
                    {"input": t[:160] + ("..." if len(t) > 160 else ""), "pred": o},
                    ensure_ascii=False,
                )
            )
        return

    ap.error("Expected --text or --file")


if __name__ == "__main__":
    main()
