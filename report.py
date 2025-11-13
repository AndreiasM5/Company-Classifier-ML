import argparse
import json
import os
from collections import Counter

import pandas as pd


def parse_top_scores(cell):
    try:
        arr = json.loads(cell)
        flat = []
        for obj in arr:
            for k, v in obj.items():
                flat.append((k, float(v)))
        return flat
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Summarize annotated classifier output")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "annotated_ml_insurance_challenge.csv"),
        help="Path to annotated CSV",
    )
    parser.add_argument("--topn", type=int, default=20, help="Top-N labels to display")
    parser.add_argument("--samples", type=int, default=3, help="Sample companies per label")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    all_assigned = []
    for labs in df.get("insurance_label", "").fillna(""):
        parts = [x.strip() for x in str(labs).split(";") if x.strip()]
        all_assigned.extend(parts)
    freq = Counter(all_assigned)
    top_labels = freq.most_common(args.topn)

    print("Top assigned labels:")
    for lab, cnt in top_labels:
        print(f"  {lab}: {cnt}")

    def gap(row):
        scores = parse_top_scores(row.get("label_scores_top5", "[]"))
        if len(scores) < 2:
            return None
        return float(scores[0][1] - scores[1][1])

    df["top_gap"] = df.apply(gap, axis=1)
    amb = df[df["top_gap"].notna()].sort_values("top_gap").head(20)

    out_md = os.path.join(os.path.dirname(args.input), "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Classification Report\n\n")
        f.write("## Top Labels\n\n")
        for lab, cnt in top_labels:
            f.write(f"- {lab}: {cnt}\n")
        f.write("\n## Most Ambiguous (smallest top1-top2 gap)\n\n")
        for _, r in amb.iterrows():
            f.write(f"- gap={r['top_gap']:.4f}, labels={r.get('insurance_label','')}\n")
    print(f"\nMarkdown report written to: {out_md}")


if __name__ == "__main__":
    main()
