import argparse
import ast
import json
import os
import re
from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP_WORDS = {
	"services",
	"service",
	"operations",
	"operation",
	"installation",
	"install",
	"maintenance",
	"management",
	"and",
	"of",
	"for",
	"the",
	"with",
}


def normalize_text(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = text.lower()
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def tokenize(text: str) -> List[str]:
	return [t for t in normalize_text(text).split() if t]


def parse_tags(cell) -> List[str]:
	if isinstance(cell, list):
		return cell
	if not isinstance(cell, str) or not cell:
		return []
	try:
		parsed = ast.literal_eval(cell)
		if isinstance(parsed, list):
			return [str(x) for x in parsed]
	except Exception:
		pass
	return [s.strip() for s in cell.split(",") if s.strip()]


def build_company_doc(row: pd.Series) -> Tuple[str, str, str, List[str]]:
	"""Return (description, tags_text, meta_text, tags_list)."""
	desc = str(row.get("description", ""))
	tags = parse_tags(row.get("business_tags", ""))
	meta_bits = [row.get("sector", ""), row.get("category", ""), row.get("niche", "")]
	meta_text = " ".join([str(x) for x in meta_bits if isinstance(x, str)])
	tag_text = " ".join(tags)
	return desc, tag_text, meta_text, tags


def label_keywords(label: str) -> List[str]:
	toks = [t for t in tokenize(label) if t not in STOP_WORDS]
	bigrams = [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
	phrases = [label.lower()]
	return list(dict.fromkeys(toks + bigrams + phrases))


def rule_score_for_label(company_meta_tokens: set, description_text: str, tags: List[str], label: str) -> float:
	keys = label_keywords(label)
	if not keys:
		return 0.0

	desc = description_text.lower()
	phrase_hit = 1.0 if any(phrase in desc for phrase in keys if " " in phrase or len(phrase) > 5) else 0.0

	label_tokens = {t for t in tokenize(label) if t not in STOP_WORDS}
	if not label_tokens:
		token_overlap = 0.0
	else:
		token_overlap = len(label_tokens & company_meta_tokens) / len(label_tokens)

	tag_texts = " | ".join([t.lower() for t in tags])
	substr_match = 1.0 if any(k in tag_texts for k in keys) else 0.0

	score = max(
		phrase_hit,
		0.7 * token_overlap + 0.3 * substr_match,
	)
	return float(np.clip(score, 0.0, 1.0))


def build_vectorizer_and_embeddings(company_docs: List[str], label_docs: List[str]):
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=1)
	all_docs = company_docs + label_docs
	X = vectorizer.fit_transform(all_docs)
	X_companies = X[: len(company_docs)]
	X_labels = X[len(company_docs) :]
	return vectorizer, X_companies, X_labels


def select_labels(scores: np.ndarray, labels: List[str], min_threshold=0.2, margin=0.05, max_labels=5) -> List[str]:
	order = np.argsort(scores)[::-1]
	top_score = scores[order[0]] if len(order) else 0.0
	picked = []
	dynamic_threshold = max(min_threshold, top_score - margin)
	for idx in order:
		if scores[idx] >= dynamic_threshold:
			picked.append(labels[idx])
		if len(picked) >= max_labels:
			break
	if not picked and len(order):
		picked = [labels[order[0]]]
	return picked


def classify(
	companies_csv: str,
	taxonomy_csv: str,
	output_csv: str,
	weights: Tuple[float, float] = (0.6, 0.4),
	tfidf_field_weights: Tuple[float, float] = (0.7, 0.3),
	select_params: Tuple[float, float, int] = (0.2, 0.05, 5),
):

	companies = pd.read_csv(companies_csv)
	taxonomy = pd.read_csv(taxonomy_csv)
	if "label" not in taxonomy.columns:
		raise ValueError("Taxonomy CSV must have a 'label' column")
	labels = taxonomy["label"].dropna().astype(str).tolist()

	desc_docs: List[str] = []
	meta_docs: List[str] = []
	tags_list: List[List[str]] = []
	meta_token_sets = []
	desc_texts = []
	for _, row in companies.iterrows():
		desc, tag_text, meta_text, tags = build_company_doc(row)
		desc_docs.append(normalize_text(desc))
		combined_meta = f"{tag_text} {meta_text}".strip()
		meta_docs.append(normalize_text(combined_meta))
		tags_list.append(tags)
		meta_token_sets.append(set(tokenize(combined_meta)))
		desc_texts.append(desc)

	label_docs = [normalize_text(l) for l in labels]

	_, X_desc, L_desc = build_vectorizer_and_embeddings(desc_docs, label_docs)
	_, X_meta, L_meta = build_vectorizer_and_embeddings(meta_docs, label_docs)
	sim_desc = cosine_similarity(X_desc, L_desc)
	sim_meta = cosine_similarity(X_meta, L_meta)
	w_desc, w_meta = tfidf_field_weights
	sim_matrix = w_desc * sim_desc + w_meta * sim_meta

	rule = np.zeros_like(sim_matrix)
	for i in range(len(desc_docs)):
		meta_tokens = meta_token_sets[i]
		desc = desc_texts[i]
		tags = tags_list[i]
		r_scores = [rule_score_for_label(meta_tokens, desc, tags, lab) for lab in labels]
		rule[i, :] = r_scores

	w_rule, w_tfidf = weights
	combined = w_rule * rule + w_tfidf * sim_matrix

	selected_labels = []
	label_scores_serialized = []
	for i in range(combined.shape[0]):
		min_thr, margin, max_labels = select_params
		sels = select_labels(combined[i, :], labels, min_threshold=min_thr, margin=margin, max_labels=max_labels)
		selected_labels.append("; ".join(sels))
		order = np.argsort(combined[i, :])[::-1][:5]
		label_scores_serialized.append(
			json.dumps([{labels[j]: float(combined[i, j])} for j in order])
		)

	out_df = companies.copy()
	out_df["insurance_label"] = selected_labels
	out_df["label_scores_top5"] = label_scores_serialized
	out_df.to_csv(output_csv, index=False)

	all_assigned = [lab for labs in selected_labels for lab in labs.split("; ") if lab]
	freq = Counter(all_assigned)
	top10 = freq.most_common(10)
	print(f"Annotated {len(out_df)} companies. Output -> {output_csv}")
	print("Top assigned labels:")
	def main():
		default_companies = os.path.join(os.path.dirname(__file__), "ml_insurance_challenge.csv")
		default_taxonomy = os.path.join(
			os.path.dirname(__file__), "insurance_taxonomy - insurance_taxonomy.csv"
		)
		default_output = os.path.join(
			os.path.dirname(__file__), "annotated_ml_insurance_challenge.csv"
		)

		parser = argparse.ArgumentParser(description="Company -> Insurance taxonomy classifier")
		parser.add_argument("--companies", default=default_companies, help="Path to companies CSV")
		parser.add_argument("--taxonomy", default=default_taxonomy, help="Path to taxonomy CSV")
		parser.add_argument("--output", default=default_output, help="Where to write annotated CSV")
		parser.add_argument(
			"--weights",
			default="0.6,0.4",
			help="Weights for rule,tfidf (e.g., 0.7,0.3)",
		)
		parser.add_argument(
			"--tfidf-field-weights",
			default="0.7,0.3",
			help="Weights for description,meta+tags TF-IDF similarities (e.g., 0.8,0.2)",
		)
		parser.add_argument(
			"--selection",
			default="0.2,0.05,5",
			help="Selection params: min_threshold,margin,max_labels",
		)
		args = parser.parse_args()

		try:
			w_rule, w_tfidf = [float(x) for x in args.weights.split(",")]
		except Exception:
			w_rule, w_tfidf = 0.6, 0.4
		try:
			w_desc, w_meta = [float(x) for x in args.tfidf_field_weights.split(",")]
		except Exception:
			w_desc, w_meta = 0.7, 0.3
		try:
			sel_parts = args.selection.split(",")
			min_thr, margin, max_labels = float(sel_parts[0]), float(sel_parts[1]), int(sel_parts[2])
		except Exception:
			min_thr, margin, max_labels = 0.2, 0.05, 5

		classify(
			args.companies,
			args.taxonomy,
			args.output,
			(w_rule, w_tfidf),
			(w_desc, w_meta),
			(min_thr, margin, max_labels),
		)

