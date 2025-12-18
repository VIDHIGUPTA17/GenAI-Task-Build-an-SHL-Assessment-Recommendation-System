import csv
import requests
from collections import defaultdict
from typing import List, Dict, Set

# ===========================
# CONFIG
# ===========================
API_BASE_URL = "http://genai-task-build-an-shl-assessment.onrender.com"  # change if deployed online
TRAIN_CSV_PATH = "Gen_AI Dataset(Train-Set).csv"   # TODO: put your actual train CSV path
TEST_CSV_PATH = "unlables.xlsx"   # TODO: put your actual test CSV path
OUTPUT_PREDICTION_CSV = "phase_c_predictions.csv"

K = 10  # top-K recommendations to request


# ===========================
# HELPER FUNCTIONS
# ===========================
def normalize_url(url: str) -> str:
    """Simple URL normalization to improve matching."""
    if not url:
        return ""
    url = url.strip()
    # remove trailing slash
    if url.endswith("/"):
        url = url[:-1]
    return url.lower()


def call_recommend_api(query: str, k: int = K) -> List[str]:
    """Call your /recommend API and return list of assessment URLs."""
    params = {"query": query, "k": k}
    resp = requests.post(f"{API_BASE_URL}/recommend", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    urls = []
    for item in data.get("recommended_assessments", []):
        url = item.get("url", "")
        if url:
            urls.append(url)
    return urls


# ===========================
# PART A: EVALUATE ON TRAIN SET
# ===========================
def load_train_relevance(train_csv_path: str) -> Dict[str, Set[str]]:
    """
    Load labelled train data.
    Expected columns: query, assessment_url
    Returns: {query_text -> set(relevant_urls)}
    """
    relevance = defaultdict(set)
    with open(train_csv_path, "r", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Query"].strip()
            url = normalize_url(row["Assessment_url"])
            if q and url:
                relevance[q].add(url)
    return relevance


def evaluate_mean_recall_at_k(train_csv_path: str, k: int = K):
    relevance = load_train_relevance(train_csv_path)
    queries = list(relevance.keys())

    per_query_recall = []

    print(f"Found {len(queries)} unique train queries")

    for q in queries:
        relevant_urls = relevance[q]
        predicted_urls_raw = call_recommend_api(q, k=k)
        predicted_urls = {normalize_url(u) for u in predicted_urls_raw if u}

        # avoid division by zero; if no labelled relevant URLs, skip
        if not relevant_urls:
            continue

        intersection = relevant_urls.intersection(predicted_urls)
        recall = len(intersection) / len(relevant_urls)

        per_query_recall.append(recall)
        print(f"Query: {q}")
        print(f"  Relevant URLs count: {len(relevant_urls)}")
        print(f"  Predicted URLs count: {len(predicted_urls)}")
        print(f"  Intersection count: {len(intersection)}")
        print(f"  Recall@{k}: {recall:.4f}\n")

    if not per_query_recall:
        print("No queries with relevant labels; cannot compute Mean Recall.")
        return

    mean_recall = sum(per_query_recall) / len(per_query_recall)
    print(f"Mean Recall@{k} over {len(per_query_recall)} queries: {mean_recall:.4f}")

import pandas as pd

# ===========================
# PART B: GENERATE TEST PREDICTION CSV
# ===========================
def load_test_queries(test_csv_path: str) -> List[str]:
    """
    Load unlabelled test queries.
    Expected column: query
    """
    df = pd.read_excel(test_csv_path)
    return df["Query"].dropna().astype(str).tolist()

def generate_submission_csv(test_csv_path: str, output_csv_path: str, k: int = K):
    """
    Generate submission CSV in required long format:
    Query,Assessment_url
    q1,url1
    q1,url2
    ...
    """
    queries = load_test_queries(test_csv_path)
    print(f"Generating predictions for {len(queries)} test queries")

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        # IMPORTANT: header must match exactly what PDF says
        writer.writerow(["Query", "Assessment_url"])

        for q in queries:
            urls = call_recommend_api(q, k=k)

            # If API returns fewer than k, just write what you have (1–10 allowed)
            for url in urls:
                writer.writerow([q, url])

            print(f"Query: {q}")
            print(f"  Returned {len(urls)} URLs")

    print(f"Submission CSV written to: {output_csv_path}")


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    # 1) Evaluate on labelled train set
    print("=== Evaluating Mean Recall@10 on train set ===")
    evaluate_mean_recall_at_k(TRAIN_CSV_PATH, k=K)

    # 2) Generate prediction CSV for test set
    print("\n=== Generating prediction CSV for test set ===")
    generate_submission_csv(TEST_CSV_PATH, OUTPUT_PREDICTION_CSV, k=K)
