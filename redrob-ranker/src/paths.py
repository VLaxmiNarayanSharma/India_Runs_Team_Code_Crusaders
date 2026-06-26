"""Default paths for hackathon data files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"

JOB_DESCRIPTION_DOCX = DATASET_ROOT / "job_description.docx"
CANDIDATES_JSONL = DATASET_ROOT / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON = DATASET_ROOT / "sample_candidates.json"

PRECOMPUTED_DIR = ROOT / "data" / "precomputed"
JD_REQUIREMENTS_JSON = PRECOMPUTED_DIR / "jd_requirements.json"
CANDIDATE_FEATURES_JSONL = PRECOMPUTED_DIR / "candidate_features.jsonl"
CANDIDATE_FEATURES_FULL_JSONL = PRECOMPUTED_DIR / "candidate_features_full.jsonl"
RETRIEVED_TOP3K_JSONL = PRECOMPUTED_DIR / "retrieved_top3000.jsonl"
HONEYPOT_ASSESSMENTS_JSONL = PRECOMPUTED_DIR / "honeypot_assessments.jsonl"
SCORED_TOP100_JSONL = PRECOMPUTED_DIR / "scored_top100.jsonl"
CANDIDATE_EMBEDDINGS_NPY = PRECOMPUTED_DIR / "candidate_embeddings.npy"
JD_EMBEDDING_NPY = PRECOMPUTED_DIR / "jd_embedding.npy"
