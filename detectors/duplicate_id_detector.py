"""
duplicate_id_detector.py
-------------------------
Prevents identity fraud where the same face is registered under
different individual details (i.e. the same person trying to enroll
under multiple ID records).

How it works:
1. Keeps a local JSON store of previously-seen face embeddings, each
   tagged with the ID it was registered under (data/known_faces.json).
2. When a new face embedding comes in (512-D vector), it is compared
   against every stored embedding using cosine similarity.
3. If the similarity to ANY embedding stored under a *different* ID
   is above the threshold (default 0.85), the face is flagged as a
   HIGH RISK duplicate identity attempt.

Dependencies: numpy (required), json (stdlib).
faiss-cpu is optional and only helps if your known_faces store grows
very large (thousands+ of records) and linear search becomes slow.
This script works fine with plain numpy for typical kiosk-scale data.

Usage as a library:
    from duplicate_id_detector import check_duplicate_identity

    result = check_duplicate_identity(
        embedding=my_512d_vector,
        id_tag="ID12345",
        store_path="data/known_faces.json",
    )
    # result is a dict matching the standard detector output schema

Usage as a script (for quick manual testing):
    python duplicate_id_detector.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------
# Config - adjust these two paths if your repo uses different locations
# ---------------------------------------------------------------------
KNOWN_FACES_PATH = os.path.join("data", "known_faces.json")
LOG_PATH = os.path.join("data", "detection_logs.json")
SIMILARITY_THRESHOLD = 0.85
EMBEDDING_DIM = 512


# ---------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------
def load_known_faces(store_path: str = KNOWN_FACES_PATH) -> list:
    """
    Loads the known_faces.json store.
    Expected format:
    [
      {"id_tag": "ID12345", "embedding": [0.01, 0.02, ...], "name": "optional"},
      {"id_tag": "ID67890", "embedding": [0.03, 0.05, ...], "name": "optional"}
    ]
    Returns an empty list if the file doesn't exist yet (first run).
    """
    if not os.path.exists(store_path):
        return []
    with open(store_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Corrupt or empty file - treat as no records rather than crashing
            return []


def save_known_faces(records: list, store_path: str = KNOWN_FACES_PATH) -> None:
    """Overwrites the known_faces.json store with the given list of records."""
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(records, f, indent=2)


def register_face(embedding, id_tag: str, store_path: str = KNOWN_FACES_PATH, name: str = None) -> None:
    """
    Adds a new face embedding to the known_faces store under the given ID tag.
    Call this AFTER check_duplicate_identity has cleared the face as safe,
    so genuinely new/unique people get added to the historical record.
    """
    records = load_known_faces(store_path)
    records.append({
        "id_tag": id_tag,
        "embedding": _to_list(embedding),
        "name": name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    save_known_faces(records, store_path)


def _to_list(embedding) -> list:
    if isinstance(embedding, np.ndarray):
        return embedding.astype(float).tolist()
    return list(embedding)


# ---------------------------------------------------------------------
# Core similarity logic
# ---------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b) -> float:
    """Standard cosine similarity between two vectors, returned as a float in [-1, 1]."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_duplicate_identity(
    embedding,
    id_tag: str,
    store_path: str = KNOWN_FACES_PATH,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Compares `embedding` against every record in the known_faces store.

    Only matches against records whose id_tag is DIFFERENT from the
    incoming id_tag count as a duplicate-identity risk - a returning
    person re-verifying under their OWN id_tag is expected and safe.

    Returns a dict following the standard detector output schema:
    {
        "detector_name": "duplicate_identity_check",
        "score": float,        # 0.0 (safe) - 1.0 (flagged), based on top match similarity
        "confidence": str,     # "high" | "medium" | "low"
        "explanation": str,
        "status": str          # "passed" | "flagged" | "failed"
    }
    """
    try:
        embedding = np.asarray(embedding, dtype=float)

        if embedding.ndim != 1 or embedding.shape[0] != EMBEDDING_DIM:
            return _build_result(
                score=0.0,
                confidence="low",
                explanation=f"Invalid embedding shape {embedding.shape}, expected ({EMBEDDING_DIM},).",
                status="failed",
            )

        known_faces = load_known_faces(store_path)

        best_match = None       # highest similarity seen against a DIFFERENT id_tag
        best_similarity = -1.0

        for record in known_faces:
            record_id_tag = record.get("id_tag")
            record_embedding = record.get("embedding")
            if record_id_tag is None or record_embedding is None:
                continue  # skip malformed records rather than crashing the kiosk

            similarity = cosine_similarity(embedding, record_embedding)

            if record_id_tag != id_tag and similarity > best_similarity:
                best_similarity = similarity
                best_match = record_id_tag

        # No comparable records at all -> nothing to flag against
        if best_match is None:
            return _build_result(
                score=0.0,
                confidence="high",
                explanation="No prior records found under a different ID. No duplicate risk detected.",
                status="passed",
            )

        score = max(0.0, min(1.0, best_similarity))  # clamp into [0, 1] for the output schema

        if best_similarity > threshold:
            margin = best_similarity - threshold
            confidence = "high" if margin > 0.05 else "medium"
            return _build_result(
                score=round(score, 4),
                confidence=confidence,
                explanation=(
                    f"HIGH RISK: Face matches an existing record under a different ID "
                    f"('{best_match}') with similarity {best_similarity:.4f}, "
                    f"above the {threshold} duplicate-identity threshold."
                ),
                status="flagged",
            )

        # Below threshold - safe, but report how close it was for transparency
        margin = threshold - best_similarity
        confidence = "high" if margin > 0.05 else "medium"
        return _build_result(
            score=round(score, 4),
            confidence=confidence,
            explanation=(
                f"No duplicate identity detected. Closest match under a different ID "
                f"('{best_match}') had similarity {best_similarity:.4f}, "
                f"below the {threshold} threshold."
            ),
            status="passed",
        )

    except Exception as e:
        return _build_result(
            score=0.0,
            confidence="low",
            explanation=f"Duplicate identity check failed to run: {e}",
            status="failed",
        )


def _build_result(score: float, confidence: str, explanation: str, status: str) -> dict:
    return {
        "detector_name": "duplicate_identity_check",
        "score": score,
        "confidence": confidence,
        "explanation": explanation,
        "status": status,
    }


# ---------------------------------------------------------------------
# Logging - appends every detection result to a running JSON log file
# ---------------------------------------------------------------------
def log_result(result: dict, id_tag: str, log_path: str = LOG_PATH) -> None:
    """
    Appends a timestamped detection result to data/detection_logs.json.
    Creates the file/folder if they don't exist yet.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id_tag": id_tag,
        **result,
    })

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


# ---------------------------------------------------------------------
# Manual test / demo when run directly: python duplicate_id_detector.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_store = os.path.join("data", "known_faces_demo.json")
    demo_log = os.path.join("data", "detection_logs_demo.json")

    # Seed the demo store with one existing face under "ID001"
    rng = np.random.default_rng(42)
    original_embedding = rng.normal(size=EMBEDDING_DIM)
    save_known_faces(
        [{"id_tag": "ID001", "embedding": _to_list(original_embedding), "name": "Demo Person"}],
        demo_store,
    )

    print("Test 1: Same face, different ID -> should FLAG")
    duplicate_attempt = original_embedding + rng.normal(scale=0.01, size=EMBEDDING_DIM)
    result_1 = check_duplicate_identity(duplicate_attempt, id_tag="ID002", store_path=demo_store)
    print(json.dumps(result_1, indent=2))
    log_result(result_1, id_tag="ID002", log_path=demo_log)

    print("\nTest 2: Different face, new ID -> should PASS")
    new_face = rng.normal(size=EMBEDDING_DIM)
    result_2 = check_duplicate_identity(new_face, id_tag="ID003", store_path=demo_store)
    print(json.dumps(result_2, indent=2))
    log_result(result_2, id_tag="ID003", log_path=demo_log)

    print(f"\nDemo logs written to: {demo_log}")

"""
duplicate_id_detector.py
-------------------------
Prevents identity fraud where the same face is registered under
different individual details (i.e. the same person trying to enroll
under multiple ID records).

How it works:
1. Keeps a local JSON store of previously-seen face embeddings, each
   tagged with the ID it was registered under (data/known_faces.json).
2. When a new face embedding comes in (512-D vector), it is compared
   against every stored embedding using cosine similarity.
3. If the similarity to ANY embedding stored under a *different* ID
   is above the threshold (default 0.85), the face is flagged as a
   HIGH RISK duplicate identity attempt.

Dependencies: numpy (required), json (stdlib).
faiss-cpu is optional and only helps if your known_faces store grows
very large (thousands+ of records) and linear search becomes slow.
This script works fine with plain numpy for typical kiosk-scale data.

Usage as a library:
    from duplicate_id_detector import check_duplicate_identity

    result = check_duplicate_identity(
        embedding=my_512d_vector,
        id_tag="ID12345",
        store_path="data/known_faces.json",
    )
    # result is a dict matching the standard detector output schema

Usage as a script (for quick manual testing):
    python duplicate_id_detector.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------
# Config - adjust these two paths if your repo uses different locations
# ---------------------------------------------------------------------
KNOWN_FACES_PATH = os.path.join("data", "known_faces.json")
LOG_PATH = os.path.join("data", "detection_logs.json")
SIMILARITY_THRESHOLD = 0.85
EMBEDDING_DIM = 512


# ---------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------
def load_known_faces(store_path: str = KNOWN_FACES_PATH) -> list:
    """
    Loads the known_faces.json store.
    Expected format:
    [
      {"id_tag": "ID12345", "embedding": [0.01, 0.02, ...], "name": "optional"},
      {"id_tag": "ID67890", "embedding": [0.03, 0.05, ...], "name": "optional"}
    ]
    Returns an empty list if the file doesn't exist yet (first run).
    """
    if not os.path.exists(store_path):
        return []
    with open(store_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Corrupt or empty file - treat as no records rather than crashing
            return []


def save_known_faces(records: list, store_path: str = KNOWN_FACES_PATH) -> None:
    """Overwrites the known_faces.json store with the given list of records."""
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(records, f, indent=2)


def register_face(embedding, id_tag: str, store_path: str = KNOWN_FACES_PATH, name: str = None) -> None:
    """
    Adds a new face embedding to the known_faces store under the given ID tag.
    Call this AFTER check_duplicate_identity has cleared the face as safe,
    so genuinely new/unique people get added to the historical record.
    """
    records = load_known_faces(store_path)
    records.append({
        "id_tag": id_tag,
        "embedding": _to_list(embedding),
        "name": name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    save_known_faces(records, store_path)


def _to_list(embedding) -> list:
    if isinstance(embedding, np.ndarray):
        return embedding.astype(float).tolist()
    return list(embedding)


# ---------------------------------------------------------------------
# Core similarity logic
# ---------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b) -> float:
    """Standard cosine similarity between two vectors, returned as a float in [-1, 1]."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_duplicate_identity(
    embedding,
    id_tag: str,
    store_path: str = KNOWN_FACES_PATH,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Compares `embedding` against every record in the known_faces store.

    Only matches against records whose id_tag is DIFFERENT from the
    incoming id_tag count as a duplicate-identity risk - a returning
    person re-verifying under their OWN id_tag is expected and safe.

    Returns a dict following the standard detector output schema:
    {
        "detector_name": "duplicate_identity_check",
        "score": float,        # 0.0 (safe) - 1.0 (flagged), based on top match similarity
        "confidence": str,     # "high" | "medium" | "low"
        "explanation": str,
        "status": str          # "passed" | "flagged" | "failed"
    }
    """
    try:
        embedding = np.asarray(embedding, dtype=float)

        if embedding.ndim != 1 or embedding.shape[0] != EMBEDDING_DIM:
            return _build_result(
                score=0.0,
                confidence="low",
                explanation=f"Invalid embedding shape {embedding.shape}, expected ({EMBEDDING_DIM},).",
                status="failed",
            )

        known_faces = load_known_faces(store_path)

        best_match = None       # highest similarity seen against a DIFFERENT id_tag
        best_similarity = -1.0

        for record in known_faces:
            record_id_tag = record.get("id_tag")
            record_embedding = record.get("embedding")
            if record_id_tag is None or record_embedding is None:
                continue  # skip malformed records rather than crashing the kiosk

            similarity = cosine_similarity(embedding, record_embedding)

            if record_id_tag != id_tag and similarity > best_similarity:
                best_similarity = similarity
                best_match = record_id_tag

        # No comparable records at all -> nothing to flag against
        if best_match is None:
            return _build_result(
                score=0.0,
                confidence="high",
                explanation="No prior records found under a different ID. No duplicate risk detected.",
                status="passed",
            )

        score = max(0.0, min(1.0, best_similarity))  # clamp into [0, 1] for the output schema

        if best_similarity > threshold:
            margin = best_similarity - threshold
            confidence = "high" if margin > 0.05 else "medium"
            return _build_result(
                score=round(score, 4),
                confidence=confidence,
                explanation=(
                    f"HIGH RISK: Face matches an existing record under a different ID "
                    f"('{best_match}') with similarity {best_similarity:.4f}, "
                    f"above the {threshold} duplicate-identity threshold."
                ),
                status="flagged",
            )

        # Below threshold - safe, but report how close it was for transparency
        margin = threshold - best_similarity
        confidence = "high" if margin > 0.05 else "medium"
        return _build_result(
            score=round(score, 4),
            confidence=confidence,
            explanation=(
                f"No duplicate identity detected. Closest match under a different ID "
                f"('{best_match}') had similarity {best_similarity:.4f}, "
                f"below the {threshold} threshold."
            ),
            status="passed",
        )

    except Exception as e:
        return _build_result(
            score=0.0,
            confidence="low",
            explanation=f"Duplicate identity check failed to run: {e}",
            status="failed",
        )


def _build_result(score: float, confidence: str, explanation: str, status: str) -> dict:
    return {
        "detector_name": "duplicate_identity_check",
        "score": score,
        "confidence": confidence,
        "explanation": explanation,
        "status": status,
    }


# ---------------------------------------------------------------------
# Logging - appends every detection result to a running JSON log file
# ---------------------------------------------------------------------
def log_result(result: dict, id_tag: str, log_path: str = LOG_PATH) -> None:
    """
    Appends a timestamped detection result to data/detection_logs.json.
    Creates the file/folder if they don't exist yet.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id_tag": id_tag,
        **result,
    })

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


# ---------------------------------------------------------------------
# Manual test / demo when run directly: python duplicate_id_detector.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_store = os.path.join("data", "known_faces_demo.json")
    demo_log = os.path.join("data", "detection_logs_demo.json")

    # Seed the demo store with one existing face under "ID001"
    rng = np.random.default_rng(42)
    original_embedding = rng.normal(size=EMBEDDING_DIM)
    save_known_faces(
        [{"id_tag": "ID001", "embedding": _to_list(original_embedding), "name": "Demo Person"}],
        demo_store,
    )

    print("Test 1: Same face, different ID -> should FLAG")
    duplicate_attempt = original_embedding + rng.normal(scale=0.01, size=EMBEDDING_DIM)
    result_1 = check_duplicate_identity(duplicate_attempt, id_tag="ID002", store_path=demo_store)
    print(json.dumps(result_1, indent=2))
    log_result(result_1, id_tag="ID002", log_path=demo_log)

    print("\nTest 2: Different face, new ID -> should PASS")
    new_face = rng.normal(size=EMBEDDING_DIM)
    result_2 = check_duplicate_identity(new_face, id_tag="ID003", store_path=demo_store)
    print(json.dumps(result_2, indent=2))
    log_result(result_2, id_tag="ID003", log_path=demo_log)

    print(f"\nDemo logs written to: {demo_log}")
