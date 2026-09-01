"""
detectors/duplicate_id_detector.py
------------------------------------
Prevents identity fraud where the same face is registered under
different individual details (i.e. the same person trying to enroll
under multiple ID records).

How it works:
1. Keeps a local JSON store of previously-seen face embeddings, each
   tagged with the person/ID it was registered under
   (data/known_faces.json).
2. When a new 512-D face embedding comes in, it is compared against
   every stored embedding using cosine similarity.
3. If similarity to an embedding stored under a DIFFERENT person_id is
   above the threshold (default 0.85) -> flagged as HIGH RISK
   duplicate identity attempt.

Dependencies: numpy (required), json (stdlib), opencv-python + insightface
(only needed for run_duplicate_id_detector's own face extraction step;
not needed if you only ever call check_duplicate_identity directly with
an embedding you already have).

Pipeline integration:
    run_duplicate_id_detector(image_path) is the entry point
    kavach_engine.py's detector_pipeline calls, matching every other
    detector's (image_path) -> dict signature.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
KNOWN_FACES_PATH = os.path.join("data", "known_faces.json")
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
      {"person_id": "ID12345", "embedding": [0.01, 0.02, ...], "registered_at": "..."},
      {"person_id": "ID67890", "embedding": [0.03, 0.05, ...], "registered_at": "..."}
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


def save_new_face_vector(person_id: str, embedding_512d, store_path: str = KNOWN_FACES_PATH) -> None:
    """
    Appends a new face embedding to data/known_faces.json under the
    given person_id.

    Call this after check_duplicate_identity() has cleared a face as
    safe, so genuinely new/unique people get added to the historical
    record for future comparisons. Do NOT call this for a flagged or
    failed check - that would write a fraud attempt into the trusted
    store.

    Args:
        person_id: the ID/tag this face should be registered under.
        embedding_512d: a 512-D face embedding (list, tuple, or numpy array).
        store_path: which known_faces.json file to append to.

    Raises:
        ValueError: if embedding_512d isn't a valid 512-D vector.
    """
    vector = np.asarray(embedding_512d, dtype=float)
    if vector.ndim != 1 or vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"embedding_512d must be a 512-D vector, got shape {vector.shape}")

    records = load_known_faces(store_path)
    records.append({
        "person_id": person_id,
        "embedding": vector.tolist(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    save_known_faces(records, store_path)


# ---------------------------------------------------------------------
# Core similarity logic
# ---------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b) -> float:
    """Standard cosine similarity between two vectors, returned as a float in [-1, 1]."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_duplicate_identity(
    embedding,
    person_id: str,
    store_path: str = KNOWN_FACES_PATH,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Compares `embedding` against every record in the known_faces store.

    Only matches against records whose person_id is DIFFERENT from the
    incoming person_id count as a duplicate-identity risk - a returning
    person re-verifying under their OWN person_id is expected and safe.

    Returns a dict in the standard detector output format:
        detector_name, score, confidence, explanation, status
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

        best_match = None       # highest similarity seen against a DIFFERENT person_id
        best_similarity = -1.0

        for record in known_faces:
            record_person_id = record.get("person_id")
            record_embedding = record.get("embedding")
            if record_person_id is None or record_embedding is None:
                continue  # skip malformed records rather than crashing the kiosk

            similarity = cosine_similarity(embedding, record_embedding)

            if record_person_id != person_id and similarity > best_similarity:
                best_similarity = similarity
                best_match = record_person_id

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
    """
    Builds the standard detector output dict:
        detector_name  -> short name of this check
        score          -> 0.0 (clean/low risk) to 1.0 (suspicious/high risk)
        confidence      -> "low" | "medium" | "high"
        explanation     -> short human sentence
        status          -> "passed" | "flagged" | "failed" | "unavailable"
    """
    return {
        "detector_name": "duplicate_identity_check",
        "score": score,
        "confidence": confidence,
        "explanation": explanation,
        "status": status,
    }


def _person_id_for(image_path: str) -> str:
    """
    Placeholder person_id source.

    kavach_engine.analyze_media(image_path) doesn't currently pass a
    real enrollment ID into the pipeline, so this uses the image's
    filename (without extension) as a stand-in person_id. Swap this
    out for a real ID as soon as one is available in analyze_media's
    call signature - filenames are NOT a real identity field.
    """
    return os.path.splitext(os.path.basename(image_path))[0]


# ---------------------------------------------------------------------
# Pipeline entry point - this is what kavach_engine.py calls
# ---------------------------------------------------------------------
def run_duplicate_id_detector(image_path: str) -> dict:
    """
    Standard detector interface: takes an image path, returns the
    standard detector_signals dict (detector_name, score, confidence,
    explanation, status). Matches every other detector in detectors/
    so it can be dropped straight into kavach_engine.py's
    detector_pipeline list.
    """
    try:
        import cv2
    except ImportError:
        return _build_result(
            score=0.0,
            confidence="low",
            explanation="opencv-python is not installed; duplicate identity check unavailable.",
            status="unavailable",
        )

    try:
        from insightface.app import FaceAnalysis  # noqa: F401 (import-check only)
    except ImportError:
        return _build_result(
            score=0.0,
            confidence="low",
            explanation="insightface is not installed; duplicate identity check unavailable.",
            status="unavailable",
        )

    try:
        img = cv2.imread(image_path)
        if img is None:
            return _build_result(
                score=0.0,
                confidence="low",
                explanation="Could not read image file.",
                status="failed",
            )

        app = _get_face_app()
        faces = app.get(img)

        if not faces:
            return _build_result(
                score=0.0,
                confidence="low",
                explanation="No face detected in image; duplicate identity check unavailable.",
                status="unavailable",
            )

        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = largest_face.normed_embedding
        person_id = _person_id_for(image_path)

        result = check_duplicate_identity(embedding, person_id=person_id)

        # Only add clean checks to the trusted known-faces store.
        # Flagged/failed/unavailable results are NOT auto-registered.
        if result["status"] == "passed":
            save_new_face_vector(person_id, embedding)

        return result

    except Exception as e:
        return _build_result(
            score=0.0,
            confidence="low",
            explanation=f"Duplicate identity check crashed during execution: {e}",
            status="failed",
        )


_FACE_APP = None


def _get_face_app():
    """Lazily loads and caches the InsightFace model (loaded once per process)."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_l")
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _FACE_APP


# ---------------------------------------------------------------------
# Manual test / demo when run directly: python detectors/duplicate_id_detector.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_store = os.path.join("data", "known_faces_demo.json")

    rng = np.random.default_rng(42)
    original_embedding = rng.normal(size=EMBEDDING_DIM)
    save_known_faces(
        [{"person_id": "ID001", "embedding": original_embedding.tolist()}],
        demo_store,
    )

    print("Test 1: Same face, different ID -> should FLAG")
    duplicate_attempt = original_embedding + rng.normal(scale=0.01, size=EMBEDDING_DIM)
    result_1 = check_duplicate_identity(duplicate_attempt, person_id="ID002", store_path=demo_store)
    print(json.dumps(result_1, indent=2))

    print("\nTest 2: Different face, new ID -> should PASS")
    new_face = rng.normal(size=EMBEDDING_DIM)
    result_2 = check_duplicate_identity(new_face, person_id="ID003", store_path=demo_store)
    print(json.dumps(result_2, indent=2))

    print("\nTest 3: save_new_face_vector helper")
    save_new_face_vector("ID003", new_face, store_path=demo_store)
    print(f"known_faces now has {len(load_known_faces(demo_store))} records in {demo_store}")