"""Load and prepare HotpotQA data."""

import json
import hashlib
import random
from pathlib import Path
from typing import Any

from config import NUM_QUESTIONS, SEED, NUM_DEV_QUESTIONS, DEV_SEED_TAG, stable_seed

HOTPOTQA_CACHE = Path(__file__).resolve().parents[1] / "hotpotqa_dev.json"


def download_hotpotqa() -> list[dict[str, Any]]:
    if HOTPOTQA_CACHE.exists():
        with open(HOTPOTQA_CACHE) as f:
            return json.load(f)
    print("Downloading HotpotQA dev set (distractor)...")
    import urllib.request
    url = (
        "http://curtis.ml.cmu.edu/datasets/hotpot/"
        "hotpot_dev_distractor_v1.json"
    )
    urllib.request.urlretrieve(url, HOTPOTQA_CACHE)
    with open(HOTPOTQA_CACHE) as f:
        return json.load(f)


def build_knowledge_base(item: dict) -> dict[str, str]:
    kb = {}
    for title, sentences in item["context"]:
        kb[title] = " ".join(sentences)
    return kb


def _prepare(sampled: list[dict]) -> list[dict[str, Any]]:
    questions = []
    for item in sampled:
        supporting_titles = set()
        for title, _ in item.get("supporting_facts", []):
            supporting_titles.add(title)
        distractor_titles = [
            title for title, _ in item["context"]
            if title not in supporting_titles
        ]
        questions.append({
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown"),
            "knowledge_base": build_knowledge_base(item),
            "supporting_titles": sorted(supporting_titles),
            "distractor_titles": distractor_titles,
        })
    return questions


def sample_questions(n: int = NUM_QUESTIONS) -> list[dict[str, Any]]:
    """Historical sample; n=300 reproduces the earlier version. Not a holdout."""
    raw = download_hotpotqa()
    rng = random.Random(SEED)
    sampled = rng.sample(raw, min(n, len(raw)))
    return _prepare(sampled)


def sample_dev_questions(n: int = NUM_DEV_QUESTIONS) -> list[dict[str, Any]]:
    """A prefix of the immutable dev100 manifest, disjoint from historical300."""
    if not 1 <= n <= NUM_DEV_QUESTIONS:
        raise ValueError(f"dev size must be 1..{NUM_DEV_QUESTIONS}; subsets use fixed manifest prefixes")
    raw = download_hotpotqa()
    main_ids = {item["_id"] for item in random.Random(SEED).sample(raw, min(NUM_QUESTIONS, len(raw)))}
    remaining = [item for item in raw if item["_id"] not in main_ids]
    rng = random.Random(stable_seed(DEV_SEED_TAG, NUM_DEV_QUESTIONS))
    sampled = rng.sample(remaining, NUM_DEV_QUESTIONS)
    manifest = {
        "version": "dev100-v2.1", "seed_tag": DEV_SEED_TAG,
        "dataset_sha256": hashlib.sha256(HOTPOTQA_CACHE.read_bytes()).hexdigest(),
        "question_ids": [x["_id"] for x in sampled],
        "development_ids": [x["_id"] for x in sampled[:20]],
        "reserved_audit_ids": [x["_id"] for x in sampled[20:]],
        "historical_ids": [x["_id"] for x in random.Random(SEED).sample(raw, NUM_QUESTIONS)],
    }
    path = HOTPOTQA_CACHE.parent / "manifests" / "dev100_v2.1.json"
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("x") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
    except FileExistsError:
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Dev manifest or dataset changed; create a new explicit version")
    return _prepare(sampled[:n])


def load_run_data(question_set="main", n=None, shuffle_mode="per_question"):
    """Shared CLI data path: subsets never change the v2 noise pool."""
    if question_set not in {"main", "dev", "test", "fresh"}:
        raise ValueError(question_set)
    full = (sample_dev_questions() if question_set == "dev" else
            sample_test_questions() if question_set == "test" else
            sample_fresh_questions() if question_set == "fresh" else sample_questions())
    n = len(full) if n is None else n
    if not 1 <= n <= len(full):
        raise ValueError(f"{question_set} size must be 1..{len(full)}")
    questions = full[:n]
    pool = build_shuffle_pool(questions) if shuffle_mode == "legacy" else build_shuffle_pool_v2(full)
    metadata = {
        "question_set": question_set, "question_ids": [q["id"] for q in questions],
        "pool_question_ids": [q["id"] for q in (questions if shuffle_mode == "legacy" else full)],
        "pool_sha256": hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
        "dataset_sha256": hashlib.sha256(HOTPOTQA_CACHE.read_bytes()).hexdigest(),
        "split_role": {"dev": "rubric_dev_and_reserved_audit", "test": "formal_test300", "fresh": "confirmatory_fresh300"}.get(question_set, "historical_replication"),
    }
    return questions, pool, metadata


def build_shuffle_pool(questions: list[dict]) -> list[str]:
    """Build a pool of paragraphs from ALL questions for shuffled condition.

    Returns a flat list of (title, paragraph) strings from all KBs,
    so shuffled observations look realistic but are from wrong questions.
    """
    pool = []
    for q in questions:
        for title, para in q["knowledge_base"].items():
            pool.append(f"[{title}] {para}")

    rng = random.Random(SEED + 1)
    rng.shuffle(pool)
    return pool


def build_shuffle_pool_v2(questions: list[dict]) -> list[dict[str, str]]:
    """Pool with provenance, for per-question shuffled sampling (v2).

    Each entry records the source question id so an environment can exclude
    the current question's own pages and log where each misleading paragraph
    came from. Order is deterministic (question order, then title order).
    """
    pool = []
    for q in questions:
        for title, para in q["knowledge_base"].items():
            pool.append({"text": f"[{title}] {para}", "source_qid": q["id"], "title": title})
    return pool

NUM_TEST_QUESTIONS = 300
TEST_SEED_TAG = "test-v1"


def sample_test_questions(n: int = NUM_TEST_QUESTIONS) -> list[dict[str, Any]]:
    """Formal held-out test set: disjoint from historical300 and dev100; used only for
    pre-registered Gate 2 evaluation. A prefix of the immutable test300 manifest."""
    if not 1 <= n <= NUM_TEST_QUESTIONS:
        raise ValueError(f"test size must be 1..{NUM_TEST_QUESTIONS}")
    raw = download_hotpotqa()
    main_ids = {item["_id"] for item in random.Random(SEED).sample(raw, min(NUM_QUESTIONS, len(raw)))}
    remaining = [item for item in raw if item["_id"] not in main_ids]
    dev_ids = {x["_id"] for x in random.Random(stable_seed(DEV_SEED_TAG, NUM_DEV_QUESTIONS)).sample(remaining, NUM_DEV_QUESTIONS)}
    pool = [item for item in remaining if item["_id"] not in dev_ids]
    sampled = random.Random(stable_seed(TEST_SEED_TAG, NUM_TEST_QUESTIONS)).sample(pool, NUM_TEST_QUESTIONS)
    manifest = {
        "version": "test300-v1", "seed_tag": TEST_SEED_TAG,
        "dataset_sha256": hashlib.sha256(HOTPOTQA_CACHE.read_bytes()).hexdigest(),
        "question_ids": [x["_id"] for x in sampled],
        "excluded": {"historical300": len(main_ids), "dev100": len(dev_ids)},
    }
    path = HOTPOTQA_CACHE.parent / "manifests" / "test300_v1.json"
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("x") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
    except FileExistsError:
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Test manifest or dataset changed; create a new explicit version")
    return _prepare(sampled[:n])


FRESH_SEED_TAG = "fresh-v1"


def sample_fresh_questions(n: int = NUM_TEST_QUESTIONS) -> list[dict[str, Any]]:
    """Confirmatory fresh set (amendment 16): disjoint from historical300, dev100 and test300; drawn
    after every earlier analysis was fixed, for the pre-registered confirmatory replication only."""
    if not 1 <= n <= NUM_TEST_QUESTIONS:
        raise ValueError(f"fresh size must be 1..{NUM_TEST_QUESTIONS}")
    raw = download_hotpotqa()
    used = {item["_id"] for item in random.Random(SEED).sample(raw, min(NUM_QUESTIONS, len(raw)))}
    remaining = [item for item in raw if item["_id"] not in used]
    used |= {x["_id"] for x in random.Random(stable_seed(DEV_SEED_TAG, NUM_DEV_QUESTIONS)).sample(remaining, NUM_DEV_QUESTIONS)}
    used |= {q["id"] for q in sample_test_questions()}
    pool = [item for item in raw if item["_id"] not in used]
    sampled = random.Random(stable_seed(FRESH_SEED_TAG, NUM_TEST_QUESTIONS)).sample(pool, NUM_TEST_QUESTIONS)
    manifest = {"version": "fresh300-v1", "seed_tag": FRESH_SEED_TAG,
                "dataset_sha256": hashlib.sha256(HOTPOTQA_CACHE.read_bytes()).hexdigest(),
                "question_ids": [x["_id"] for x in sampled], "excluded": len(used)}
    path = HOTPOTQA_CACHE.parent / "manifests" / "fresh300_v1.json"
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("x") as f:
            json.dump(manifest, f, indent=2); f.write("\n")
    except FileExistsError:
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Fresh manifest or dataset changed; create a new explicit version")
    return _prepare(sampled[:n])
