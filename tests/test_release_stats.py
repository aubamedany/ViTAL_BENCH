import importlib.util
from pathlib import Path


module_path = Path(__file__).parents[1] / "scripts" / "check_release_stats.py"
spec = importlib.util.spec_from_file_location("check_release_stats", module_path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_release_statistics_match_canonical_files():
    stats = module.load_stats(Path(__file__).parents[1])

    assert stats["questions"] == 4536
    assert stats["detection_questions"] == 2122
    assert stats["law_questions"] == 2414
    assert stats["images"] == 551
    assert stats["image_categories"] == 11
    assert stats["law_corpus_chunks"] == 990
    assert stats["law_categories"] == {
        "Violation identification": 482,
        "Regulation Application": 1601,
        "Multi-step legal Reasoning": 331,
    }
