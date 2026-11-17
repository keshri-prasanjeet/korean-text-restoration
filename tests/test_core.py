import random

from ktr.corrupt import corrupt
from ktr.metrics import boundaries, project_spacing, score
from ktr.ngram import NgramSpacer
from ktr.text import is_good_sentence, normalize

SENT = "나는 오늘 학교에 갔다."


def test_boundaries_roundtrip():
    chars, bounds = boundaries(SENT)
    assert chars == "나는오늘학교에갔다."
    assert bounds == {1, 3, 6}


def test_corruptions_keep_characters():
    rng = random.Random(0)
    for cond in ("nospace", "mixed", "clean"):
        out = corrupt(SENT, cond, rng)
        assert boundaries(out)[0] == boundaries(SENT)[0]
    assert corrupt(SENT, "nospace", rng) == "나는오늘학교에갔다."
    assert corrupt(SENT, "clean", rng) == SENT
    assert corrupt(SENT, "mixed", rng) != SENT


def test_perfect_and_empty_scores():
    perfect = score([(SENT, SENT)])
    assert perfect["boundary_f1"] == 1.0
    assert perfect["sentence_accuracy"] == 1.0
    nospace = score([("나는오늘학교에갔다.", SENT)])
    assert nospace["boundary_recall"] == 0.0
    assert nospace["sentence_accuracy"] == 0.0


def test_partial_score():
    s = score([("나는 오늘학교에 갔다.", SENT)])
    assert s["boundary_precision"] == 1.0
    assert abs(s["boundary_recall"] - 2 / 3) < 1e-9
    # gold words: 나는/오늘/학교에/갔다. ; predicted: 나는/오늘학교에/갔다. -> 2 shared
    assert abs(s["word_f1"] - (2 * (2 / 3) * (2 / 4)) / ((2 / 3) + (2 / 4))) < 1e-9


def test_content_change_counts_as_miss():
    s = score([("나는 오늘 학교로 갔다.", SENT)])
    assert s["content_changed_rate"] == 1.0
    assert s["boundary_recall"] == 0.0


def test_project_spacing_repairs_small_edits():
    # prediction dropped "에" but spaced correctly elsewhere
    repaired = project_spacing("나는오늘학교에갔다.", "나는 오늘 학교 갔다.")
    assert boundaries(repaired)[0] == "나는오늘학교에갔다."
    assert repaired.startswith("나는 오늘 ")


def test_ngram_learns_simple_pattern():
    train = ["나는 학교에 갔다.", "너는 학교에 왔다.", "그는 학교에 갔다."] * 5
    model = NgramSpacer(min_count=2).fit(train)
    assert model.predict("나는학교에갔다.") == "나는 학교에 갔다."


def test_sentence_filter():
    assert is_good_sentence("서울은 대한민국의 수도이며 가장 큰 도시이다.")
    assert not is_good_sentence("배리 페퍼 등이 주연으로 출연하였고 로버트 F.")
    assert not is_good_sentence("짧은 문장.")
    assert normalize("가  나\t다") == "가 나 다"
