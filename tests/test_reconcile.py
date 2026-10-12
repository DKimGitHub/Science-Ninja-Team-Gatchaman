from pipeline.reconcile import vote, snap_names

G = ["Zoltar", "Keyop", "Mark", "7-Zark-7"]


def test_majority_wins():
    t = {"whisper": "go G-Force now".split(),
         "parakeet": "go G-Force now".split(),
         "canary": "go G-Force wow".split()}
    words, flags = vote(t, G)
    assert " ".join(words) == "go G-Force now" and flags == []


def test_name_snap():
    assert snap_names("Zoltaar", G) == "Zoltar"
    assert snap_names("Keyup", G) == "Keyop"
    assert snap_names("mask", G) == "mask"
    assert snap_names("the", G) == "the"


def test_three_way_disagreement_is_flagged():
    t = {"whisper": "we fly at dawn".split(),
         "parakeet": "we fry at down".split(),
         "canary": "we pry at dun".split()}
    words, flags = vote(t, G)
    assert len(flags) >= 2


def test_hallucinated_insert_flagged():
    t = {"whisper": "hold on thanks for watching".split(),
         "parakeet": "hold on".split(),
         "canary": "hold on".split()}
    words, flags = vote(t, G)
    assert any("only whisper" in f[1] or "dropped" in f[1] for f in flags)
    assert " ".join(words).startswith("hold on")


def test_punctuation_tokens_are_not_flagged():
    t = {"whisper": ["hold", "it,", "Jason"],
         "parakeet": ["hold", "it", "Jason"],
         "canary": ["hold", "it", ",", "Jason", "?"]}
    words, flags = vote(t, G)
    assert flags == [] and len(words) == 3
