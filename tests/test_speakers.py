from pipeline.speakers import group_lines, smooth, time_words


def test_smooth_removes_single_word_flicker():
    assert smooth(["A", "A", "B", "A", "A"]) == ["A"] * 5
    assert smooth(["A", "A", "B", "B", "B"]) == ["A", "A", "B", "B", "B"]


def test_group_lines_splits_on_speaker_change():
    out = group_lines(["hi", "there", "go"], ["A", "A", "B"])
    assert out == [("A", ["hi", "there"]), ("B", ["go"])]


def test_time_words_aligns_and_interpolates():
    ref = [("hold", 1.0, 1.2), ("it", 1.3, 1.5), ("jason", 2.0, 2.4)]
    t = time_words(["Hold", "it,", "now", "Jason"], ref, 0, 5)
    assert abs(t[0] - 1.1) < 1e-6 and abs(t[3] - 2.2) < 1e-6
    assert t[1] < t[2] < t[3]


def test_snap_moves_turn_change_to_sentence_end():
    from pipeline.speakers import snap_to_sentences
    w = ["Go", "now.", "Look,", "Jason", "wait."]
    assert snap_to_sentences(w, ["A", "A", "A", "B", "B"]) == ["A", "A", "B", "B", "B"]
    assert snap_to_sentences(w, ["A", "A", "B", "B", "B"]) == ["A", "A", "B", "B", "B"]
