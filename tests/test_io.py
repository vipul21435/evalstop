"""Reading per-item results from JSON Lines and CSV."""

from __future__ import annotations

import io

import pytest

from evalstop.io import InputError, Record, iter_records, model_names, read_records, scores_for

JSONL = (
    '{"item": "q1", "model": "m", "score": 1}\n'
    '{"item": "q2", "model": "m", "score": 0.5}\n'
    "\n"
    '{"item": 3, "model": "m", "score": false}\n'
)
CSV = "Item, Model ,SCORE\nq1,m,1\nq2,m,0.5\n\n3,m,false\n"
EXPECTED = [Record("q1", "m", 1.0), Record("q2", "m", 0.5), Record("3", "m", 0.0)]


def test_jsonl_records_keep_file_order():
    assert list(iter_records(JSONL.splitlines())) == EXPECTED


def test_csv_header_is_case_and_space_insensitive():
    assert list(iter_records(CSV.splitlines())) == EXPECTED


def test_csv_quoted_cells_and_extra_columns():
    text = 'item,model,score,note\n"q,1","m, large",1,"said ""ok"""\nq2,"m, large",0,\n'
    records = list(iter_records(io.StringIO(text)))
    assert records == [Record("q,1", "m, large", 1.0), Record("q2", "m, large", 0.0)]


def test_missing_item_and_model_get_defaults():
    records = list(iter_records(['{"score": 1}', '{"score": 0, "item": ""}']))
    assert records == [Record("1", "", 1.0), Record("2", "", 0.0)]
    assert list(iter_records(["score", "0.25", "0.75"])) == [
        Record("1", "", 0.25),
        Record("2", "", 0.75),
    ]


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("true", 1.0),
        ("True", 1.0),
        ("false", 0.0),
        (" PASS ", 1.0),
        ("fail", 0.0),
        ("0.125", 0.125),
        ("1e-1", 0.1),
    ],
)
def test_score_words_and_numbers(raw, value):
    assert list(iter_records(["score", raw]))[0].score == value
    assert list(iter_records([f'{{"score": "{raw}"}}']))[0].score == value


def test_json_booleans_and_numbers():
    records = list(iter_records(['{"score": true}', '{"score": 2}', '{"score": -0.5}']))
    assert [record.score for record in records] == [1.0, 2.0, -0.5]


def test_format_comes_from_the_suffix_then_the_content(tmp_path):
    for name, text in [("a.jsonl", JSONL), ("a.ndjson", JSONL), ("a.csv", CSV), ("a.txt", JSONL)]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert read_records(path) == EXPECTED
        assert read_records(str(path)) == EXPECTED
    other = tmp_path / "b.dat"
    other.write_text(CSV, encoding="utf-8")
    assert read_records(other) == EXPECTED


def test_explicit_format_overrides_the_suffix(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(JSONL, encoding="utf-8")
    assert read_records(path, "jsonl") == EXPECTED
    with pytest.raises(InputError, match="score column"):
        read_records(path)


def test_dash_reads_standard_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(CSV))
    assert read_records("-") == EXPECTED


def test_open_text_files_are_accepted():
    assert read_records(io.StringIO(JSONL)) == EXPECTED


def test_leading_blank_lines_and_a_byte_order_mark_are_skipped():
    assert list(iter_records(["", "  ", "﻿" + CSV.splitlines()[0], "q1,m,1"])) == EXPECTED[:1]
    assert list(iter_records(["﻿" + '{"score": 1}'])) == [Record("1", "", 1.0)]


def test_windows_line_endings(tmp_path):
    path = tmp_path / "results.csv"
    path.write_bytes(CSV.replace("\n", "\r\n").encode("utf-8"))
    assert read_records(path) == EXPECTED
    path = tmp_path / "results.jsonl"
    path.write_bytes(JSONL.replace("\n", "\r\n").encode("utf-8"))
    assert read_records(path) == EXPECTED


def test_empty_input_gives_no_records():
    assert list(iter_records([])) == []
    assert list(iter_records(["", "\n"])) == []
    assert list(iter_records(["item,model,score"])) == []


def test_records_are_parsed_lazily():
    def lines():
        yield '{"score": 1}'
        yield '{"score": 0}'
        raise AssertionError("read past the second line")

    stream = iter_records(lines())
    assert next(stream).score == 1.0
    assert next(stream).score == 0.0


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"score": 1}', "{broken"], "line 2: not valid JSON"),
        (['{"score": 1}', "[1, 2]"], "line 2: expected a JSON object"),
        (['{"item": "q1"}'], "line 1: no score field"),
        (['{"score": null}'], "line 1: no score field"),
        (['{"score": "high"}'], "line 1: score 'high' is not a number"),
        (['{"score": [1]}'], r"line 1: score \[1\] is not a number"),
        (['{"score": NaN}'], "line 1: score nan is not finite"),
        (['{"score": 1e999}'], "line 1: score inf is not finite"),
        (["", '{"score": 1}', '{"score": "x"}'], "line 3: score 'x'"),
        (["item,model", "q1,m"], "line 1: the CSV header has no score column"),
        (["item,score", "q1,1", "q2,1,extra"], "line 3: expected 2 columns, found 3"),
        (["item,score", "q1,"], "line 2: no score field"),
        (["item,score", "q1,1", "", "q2,abc"], "line 4: score 'abc' is not a number"),
        (["", "item,score", "q1,inf"], "line 3: score 'inf' is not finite"),
    ],
)
def test_bad_input_is_reported_with_its_line(lines, message):
    with pytest.raises(InputError, match=message):
        list(iter_records(lines))


def test_unknown_format_is_rejected():
    with pytest.raises(InputError, match="unknown format"):
        list(iter_records(["score", "1"], "xml"))


def test_model_names_keep_first_appearance_order():
    records = [Record("1", "b", 1.0), Record("1", "a", 0.0), Record("2", "b", 0.0)]
    assert model_names(records) == ["b", "a"]


def test_scores_for_a_single_or_a_chosen_model():
    records = [Record("1", "b", 1.0), Record("1", "a", 0.0), Record("2", "b", 0.25)]
    assert scores_for(records, "b") == ("b", [1.0, 0.25])
    assert scores_for(records[:1]) == ("b", [1.0])
    assert scores_for(iter(records), "a") == ("a", [0.0])
    assert scores_for([]) == ("", [])


def test_scores_for_needs_a_choice_between_several_models():
    records = [Record("1", "b", 1.0), Record("1", "a", 0.0)]
    with pytest.raises(InputError, match=r"several models \('b', 'a'\)"):
        scores_for(records)
    with pytest.raises(InputError, match="no results for model 'c'; found 'b', 'a'"):
        scores_for(records, "c")
    with pytest.raises(InputError, match="no results for model 'c'; found none"):
        scores_for([], "c")
