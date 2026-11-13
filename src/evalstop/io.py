"""Read per-item results in evaluation order.

Two formats are understood, both with the fields ``item``, ``model`` and
``score``:

* JSON Lines: one object per line, ``{"item": "q1", "model": "m", "score": 1}``;
* CSV with a header row: ``item,model,score``.

Only ``score`` is required. A missing ``item`` becomes the row number and a
missing ``model`` becomes the empty string. A score is a number, a JSON boolean
or one of the words ``true``, ``false``, ``pass``, ``fail``.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

_WORDS = {"true": 1.0, "pass": 1.0, "false": 0.0, "fail": 0.0}
_FORMAT_BY_SUFFIX = {".jsonl": "jsonl", ".ndjson": "jsonl", ".csv": "csv"}
FORMATS = ("jsonl", "csv")


class InputError(ValueError):
    """A results file that cannot be read as per-item results."""


@dataclass(frozen=True)
class Record:
    """The score one model got on one item."""

    item: str
    model: str
    score: float


def _score(value: object, where: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            raise InputError(f"{where}: score is too large") from None
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _WORDS:
            return _WORDS[text]
        try:
            number = float(text)
        except ValueError:
            raise InputError(f"{where}: score {value!r} is not a number") from None
    else:
        raise InputError(f"{where}: score {value!r} is not a number")
    if not math.isfinite(number):
        raise InputError(f"{where}: score {value!r} is not finite")
    return number


def _record(fields: dict[str, object], row: int, where: str) -> Record:
    if fields.get("score") is None:
        raise InputError(f"{where}: no score field")
    item = fields.get("item")
    model = fields.get("model")
    return Record(
        item=str(row) if item is None or item == "" else str(item),
        model="" if model is None else str(model),
        score=_score(fields["score"], where),
    )


def _iter_jsonl(lines: Iterable[str], first_line: int) -> Iterator[Record]:
    row = 0
    for number, line in enumerate(lines, start=first_line):
        if not line.strip():
            continue
        where = f"line {number}"
        try:
            fields = json.loads(line)
        except json.JSONDecodeError as err:
            raise InputError(f"{where}: not valid JSON ({err.msg})") from None
        if not isinstance(fields, dict):
            raise InputError(f"{where}: expected a JSON object")
        row += 1
        yield _record(fields, row, where)


def _iter_csv(lines: Iterable[str], first_line: int) -> Iterator[Record]:
    reader = csv.reader(lines)
    header = next(reader, None)
    if header is None:
        return
    names = [name.strip().lower() for name in header]
    if "score" not in names:
        raise InputError(f"line {first_line}: the CSV header has no score column")
    row = 0
    for cells in reader:
        if not any(cell.strip() for cell in cells):
            continue
        where = f"line {first_line + reader.line_num - 1}"
        if len(cells) != len(names):
            raise InputError(f"{where}: expected {len(names)} columns, found {len(cells)}")
        row += 1
        fields: dict[str, object] = {
            name: cell.strip() for name, cell in zip(names, cells, strict=True)
        }
        if fields["score"] == "":
            raise InputError(f"{where}: no score field")
        yield _record(fields, row, where)


def iter_records(lines: Iterable[str], fmt: str | None = None) -> Iterator[Record]:
    """Parse lines of JSON Lines or CSV lazily, one record at a time.

    With ``fmt=None`` the format is taken from the first non-blank line: a line
    starting with ``{`` means JSON Lines, anything else CSV.
    """
    if fmt is not None and fmt not in FORMATS:
        raise InputError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    source = iter(lines)
    first_line = 1
    first = None
    for line in source:
        if line.strip():
            first = line
            break
        first_line += 1
    if first is None:
        return
    first = first.lstrip("﻿")
    if fmt is None:
        fmt = "jsonl" if first.lstrip().startswith("{") else "csv"
    rest = itertools.chain([first], source)
    if fmt == "jsonl":
        yield from _iter_jsonl(rest, first_line)
    else:
        yield from _iter_csv(rest, first_line)


def read_records(source: str | Path | TextIO, fmt: str | None = None) -> list[Record]:
    """Read all records from a path, from standard input (``"-"``) or from an open text file.

    For a path the format is taken from the suffix (``.jsonl``, ``.ndjson``,
    ``.csv``) unless ``fmt`` is given; otherwise from the content.
    """
    if isinstance(source, (str, Path)):
        if str(source) == "-":
            return list(iter_records(sys.stdin, fmt))
        path = Path(source)
        if fmt is None:
            fmt = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
        with path.open(encoding="utf-8", newline="") as handle:
            return list(iter_records(handle, fmt))
    return list(iter_records(source, fmt))


def model_names(records: Iterable[Record]) -> list[str]:
    """Distinct model names in order of first appearance."""
    return list(dict.fromkeys(record.model for record in records))


def scores_for(records: Iterable[Record], model: str | None = None) -> tuple[str, list[float]]:
    """Return the chosen model's name and its scores in file order.

    With ``model=None`` the records must all belong to one model.
    """
    records = list(records)
    names = model_names(records)
    if model is None:
        if len(names) > 1:
            listed = ", ".join(repr(name) for name in names)
            raise InputError(f"the results hold several models ({listed}); choose one")
        model = names[0] if names else ""
    elif model not in names:
        listed = ", ".join(repr(name) for name in names) or "none"
        raise InputError(f"no results for model {model!r}; found {listed}")
    return model, [record.score for record in records if record.model == model]
