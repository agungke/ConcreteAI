import csv
import io
from pathlib import Path

import pytest

from src.prepare_concrete_dataset import COLUMNS, ORIGINAL, prepare


ROW = ["300", "0", "0", "180", "2", "1000", "700", "28", "35"]


def make_raw(rows, header=ORIGINAL):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def test_prepare_preserves_values_and_renames_columns(tmp_path):
    rows = [ROW] * 1030  # artificial fixture: validation only, never used for training
    output = tmp_path / "concrete.csv"
    prepare(make_raw(rows), output)
    with output.open(newline="", encoding="utf-8") as f:
        result = list(csv.reader(f))
    assert result[0] == COLUMNS
    assert result[1:] == rows


def test_prepare_rejects_wrong_header():
    with pytest.raises(ValueError, match="Unexpected source columns"):
        prepare(make_raw([ROW] * 1030, ["wrong"]), "unused.csv")


def test_prepare_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="Expected 1030 rows"):
        prepare(make_raw([ROW]), "unused.csv")


def test_prepare_rejects_non_finite_values():
    bad = ROW.copy()
    bad[0] = "NaN"
    with pytest.raises(ValueError, match="non-finite"):
        prepare(make_raw([bad] + [ROW] * 1029), "unused.csv")


def test_prepare_rejects_invalid_age():
    bad = ROW.copy()
    bad[7] = "0"
    with pytest.raises(ValueError, match="Out-of-range"):
        prepare(make_raw([bad] + [ROW] * 1029), "unused.csv")
