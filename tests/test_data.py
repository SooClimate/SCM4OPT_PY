"""Scenario input data (scm4opt/io/datasets.py): the files match their checksum manifest,
and a pack -> fetch round trip restores a set unchanged."""

import shutil

import pytest

from scm4opt.core.config import DAT_DIR
from scm4opt.io.datasets import MANIFEST, fetch, pack, read_manifest, verify, write_manifest


def test_inputs_match_manifest():
    if not (DAT_DIR / MANIFEST).exists():
        pytest.skip(f"no inputs manifest in {DAT_DIR}")
    res = verify(DAT_DIR, quick=True)
    present = res["checked"] - len(res["missing"])
    if present == 0:
        pytest.skip("scenario inputs not present")
    assert res["changed"] == []


def test_pack_fetch_round_trip(tmp_path):
    src = DAT_DIR / "NOAA"
    if not src.is_dir():
        pytest.skip("NOAA inputs not present")
    data = tmp_path / "data"
    shutil.copytree(src, data / "NOAA")
    write_manifest(data, ["NOAA"])
    pack(data, tmp_path / "host", ["NOAA"])
    res = fetch(tmp_path / "host", tmp_path / "restored", log=lambda m: None)
    assert res["missing"] == res["changed"] == [] and res["checked"] == len(read_manifest(data))
