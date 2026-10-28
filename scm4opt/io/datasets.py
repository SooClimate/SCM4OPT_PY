# SPDX-License-Identifier: GPL-3.0-or-later
"""Scenario input data: checksum manifest, verification, archives for hosting, download.

The scenario inputs (emissions, concentrations, land cover, observations, forcing data)
live in sets, the top-level folders of the data folder (RCMIP3/, AGAGE/, NOAA/,
contrib/). They are described by

  <data>/inputs_manifest.csv   set, path (relative to <data>), bytes, sha256 of every file

and can be distributed as one archive per set:

  pack   <data> -> <out>/scm4opt-inputs-<set>.tar.gz + <out>/archives_manifest.csv
                   (archive, set, bytes, sha256, files)
  fetch  <source>/archives_manifest.csv + archives (a URL or a folder) -> <data>:
         each archive is checked against its sha256 before it is unpacked, and the
         unpacked files against inputs_manifest.csv
  verify <data> against inputs_manifest.csv (missing, changed and unlisted files)

Usage:
  python3 -m scm4opt.io.datasets verify [--data DIR] [--set RCMIP3 ...]
  python3 -m scm4opt.io.datasets manifest [--data DIR] [--set ...]    # (re)write the manifest
  python3 -m scm4opt.io.datasets pack --out DIR [--data DIR] [--set ...]
  python3 -m scm4opt.io.datasets fetch --source URL_OR_DIR [--data DIR] [--set ...]
--data defaults to the model's data folder (config.DAT_DIR: $SCM4OPT_DATA, ...). Exit code 1
if verification fails. fetch takes all sets by default; the sets are not independent:
--scen=rcmip3 runs read RCMIP3/ and the land cover in contrib/LC_*, --load_AGAGE /
--load_NOAA the observation sets.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

MANIFEST = "inputs_manifest.csv"
ARCHIVES = "archives_manifest.csv"
ARCHIVE_NAME = "scm4opt-inputs-{set}.tar.gz"
SETS = ("RCMIP3", "AGAGE", "NOAA", "contrib")      # scenario input sets (top-level folders)
_CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _files(data_dir, sets):
    data_dir = Path(data_dir)
    for s in sets:
        root = data_dir / s
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    yield s, p


def build_manifest(data_dir, sets=SETS):
    """The manifest of the files of `sets` under data_dir, as a DataFrame."""
    rows = [dict(set=s, path=p.relative_to(data_dir).as_posix(), bytes=p.stat().st_size,
                 sha256=sha256(p)) for s, p in _files(data_dir, sets)]
    return pd.DataFrame(rows, columns=["set", "path", "bytes", "sha256"])


def write_manifest(data_dir, sets=SETS):
    m = build_manifest(data_dir, sets)
    m.to_csv(Path(data_dir) / MANIFEST, index=False)
    return m


def read_manifest(data_dir):
    return pd.read_csv(Path(data_dir) / MANIFEST, dtype={"bytes": "int64"})


def verify(data_dir, sets=None, manifest=None, quick=False):
    """Compare data_dir with the manifest. Returns dict(missing, changed, unlisted, checked)
    (lists of paths). quick: compare sizes only."""
    data_dir = Path(data_dir)
    m = read_manifest(data_dir) if manifest is None else manifest
    if sets:
        m = m[m["set"].isin(sets)]
    missing, changed = [], []
    for r in m.itertuples():
        p = data_dir / r.path
        if not p.is_file():
            missing.append(r.path)
        elif p.stat().st_size != r.bytes or (not quick and sha256(p) != r.sha256):
            changed.append(r.path)
    listed = set(m["path"])
    unlisted = [p.relative_to(data_dir).as_posix() for _, p in _files(data_dir, sets or sorted(set(m["set"])))
                if p.relative_to(data_dir).as_posix() not in listed]
    return dict(missing=missing, changed=changed, unlisted=unlisted, checked=len(m))


def pack(data_dir, out_dir, sets=SETS):
    """One tar.gz per set (paths relative to data_dir) and the archives manifest."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = read_manifest(data_dir)
    rows = []
    for s in sets:
        files = m.loc[m["set"] == s, "path"].tolist()
        if not files:
            continue
        arc = out_dir / ARCHIVE_NAME.format(set=s)
        with tarfile.open(arc, "w:gz") as tar:
            for rel in files:
                tar.add(data_dir / rel, arcname=rel, recursive=False)
        rows.append(dict(archive=arc.name, set=s, bytes=arc.stat().st_size, sha256=sha256(arc),
                         files=len(files)))
    a = pd.DataFrame(rows, columns=["archive", "set", "bytes", "sha256", "files"])
    a.to_csv(out_dir / ARCHIVES, index=False)
    shutil.copy2(data_dir / MANIFEST, out_dir / MANIFEST)
    return a


def _get(source, name, dst):
    """Copy <source>/<name> (URL or folder) to dst."""
    if str(source).startswith(("http://", "https://")):
        url = str(source).rstrip("/") + "/" + name
        with urllib.request.urlopen(url) as r, open(dst, "wb") as fh:
            shutil.copyfileobj(r, fh, _CHUNK)
    else:
        shutil.copy2(Path(source) / name, dst)


def _safe_extract(tar, dest):
    """Extract regular files and folders only, inside dest (no links, no absolute or ..)."""
    dest = Path(dest).resolve()
    for mbr in tar.getmembers():
        target = (dest / mbr.name).resolve()
        if not (mbr.isfile() or mbr.isdir()) or dest not in target.parents and target != dest:
            raise ValueError(f"unsafe archive member: {mbr.name}")
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
    else:                                            # Python < 3.12: members checked above
        tar.extractall(dest)


def fetch(source, data_dir, sets=None, log=print):
    """Download (or copy) the archives listed in <source>/archives_manifest.csv, check
    their sha256, unpack them into data_dir and verify the files. Returns verify()'s dict."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _get(source, ARCHIVES, tmp / ARCHIVES)
        _get(source, MANIFEST, tmp / MANIFEST)
        arcs = pd.read_csv(tmp / ARCHIVES)
        if sets:
            arcs = arcs[arcs["set"].isin(sets)]
        for a in arcs.itertuples():
            log(f"  {a.archive}: {a.bytes / 2**20:.0f} MB, {a.files} files")
            _get(source, a.archive, tmp / a.archive)
            if sha256(tmp / a.archive) != a.sha256:
                raise ValueError(f"{a.archive}: checksum mismatch (download damaged or changed)")
            with tarfile.open(tmp / a.archive, "r:gz") as tar:
                _safe_extract(tar, data_dir)
            (tmp / a.archive).unlink()
        m = pd.read_csv(tmp / MANIFEST, dtype={"bytes": "int64"})
        shutil.copy2(tmp / MANIFEST, data_dir / MANIFEST)
    return verify(data_dir, sets=sorted(set(arcs["set"])), manifest=m)


# ---------------------------------------------------------------- command line
def _report(res, label):
    bad = res["missing"] + res["changed"]
    print(f"{label}: {res['checked']} files listed, {len(res['missing'])} missing, "
          f"{len(res['changed'])} changed, {len(res['unlisted'])} not in the manifest")
    for k in ("missing", "changed", "unlisted"):
        for p in res[k][:10]:
            print(f"  {k}: {p}")
        if len(res[k]) > 10:
            print(f"  ... {len(res[k]) - 10} more {k}")
    return 1 if bad else 0


def main(argv=None):
    from ..core.config import DAT_DIR
    ap = argparse.ArgumentParser(prog="datasets", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("verify", "manifest", "pack", "fetch"))
    ap.add_argument("--data", type=Path, default=None, help="data folder (default: the model's)")
    ap.add_argument("--set", action="append", default=None, help=f"input set(s), default {', '.join(SETS)}")
    ap.add_argument("--out", type=Path, default=None, help="pack: output folder")
    ap.add_argument("--source", default=None, help="fetch: URL or folder with archives_manifest.csv")
    ap.add_argument("--quick", action="store_true", help="verify: compare sizes only")
    a = ap.parse_args(argv)
    data = (a.data or DAT_DIR).expanduser()
    sets = a.set or list(SETS)
    if a.command == "manifest":
        m = write_manifest(data, sets)
        print(f"{data / MANIFEST}: {len(m)} files, {m['bytes'].sum() / 2**30:.2f} GB")
        return 0
    if a.command == "verify":
        if not (data / MANIFEST).exists():
            sys.exit(f"no {MANIFEST} in {data}")
        return _report(verify(data, a.set, quick=a.quick), str(data))
    if a.command == "pack":
        if not a.out:
            sys.exit("pack needs --out")
        arcs = pack(data, a.out, sets)
        for r in arcs.itertuples():
            print(f"  {r.archive}: {r.files} files, {r.bytes / 2**20:.0f} MB, sha256 {r.sha256[:16]}...")
        print(f"{a.out / ARCHIVES} + {MANIFEST}: upload the folder's files together")
        return 0
    if not a.source:
        sys.exit("fetch needs --source")
    print(f"fetching {', '.join(a.set or ['all sets'])} from {a.source} into {data}")
    try:
        return _report(fetch(a.source, data, a.set), str(data))
    except (ValueError, OSError) as e:
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
