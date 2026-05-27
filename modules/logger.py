# Copyright (C) 2026 Lorenzo Orsini
# Author(s):
#   - Lorenzo Orsini, lorenzo.orsini3@gmail.com
#   - Alex Carnoli, alex.carnoli@gmail.com
# This file is part of CpG Island Predictor (CIP).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Logger setup for CIP with log rotation and background tar.gz archiving.

Crash-safety model
------------------
tar.gz does not support native append, so we maintain a staging **.tar**
(uncompressed) as the "live" archive.  Each append is done atomically:

  1. Build the new tar content into a temp file  ``*.tar.tmp``.
  2. ``os.replace()`` (atomic on POSIX and NTFS) swaps it over the staging
     ``.tar``.  At every point in time the staging file is either the old
     valid tar or the new valid tar — never a partial write.
  3. Only after the atomic swap are the source ``.log`` files deleted.

On the next startup ``_recover_tmp()`` cleans up any leftover ``*.tar.tmp``
files (process was killed mid-write).  A leftover tmp is either:
  - valid tar  → rename it to the staging ``.tar`` (recover)
  - empty/corrupt → delete it silently

When the staging ``.tar`` is promoted (size threshold or clean exit) it is
gzip-compressed to ``.tar.gz``.  The old frozen ``.tar.gz`` is deleted first.

What cannot be saved: log entries that were queued in RAM (``_pending_files``)
but not yet written to disk at the moment of a hard kill.  This is the
accepted trade-off for a non-journalled system.

Public API
----------
log               : logging.Logger  – the CIP logger
_SCRIPT_DIR       : Path            – project root
wait_for_archiver()                 – call before exit to flush pending work
"""

import gzip, logging, os, re, shutil, sys, tarfile, threading
from datetime import datetime
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_LOG_FILES  = 10
_MAX_ARCHIVE_MB = 100
_MAX_ARCHIVE_B  = _MAX_ARCHIVE_MB * 1024 * 1024
_ARCHIVE_PREFIX = "cip_persist_"
_FROZEN_RE      = re.compile(r"^cip_persist_(\d{8})-(\d{8})\.tar\.gz$")
_STAGING_RE     = re.compile(r"^cip_persist_(\d{8})-(\d{8})\.tar$")
_TMP_RE         = re.compile(r"^cip_persist_(\d{8})-(\d{8})\.tar\.tmp$")

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_log_dir    = _SCRIPT_DIR / "logs"
_log_dir.mkdir(exist_ok=True)

# ── Archiver state ─────────────────────────────────────────────────────────────
_archiver_lock   = threading.Lock()
_archiver_thread: threading.Thread | None = None
_archiver_event  = threading.Event()
_pending_files: list[Path] = []
_stop_archiver   = False
_archiver_busy   = False   # True while the worker is actively processing a batch


# ── Crash recovery ─────────────────────────────────────────────────────────────
def _recover_tmp() -> None:
    """Clean up leftover ``*.tar.tmp`` files from a previous hard kill.

    A tmp file is the result of an atomic write that was interrupted before
    ``os.replace()`` completed.  We try to recover it as a valid tar; if it
    is empty or corrupt we discard it silently.
    """
    for tmp in list(_log_dir.glob("cip_persist_*.tar.tmp")):
        m = _TMP_RE.match(tmp.name)
        if not m:
            tmp.unlink(missing_ok=True)
            continue

        recovered = False
        if tmp.stat().st_size > 0:
            try:
                with tarfile.open(tmp, "r") as tf:
                    tf.getmembers()
                target = tmp.parent / tmp.stem
                tmp.rename(target)
                recovered = True
            except Exception:
                pass

        if not recovered:
            tmp.unlink(missing_ok=True)


# ── Archive helpers ────────────────────────────────────────────────────────────
def _list_frozen() -> list[Path]:
    return sorted(p for p in _log_dir.iterdir()
                  if p.is_file() and _FROZEN_RE.match(p.name))


def _find_staging() -> Path | None:
    candidates = sorted(p for p in _log_dir.iterdir()
                        if p.is_file() and _STAGING_RE.match(p.name))
    return candidates[-1] if candidates else None


def _date_range(path: Path) -> tuple[str, str]:
    for pattern in (_FROZEN_RE, _STAGING_RE, _TMP_RE):
        m = pattern.match(path.name)
        if m:
            return m.group(1), m.group(2)
    return "00000000", "00000000"


def _staging_name(first_date: str, end_date: str | None = None) -> Path:
    end = end_date or datetime.now().strftime("%Y%m%d")
    return _log_dir / f"{_ARCHIVE_PREFIX}{first_date}-{end}.tar"


def _compress_to_gz(tar_path: Path) -> Path:
    """Gzip-compress *tar_path* → *.tar.gz atomically, remove the original."""
    gz_path = tar_path.parent / (tar_path.stem + ".tar.gz")
    tmp_gz = tar_path.with_suffix(".gz.tmp")
    try:
        with open(tar_path, "rb") as f_in, gzip.open(tmp_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(tmp_gz, gz_path)
        tar_path.unlink()
    except Exception:
        tmp_gz.unlink(missing_ok=True)
        raise
    return gz_path


def _append_to_staging_atomic(staging: Path, files: list[Path]) -> Path:
    """Append *files* to *staging* using a write-then-replace strategy.

    Steps
    -----
    1. Copy the existing staging tar (if any) into a temp file.
    2. Append the new log files to the temp file.
    3. ``os.replace()`` the temp file over the staging path (atomic).
    4. Delete the source log files only after the swap succeeds.

    Returns the (possibly renamed) staging path.
    """
    start, current_end = _date_range(staging)

    # Derive start/end from the actual files being archived.
    # Log files are named cip_YYYYMMDD.log.
    _log_date_re = re.compile(r"^cip_(\d{8})\.log$")
    dates = [m.group(1) for f in files if (m := _log_date_re.match(f.name))]

    # start = oldest file date (or keep existing start if no dates found)
    new_start = min(dates) if dates else start
    # end   = newest file date among those being added, or keep current end
    new_end   = max(dates) if dates else current_end

    # If the staging already has content, preserve its original start date
    if staging.exists() and staging.stat().st_size > 0:
        new_start = start

    final_staging = _log_dir / f"{_ARCHIVE_PREFIX}{new_start}-{new_end}.tar"
    tmp_path      = _log_dir / f"{_ARCHIVE_PREFIX}{new_start}-{new_end}.tar.tmp"

    try:
        if staging.exists() and staging.stat().st_size > 0:
            shutil.copy2(staging, tmp_path)

        with tarfile.open(tmp_path, "a") as tf:
            for f in files:
                if f.exists():
                    tf.add(f, arcname=f.name)

        os.replace(tmp_path, final_staging)

        for f in files:
            if f.exists():
                f.unlink()

        if staging != final_staging and staging.exists():
            staging.unlink()

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return final_staging


def _rotate_archives(files: list[Path]) -> None:
    """Core rotation logic — runs inside the archiver thread."""
    with _archiver_lock:
        staging = _find_staging()

        if staging is None:
            # first_date will be determined from the files themselves
            # inside _append_to_staging_atomic; pass a sentinel path
            first_date = "00000000"
            staging = _staging_name(first_date)

        staging = _append_to_staging_atomic(staging, files)

        if staging.stat().st_size >= _MAX_ARCHIVE_B:
            frozen_list = _list_frozen()
            if frozen_list:
                frozen_list[0].unlink()
            _compress_to_gz(staging)


# ── Background worker ──────────────────────────────────────────────────────────
def _archiver_worker() -> None:
    global _stop_archiver, _archiver_busy
    while True:
        _archiver_event.wait()
        _archiver_event.clear()

        with _archiver_lock:
            batch = list(_pending_files)
            _pending_files.clear()

        if batch:
            _archiver_busy = True
            try:
                _rotate_archives(batch)
            except Exception as exc:
                print(f"[logger] archiver error: {exc}", file=sys.stderr)
            finally:
                _archiver_busy = False

        if _stop_archiver and not _pending_files:
            break


def _ensure_archiver_running() -> None:
    global _archiver_thread
    if _archiver_thread is None or not _archiver_thread.is_alive():
        _archiver_thread = threading.Thread(
            target=_archiver_worker,
            name="cip-log-archiver",
            daemon=True,
        )
        _archiver_thread.start()


def _schedule_archiving(files: list[Path]) -> None:
    with _archiver_lock:
        _pending_files.extend(files)
    _ensure_archiver_running()
    _archiver_event.set()


# ── Log rotation ───────────────────────────────────────────────────────────────
def _rotate_log_files() -> None:
    """Enforce _MAX_LOG_FILES; send excess (oldest first) to the archiver."""
    log_files = sorted(
        p for p in _log_dir.iterdir()
        if p.is_file() and p.suffix == ".log" and p.name.startswith("cip_")
    )
    excess = len(log_files) - (_MAX_LOG_FILES)
    if excess > 0:
        _schedule_archiving(log_files[:excess])


# ── Public shutdown helper ─────────────────────────────────────────────────────
def wait_for_archiver() -> None:
    """Block until the archiver thread finishes; compress staging on exit.

    Prints ``Cleaning log directory, closing upon completion`` only when
    there is actually work in progress.
    """
    global _stop_archiver

    _stop_archiver = True

    if _archiver_thread is not None and _archiver_thread.is_alive():
        if _pending_files or _archiver_busy:
            print("Cleaning log directory, closing upon completion")
        _archiver_event.set()
        _archiver_thread.join()

    staging = _find_staging()
    if staging is not None and staging.stat().st_size >= _MAX_ARCHIVE_B:
        frozen_list = _list_frozen()
        if frozen_list:
            frozen_list[0].unlink()
        _compress_to_gz(staging)


# ── Logger setup ───────────────────────────────────────────────────────────────
log = logging.getLogger("CIP")
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler(
    _log_dir / f"cip_{datetime.now().strftime('%Y%m%d')}.log",
    encoding="utf-8",
)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)


# ── Log Rotation startup ───────────────────────────────────────────────────────
_recover_tmp()
_rotate_log_files()