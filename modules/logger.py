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

"""Logger setup for CIP. Exposes ``log`` and ``_SCRIPT_DIR``."""

import logging
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent

_log_dir = _SCRIPT_DIR / "logs"
_log_dir.mkdir(exist_ok=True)

log = logging.getLogger("CIP")
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler(
    _log_dir / f"cip_{datetime.now().strftime('%Y%m%d')}.log",
    encoding="utf-8",
)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)
