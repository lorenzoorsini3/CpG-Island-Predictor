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

"""CIP modules package."""

from .features_extractor import FEATURES_ORDER, extract_features
from .exception_handler import _handle_error, _handle_warning
from .logger import _SCRIPT_DIR, log, wait_for_archiver

__version__ = "v4.2.0"

__all__ = [
    "FEATURES_ORDER",
    "extract_features",
    "_handle_error",
    "_handle_warning",
    "_SCRIPT_DIR",
    "log",
    "wait_for_archiver",
    "__version__",
]
