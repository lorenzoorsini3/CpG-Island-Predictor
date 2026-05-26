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

"""Shared error and warning helpers for CIP."""

import json

_ERRORS: dict[type, str] = {
    FileNotFoundError:    "File not found.",
    IsADirectoryError:    "Path is a directory, not a file.",
    PermissionError:      "Permission denied.",
    UnicodeDecodeError:   "File is not valid text (encoding error).",
    OSError:              "OS-level I/O error.",
    ValueError:           "Invalid value or data format.",
    TypeError:            "Unexpected data type.",
    MemoryError:          "Not enough memory to complete the operation.",
    json.JSONDecodeError: "Invalid JSON format.",
}


def _handle_error(e: Exception, context: str) -> str:
    """Return a formatted ERROR string for a known exception type."""
    return f"ERROR: '{context}': {_ERRORS.get(type(e), str(e))}"


def _handle_warning(level: str, msg: str, log) -> None:
    """Print ``msg`` to stdout and log it at ``level``."""
    print(f"{level.upper()}: {msg}")
    if level == "warning":
        log.warning(msg)
    elif level == "error":
        log.error(msg)
    else:
        log.info("[unknown level '%s'] %s", level, msg)
