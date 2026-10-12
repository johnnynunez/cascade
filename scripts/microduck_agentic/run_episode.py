# SPDX-License-Identifier: Apache-2.0
"""Entry point for the Isaac Lab venv: optional extra site dirs APPENDED (never shadowing the venv), then the app."""
import os
import sys

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)

from cascade.apps.microduck_agentic import main  # noqa: E402

sys.exit(main())
