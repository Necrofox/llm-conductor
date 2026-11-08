# -*- coding: utf-8 -*-
"""
Repo-root entry wrapper (scenario C — kept on purpose).

Canonical CLI logic lives in ``llm_conductor.cli.main``. This file stays at the
repository root so that:

* PyInstaller (``llm-conductor.spec``) has a stable one-file script entry;
* ``python llm-conductor.py …`` from a source checkout resolves ``conductor.toml``
  next to the repo root (same “config beside the program” rule as a portable
  ``.exe``), without requiring an editable install first.

Day-to-day launches after ``pip install`` / a frozen build should use the
``llm-conductor`` command or the ``.exe``. ``python -m llm_conductor`` works but
puts configs next to the package ``__main__.py`` unless ``LLMC_CONFIG`` is set —
prefer the wrapper or the console script for local work.

Portable entry for checkout / PyInstaller; configs resolve next to this file
(see ``doc/concepts.md``). Wrapper removal is tracked as a post-1.0 task.
"""
from llm_conductor.cli.main import main

if __name__ == "__main__":
    main()
