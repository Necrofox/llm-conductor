# -*- coding: utf-8 -*-
"""LLM-Conductor: JSON localization with language models."""

try:
    from llm_conductor._version import __version__ as __version__
except ModuleNotFoundError as exc:
    if exc.name != "llm_conductor._version":
        raise
    from llm_conductor.constants import UNKNOWN_VERSION

    __version__ = UNKNOWN_VERSION
