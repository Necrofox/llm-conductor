# -*- coding: utf-8 -*-
"""
Where the configuration files live.

conductor.toml sits next to the executable - not in the working directory,
which a scheduler does not control. The LLMC_CONFIG environment variable
overrides its path (meant for tests). llm.toml defaults to the directory of
conductor.toml; --llm-config replaces it.
"""
import os
import sys
from dataclasses import dataclass
from typing import Mapping, Optional

from llm_conductor.constants import (
    CONDUCTOR_CONFIG_FILENAME,
    CONFIG_PATH_ENV_VAR,
    LLM_CONFIG_FILENAME,
)


def resolve_app_dir(entry_file: Optional[str] = None) -> str:
    """
    Directory of the running program (where ``conductor.toml`` lives).

    A one-file PyInstaller build unpacks into a temporary directory, so a
    frozen program is located by ``sys.executable``. An installed console
    script uses ``sys.argv[0]`` (the shim next to which users keep configs),
    not the package path in site-packages. The repo-root ``llm-conductor.py``
    wrapper makes ``argv[0]`` the checkout root — portable-app layout without
    an install. See ``doc/concepts.md`` and ``doc/configuration.md``.

    Args:
        entry_file: Fallback path (usually ``__file__`` of the CLI module)
            when argv[0] is not a real file.

    Returns:
        Absolute directory path.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and not argv0.startswith("-") and os.path.isfile(argv0):
        return os.path.dirname(os.path.abspath(argv0))

    if entry_file:
        return os.path.dirname(os.path.abspath(entry_file))

    return os.getcwd()


@dataclass(frozen=True)
class ConfigLocations:
    """Resolved absolute paths of the two configuration files."""

    conductor_path: str
    llm_path: str

    @classmethod
    def resolve(cls, app_dir: str, llm_override: Optional[str] = None,
                environ: Optional[Mapping[str, str]] = None) -> 'ConfigLocations':
        """
        Resolve both paths.

        Args:
            app_dir: Directory of the program (resolve_app_dir()).
            llm_override: Value of --llm-config, relative to the working directory.
            environ: Environment holding LLMC_CONFIG; os.environ by default.

        Returns:
            The resolved locations.
        """
        env = os.environ if environ is None else environ
        conductor_override = env.get(CONFIG_PATH_ENV_VAR, "").strip()
        conductor_path = (os.path.abspath(conductor_override) if conductor_override
                          else os.path.join(app_dir, CONDUCTOR_CONFIG_FILENAME))
        llm_path = (os.path.abspath(llm_override) if llm_override
                    else os.path.join(os.path.dirname(conductor_path), LLM_CONFIG_FILENAME))
        return cls(conductor_path, llm_path)
