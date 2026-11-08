# -*- coding: utf-8 -*-
"""
CLI orchestration: argparse, subcommands, the single process exit point.

conductor.toml is looked up next to the running program (frozen exe,
console-script shim, or the repo-root ``llm-conductor.py`` wrapper — see
``doc/concepts.md``); ``LLMC_CONFIG`` overrides the path. The model
profile defaults to ``llm.toml`` in the same directory; ``--llm-config``
replaces it.

main() is the only place that exits the process; every subcommand returns
an exit code of the contract in llm_conductor.constants (EXIT_*).
"""
import argparse
import atexit
import os
import signal
import sys

from llm_conductor import __version__
from llm_conductor.cli.glossary import GlossaryResolver
from llm_conductor.cli.init import create_config_files
from llm_conductor.cli.locations import ConfigLocations, resolve_app_dir
from llm_conductor.cli.runner import RunOptions, TargetRunner
from llm_conductor.cli.targets import ResultLayout, TargetResolver
from llm_conductor.config import Config, ConfigFileLoader, RunOverrides
from llm_conductor.constants import (
    EXIT_CONFIG_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    RUN_STAGE_ALL,
    TARGET_CACHE_DIR_NAME,
    VALID_LOG_LEVELS,
    VALID_RUN_STAGES,
)
from llm_conductor.core.shutdown import (
    _shutdown_manager,
    cleanup_on_exit,
    signal_handler,
)
from llm_conductor.exceptions import (
    ConfigValidationError,
    FileOperationError,
    RunInterrupted,
    TargetError,
)


def _build_parser() -> argparse.ArgumentParser:
    """Command line of the three subcommands."""
    parser = argparse.ArgumentParser(
        prog='llm-conductor',
        description='JSON localization with language models (Ollama and OpenAI-compatible APIs)',
        epilog='Example: llm-conductor run --target strings.json --stage translate --strict',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )
    subcommands = parser.add_subparsers(dest='command', metavar='{run,init,check}')

    llm_config_help = 'Model profile to use instead of llm.toml next to conductor.toml'

    run = subcommands.add_parser('run', help='Translate and verify a JSON file or a .list manifest')
    run.add_argument('--target', required=True, metavar='PATH',
                     help='Input .json file, or a .list manifest with one input path per line')
    run.add_argument('--stage', choices=VALID_RUN_STAGES, default=RUN_STAGE_ALL,
                     help='Stages to run (default: all); verify reads result/translated/ of an earlier run')
    run.add_argument('--strict', action='store_true',
                     help='Strict translation mode (precise, technical) for the whole run')
    run.add_argument('--no-cache', action='store_true',
                     help='Neither read nor write the disk cache (in-memory deduplication still works)')
    run.add_argument('--llm-config', metavar='PATH', help=llm_config_help)
    run.add_argument('--glossary', metavar='PATH',
                     help='Glossary JSON; default: glossary.json next to the target, if present')
    run.add_argument('--verb', metavar='LEVEL', type=str.upper, choices=VALID_LOG_LEVELS,
                     help=f"Log level instead of [logging].level: {', '.join(VALID_LOG_LEVELS)}")
    run.add_argument('--result2target', action='store_true',
                     help='Write result/ next to the target instead of the shared result/<target name>/')
    run.add_argument('--cache2target', action='store_true',
                     help='Keep the cache in cache/ next to the target instead of [cache].dir')

    subcommands.add_parser('init', help='Create conductor.toml and llm.toml templates and exit')

    check = subcommands.add_parser('check', help='Validate conductor.toml and the model profile and exit')
    check.add_argument('--llm-config', metavar='PATH', help=llm_config_help)

    return parser


def _command_init(locations: ConfigLocations) -> int:
    """Write the configuration templates; returns the exit code."""
    try:
        create_config_files(locations.conductor_path, locations.llm_path)
    except (ConfigValidationError, FileOperationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    return EXIT_SUCCESS


def _command_check(locations: ConfigLocations) -> int:
    """Validate both configuration files without creating logs or caches; returns the exit code."""
    try:
        data = ConfigFileLoader().load(locations.conductor_path, locations.llm_path)
    except (FileOperationError, ConfigValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(f"OK: {locations.conductor_path}")
    print(f"OK: {locations.llm_path} (profile: {data.profile_name})")
    return EXIT_SUCCESS


def _run_overrides(args: argparse.Namespace) -> RunOverrides:
    """conductor.toml settings replaced by the run command line."""
    target_dir = os.path.dirname(os.path.abspath(args.target))
    return RunOverrides(
        log_level=args.verb,
        cache_enabled=False if args.no_cache else None,
        cache_dir=os.path.join(target_dir, TARGET_CACHE_DIR_NAME) if args.cache2target else None,
    )


def _execute_run(config: Config, args: argparse.Namespace, locations: ConfigLocations) -> int:
    """
    Resolve the target, the glossary and the result layout, then run the pipeline.

    Returns:
        The exit code of the run.

    Raises:
        TargetError: If the target, its manifest or the glossary is unusable.
    """
    config.logger.info("=" * 70)
    config.logger.info(f"LLM-CONDUCTOR {__version__} - JSON LOCALIZATION TOOL")
    config.logger.info("=" * 70)
    config.logger.info(f"  - Model profile: {config.profile_name} ({locations.llm_path})")
    config.logger.info(f"  - Configuration: {locations.conductor_path}")

    target = TargetResolver(config.file_ops, config.logger).resolve(args.target)
    glossary = GlossaryResolver(config.file_ops, config.logger).resolve(args.glossary, target)
    layout = ResultLayout.for_target(target, os.path.dirname(locations.conductor_path), args.result2target)
    layout.check_disjoint(target)

    options = RunOptions(stage=args.stage, strict=args.strict)
    return TargetRunner(config, options, target, layout, glossary).run()


def _command_run(args: argparse.Namespace, locations: ConfigLocations) -> int:
    """
    Run the pipeline on the target of the command line.

    Returns:
        The exit code; the reason of a non-zero code is already logged
        (or printed to stderr when the configuration itself failed to load).
    """
    try:
        config = Config.create_from_files(locations.conductor_path, locations.llm_path, log_to_file=True,
                                          overrides=_run_overrides(args),
                                          log_name=os.path.basename(args.target))
    except (FileOperationError, ConfigValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    _shutdown_manager.register_config(config)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)

    code = EXIT_INTERNAL_ERROR
    try:
        code = _execute_run(config, args, locations)
    except TargetError as e:
        config.logger.error(f"{e}. Nothing was processed.")
        code = EXIT_USAGE_ERROR
    except RunInterrupted:
        # ShutdownManager has already logged the signal and saved the caches.
        config.logger.warning(f"Run interrupted; exiting with code {EXIT_INTERRUPTED}")
        code = EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001 - last resort: log the traceback, exit with EXIT_INTERNAL_ERROR
        config.logger.error("=" * 70)
        config.logger.error(f"Unexpected failure: {type(e).__name__}: {e}")
        config.logger.error("Full traceback:", include_traceback=True)
        config.logger.error("=" * 70)
        code = EXIT_INTERNAL_ERROR
    finally:
        _shutdown_manager.register_config(None)
        # A non-zero exit can reflect a problem logged below ERROR (e.g. the
        # WARNING of an interrupted run) - the log of such a run is kept too.
        config.close_log_files(always_keep=code != EXIT_SUCCESS)

    return code


def _dispatch(argv=None) -> int:
    """Parse the command line and run the subcommand; returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE_ERROR

    app_dir = resolve_app_dir(__file__)
    locations = ConfigLocations.resolve(app_dir, getattr(args, 'llm_config', None))

    if args.command == 'init':
        return _command_init(locations)
    if args.command == 'check':
        return _command_check(locations)
    return _command_run(args, locations)


def main() -> None:
    """The single exit point of the process."""
    try:
        code = _dispatch()
    except RunInterrupted:
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    main()
