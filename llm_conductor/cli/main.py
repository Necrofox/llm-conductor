# -*- coding: utf-8 -*-
"""
CLI orchestration: argparse, subcommands, the single process exit point.

conductor.toml is looked up next to the running program (frozen exe,
console-script shim, or the repo-root ``llm-conductor.py`` wrapper — see
``doc/concepts.md``); ``LLMC_CONFIG`` overrides the path. The model
profile defaults to ``llm.toml`` in the same directory; ``--llm-config``
replaces it (JSON targets only; TOML run manifests use precedence B).

main() is the only place that exits the process; every subcommand returns
an exit code of the contract in llm_conductor.constants (EXIT_*).
"""
import argparse
import atexit
import os
import signal
import sys
from typing import Dict, List, Tuple

from llm_conductor import __version__
from llm_conductor.cli.dry_run import run_dry_run
from llm_conductor.cli.init import create_config_files
from llm_conductor.cli.locations import ConfigLocations, resolve_app_dir
from llm_conductor.cli.preflight import validate_target_inputs
from llm_conductor.cli.run_manifest import is_legacy_list_path, is_run_manifest_path
from llm_conductor.cli.runner import (
    ProfileEndpoints,
    RunOptions,
    TargetRunner,
    load_glossaries_for_target,
)
from llm_conductor.cli.targets import ResultLayout, TargetResolver
from llm_conductor.config import (
    Config,
    ConfigData,
    ConfigFileLoader,
    ConfigLogger,
    RunOverrides,
    RuntimeConfigFactory,
)
from llm_conductor.constants import (
    EXIT_CONFIG_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    LEGACY_LIST_EXTENSION,
    MANIFEST_EXTENSION,
    RUN_STAGE_ALL,
    TARGET_CACHE_DIR_NAME,
    VALID_LOG_LEVELS,
    VALID_RUN_STAGES,
)
from llm_conductor.core.io import FileOperations
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

# Job flags forbidden with a TOML run manifest (precedence B). Session flags
# (--verb, --no-cache, --dry-run) remain allowed.
_JOB_FLAG_ATTRS: Tuple[Tuple[str, str], ...] = (
    ('stage', '--stage'),
    ('strict', '--strict'),
    ('llm_config', '--llm-config'),
    ('glossary', '--glossary'),
    ('result2target', '--result2target'),
    ('cache2target', '--cache2target'),
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

    run = subcommands.add_parser(
        'run',
        help=f'Translate and verify a JSON file or a {MANIFEST_EXTENSION} run manifest',
    )
    run.add_argument(
        '--target', required=True, metavar='PATH',
        help=f'Input .json file, or a {MANIFEST_EXTENSION} run manifest with [[targets]]',
    )
    # default=None so a TOML manifest can detect a *passed* --stage (precedence B).
    run.add_argument(
        '--stage', choices=VALID_RUN_STAGES, default=None,
        help='Stages to run (default: all); verify reads result/translated/ of an earlier run '
             f'(JSON targets only; forbidden with a {MANIFEST_EXTENSION} manifest)',
    )
    run.add_argument(
        '--strict', action='store_true',
        help='Strict translation mode (precise, technical) for the whole run '
             f'(JSON targets only; forbidden with a {MANIFEST_EXTENSION} manifest)',
    )
    run.add_argument(
        '--no-cache', action='store_true',
        help='Neither read nor write the disk cache (in-memory deduplication still works)',
    )
    run.add_argument(
        '--dry-run', action='store_true',
        help='Validate the target / manifest and print effective settings without API calls '
             'or result writes (session flag; allowed with a run manifest)',
    )
    run.add_argument('--llm-config', metavar='PATH', help=llm_config_help)
    run.add_argument(
        '--glossary', metavar='PATH',
        help='Glossary JSON; default: glossary.json next to the target, if present '
             f'(JSON targets only; forbidden with a {MANIFEST_EXTENSION} manifest)',
    )
    run.add_argument(
        '--verb', metavar='LEVEL', type=str.upper, choices=VALID_LOG_LEVELS,
        help=f"Log level instead of [logging].level: {', '.join(VALID_LOG_LEVELS)}",
    )
    run.add_argument(
        '--result2target', action='store_true',
        help='Write result/ next to the target instead of the shared result/<target name>/ '
             f'(JSON targets only; forbidden with a {MANIFEST_EXTENSION} manifest)',
    )
    run.add_argument(
        '--cache2target', action='store_true',
        help='Keep the cache in cache/ next to the target instead of [cache].dir '
             f'(JSON targets only; forbidden with a {MANIFEST_EXTENSION} manifest)',
    )

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


def _passed_job_flags(args: argparse.Namespace) -> List[str]:
    """Names of job CLI flags that were passed (precedence B)."""
    passed: List[str] = []
    for attr, flag in _JOB_FLAG_ATTRS:
        value = getattr(args, attr, None)
        if attr in ('strict', 'result2target', 'cache2target'):
            if value:
                passed.append(flag)
        elif value is not None:
            passed.append(flag)
    return passed


def _reject_job_flags_for_manifest(args: argparse.Namespace) -> None:
    """
    Precedence B: any passed job flag with a ``.toml`` target → exit 2.

    Raises:
        TargetError: Naming the forbidden flags.
    """
    passed = _passed_job_flags(args)
    if not passed:
        return
    raise TargetError(
        f"Job flags {', '.join(passed)} are not allowed with a {MANIFEST_EXTENSION} "
        f"run manifest (set them in the manifest instead; session flags "
        f"--verb / --no-cache / --dry-run still apply)"
    )


def _run_overrides(args: argparse.Namespace, *, cache2target: bool,
                   target_dir: str) -> RunOverrides:
    """conductor.toml settings replaced by session / layout choices."""
    return RunOverrides(
        log_level=args.verb,
        cache_enabled=False if args.no_cache else None,
        cache_dir=os.path.join(target_dir, TARGET_CACHE_DIR_NAME) if cache2target else None,
    )


def _load_profiles(conductor_path: str, llm_paths: List[str],
                   overrides: RunOverrides) -> Dict[str, ConfigData]:
    """
    Load unique llm profiles against the same conductor.toml.

    Raises:
        FileOperationError, ConfigValidationError: On load failure.
    """
    loader = ConfigFileLoader()
    profiles: Dict[str, ConfigData] = {}
    for llm_path in llm_paths:
        if llm_path in profiles:
            continue
        # Session overrides (verb / no-cache / cache dir) apply to every profile
        # copy so RuntimeConfigFactory sees the same conductor-side settings.
        profiles[llm_path] = loader.load(conductor_path, llm_path, overrides)
    return profiles


def _execute_run(config: Config, args: argparse.Namespace,
                 locations: ConfigLocations, target,
                 profiles: Dict[str, ConfigData],
                 result2target: bool) -> int:
    """
    Resolve glossary and layout, then run the pipeline.

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

    glossaries = load_glossaries_for_target(
        target, config.file_ops, config.logger, cli_glossary=args.glossary,
    )
    layout = ResultLayout.for_target(
        target, os.path.dirname(locations.conductor_path), result2target,
    )
    layout.check_disjoint(target)

    stage = args.stage if args.stage is not None else RUN_STAGE_ALL
    options = RunOptions(stage=stage, strict=args.strict)

    # Shared input preflight before locks / API (and before dry-run table).
    validate_target_inputs(target, config.file_ops, config.logger)

    if args.dry_run:
        return run_dry_run(
            config, target, layout, profiles, glossaries,
            default_options=options, bootstrap_llm=locations.llm_path,
        )

    factories = {
        path: RuntimeConfigFactory(data) for path, data in profiles.items()
    }
    endpoints = {
        path: ProfileEndpoints(
            translation=data.translation_api,
            verification=data.verification_api,
        )
        for path, data in profiles.items()
    }
    return TargetRunner(
        config, target, layout, factories, endpoints, glossaries,
        default_options=options,
    ).run()


def _command_run(args: argparse.Namespace, app_dir: str) -> int:
    """
    Run the pipeline on the target of the command line.

    Returns:
        The exit code; the reason of a non-zero code is already logged
        (or printed to stderr when the configuration itself failed to load).
    """
    target_path = os.path.abspath(args.target)

    # Early contract checks that do not need Config / logs.
    if is_legacy_list_path(target_path):
        print(
            f"Error: {LEGACY_LIST_EXTENSION} manifests are no longer supported "
            f"(target was {target_path}).\n"
            f"  Migration: create a {MANIFEST_EXTENSION} run manifest; each former "
            f"line becomes:\n"
            f"    [[targets]]\n"
            f"    original = \"path/from/line.json\"\n"
            f"  Then: llm-conductor run --target that{MANIFEST_EXTENSION}\n"
            f"  Job options (stage, glossary, llm_config, …) go in the manifest; "
            f"see doc/cli.md.",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    if is_run_manifest_path(target_path):
        try:
            _reject_job_flags_for_manifest(args)
        except TargetError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    # Resolve conductor location without --llm-config for manifests (job flag).
    llm_override = None if is_run_manifest_path(target_path) else args.llm_config
    locations = ConfigLocations.resolve(app_dir, llm_override)
    conductor_dir = os.path.dirname(locations.conductor_path)

    # Resolve the target before creating the full Config so schema/path errors
    # can exit 2 without a half-initialized logger.
    early_logger = ConfigLogger.create(log_to_file=False)
    early_ops = FileOperations(logger=early_logger)
    try:
        target = TargetResolver(early_ops, early_logger, conductor_dir=conductor_dir).resolve(
            args.target,
        )
    except TargetError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if target.is_manifest and target.settings is not None:
        result2target = target.settings.result2target
        cache2target = target.settings.cache2target
        # Manifest verb is the base; CLI --verb overrides (session flag).
        if args.verb is None and target.settings.verb is not None:
            args.verb = target.settings.verb
        llm_paths = []
        for entry in target.entries:
            path = entry.llm_config_path or locations.llm_path
            if path not in llm_paths:
                llm_paths.append(path)
        bootstrap_llm = llm_paths[0]
        locations = ConfigLocations(locations.conductor_path, bootstrap_llm)
    else:
        result2target = args.result2target
        cache2target = args.cache2target
        llm_paths = [locations.llm_path]

    overrides = _run_overrides(
        args, cache2target=cache2target,
        target_dir=os.path.dirname(target_path),
    )

    try:
        profiles = _load_profiles(locations.conductor_path, llm_paths, overrides)
        config = Config.create_from_files(
            locations.conductor_path, locations.llm_path,
            log_to_file=not args.dry_run,
            overrides=overrides, log_name=os.path.basename(args.target),
        )
    except (FileOperationError, ConfigValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Ensure bootstrap profile is in the map (same path key).
    if locations.llm_path not in profiles:
        profiles[locations.llm_path] = profiles[llm_paths[0]]

    _shutdown_manager.register_config(config)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)

    code = EXIT_INTERNAL_ERROR
    try:
        code = _execute_run(config, args, locations, target, profiles, result2target)
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

    if args.command == 'init':
        locations = ConfigLocations.resolve(app_dir, None)
        return _command_init(locations)
    if args.command == 'check':
        locations = ConfigLocations.resolve(app_dir, getattr(args, 'llm_config', None))
        return _command_check(locations)
    return _command_run(args, app_dir)


def main() -> None:
    """The single exit point of the process."""
    try:
        code = _dispatch()
    except RunInterrupted:
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    main()
