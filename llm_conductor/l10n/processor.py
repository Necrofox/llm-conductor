# -*- coding: utf-8 -*-
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from llm_conductor.constants import (
    FAILURE_KIND_REJECTED,
    GLOSSARY_MARKER,
    LINGUISTIC_CONTEXT_BLOCK_TEMPLATE,
    LINGUISTIC_CONTEXT_MARKER,
    MODE_STYLE,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    ORIGINAL_TEXT_BLOCK_NOTE,
    ORIGINAL_TEXT_BLOCK_TEMPLATE,
    ORIGINAL_TEXT_MARKER,
    REJECTED_REPLIES_SUBDIR,
    REJECTED_REPLY_LOG_LENGTH,
    STATISTICS_LOG_LEVELS,
    TRANSLATION_TEXT_MARKER,
    USAGE_NON_API_CACHE_HIT,
    USAGE_NON_API_DEDUP,
    VERIFICATION_TEXT_MARKER,
)
from llm_conductor.core.cache import build_sampling_identity
from llm_conductor.core.io import safe_write_file
from llm_conductor.core.leaf_report import (
    make_degraded,
    make_ok,
    make_skipped,
    make_verify_skipped,
)
from llm_conductor.core.shutdown import is_shutdown_requested
from llm_conductor.exceptions import FileOperationError
from llm_conductor.l10n.context import (
    KeyPath,
    LinguisticContext,
    LinguisticHint,
    build_glossary_string,
)
from llm_conductor.l10n.validation import ReplyValidationResult, validate_model_reply
from llm_conductor.utils import (
    is_format_only_leaf,
    mask_tags,
    split_boundary_whitespace,
    unmask_tags,
)

if TYPE_CHECKING:
    from llm_conductor.interfaces import (
        HeartbeatProgressProtocol,
        ProcessorConfigProtocol,
    )


def count_string_leaves(data: Any, max_depth: int, current_depth: int = 0) -> int:
    """
    Count every string leaf of a tree with the same depth cut-off as traverse().

    Blank strings are included: they still enter ``_process_leaf`` and must
    advance the heartbeat fraction so percent can reach 100.

    Args:
        data: Root of the tree.
        max_depth: Same limit as ``[limits].max_recursion_depth``.
        current_depth: Depth of ``data`` (0 at the root).

    Returns:
        Number of string leaves that traverse() would hand to ``_process_leaf``.
    """
    if current_depth > max_depth:
        return 0
    if isinstance(data, dict):
        return sum(
            count_string_leaves(value, max_depth, current_depth + 1)
            for value in data.values()
        )
    if isinstance(data, list):
        return sum(
            count_string_leaves(value, max_depth, current_depth + 1)
            for value in data
        )
    if isinstance(data, str):
        return 1
    return 0


class RecursiveProcessor:
    """
    Thread-safe recursive traversal of a JSON tree.

    Settings are read from the config once, in the constructor; the mutable
    run state (dedup index, statistics, auto-save counter) is guarded by
    one lock per structure.
    """

    def __init__(self, config: 'ProcessorConfigProtocol',
                 context_analyzer: 'LinguisticContext',
                 heartbeat: Optional['HeartbeatProgressProtocol'] = None):
        """
        Args:
            config: Processor-facing configuration.
            context_analyzer: Linguistic analyzer of key paths.
            heartbeat: Optional progress ticks (begin_stage / note_leaf).
        """
        self.config = config

        self.logger = config.logger
        self.is_debug = config.is_debug_mode
        self.cache_enabled = config.cache_enabled
        self.dedup_enabled = config.value_deduplication_enabled
        self.cache_manager = config.cache_manager
        self.max_recursion_depth = config.max_recursion_depth
        self.dedup_log_interval = config.dedup_log_interval
        self._heartbeat = heartbeat

        self.context_analyzer = context_analyzer

        # {full cache key: (KeyPath, translated value)} - the same key as the
        # cache, because one text gets different hints on different paths.
        self._dedup_index: Dict[str, Tuple[KeyPath, Any]] = {}
        self._dedup_lock = threading.Lock()

        # Counts leaves that added a NEW cache entry. A separate lock, so
        # _stats_lock/_dedup_lock are never held during a slow save().
        self._autosave_lock = threading.Lock()
        self._leaves_since_save = 0

        # Processing statistics, guarded by _stats_lock
        self.stats = {
            'total_nodes': 0,
            'dict_nodes': 0,
            'list_nodes': 0,
            'leaf_nodes': 0,
            'processed_leaves': 0,
            'skipped_leaves': 0,
            'deduplicated_leaves': 0,
            'failed_leaves': 0,
            'max_depth_reached': 0
        }
        self._stats_lock = threading.Lock()
    
    def build_dedup_index(self, data: Any):
        """
        Count the unique string values of a tree and log the number.

        The dedup index itself is filled in _process_leaf(): its key is the
        full request hash, known only once the hints of a path are analyzed.

        Args:
            data: Root of the tree.
        """
        if not self.dedup_enabled:
            return

        unique_values: set = set()

        def scan_node(node: Any):
            """Recursive tree scan for unique non-blank string leaves."""
            if isinstance(node, dict):
                for value in node.values():
                    scan_node(value)

            elif isinstance(node, list):
                for value in node:
                    scan_node(value)

            elif isinstance(node, str) and node.strip():
                unique_values.add(node)

        if self.config.is_debug_mode:
            self.config.logger.debug("Scanning tree for unique values (deduplication stats)...")

        scan_node(data)

        unique_count = len(unique_values)

        if self.config.is_debug_mode:
            self.config.logger.debug(f"Deduplication scan complete: {unique_count} unique values")
        else:
            self.config.logger.info(
                f"Deduplication: {unique_count} unique values found (indexed during processing)"
            )
    
    def traverse(
        self,
        data: Any,
        current_path: KeyPath,
        process_func: Callable,
        glossary: dict,
        runtime_config: dict,
        mode: str,
        original_data: Optional[Any] = None,
        is_verification: bool = False,
        current_depth: int = 0
    ) -> Any:
        """
        Process a subtree recursively, preserving its structure and key order.

        The input is never mutated; a new tree is built. A subtree deeper than
        max_recursion_depth, or any node reached after a shutdown request, is
        returned unchanged.

        Args:
            data: Current node.
            current_path: Path of the node.
            process_func: Leaf function (translate_key / verify_key).
            glossary: Glossary {term: translation}.
            runtime_config: Flat API parameters.
            mode: Translation mode.
            original_data: Matching node of the English source (verification only).
            is_verification: True on the verification stage.
            current_depth: Depth of the node.

        Returns:
            Processed copy of the node, of the same type.
        """
        operation_type = OPERATION_VERIFICATION if is_verification else OPERATION_TRANSLATION
        if self.config.circuit_breaker.is_tripped(operation_type):
            # The trip itself was already logged once, from _process_leaf();
            # every subsequent node is returned unchanged, same as shutdown.
            return data

        if current_depth > self.max_recursion_depth:
            self.config.logger.error(
                f"Maximum recursion depth ({self.max_recursion_depth}) exceeded at {current_path}. "
                f"Returning original data."
            )
            with self._stats_lock:
                self.stats['max_depth_reached'] = max(
                    self.stats['max_depth_reached'],
                    current_depth
                )
            return data
        
        with self._stats_lock:
            self.stats['total_nodes'] += 1
            self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], current_depth)
        
        if is_shutdown_requested():
            self.config.logger.info(f"Shutdown requested at {current_path}, stopping traversal")
            return data
        
        if isinstance(data, dict):
            with self._stats_lock:
                self.stats['dict_nodes'] += 1
            return self._traverse_dict(
                data, current_path, process_func, glossary,
                runtime_config, mode, original_data, is_verification,
                current_depth
            )
        
        elif isinstance(data, list):
            with self._stats_lock:
                self.stats['list_nodes'] += 1
            return self._traverse_list(
                data, current_path, process_func, glossary,
                runtime_config, mode, original_data, is_verification,
                current_depth
            )
        
        elif isinstance(data, str):
            with self._stats_lock:
                self.stats['leaf_nodes'] += 1
            return self._process_leaf(
                data, current_path, process_func, glossary,
                runtime_config, mode, original_data, is_verification
            )
        
        else:
            # Immutable primitives (int, float, bool, None)
            return data
    
    def _traverse_dict(
        self,
        data: dict,
        path: KeyPath,
        process_func: Callable,
        glossary: dict,
        runtime_config: dict,
        mode: str,
        original_data: Optional[Any],
        is_verification: bool,
        depth: int
    ) -> dict:
        """
        Walk a dict, preserving key order.

        Returns:
            A new dict with processed values.
        """
        result = {}

        for key, value in data.items():
            child_path = path.append(key)

            # Matching node in original_data (verification stage).
            original_value = None
            if is_verification and isinstance(original_data, dict):
                original_value = original_data.get(key)
            
            result[key] = self.traverse(
                data=value,
                current_path=child_path,
                process_func=process_func,
                glossary=glossary,
                runtime_config=runtime_config,
                mode=mode,
                original_data=original_value,
                is_verification=is_verification,
                current_depth=depth + 1
            )
        
        return result
    
    def _traverse_list(
        self,
        data: list,
        path: KeyPath,
        process_func: Callable,
        glossary: dict,
        runtime_config: dict,
        mode: str,
        original_data: Optional[Any],
        is_verification: bool,
        depth: int
    ) -> list:
        """
        Walk a list, preserving element order.

        Returns:
            A new list with processed elements.
        """
        result = []

        for idx, value in enumerate(data):
            child_path = path.append(idx)

            # Matching element in original_data (verification stage).
            original_value = None
            if is_verification and isinstance(original_data, list) and idx < len(original_data):
                original_value = original_data[idx]
            
            result.append(
                self.traverse(
                    data=value,
                    current_path=child_path,
                    process_func=process_func,
                    glossary=glossary,
                    runtime_config=runtime_config,
                    mode=mode,
                    original_data=original_value,
                    is_verification=is_verification,
                    current_depth=depth + 1
                )
            )
        
        return result
    
    def _process_leaf(
        self,
        value: str,
        path: KeyPath,
        process_func: Callable,
        glossary: dict,
        runtime_config: dict,
        mode: str,
        original_data: Optional[Any],
        is_verification: bool
    ) -> str:
        """
        Process one string leaf: cache, dedup, tag masking, the model call
        and the model-reply sanity checks of llm_conductor/l10n/validation.py.

        Always returns a str: the CallOutcome of translate_key/verify_key is
        a transport format and never reaches the tree or the cache. A failed
        call, a reply the validator rejects, and an unexpected exception
        from process_func all feed config.circuit_breaker and are never
        cached; only an accepted reply updates the cache, the dedup index
        and resets the breaker's streak.

        Order matters: the glossary, the linguistic hints and the cache key
        are computed from the original, unmasked text; the model receives the
        masked text; the cache stores the unmasked reply. On verification,
        original_data is also masked into the {ORIGINAL_TEXT} reference
        block before the cache key is built, so the key already reflects it.

        Args:
            value: Leaf text.
            path: Path of the leaf.
            process_func: translate_key or verify_key.
            glossary: Glossary {term: translation}.
            runtime_config: Flat API parameters.
            mode: Translation mode.
            original_data: English source of the leaf (verification only).
            is_verification: True on the verification stage.

        Returns:
            The processed text, or the original value on any failure or
            rejection.
        """
        try:
            return self._process_leaf_body(
                value, path, process_func, glossary, runtime_config,
                mode, original_data, is_verification,
            )
        finally:
            if self._heartbeat is not None:
                self._heartbeat.note_leaf()

    def _process_leaf_body(
        self,
        value: str,
        path: KeyPath,
        process_func: Callable,
        glossary: dict,
        runtime_config: dict,
        mode: str,
        original_data: Optional[Any],
        is_verification: bool
    ) -> str:
        """Body of ``_process_leaf``; the outer wrapper always notes the leaf."""
        operation_type = OPERATION_VERIFICATION if is_verification else OPERATION_TRANSLATION
        leaf_path = str(path)
        outcomes = self.config.leaf_outcomes

        if not value.strip() or is_format_only_leaf(value):
            with self._stats_lock:
                self.stats['skipped_leaves'] += 1
            outcomes.record(leaf_path, operation_type, make_skipped(value))
            return value

        # Skip normal verification when translation already degraded — decided
        # from the prior stage outcome, never from value == original_data.
        if is_verification and outcomes.is_translation_degraded(leaf_path):
            with self._stats_lock:
                self.stats['skipped_leaves'] += 1
            outcomes.record(leaf_path, operation_type, make_verify_skipped(value))
            if self.config.is_debug_mode:
                self.config.logger.debug(
                    f"Skipping verification for {leaf_path}: translation_degraded"
                )
            return value

        prefix, core, suffix = split_boundary_whitespace(value)

        # Verification matches glossary terms against the English source, not the translation.
        if is_verification and isinstance(original_data, str):
            glossary_string = build_glossary_string(original_data, glossary, self.config)
        else:
            glossary_string = build_glossary_string(value, glossary, self.config)

        # The proofreader gets the English source for alignment reference
        # — masked with its own independent tag_map, never unmasked back,
        # so a divergent [Tn] numbering from the translated value's own
        # masking (below) cannot corrupt either side.
        original_text_block = ""
        if is_verification and isinstance(original_data, str) and original_data.strip():
            masked_original, original_tag_map = mask_tags(original_data)
            original_text_block = ORIGINAL_TEXT_BLOCK_TEMPLATE.format(
                masked_original=masked_original
            )
            # Note about marker numbering only when the ORIGINAL block itself
            # carries markers; otherwise it is noise that can provoke
            # fabricated_tag / tag_marker_mismatch.
            if original_tag_map:
                original_text_block += ORIGINAL_TEXT_BLOCK_NOTE

        # LRU-cached and offline, so it is cheap to run before the cache lookup.
        linguistic_hints = self.context_analyzer.analyze(path, value)
        hints_text = "; ".join(hint.instruction for hint in linguistic_hints)

        prompt_template = runtime_config['prompt']

        enhanced_prompt = self._build_enhanced_prompt(
            base_prompt=prompt_template,
            hints=linguistic_hints,
            glossary_string=glossary_string,
            operation_type=operation_type,
            original_text_block=original_text_block
        )

        # The key hashes the whole request, so identical text with different
        # hints (`_male` / `_female`) or sampling never shares a cache/dedup entry.
        cache_manager = self.cache_manager
        cache_key: Optional[str] = None
        fingerprints: dict = {}
        if cache_manager is not None:
            sampling_identity = build_sampling_identity(runtime_config)
            fingerprints = cache_manager.compute_fingerprints(
                operation_type=operation_type,
                model=runtime_config['model'],
                system_prompt=runtime_config['system_prompt'],
                prompt_template=prompt_template,
                hints_text=hints_text,
                glossary_string=glossary_string,
                mode=mode,
                sampling_identity=sampling_identity,
            )
            cache_key = cache_manager.create_cache_key(
                operation_type=operation_type,
                model=runtime_config['model'],
                system_prompt=runtime_config['system_prompt'],
                enhanced_prompt=enhanced_prompt,
                mode=mode,
                value=value,
                sampling_identity=sampling_identity,
            )

        if self.cache_enabled and cache_manager is not None and cache_key is not None:
            cached_result = cache_manager.get(operation_type, cache_key)

            if cached_result is not None:
                if self.config.is_debug_mode:
                    self.config.logger.debug(f"Cache hit for {path}")

                with self._stats_lock:
                    self.stats['processed_leaves'] += 1

                # A (bool, str) verification tuple comes back from JSON as a list.
                if isinstance(cached_result, (tuple, list)):
                    if len(cached_result) >= 2:
                        sanitized_result = cached_result[1]
                    else:
                        self.config.logger.warning(
                            f"Malformed tuple in cache for {path}: {cached_result!r}. "
                            f"Expected (bool, str), got {len(cached_result)} elements."
                        )
                        sanitized_result = cached_result[0] if cached_result else value
                else:
                    sanitized_result = cached_result

                if not isinstance(sanitized_result, str):
                    self.config.logger.error(
                        f"Cache returned non-string value for {path}: "
                        f"type={type(sanitized_result).__name__}, value={sanitized_result!r}. "
                        f"Clearing corrupted cache entry and reprocessing."
                    )
                    try:
                        cache_manager.delete(operation_type, cache_key)
                    except Exception as delete_error:  # noqa: BLE001 - continue with a fresh model call
                        self.config.logger.debug(f"Failed to delete cache entry: {delete_error}")
                    # No return: fall through to a fresh model call.
                else:
                    outcomes.record(
                        leaf_path, operation_type, make_ok(value, sanitized_result, cache_hit=True)
                    )
                    self.config.usage_ledger.record_non_api(
                        operation_type, USAGE_NON_API_CACHE_HIT
                    )
                    return sanitized_result
            elif self.config.is_debug_mode:
                # Explains the miss only; the full key has already decided it.
                cache_manager.diagnose_miss(operation_type, value, fingerprints)

        if self.dedup_enabled and not is_verification and cache_key is not None:
            with self._dedup_lock:
                if cache_key in self._dedup_index:
                    _, translated = self._dedup_index[cache_key]

                    if translated is not None:
                        with self._stats_lock:
                            self.stats['deduplicated_leaves'] += 1

                            if self.stats['deduplicated_leaves'] % self.dedup_log_interval == 0:
                                self.config.logger.info(
                                    f"Deduplication: {self.stats['deduplicated_leaves']} "
                                    f"duplicate values reused"
                                )

                        outcomes.record(
                            leaf_path, operation_type, make_ok(value, translated, dedup=True)
                        )
                        self.config.usage_ledger.record_non_api(
                            operation_type, USAGE_NON_API_DEDUP
                        )
                        return translated

        enhanced_config = runtime_config.copy()
        enhanced_config['prompt'] = enhanced_prompt

        # Model sees the peeled core; outer whitespace is restored after accept.
        masked_text, tag_map = mask_tags(core)

        try:
            outcome = process_func(
                masked_text,
                self.config,
                enhanced_config,
                "",  # the glossary is already part of the prompt
                mode=mode
            )

            if not outcome.success:
                if self.config.is_debug_mode:
                    self.config.logger.warning(f"Model call failed for {path}: {outcome.failure_kind}")

                if self.config.circuit_breaker.record_failure(operation_type):
                    self._log_circuit_breaker_trip(operation_type)

                with self._stats_lock:
                    self.stats['failed_leaves'] += 1

                outcomes.record(
                    leaf_path, operation_type, make_degraded(value, outcome.failure_kind)
                )
                return value

            unmask_result = unmask_tags(outcome.text, tag_map, self.config)

            validation = (
                validate_model_reply(
                    core, outcome.text, unmask_result.text, tag_map,
                    unmask_result.missing_markers, unmask_result.orphaned_markers,
                    glossary, self.config,
                )
                if self.config.validation.enabled
                else ReplyValidationResult(True, None, None)
            )

            if validation.warning:
                self.config.logger.warning(f"{path}: {validation.warning}")

            if not validation.accepted:
                reason_text = validation.reason or ''
                if validation.detail:
                    reason_text = f"{reason_text}[{validation.detail}]"
                self.config.logger.warning(
                    f"Rejected model reply for {path}: {reason_text}. "
                    f"Reply: {self._preview_rejected_reply(outcome.text)}"
                )
                if self.config.validation.save_rejected_replies:
                    self._save_rejected_reply(path, operation_type, outcome.text)
                self.config.metrics.record_rejection(operation_type)

                if self.config.circuit_breaker.record_failure(operation_type):
                    self._log_circuit_breaker_trip(operation_type)

                with self._stats_lock:
                    self.stats['failed_leaves'] += 1

                error_kind = validation.reason or FAILURE_KIND_REJECTED
                outcomes.record(
                    leaf_path, operation_type, make_degraded(value, error_kind)
                )
                return value

            self.config.circuit_breaker.record_success(operation_type)
            unmasked_result = prefix + unmask_result.text + suffix

            if self.cache_enabled and cache_manager is not None and cache_key is not None:
                metadata = {
                    'original_key': [value, glossary_string],
                    'path': str(path),
                    'fingerprints': fingerprints,
                }

                # Only the string is cached; a tuple would turn into a list on disk.
                cache_manager.set(
                    operation_type, cache_key, unmasked_result, metadata, source_value=value
                )

                self._maybe_auto_save(operation_type)

            if self.dedup_enabled and not is_verification and cache_key is not None:
                with self._dedup_lock:
                    self._dedup_index[cache_key] = (path, unmasked_result)

            with self._stats_lock:
                self.stats['processed_leaves'] += 1

            if not isinstance(unmasked_result, str):
                self.config.logger.error(
                    f"Internal error: unmasked_result is not a string for {path}: "
                    f"type={type(unmasked_result).__name__}, value={unmasked_result!r}. "
                    f"This indicates a bug in unmask_tags() or API client. "
                    f"Using original value as fallback."
                )
                outcomes.record(
                    leaf_path, operation_type, make_degraded(value, 'internal_error')
                )
                return value

            outcomes.record(leaf_path, operation_type, make_ok(value, unmasked_result))
            return unmasked_result

        except Exception as e:  # noqa: BLE001 - leaf degrades to the original value (§4.3)
            self.config.logger.error(f"Error processing leaf at {path}: {e}")

            if self.config.is_debug_mode:
                self.config.logger.error("Full traceback:", include_traceback=True)

            # Same decision class as CallOutcome.success=False / validator reject:
            # unexpected process_func failures feed the consecutive-failure streak.
            if self.config.circuit_breaker.record_failure(operation_type):
                self._log_circuit_breaker_trip(operation_type)

            with self._stats_lock:
                self.stats['failed_leaves'] += 1

            outcomes.record(
                leaf_path, operation_type, make_degraded(value, type(e).__name__)
            )
            return value
    
    def _inject_original_text(self, prompt: str, original_text_block: str, operation_type: str) -> str:
        """
        Place the verification reference block, before any other marker is resolved.

        Runs first in _build_enhanced_prompt(), so the {LINGUISTIC_CONTEXT}/
        {GLOSSARY} scenarios below always see the result and keep anchoring
        themselves relative to it - that is what keeps the trailing
        "Text to translate:"/"Text to proofread:" label immediately followed
        by the leaf value (nothing else gets inserted after this step runs).
        Always a no-op for translation, since original_text_block is "" there.

        Replaced at {ORIGINAL_TEXT} if the prompt declares it; otherwise
        inserted before the sentence containing the operation's keyword
        ("verify" here - translation never reaches this branch), same anchor
        _find_sentence_start() gives the context/glossary fallback below; with
        no keyword either, appended at the end as a last resort.

        Args:
            prompt: base_prompt, not yet touched by any other marker.
            original_text_block: ORIGINAL_TEXT_BLOCK_TEMPLATE (+ optional note),
                or "".
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.

        Returns:
            Prompt with the reference block placed, or unchanged if there is
            neither a marker nor a block to insert.
        """
        if ORIGINAL_TEXT_MARKER in prompt:
            return self._replace_last_marker(prompt, ORIGINAL_TEXT_MARKER, original_text_block)
        if not original_text_block:
            return prompt

        keyword = "translate" if operation_type == OPERATION_TRANSLATION else "verify"
        keyword_pos = prompt.lower().rfind(keyword)
        if keyword_pos != -1:
            sentence_start = self._find_sentence_start(prompt, keyword_pos)
            return prompt[:sentence_start] + original_text_block + prompt[sentence_start:]
        return prompt + original_text_block

    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        hints: List[LinguisticHint],
        glossary_string: str,
        operation_type: str = OPERATION_TRANSLATION,
        original_text_block: str = ""
    ) -> str:
        """
        Inject linguistic hints, the glossary and (verification only) the
        original-text reference block into the prompt.

        {ORIGINAL_TEXT} (verification only) is resolved first
        (_inject_original_text()), independently of the scenarios below -
        it is orthogonal to the context/glossary pairing. Everything that
        follows then operates on the result of that step.

        Injection strategy for {LINGUISTIC_CONTEXT}/{GLOSSARY} (4 scenarios):
        1. Both markers present -> each replaced independently.
        2. Only {LINGUISTIC_CONTEXT} -> context at the marker, glossary right after it.
        3. Only {GLOSSARY} -> context before the marker, glossary at the marker.
        4. No markers -> fallback (insert before "Translate"/"Verify", or append).

        Rules for {LINGUISTIC_CONTEXT}/{GLOSSARY}:
        - Only the last occurrence of each marker is replaced.
        - Other occurrences are removed ("the clarification goes at the end").
        - An empty block is not inserted and its marker is stripped.

        Context block format: LINGUISTIC_CONTEXT_BLOCK_TEMPLATE
        (instructions joined as "<hint1>; <hint2>; <hint3>").

        Args:
            base_prompt: Base prompt from the configuration.
            hints: Hints, sorted by priority.
            glossary_string: Glossary definitions block.
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            original_text_block: ORIGINAL_TEXT_BLOCK_TEMPLATE (+ optional note),
                or "" (translation always passes "").

        Returns:
            The prompt with context, glossary and the original-text block injected.
        """
        base_prompt = self._inject_original_text(base_prompt, original_text_block, operation_type)

        if hints:
            hint_instructions = "; ".join(hint.instruction for hint in hints)
            context_block = LINGUISTIC_CONTEXT_BLOCK_TEMPLATE.format(
                instructions=hint_instructions
            )
        else:
            context_block = ""
        
        has_context_marker = LINGUISTIC_CONTEXT_MARKER in base_prompt
        has_glossary_marker = GLOSSARY_MARKER in base_prompt

        # === SCENARIO 1: both markers present ===
        if has_context_marker and has_glossary_marker:
            if self.config.is_debug_mode:
                context_pos = base_prompt.rfind(LINGUISTIC_CONTEXT_MARKER)
                glossary_pos = base_prompt.rfind(GLOSSARY_MARKER)

                if glossary_pos < context_pos:
                    self.config.logger.debug(
                        "WARNING: Unusual marker order detected: {GLOSSARY} appears before "
                        "{LINGUISTIC_CONTEXT}. Consider placing context hints before glossary "
                        "for better model comprehension."
                    )

                self.config.logger.debug("Prompt injection: using both markers")

            result = self._replace_last_marker(base_prompt, LINGUISTIC_CONTEXT_MARKER, context_block)
            result = self._replace_last_marker(result, GLOSSARY_MARKER, glossary_string)
            return result

        # === SCENARIO 2: context marker only ===
        if has_context_marker and not has_glossary_marker:
            if self.config.is_debug_mode:
                self.config.logger.debug("Prompt injection: using context marker, glossary after it")

            # Glossary goes right after the context.
            combined_block = context_block + glossary_string
            result = self._replace_last_marker(base_prompt, LINGUISTIC_CONTEXT_MARKER, combined_block)
            return result

        # === SCENARIO 3: glossary marker only ===
        if not has_context_marker and has_glossary_marker:
            if self.config.is_debug_mode:
                self.config.logger.debug("Prompt injection: using glossary marker, context before it")

            # Context goes right before the glossary.
            combined_block = context_block + glossary_string
            result = self._replace_last_marker(base_prompt, GLOSSARY_MARKER, combined_block)
            return result

        # === SCENARIO 4: no markers (fallback) ===

        # Look for the operation's keyword from the end (case-insensitive).
        keyword = "translate" if operation_type == OPERATION_TRANSLATION else "verify"
        keyword_pos = base_prompt.lower().rfind(keyword)

        if keyword_pos != -1:
            sentence_start = self._find_sentence_start(base_prompt, keyword_pos)

            if self.config.is_debug_mode:
                self.config.logger.debug(
                    f"Prompt injection: detected '{keyword}' at position {keyword_pos}, "
                    f"sentence starts at {sentence_start} (operation: {operation_type})"
                )

            # Insert context + glossary before the sentence with the keyword.
            injection = context_block + glossary_string
            result = base_prompt[:sentence_start] + injection + base_prompt[sentence_start:]
            return result

        # Full fallback: no markers and no keyword found.
        if self.config.is_debug_mode:
            self.config.logger.debug(
                "Prompt injection: fallback mode (no markers or 'Translate' found), "
                "appending to end with explicit marker"
            )

        result = base_prompt

        if context_block or glossary_string:
            result += "\n\n"  # Visual separation

            if context_block:
                result += context_block

            result += glossary_string
            result += TRANSLATION_TEXT_MARKER if operation_type == OPERATION_TRANSLATION else VERIFICATION_TEXT_MARKER

        return result
    
    def _replace_last_marker(self, text: str, marker: str, replacement: str) -> str:
        """
        Replace the last occurrence of ``marker`` with ``replacement``;
        remove every earlier occurrence.

        Thread-safe: operates only on immutable strings.

        Args:
            text: Source text.
            marker: Marker to replace.
            replacement: Replacement string (may be empty).

        Returns:
            Text with the last marker replaced and the rest removed.
        """
        if not replacement:
            return text.replace(marker, "")

        last_pos = text.rfind(marker)

        if last_pos == -1:
            return text

        text_without_markers = text.replace(marker, "")

        # Adjust the insert index for markers removed before last_pos.
        markers_before = text[:last_pos].count(marker)
        adjusted_pos = last_pos - (markers_before * len(marker))

        return (
            text_without_markers[:adjusted_pos]
            + replacement
            + text_without_markers[adjusted_pos:]
        )

    def _find_sentence_start(self, text: str, position: int) -> int:
        """
        Find the start of the sentence that contains ``position``.

        Searches backward for the nearest delimiter: a double newline, a
        single newline, ``". "``, or the start of the text.

        Thread-safe: operates only on immutable strings.

        Args:
            text: Text to search.
            position: Offset inside the sentence.

        Returns:
            Index of the sentence start.
        """
        double_newline = text.rfind("\n\n", 0, position)
        if double_newline != -1:
            return double_newline + 2  # Skip past \n\n.

        single_newline = text.rfind("\n", 0, position)
        if single_newline != -1:
            return single_newline + 1  # Skip past \n.

        dot_space = text.rfind(". ", 0, position)
        if dot_space != -1:
            return dot_space + 2  # Skip past ". ".

        return 0

    def _maybe_auto_save(self, operation_type: str) -> None:
        """
        Flush the cache to disk every ``cache_auto_save_interval`` new
        entries so a hard process kill (SIGKILL, BSOD, power loss) does
        not erase already paid-for translations.

        Counts only leaves that added a new cache entry: called from
        ``_process_leaf`` right after ``cache_manager.set()``, not on every
        processed leaf — a cache hit does not change the on-disk file and
        needs no flush. ``interval <= 0`` disables the mechanism.

        The write itself is ordered against reordering inside
        ``CacheManager.save()`` (see ``_CacheWriteCoordinator``) — this
        method must not add its own write lock.

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION
                — which cache to save.
        """
        interval = self.config.cache_auto_save_interval
        if interval <= 0:
            return

        with self._autosave_lock:
            self._leaves_since_save += 1
            should_save = self._leaves_since_save >= interval
            if should_save:
                self._leaves_since_save = 0

        if should_save:
            if self.config.is_debug_mode:
                self.config.logger.debug(
                    f"Auto-save: {interval} new {operation_type} cache entries "
                    f"since last save, flushing to disk"
                )
            self.config.cache_manager.save(operation_type)

    def _log_circuit_breaker_trip(self, operation_type: str) -> None:
        """
        Log that the consecutive-failure circuit breaker just tripped.

        Called only on the one record_failure() call that returns True for
        operation_type - every later failure is silent, and traverse()
        already stops recursing into that operation's remaining nodes.

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
        """
        self.config.logger.error(
            f"Circuit breaker tripped for {operation_type}: consecutive failure "
            f"threshold reached ([resilience].max_consecutive_failures). "
            f"Aborting the stage; already processed leaves are kept."
        )

    @staticmethod
    def _preview_rejected_reply(masked_reply: str) -> str:
        """
        Format a rejected model reply for the WARNING log line.

        repr() surfaces literal newlines and other invisible characters
        that make the reply worth logging in the first place (the rejection
        reason alone does not show what the model actually wrote); masked_reply
        is the model's raw text, still carrying [Tn] markers, not the
        unmasked result - the markers are part of what the validator checked.

        Args:
            masked_reply: outcome.text of the rejected CallOutcome.

        Returns:
            repr() of the reply, truncated to REJECTED_REPLY_LOG_LENGTH.
        """
        truncated = len(masked_reply) > REJECTED_REPLY_LOG_LENGTH
        preview = repr(masked_reply[:REJECTED_REPLY_LOG_LENGTH])
        return preview + '...' if truncated else preview

    @staticmethod
    def _sanitize_rejected_reply_filename(path: str) -> str:
        """
        Turn a leaf path into a single path-segment filename.

        Args:
            path: Human-readable KeyPath string (e.g. ``ui.title``).

        Returns:
            A ``.txt`` filename safe for Windows and POSIX.
        """
        sanitized = path
        for char in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', '[', ']'):
            sanitized = sanitized.replace(char, '_')
        return f"{sanitized}.txt"

    def _save_rejected_reply(self, path: KeyPath, operation_type: str, masked_reply: str) -> None:
        """
        Persist the full masked reply under ``[logging].dir/rejected/``.

        Opt-in via ``[validation].save_rejected_replies``. Failures are logged
        and swallowed so rejection handling never raises from diagnostics I/O.

        Args:
            path: Leaf KeyPath used for the filename.
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            masked_reply: outcome.text of the rejected CallOutcome.
        """
        file_ops = self.config.file_ops
        log_dir = self.config.log_dir
        if file_ops is None or not log_dir:
            self.config.logger.warning(
                f"Cannot save rejected reply for {path}: log_dir or file_ops unavailable"
            )
            return

        target = os.path.join(
            log_dir,
            REJECTED_REPLIES_SUBDIR,
            operation_type,
            self._sanitize_rejected_reply_filename(str(path)),
        )
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            safe_write_file(
                target, masked_reply, f"rejected reply for {path}",
                file_ops, self.config.logger,
            )
        except FileOperationError as exc:
            self.config.logger.warning(
                f"Failed to save rejected reply for {path}: {exc}"
            )
        except OSError as exc:
            self.config.logger.warning(
                f"Failed to save rejected reply for {path}: {exc}"
            )

    def get_stats_summary(self) -> dict:
        """
        Return a thread-safe shallow copy of processing stats.

        Returns:
            Dict of processing metrics.
        """
        with self._stats_lock:
            return self.stats.copy()

def process_stage_recursive(
    output_path: str,
    stage_name: str,
    input_data: Any,
    original_data: Any,
    process_func: Callable,
    is_verification: bool,
    config: 'ProcessorConfigProtocol',
    glossary: dict,
    config_prefix: str,
    runtime_config: dict,
    heartbeat: Optional['HeartbeatProgressProtocol'] = None,
) -> Any:
    """
    Run one pipeline stage over a JSON tree and write the stage output file.

    Args:
        output_path: Where the stage output file is written; its directory must exist.
        stage_name: STAGE_TRANSLATED or STAGE_REDACTED.
        input_data: Tree to process (any JSON value).
        original_data: English source tree; glossary terms of the
            verification stage are matched against it.
        process_func: Leaf function (translate_key / verify_key).
        is_verification: True on the verification stage.
        config: Processor-facing configuration.
        glossary: Glossary {term: translation}.
        config_prefix: Cache type whose statistics are printed
            (OPERATION_TRANSLATION / OPERATION_VERIFICATION).
        runtime_config: Flat API parameters, including 'mode'.
        heartbeat: Optional progress writer for begin_stage / note_leaf.

    Returns:
        The processed tree, with the structure of input_data.
    """
    current_mode = runtime_config.get('mode', MODE_STYLE)

    config.logger.info(f"Starting {stage_name} stage (recursive mode)...")

    context_analyzer = LinguisticContext(config)
    processor = RecursiveProcessor(config, context_analyzer, heartbeat=heartbeat)

    if config.value_deduplication_enabled and not is_verification:
        processor.build_dedup_index(input_data)
    elif is_verification and config.is_debug_mode:
        config.logger.debug("Deduplication disabled for verification stage")

    if heartbeat is not None:
        heartbeat.begin_stage(
            config_prefix,
            count_string_leaves(input_data, config.max_recursion_depth),
        )

    start_time = time.time()

    result_data = processor.traverse(
        data=input_data,
        current_path=KeyPath(segments=()),
        process_func=process_func,
        glossary=glossary,
        runtime_config=runtime_config,
        mode=current_mode,
        original_data=original_data,
        is_verification=is_verification,
        current_depth=0
    )

    end_time = time.time()

    if safe_write_file(output_path, result_data, f"{stage_name} output", config.file_ops, config.logger):
        config.logger.info(f"Saved {stage_name} results to {output_path}")
    else:
        config.logger.error(f"Failed to save {stage_name} results")

    _print_recursive_statistics(
        stage_name=stage_name,
        stats=processor.get_stats_summary(),
        start_time=start_time,
        end_time=end_time,
        config=config
    )

    config.cache_manager.print_stats(config_prefix)

    unknown_patterns = context_analyzer.get_unknown_patterns()
    if unknown_patterns:
        config.logger.warning(
            f"Encountered {len(unknown_patterns)} unknown key pattern(s) during processing. "
            f"Review WARNING logs for details and consider extending pattern detection."
        )

        if config.is_debug_mode:
            config.logger.debug(
                f"Unknown patterns: {', '.join(list(unknown_patterns)[:config.max_error_display])}")

    return result_data

def _print_recursive_statistics(
    stage_name: str,
    stats: dict,
    start_time: float,
    end_time: float,
    config: 'ProcessorConfigProtocol'
):
    """
    Log the traversal and leaf statistics of a finished stage.

    Args:
        stage_name: Stage name.
        stats: Snapshot from RecursiveProcessor.get_stats_summary().
        start_time: Stage start, time.time().
        end_time: Stage end, time.time().
        config: Supplies the logger and the log level.
    """
    if config.log_level not in STATISTICS_LOG_LEVELS:
        return

    duration = end_time - start_time

    config.logger.info(f"{stage_name.capitalize()} stage completed in {duration:.2f} seconds")

    config.logger.info(
        f"Tree statistics:\n"
        f"  Total nodes traversed: {stats['total_nodes']}\n"
        f"  - Dictionary nodes: {stats['dict_nodes']}\n"
        f"  - List nodes: {stats['list_nodes']}\n"
        f"  - Leaf nodes (strings): {stats['leaf_nodes']}\n"
        f"  Maximum depth reached: {stats['max_depth_reached']}"
    )

    config.logger.info(
        f"Leaf processing:\n"
        f"  Processed: {stats['processed_leaves']}\n"
        f"  Skipped (empty/format): {stats['skipped_leaves']}\n"
        f"  Deduplicated: {stats['deduplicated_leaves']}\n"
        f"  Failed: {stats['failed_leaves']}"
    )

    if stats['processed_leaves'] > 0 and duration > 0:
        avg_time = duration / stats['processed_leaves']
        speed = stats['processed_leaves'] / duration * 60

        config.logger.info(
            f"Performance:\n"
            f"  Average time per leaf: {avg_time:.4f} seconds\n"
            f"  Processing speed: {speed:.2f} leaves/minute"
        )

    if stats['failed_leaves'] > 0:
        failure_rate = stats['failed_leaves'] / stats['leaf_nodes'] * 100
        config.logger.warning(
            f"Warning: {stats['failed_leaves']} leaf(ves) failed processing ({failure_rate:.2f}%)"
        )
