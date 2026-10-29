"""The one test that keeps removed code removed (#540, J1).

Removing something later = add a line to ``REMOVED`` below. Do not write a new "this name is gone" test.

Each entry names what was removed, why, and the commit or issue that removed it, and says where to look:

* ``kind="text"``: the text must not contain the name anywhere (code, comments and docstrings alike). A plain string
  is a substring, so ``cost_period`` also catches ``cost_period_v2``; set ``whole_word=True`` for a short name that
  other words contain. A ``re.Pattern`` is searched as given.
* ``kind="code"``: the name must not be used as a code identifier (a name, an attribute, a def or class, an import, a
  parameter or keyword argument) or equal a string literal. Prose that explains the history (a comment, a docstring)
  is not looked at. Use this when the module's own comments legitimately say what used to be there.
* ``kind="file"``: the file must not exist (a removed module). ``names`` are repository-relative paths.

``scope`` is the repository-relative files or directories scanned (default: all of ``brain/``). ``allow`` lists files
inside the scope where the name may appear (a migration that deletes old keys by name, for instance). A name that
other modules could reasonably reuse (``_rewrite``, ``_README``) gets a one-file scope. ``ignore_case`` applies to
``text`` entries only. A scope that no longer exists fails the test (a typo would otherwise scan nothing): delete the
entry then.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.source_tree import REPO, tree_asts, tree_sources

_SUFFIXES = (".py", ".toml", ".json", ".ts", ".tsx", ".rs")
_ROOTS = ("brain", "app/src")
_KINDS = ("text", "code", "file")


@dataclass(frozen=True)
class Removed:
    names: tuple[str | re.Pattern[str], ...]
    why: str
    ref: str
    kind: str = "text"
    scope: tuple[str, ...] = ("brain",)
    allow: tuple[str, ...] = ()
    ignore_case: bool = False
    whole_word: bool = False


_THINKING_REF = "9dd1320c bump to v0.0.26 (inner monologue, tool-call architecture); 7964e111 finish Phase A cleanup + add regression gate"
_JUDGE_REF = "ba2ca98a refactor(judge_selftune): collapse to one evolving model per persona (F2c inc7)"
_ATOMIC_REF = "e64cabec Move the identity key, judge pointer, legacy copy, blob, plist and unit-refresh writers onto the shared writer (#465)"
_FLOOR_REF = "0c4e7b23 fold pre-flip revision F2a fixes into #274"
_ST_REF = "108bb6c5 fix(recall): per-message rerank width, always-merged keyword search, known-names list (#321)"
_TIER_REF = "44af2447 Choose each job's model with one line in the user tunables file"
_IDLE_REF = "1f69f189 feat(ram-spike-fix): one shared chat-idle lull + is_chat_idle gate for every background caller (INC-6)"
_DEVCONST_REF = "#531 shared-settings guard, the constants move into brain/dev_constants.py (draft PR #553)"
_GUARD_REF = "#531 shared-settings guard (draft PR #553)"
_JSON_REF = "b826f01b refactor(brain): one shared JSON reader/writer for every JSON file read and write (#439)"

# Every public writer and helper of brain/utils/atomic_write.py except save_with_backup_text (attempt_heal imports that).
_MOVED_WRITERS = (
    "_REPLACE_RETRIES",
    "_REPLACE_RETRY_SLEEP_S",
    "_retry_perm",
    "_replace_with_retry",
    "_copy_for_backup",
    "_INFLIGHT",
    "_INFLIGHT_LOCK",
    "_owner_alive",
    "_remove_stale_temps",
    "_temp_scope",
    "_save_payload",
    "fsync_dir",
    "_write_temp_file",
    "save_bytes_atomic",
    "save_blob_once",
)
# A route through attempt_heal to ANY writer of the shared module, save_with_backup_text included, is a shim.
_MOVED_ALTERNATION = "|".join(re.escape(n) for n in (*_MOVED_WRITERS, "save_with_backup_text"))

REMOVED: tuple[Removed, ...] = (
    # --- extended thinking (v0.0.26) ---------------------------------------------------------------------------------
    Removed(
        names=(
            "thinking_budget_tokens",
            "--thinking",
            "--budget-tokens",
            "_write_thinking_log",
            "thinking_log.jsonl",
            "thinking_blocks",
            "setPersonaThinking",
            "/persona/config/thinking",
        ),
        why="extended-thinking plumbing, removed because the CLI never surfaces thinking blocks (spec 2026-05-30-inner-monologue-tool-call-design.md section 6)",
        ref=_THINKING_REF,
        scope=("brain", "app/src"),
    ),
    Removed(
        names=("thinking_log",),
        why="the health walker no longer reads the thinking log",
        ref=_THINKING_REF,
        scope=("brain/health/walker.py",),
    ),
    # --- voice reflection, pass 2, reply scaffold -----------------------------------------------------------------------
    Removed(
        names=("_read_recent_message_tones",),
        why="the message-tone placeholder was retired when voice reflection got real evidence text",
        ref="a20d82ba fix(voice): feed voice reflection real evidence text; retire the tone placeholder",
    ),
    Removed(
        names=("_ensure_worker", "_worker_loop", "_worker_thread", "_worker_inhibited"),
        why="the pass-2 worker thread is gone; pass 2 runs as a job of the central cadence function",
        ref="e1cd5047 Run gated background jobs from one central cadence function",
        scope=("brain/chat/pass2_queue.py",),
    ),
    Removed(
        names=("reply_scaffold", "strip_scaffold_tail"),
        why="the filter that stripped leaked scaffold tokens off the end of a reply was reverted",
        ref="99997bd7 revert the tail-stripping filter for leaked scaffold tokens (#227)",
    ),
    Removed(
        names=("brain/chat/reply_scaffold.py",),
        why="the scaffold filter module was reverted",
        ref="99997bd7 (#227)",
        kind="file",
    ),
    Removed(
        names=("_claude_context_jsonl_lines",),
        why="chat history renders as narrative lines; the JSONL context renderer is gone",
        ref="5e26300d render chat history as narrative lines instead of JSONL (#227)",
    ),
    # --- cost series, floor calibration, judge ---------------------------------------------------------------------------
    Removed(
        names=("cost_period",),
        why="S146 removed the per-period cost series and its tick hook",
        ref=_ST_REF,
    ),
    Removed(
        names=("brain/memory/cost_period.py",),
        why="S146 removed the per-period cost series module",
        ref=_ST_REF,
        kind="file",
    ),
    Removed(
        names=("start_new_period",),
        why="S146 removed the per-period reset at the compaction tick",
        ref=_ST_REF,
    ),
    Removed(
        names=("BACKOFF",),
        why="S85 (revised): the cosine bootstrap's retry rule has no time constants",
        ref="named in test_cosine_bootstrap_job.py (no commit found)",
        scope=("brain/memory/floor_calibration.py",),
        ignore_case=True,
    ),
    Removed(
        names=(
            "ema_update",
            "stability_gate_accepts",
            "_bootstrap_ci",
            "FLOOR_EMA_WINDOW_DAYS",
            "FLOOR_STABILITY_CI",
            "FLOOR_STABILITY_BOOTSTRAP_ITERATIONS",
            "_worst_case_daily_labeled_pairs",
        ),
        why="the floor's EMA layer and bootstrap-CI stability gate were removed entirely (Change 1); the floor is the raw fit",
        ref=_FLOOR_REF,
        kind="code",
        scope=("brain/memory/floor_calibration.py",),
    ),
    Removed(
        names=(
            "full_champion_dir",
            "resolve_serving_tuned_judge",
            "LoraAdapterJudge",
            "load_lora_scorer",
            "save_adapter_dir",
            "sibling_root",
        ),
        why="the adapter-era judge store: one evolving checkpoint per persona replaced per-tier adapters (F2c inc7)",
        ref=_JUDGE_REF,
    ),
    Removed(
        names=("load_adapter",),
        why="a judge adapter is never reloaded: weights are merged and stored as a plain checkpoint (F2c inc7, M4)",
        ref=_JUDGE_REF,
        kind="code",
    ),
    Removed(
        names=("adapter_dir",),
        why="build_judge_provider and label_calibration_sample no longer take an adapter directory (F2c inc7)",
        ref=_JUDGE_REF,
        kind="code",
        scope=("brain/memory",),
    ),
    Removed(
        names=(re.compile(r"PeftModel\s*\.\s*from_pretrained\s*\("),),
        why="a judge adapter is never reloaded from disk (F2c inc7, M4)",
        ref=_JUDGE_REF,
    ),
    Removed(
        names=(re.compile(r"noqa:\s*[\w, ]*\bARG001\b"),),
        why="_keep_after_accept lost its unused parameter, so the unused-argument suppression went too (F2c inc9)",
        ref="499b637c fix(judge_selftune): bind the knob to its checkpoint and close inc7/inc8 review findings (F2c inc9)",
        scope=("brain/memory/judge_selftune.py",),
    ),
    Removed(
        names=("lora_available", "_downgrade_for_missing_lora_extra", "_F2C_TRAINING_EXTRA_HINT"),
        why="peft and datasets are base dependencies now, so the optional-extra plumbing (inc5a) is gone",
        ref="f7c5f359 feat(judge_selftune): wire the LoRA/full-FT weight-retrain tick lifecycle (F2c inc5b-2)",
    ),
    Removed(
        names=("_Deadline", "interrupt", "deadline"),
        why="the name-budget search no longer interrupts a statement or tracks a deadline (CS8b)",
        ref=_ST_REF,
        scope=("brain/memory/store.py",),
        ignore_case=True,
    ),
    Removed(
        names=("threading",),
        why="the memory store holds no thread machinery after the name-budget rework (CS8b)",
        ref=_ST_REF,
        kind="code",
        scope=("brain/memory/store.py",),
    ),
    # --- RAM figure, rerank width, precision self-check ---------------------------------------------------------------------
    Removed(
        names=(
            "ram_bytes_per_token",
            "peak_padded_tokens",
            "ram_figure_kept",
            "_RamState",
            "_ram_state",
            "_ram_kept",
            "_note_ram_figure",
            "_full_pair_tokens",
            "_ram_figure_would_refuse",
            "_ram_figure_refusal",
            "_pair_ran_unrecorded",
            "THROUGH_ORIGIN_SERIES",
            "RAM_SERIES",
            "_ram_cost",
            "_current_rss_bytes",
            "_rss_delta",
            "rss_delta_bytes",
            "peak_x",
        ),
        why="S220 H10: the per-token RAM figure and everything built around it were replaced by the RAM fit",
        ref=_ST_REF,
        whole_word=True,
    ),
    Removed(
        names=(
            "get_rerank_width",
            "_latency_cache",
            "_memory_cache",
            "_warm_per_doc_latency",
            "_warm_per_doc_memory",
            "_measure_warm_per_doc_latency",
            "_measure_warm_per_doc_memory",
            "_LATENCY_RECOMPUTE_INTERVAL_SECONDS",
            "_MEMORY_RECOMPUTE_INTERVAL_SECONDS",
            "_MEASURE_RERANKS",
            "CALIBRATION_SAMPLE_SIZE",
        ),
        why="the hourly rerank-width measurement was replaced by a per-message width",
        ref=_ST_REF,
        kind="code",
        scope=("brain/memory/reranker.py",),
    ),
    Removed(
        names=(
            "_precision_decision_cache",
            "_choose_reranker_model_id",
            "_run_precision_selfcheck",
            "reset_precision_decision_for_floor_change",
            "agreement bar",
        ),
        why="the first-use precision self-check was removed; fp16 is the one quantisation lever (AC4)",
        ref=_FLOOR_REF,
        scope=("brain/memory/reranker.py", "brain/bridge/supervisor.py"),
    ),
    Removed(
        names=("int8",),
        why="int8 quantisation was dropped: a 278M model has less redundancy to absorb its accuracy loss (AC3, owner catch)",
        ref=_FLOOR_REF,
        scope=("brain/memory/reranker.py", "brain/bridge/model_tier.py"),
        ignore_case=True,
    ),
    Removed(
        names=("inactive_file",),
        why="inactive_file is not the reclaimable page-cache term here (G12); the formula uses the file counters",
        ref=_ST_REF,
    ),
    # --- model tiers, name rules ---------------------------------------------------------------------------------------------
    # The other removed model-tier names are swept, code and comments, everywhere, by
    # tests/unit/brain/bridge/test_model_tier.py::test_removed_names_exist_nowhere_in_code_or_comments.
    Removed(
        names=("is_valid_model", "KNOWN_MODELS"),
        why="persona_config holds no model name; each job's model comes from the tunables file",
        ref=_TIER_REF,
        kind="code",
        scope=("brain/persona_config.py",),
    ),
    Removed(
        names=("is_sentence_start",),
        why="S151 superseded S104/S121: no position rule is left in the shared name definition",
        ref="named in test_name_rules.py (no commit found)",
        kind="code",
        scope=("brain/memory/name_rules.py",),
    ),
    # --- ingest buffer, idle gate, images ---------------------------------------------------------------------------------
    Removed(
        names=("_fsync_dir",),
        why="the buffer's private directory fsync moved to the shared atomic writer (#465)",
        ref=_ATOMIC_REF,
        scope=("brain/ingest/buffer.py",),
    ),
    Removed(
        names=(
            "throttle.background_min_idle_seconds",
            "chat.pass2_min_idle_seconds",
            "self_model.articulate_min_idle_seconds",
        ),
        why="the per-caller idle tunables were retired for one shared lull key, chat.idle_lull_seconds (ram-spike-fix INC-6)",
        ref=_IDLE_REF,
        kind="code",
        allow=("brain/tunables_migration.py",),
    ),
    Removed(
        names=("_ACTIVE_CHAT_IDLE_MINUTES",),
        why="the idle-minutes constant was retired with the shared chat-idle lull (INC-6)",
        ref=_IDLE_REF,
        kind="code",
    ),
    Removed(
        names=("silence_minutes",),
        why="run_folded's silence_minutes kwarg was retired with the shared lull; the server forwards none (INC-6)",
        ref=_IDLE_REF,
        kind="code",
        scope=("brain/bridge/server.py",),
    ),
    Removed(
        names=("busy_timeout = 5000",),
        why="the memories.db open sequence and the other stores' busy timeout come from brain/utils/sqlite_open.py and dev_constants, not a written-out 5000; brain/initiate/resonance.py still has its own literal and a store is added here in the commit that migrates it (#465 C9)",
        ref="25004f83 refactor: open WorksStore through the shared SQLite helper (#465 C9)",
        scope=(
            "brain/memory/hebbian.py",
            "brain/soul/store.py",
            "brain/kindled_link/store.py",
            "brain/memory/store.py",
            "brain/works/store.py",
            "brain/memory/embedding_matrix.py",
        ),
    ),
    Removed(
        names=("_chat_with_images", "_message_has_image", "_build_stream_json_user_message"),
        why="the image/text builder fork was dissolved; shared files route through read_file (P0 image-tool route)",
        ref="634cf0b3 feat(files): P0 image-tool route",
    ),
    Removed(
        names=("ImageBlock", "if image_shas", "image_shas=", "build_system_message("),
        why="the engine no longer forks on per-turn images; every turn takes the static plus volatile prompt split (P0, C6)",
        ref="634cf0b3 feat(files): P0 image-tool route",
        scope=("brain/chat/engine.py",),
    ),
    # --- name-extraction wording, toml, tunables ----------------------------------------------------------------------------
    Removed(
        names=("placeholder",),
        why="every placeholder in the name-extraction wording was replaced by the approved text (S107)",
        ref="named in test_name_extraction_placeholders.py (no commit found)",
        scope=(
            "brain/tools/schemas.py",
            "brain/tools/impls/add_known_name.py",
            "brain/tools/impls/remove_known_name.py",
            "brain/chat/tool_recruit.py",
        ),
        ignore_case=True,
    ),
    Removed(
        names=(re.compile(r"(?is)\[engines\.consolidation\](?:(?!\n\[).)*?placeholder"),),
        why="the consolidation prompts carry the approved text, no placeholder (S107)",
        ref="named in test_name_extraction_placeholders.py (no commit found)",
        scope=("brain/prompt_strings.toml",),
    ),
    Removed(
        names=("PLACEHOLDER",),
        why="the last wording placeholder in the app's job labels was replaced (wording file section 8)",
        ref="9907316a Set owner wording/layout/colour for the NellFace background-jobs line",
        scope=("app/src/App.tsx",),
    ),
    Removed(
        names=(re.compile(r"""(?i)["'](tomli|tomli[-_]w|tomlkit|toml)(?![\w.\-/])"""),),
        why="reading and writing toml goes through brain/toml_io.py on the standard library; no toml dependency is added",
        ref="named in test_toml_io.py (no commit found)",
        scope=("pyproject.toml",),
    ),
    Removed(
        names=("set_overrides", "get_override", "overrides_mtime", "write_defaults_section", "_README", "_overrides_is_object",
               "_load_overrides_locked", "_record_heal", "_same_content", "quarantine_damaged"),
        why="tunables moved from a JSON overrides file to the toml file; the JSON file's helpers and its heal notices are gone",
        ref=_TIER_REF,
        scope=("brain/tunables.py",),
    ),
    Removed(
        names=("migrate_idle_keys", "_read_state", "_rewrite", "_BACKUP_SUFFIX"),
        why="the idle-key migration was rewritten as one pre-check plus a locked rewrite; its old helpers are gone",
        ref=_TIER_REF,
        scope=("brain/tunables_migration.py",),
    ),
    # --- json io, atomic writer ---------------------------------------------------------------------------------------------
    Removed(
        names=("attempt_heal", "_heal_from_baks", "_load_and_validate", "__all__"),
        why="the load-and-heal functions were replaced by the shared JSON reader (#439); attempt_heal_text and save_with_backup stay, with no __all__ shim",
        ref=_JSON_REF,
        kind="code",
        scope=("brain/health/attempt_heal.py",),
    ),
    Removed(
        names=("read_jsonl_skipping_corrupt", "iter_jsonl_streaming", "read_last_n_jsonl_lines"),
        why="the jsonl readers were replaced by iter_jsonl_skipping_corrupt and the shared reader (#439)",
        ref=_JSON_REF,
        kind="code",
        scope=("brain/health/jsonl_reader.py",),
    ),
    Removed(
        names=("mkstemp",),
        why="every atomic write names its own temp through the shared writer; no mkstemp (#465, G12)",
        ref=_ATOMIC_REF,
    ),
    Removed(
        names=("O_DIRECTORY",),
        why="the directory-fsync flag is named only in the shared writer (#465, G3b)",
        ref=_ATOMIC_REF,
        kind="code",
        allow=("brain/utils/atomic_write.py",),
    ),
    Removed(
        names=("os.replace", "uuid", ".new"),
        why="the blob modules write through save_blob_once: no replace, no uuid and no new temp name of their own (#465, G4a)",
        ref=_ATOMIC_REF,
        scope=("brain/images.py", "brain/file_store.py"),
    ),
    Removed(
        names=_MOVED_WRITERS,
        why="the writers live in brain/utils/atomic_write.py; attempt_heal holds none of them (#465, G20a)",
        ref=_ATOMIC_REF,
        kind="code",
        scope=("brain/health/attempt_heal.py",),
    ),
    Removed(
        names=(re.compile(r"import\s+\*"),),
        why="attempt_heal re-exports nothing by star import: no shim for the moved writers (#465, G20a)",
        ref=_ATOMIC_REF,
        scope=("brain/health/attempt_heal.py",),
    ),
    Removed(
        names=(
            re.compile(rf"attempt_heal\s*\.\s*({_MOVED_ALTERNATION})\b"),
            re.compile(rf"from\s+brain\.health\.attempt_heal\s+import\s+[^(\n]*\b({_MOVED_ALTERNATION})\b"),
            re.compile(rf"from\s+brain\.health\.attempt_heal\s+import\s*\([^)]*\b({_MOVED_ALTERNATION})\b"),
            re.compile(rf"from\s+brain\.health\.attempt_heal\s+import\s+[^(\n]*\\\s*\n[^\n]*\b({_MOVED_ALTERNATION})\b"),
            re.compile(r"from\s+brain\.health\.attempt_heal\s+import\s+\*"),
            # An alias would hide the dotted route above: attempt_heal is imported under its own name or not at all.
            re.compile(r"import\s+brain\.health\.attempt_heal\s+as\b"),
            re.compile(r"from\s+brain\.health\s+import\s+[^\n]*\battempt_heal\s+as\b"),
        ),
        why="no module reaches a moved writer through attempt_heal: no import of one, no dotted name, no star import, no alias of the module (#465, G20b)",
        ref=_ATOMIC_REF,
    ),
    # --- shared constants moved to brain/dev_constants.py (#531) ----------------------------------------------------------
    # Only names that CHANGED are listed. A constant that kept its name (FADE_THRESHOLD, MAX_ARC_MEMBERS, ...) is the same
    # identifier in its new home, so a name scan cannot tell the old definition from the new read.
    Removed(
        names=("brain/kindled_link/limits.py",),
        why="the peer-link caps and disclosure-budget constants live in brain/dev_constants.py under a KINDLED_LINK_ prefix; the constants-only module is gone, no shim",
        ref=_DEVCONST_REF,
        kind="file",
    ),
    Removed(
        names=(
            "MIN_OUTBOUND_GAP_SECONDS", "SESSION_MSG_CAP", "SESSION_COOLDOWN_HOURS", "DAILY_OUTBOUND_CAP", "DAILY_PROVIDER_CAP",
            "BUDGET_MAX", "BUDGET_REFILL_PER_DAY", "BUDGET_TIGHTEN_THRESHOLD", "MIN_SEND_DEBIT", "BUDGET_DEPLETED_THRESHOLD",
            "PEER_EMOTION_WINDOW_CAP", "PEER_EMOTION_WINDOW_HOURS", "INBOUND_FLOOD_CAP",
            "_MIN_OUTBOUND_GAP_SECONDS", "_SESSION_MSG_CAP", "_SESSION_COOLDOWN_HOURS", "_DAILY_OUTBOUND_CAP", "_DAILY_PROVIDER_CAP",
        ),
        why="the peer-link constants are read as dev_constants.KINDLED_LINK_*; no local alias of a constant (#531)",
        ref=_DEVCONST_REF,
        kind="code",
        scope=("brain/kindled_link",),
    ),
    Removed(
        names=("SCHEMA_VERSION", "MATURITY_FORMING_MIN", "MATURITY_KNOWN_MIN", "MATURITY_FALSIFIED_MAX", "ADDRESS_COOLDOWN_HOURS", "DAILY_BUDGET_DEFAULT"),
        why="the attunement constants live in brain/dev_constants.py under an ATTUNEMENT_ prefix (#531)",
        ref=_DEVCONST_REF,
        kind="code",
        scope=("brain/attunement",),
    ),
    Removed(
        names=("TUNE_GRADE_KNOB_REFIT", "TUNE_GRADE_LORA", "TUNE_GRADE_FULL_FT", "_TUNE_GRADE_ORDER"),
        why="the judge self-tune grades and their order live in brain/dev_constants.py as JUDGE_TUNE_GRADE_* (#531)",
        ref=_DEVCONST_REF,
        kind="code",
        scope=("brain/memory", "brain/bridge"),
    ),
    Removed(
        names=("tests/unit/brain/bridge/test_model_guard.py",),
        why="the model guard is the model case of the shared-settings guard (tests/shared_settings.py and tests/unit/test_shared_settings_guard.py); one guard, not two",
        ref=_GUARD_REF,
        kind="file",
    ),
    Removed(
        names=("JUDGE_TUNE_GATE_HANDFUL_DECISIONS",),
        why="a second name for dev_constants.JUDGE_SELFTUNE_GATE_HANDFUL_DECISIONS in judge_selftune.py; readers use the dev constant",
        ref=_GUARD_REF,
        scope=("brain",),
    ),
    Removed(
        names=("offline_load_scope", "_hf_telemetry_lock", "_hf_telemetry_depth", "_hf_telemetry_before"),
        why="huggingface_hub's telemetry opt-out is one process-wide default set in brain/__init__.py, not a scoped flip of the library's flag around each offline load (#554)",
        ref="#554 (PR #561): the first commits of that PR added a scoped flip of the flag around each offline load, replaced by the process-wide default before merge",
        kind="code",
    ),
    # --- the archive-with-dedup write moved to the conversation buffer (#556) -------------------------------------------
    Removed(
        names=("_archive_once", "_archive_batch_key", "_turn_identity"),
        why="the dedup archive write and the turn identity moved to brain/ingest/buffer.py (archive_once, turn_identity) so the rollover archives through the same path as compaction and compaction keeps no private copy of it",
        ref="#556 rollover keeps every recent turn (the pull request that moved them)",
        kind="code",
        scope=("brain/chat",),
    ),
)


def _flags(entry: Removed) -> int:
    return re.IGNORECASE if entry.ignore_case else 0


def _text_regexes(entry: Removed) -> list[re.Pattern[str]]:
    """One alternation of the string names (substring, or whole word), then each ``re.Pattern`` as the entry gave it."""
    strings = [re.escape(n) for n in entry.names if isinstance(n, str)]
    out: list[re.Pattern[str]] = []
    if strings:
        body = "|".join(strings)
        if entry.whole_word:
            body = rf"(?<![A-Za-z0-9_])(?:{body})(?![A-Za-z0-9_])"
        out.append(re.compile(body, _flags(entry)))
    out += [n for n in entry.names if isinstance(n, re.Pattern)]
    return out


def _identifier_index(tree: ast.AST) -> dict[str, int]:
    """``{identifier or string literal: first line}`` for every code use in ``tree``. Comments and docstring prose are not code."""
    found: dict[str, int] = {}

    def add(name: str, line: int) -> None:
        found.setdefault(name, line)

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            add(node.id, line)
        elif isinstance(node, ast.Attribute):
            add(node.attr, line)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name, line)
        elif isinstance(node, ast.arg):
            add(node.arg, line)
        elif isinstance(node, ast.keyword) and node.arg:
            add(node.arg, line)
        elif isinstance(node, ast.alias):
            for part in node.name.split("."):
                add(part, line)
            if node.asname:
                add(node.asname, line)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for part in node.module.split("."):
                add(part, line)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, line)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                add(n, line)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            add(node.value, line)
    return found


def _under(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in prefixes)


def find_reappeared(entries: tuple[Removed, ...], texts: dict[str, str], trees: dict[str, ast.Module], repo: Path) -> list[str]:
    """One message per removed name found again, naming where, why it was removed and the reference."""
    problems: list[str] = []
    index: dict[str, dict[str, int]] = {}  # parsed identifiers per file, built once when a code entry first needs it
    for entry in entries:
        label = f"{entry.why} ({entry.ref})"
        assert entry.kind in _KINDS and entry.names, f"malformed REMOVED entry: {entry}"
        if entry.kind == "file":
            problems += [f"{p} exists again: {label}" for p in entry.names if (repo / str(p)).exists()]
            continue
        assert entry.kind == "text" or all(isinstance(n, str) for n in entry.names), f"a code entry takes plain names: {entry}"
        for scope in entry.scope:
            if not (repo / scope).exists():
                problems.append(f"entry scope {scope!r} no longer exists: delete or re-scope the entry ({label})")
        files = [rel for rel in sorted(texts) if _under(rel, entry.scope) and not _under(rel, entry.allow)]
        if not files or (entry.kind == "code" and not any(rel in trees for rel in files)):
            problems.append(f"entry scope {entry.scope} holds no scanned {'python ' if entry.kind == 'code' else ''}file: fix the scope ({label})")
            continue
        if entry.kind == "text":
            regexes = _text_regexes(entry)
            for rel in files:
                for regex in regexes:
                    for m in regex.finditer(texts[rel]):
                        line = texts[rel].count("\n", 0, m.start()) + 1
                        problems.append(f"{m.group(0)[:60]!r} is back in {rel}:{line}: {label}")
        else:
            wanted = set(entry.names)
            for rel in files:
                tree = trees.get(rel)
                if tree is None:
                    continue
                idx = index.get(rel)
                if idx is None:
                    idx = index[rel] = _identifier_index(tree)
                problems += [f"{name!r} is back in {rel}:{idx[name]}: {label}" for name in sorted(wanted & idx.keys())]
    return problems


def _scanned_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for root in _ROOTS:
        texts.update(tree_sources(root, _SUFFIXES, ("node_modules",)))
    # A scope that is a single file outside brain/ and app/src (pyproject.toml) is read directly.
    for entry in REMOVED:
        for scope in entry.scope:
            path = REPO / scope
            if scope not in texts and path.is_file():
                texts[scope] = path.read_text(encoding="utf-8")
    return texts


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_removed_names_stay_removed() -> None:
    """Nothing listed in ``REMOVED`` has come back; a failure names each name, where it reappeared and why it was removed."""
    texts = _scanned_texts()
    assert "brain/tunables.py" in texts and "app/src/App.tsx" in texts, "the scan did not read the source tree"
    # The consolidation entry is a regex over this section header; if the header is renamed the regex would match nothing.
    assert "[engines.consolidation]" in texts.get("brain/prompt_strings.toml", ""), "update the consolidation-section entry of REMOVED"
    problems = find_reappeared(REMOVED, texts, dict(tree_asts("brain")), REPO)
    assert not problems, (
        "Removed code has reappeared. If it is intended, delete its line from REMOVED in tests/unit/test_removed_names_stay_removed.py; "
        "otherwise remove it again:\n  " + "\n  ".join(problems)
    )
