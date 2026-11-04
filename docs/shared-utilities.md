# Shared utilities

The helpers that any part of `brain/` may call, so the existing function for a job can be found without searching the whole codebase. One row per function: its name, where it is, and what it is for. Classes are marked `(class)`.

Maintained by hand: a PR that adds, moves or removes a shared function, or changes what one is for, updates this list in the same PR. Line numbers were last refreshed by the shared file accessors change (#643) and drift as files change; the path and name are what to search for.

New code uses these instead of writing its own copy. Where none fits, extend the shared one rather than working around it.

## File access: JSON and JSONL

Use these for every JSON/JSONL read or write; do not call json.load/open/os.replace directly.

### `brain/health/json_io.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `Recovery` (class) | brain/health/json_io.py:151 | Enum of what a JSON read does with a bad file: DEFAULT (use the default), CONTENT (default for bad content, raise on I/O faults), RAISE, RESTORE (from the .bak chain), HEAL (restore and record an anomaly). |
| `JsonFileError` (class) | brain/health/json_io.py:163 | Exception for an unusable JSON file; `.kind` is missing, io, empty, malformed, encoding, shape, schema or line; is both an OSError and a ValueError. |
| `JsonFileNotFound` (class) | brain/health/json_io.py:191 | The `kind == "missing"` JsonFileError; also a FileNotFoundError. |
| `JsonRead` (class) | brain/health/json_io.py:200 | Result record of read_json_result: `data`, `outcome` (ok, missing, defaulted, restored, reset), `kind` and any `anomaly`. |
| `read_json` | brain/health/json_io.py:269 | Read a whole JSON file under a Recovery policy (default factory, expected type, validator, fallback files) and return its data. |
| `read_json_result` | brain/health/json_io.py:289 | Same read as read_json but returns a JsonRead, so the caller can see whether the file was missing, defaulted or restored. |
| `write_json` | brain/health/json_io.py:703 | Atomically write data as JSON: strict encoding (no NaN), optional .bak1..N chain, file mode, read-back verify, create-only, and log-instead-of-raise; returns True if written. |
| `encode_jsonl` | brain/health/json_io.py:821 | The one JSONL encoder: dict records to ASCII lines (newline terminated) as bytes; raises before anything is written. |
| `parse_jsonl` | brain/health/json_io.py:849 | Parse JSONL text or bytes already in memory, skipping bad lines (or raising when strict); a line the JSON parser cannot load stops the read. |
| `iter_jsonl` | brain/health/json_io.py:879 | Yield the dict records of a JSONL file (.gz by suffix), skipping bad lines with a warning (a line the JSON parser cannot load stops the read); a missing file yields nothing. |
| `read_jsonl` | brain/health/json_io.py:914 | iter_jsonl collected into a list. |
| `iter_jsonl_raw` | brain/health/json_io.py:920 | Like iter_jsonl but yields (line_text, record), for callers that must rewrite the lines exactly as read or still see unparsable lines. |
| `read_jsonl_tail` | brain/health/json_io.py:1033 | Return the records on the last n non-blank lines, reading backwards in byte chunks so the I/O is bounded. |
| `append_jsonl` | brain/health/json_io.py:1123 | Append dict records as JSONL in one write under the file lock; starts with a newline if the file ends in a torn line; returns bytes written. |
| `write_jsonl` | brain/health/json_io.py:1190 | Atomically replace a JSONL file with the given records (same writer as write_json). |
| `read_jsonl_from` | brain/health/json_io.py:1001 | Read a JSONL file from a line position: skip the first `start_line` complete lines, return each later complete line as a `JsonlRecord` (line number, object or bad-line reason) and `end_line`, the position to pass next time; a torn last line is left for the next read. |
| `JsonlRecord` (class) | brain/health/json_io.py:981 | One line from read_jsonl_from: `line` (1-based), `data` (the object, or None for a bad line) and `reason` (why it is bad, never the line's content). |
| `JsonlRead` (class) | brain/health/json_io.py:994 | Result of read_jsonl_from: `records` and `end_line` (the number of complete lines in the file). |

### `brain/health/jsonl_reader.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `JsonlLineError` (class) | brain/health/jsonl_reader.py:25 | ValueError for a bad line met in strict mode: carries `line` (1-based), `reason` and `path`; json_io turns it into `JsonFileError(kind="line")`. |
| `UnloadableJsonlLineError` (class) | brain/health/jsonl_reader.py:38 | ValueError (stable `code` "jsonl_unloadable_line", plus `path`, `line`, `reason`) that stops a read at a line the JSON parser cannot load (integer past the digit limit, nesting too deep); a known abort, see #646. |
| `classify_line` | brain/health/jsonl_reader.py:91 | The one place a JSONL line is judged: a usable object, a blank line, or a bad line (invalid UTF-8, unparsable JSON, not an object) with the reason; every JSONL reader calls it. |
| `LineVerdict` (class) | brain/health/jsonl_reader.py:64 | What classify_line returns: the line's text, its object, the kind of fault and a `reason` property. |

## File access: atomic writes, locks, TOML, SQLite

### `brain/utils/file_ops.py`

Use these for retried replace and delete, whole-file byte reads, folder listing and path checks; do not call `os.replace`, `unlink`, `os.listdir` or `os.stat` directly in new code. The retry schedule is `FILE_RETRY_ATTEMPTS`, `FILE_RETRY_SLEEP_S` and `FILE_READ_ATTEMPTS` in `brain/dev_constants.py`.

| function | where (path:line) | what it is for |
|---|---|---|
| `retry_transient` | brain/utils/file_ops.py:32 | The one retry loop: call a function again after a Windows `PermissionError` (sharing violation) on a short growing sleep, and raise if it persists; `windows_only=True` makes one attempt elsewhere. |
| `stat_key` | brain/utils/file_ops.py:64 | The `(inode, mtime_ns, size)` triple a cache compares to see that a file has not changed; the one definition. |
| `read_bytes_stat` | brain/utils/file_ops.py:75 | Read a whole file and return its bytes with the stat key taken from the open handle; retries a Windows sharing violation. |
| `read_bytes` | brain/utils/file_ops.py:85 | Read a whole file's bytes (read_bytes_stat without the key). |
| `replace_file` | brain/utils/file_ops.py:90 | `os.replace` of a file or folder with the transient-failure retry. |
| `delete_file` | brain/utils/file_ops.py:95 | Delete a file with the retry; a missing file is not an error, a failure that persists raises, a folder raises IsADirectoryError at once on every system (never retried). |
| `delete_tree` | brain/utils/file_ops.py:116 | Delete a folder and its content with the retry; a missing folder is not an error; a file or a symbolic link is refused, never followed. |
| `list_dir` | brain/utils/file_ops.py:126 | The sorted names of the entries in a folder; a missing folder is `[]` and is not created. |
| `PathStat` (class) | brain/utils/file_ops.py:134 | One `os.stat` of a path: inode, mtime_ns, size, is_file, is_dir, and a `key` property equal to stat_key. |
| `stat_path` | brain/utils/file_ops.py:149 | One `os.stat` of a path as a PathStat, or None if the path is missing; any other OSError raises. The accessor for exists, size, file-or-folder and a cache's currency key. |

### `brain/utils/atomic_write.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `fsync_dir` | brain/utils/atomic_write.py:223 | fsync a directory so a new or renamed entry is durable; POSIX only, best effort, never fails a save. |
| `save_with_backup_text` | brain/utils/atomic_write.py:275 | Atomic raw-text save with a rotating .bak1..N chain; the live path always holds a complete file. |
| `save_bytes_atomic` | brain/utils/atomic_write.py:318 | Atomic raw-bytes save: unique temp in the same directory, optional mode and fsync, one retried replace, no backup chain. |
| `save_blob_once` | brain/utils/atomic_write.py:331 | Write a content-addressed blob once; skips if the target exists and tolerates a concurrent writer of the same content (used by images and file_store). |

### `brain/utils/file_lock.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `file_lock` | brain/utils/file_lock.py:45 | Context manager: exclusive cross-process lock on the sidecar `<path>.lock`; blocks by default, or with `blocking=False` tries once and yields whether it got the lock. |
| `LockTimeoutError` (class) | brain/utils/file_lock.py:96 | TimeoutError raised by locked_within when another holder kept the lock for the whole wait. |
| `locked_within` | brain/utils/file_lock.py:101 | Context manager: hold the path lock, waiting at most `seconds`; on timeout raises LockTimeoutError and the body never runs. |

### `brain/utils/toml_io.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `TomlFileError` (class) | brain/utils/toml_io.py:49 | Exception for an unusable TOML file or value; `kind` is missing, io, damaged or unsupported, with line and column when known. |
| `TomlRead` (class) | brain/utils/toml_io.py:86 | Result record of read_toml_result: data (empty unless outcome is ok) plus outcome. |
| `parse_toml` | brain/utils/toml_io.py:134 | Parse TOML text; raises TomlFileError (damaged) with line and column when known. |
| `read_toml_result` | brain/utils/toml_io.py:165 | Read and parse a TOML file without ever raising; returns a TomlRead; retries a Windows sharing violation. |
| `read_toml` | brain/utils/toml_io.py:177 | Read and parse a TOML file; raises TomlFileError; a zero-byte file is `{}`. |
| `read_toml_cached` | brain/utils/toml_io.py:189 | read_toml that re-parses only when the file's mtime, size or inode changes; thread-safe; the returned dict is shared, treat as read only. |
| `dumps_value` | brain/utils/toml_io.py:240 | Render one TOML value (bool, int, float, str, list/tuple of those) on one line; anything else raises TomlFileError. |
| `flatten_toml` | brain/utils/toml_io.py:264 | Flatten nested tables and dotted keys into one dict of dotted keys; the shallowest spelling wins. |
| `same_value` | brain/utils/toml_io.py:287 | Type-strict equality for TOML values (1, 1.0 and True differ; NaN equals NaN; list equals tuple). |
| `write_toml_text` | brain/utils/toml_io.py:316 | Atomically replace a TOML file with bytes through save_bytes_atomic (no newline translation, BOM kept). |
| `TomlDocument` (class) | brain/utils/toml_io.py:496 | A TOML file held as lines so single entries can be edited in place without disturbing comments or the rest of the file; `to_bytes()` gives it back. |

### `brain/utils/sqlite_open.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `OpenProfile` (class) | brain/utils/sqlite_open.py:39 | Settings that differ per store: busy timeout and WAL-failure policy. |
| `close_on_error` | brain/utils/sqlite_open.py:53 | Context manager: if the body raises anything, close the sqlite connection and re-raise the same exception. |
| `check_integrity` | brain/utils/sqlite_open.py:72 | Run PRAGMA integrity_check (with transient-error retry) and raise BrainIntegrityError unless the result is ok; does not close the connection. |
| `open_store_connection` | brain/utils/sqlite_open.py:87 | Open a SQLite file with the store's busy-timeout and WAL profile, optionally run the integrity check, then the store's own setup callback; on any failure close the connection and re-raise. |

## File access: log files on disk

### `brain/health/log_rotation.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `rotate_rolling_size` | brain/health/log_rotation.py:79 | Once a log exceeds max_bytes, gzip it to `<log>.1.gz`, shift older archives up (keeping archive_keep), truncate the live log; returns the new archive path or None. |

### `brain/health/parse_failures.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `record_parse_failure` | brain/health/parse_failures.py:43 | Append a failed LLM reply to the persona's parse-failure log as evidence; never raises; can store only length and error class. |

## Paths

`brain/dev_constants.py` has no public functions or classes (constants only), so it has no rows.

### `brain/paths.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `validate_persona_name` | brain/paths.py:27 | Raise ValueError unless the name matches `[A-Za-z0-9_-]{1,40}`; the single persona-name validator. |
| `home_for_env` | brain/paths.py:79 | Resolve the app home directory for this process or for another process's environment: KINDLED_HOME, then deprecated NELLBRAIN_HOME, then the platform default. |
| `get_home` | brain/paths.py:115 | Root directory for all app state (home_for_env for this process). |
| `get_persona_dir` | brain/paths.py:122 | Directory for one persona's private data, after validating the name. |
| `list_persona_names` | brain/paths.py:134 | Sorted names of valid persona directories under `<home>/personas/`; `[]` if none. |
| `get_cache_dir` | brain/paths.py:159 | Cache directory (embeddings, computed matrices), resolved like get_home. |
| `get_log_dir` | brain/paths.py:171 | Log file directory, resolved like get_home. |
| `cadence_state_path` | brain/paths.py:186 | Path of a persona's cadence state file under `<persona>/cadence/`, moving a legacy root-level file into place on first use. |
| `presence_state_path` | brain/paths.py:214 | Path of the persona's `presence_state.json` sidecar (directly under the persona dir). |

## Settings

### `brain/tunables.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `TunablesWriteError` (class) | brain/tunables.py:83 | OSError raised when `tunables.toml` could not be changed (unreadable, uneditable entry, unwritable, or lock wait expired); nothing was written. |
| `register` | brain/tunables.py:95 | Declare a setting (key, default, description) in the registry at import time and return the default. |
| `get_tunable` | brain/tunables.py:118 | Value of a setting from `tunables.toml` if present and type-compatible, else the default. |
| `positive_or_default` | brain/tunables.py:136 | Guard for duration or size settings: return the value only if it is a finite number above 0 (and optionally below a ceiling), else the default. |
| `get_raw` | brain/tunables.py:172 | The raw file value for a key with no type guard and no default, or None. |
| `sync_file` | brain/tunables.py:526 | Boot pass: create `tunables.toml` with every registered key, add new keys, move untouched values to new defaults; fail-open, under the file lock. |
| `set_values` | brain/tunables.py:584 | Change named settings in `tunables.toml` in place under the lock and return those actually written; None resets a key to its default. |

## Time

Storage is UTC everywhere; local time is for display only.

### `brain/utils/time.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `iso_utc` | brain/utils/time.py:52 | Format a tz-aware UTC datetime as ISO-8601 with a Z suffix; raises on a naive datetime. |
| `parse_iso_utc` | brain/utils/time.py:64 | Parse an ISO-8601 string (Z suffix or offset) to a tz-aware datetime; naive means UTC; raises on bad text. |
| `lenient_utc` | brain/utils/time.py:74 | Any stored timestamp (datetime or ISO string) to an aware UTC datetime, or None; never raises; the one policy for damaged stamps. |
| `hours_since` | brain/utils/time.py:96 | Hours from a stored stamp to `now` (never negative), or None if either is unusable. |
| `to_local` | brain/utils/time.py:108 | Convert a datetime to OS local wall-clock time, for display to the companion only. |
| `format_local` | brain/utils/time.py:136 | Render a datetime as local-zone ISO-8601 at seconds precision with no offset suffix. |
| `local_display` | brain/utils/time.py:150 | Turn a stored UTC ISO string into a local display string; total, returns the raw input if it cannot render. |

## Text, prompts and LLM output

### `brain/utils/llm_output.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `extract_json_object` | brain/utils/llm_output.py:22 | Return the JSON object substring inside a model reply; raises ValueError if none can be found. |
| `loads_json_object_ex` | brain/utils/llm_output.py:36 | Parse a reply that should be one JSON object and also say whether the object was found inside prose. |
| `loads_json_object` | brain/utils/llm_output.py:62 | Parse a reply that should be one JSON object: whole reply first, then the tolerant scrape; raises ValueError. |
| `find_json_object_block` | brain/utils/llm_output.py:176 | Find the JSON object that is the model's answer in text, tolerant of preamble and braces in strings, optionally preferring an object with a given key. |

### `brain/utils/emotion.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `format_emotion_summary` | brain/utils/emotion.py:8 | Top 5 emotions as `- name: X.X/10` lines for LLM prompt context. |

### `brain/prompt_strings.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `register` | brain/prompt_strings.py:94 | Declare that a call site uses a shipped prompt-string key and return its resolved text now, so a bad key fails at import. |
| `register_segments` | brain/prompt_strings.py:136 | Same as register for a straight-line interpolation stored as a list of literal segments. |

## Numbers and data shapes

### `brain/utils/finite.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `clamp_nonfinite` | brain/utils/finite.py:33 | Copy a nested value with every NaN/Infinity float replaced (default 0.0) and log one warning if any were replaced. |

### `brain/memory/embeddings.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `cosine_similarity` | brain/memory/embeddings.py:497 | Cosine similarity of two numpy vectors in [-1, 1]; 0.0 if either has zero norm. |

### `brain/state_compat.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `from_known_fields` | brain/state_compat.py:25 | Build a dataclass from a mapping, ignoring keys the class does not declare; a missing required field still raises TypeError. |

### `brain/kindled_link/codec.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `canonical_json` | brain/kindled_link/codec.py:10 | Deterministic JSON bytes (sorted keys, compact separators, UTF-8) for signing and hashing. |

## Logging

### `brain/utils/log_throttle.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `warn_throttled` | brain/utils/log_throttle.py:38 | Log a message at WARNING once per interval per key and at DEBUG in between; for errors that can repeat every tick. |
| `count_throttled` | brain/utils/log_throttle.py:48 | Count one firing of an expected cap or gate under a key, log with the running count at a throttled level, and return the count. |

## Process and network

### `brain/process_identity.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `token_of` | brain/process_identity.py:134 | Identity token of a pid that tells a reused pid apart from the original process (per platform), or GONE or UNKNOWN. |
| `own_token` | brain/process_identity.py:154 | This process's identity token (cached), or None if unreadable. |
| `holder_alive` | brain/process_identity.py:174 | Is the process recorded in a lock or marker file still alive? Uses the recorded token if present, else pid only; UNKNOWN counts as alive. |

### `brain/bridge/state_file.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `pid_is_alive` | brain/bridge/state_file.py:191 | True if the pid is a live process (Windows-safe; a pid owned by another user counts as alive). |

### `brain/bridge/loopback.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `direct_ws_connect` | brain/bridge/loopback.py:48 | websockets sync `connect` with no proxy lookup, where the installed version supports it. |

## Failure classification

### `brain/utils/failure_scope.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `ServiceCallFailed` (class) | brain/utils/failure_scope.py:38 | Marker exception: a call to an outside service (an LLM provider) failed for any reason; counts as an environment failure. |
| `failure_scope` | brain/utils/failure_scope.py:49 | Classify an exception by type as `environment` (keep the item, retry later) or `item` (set it aside). |
| `settle_scope` | brain/utils/failure_scope.py:65 | failure_scope plus a one-read store health check that upgrades an item verdict to environment if the store itself is closed or broken. |

## Memory-derived helpers

Live in utils but read the MemoryStore.

### `brain/utils/memory.py`

| function | where (path:line) | what it is for |
|---|---|---|
| `days_since_human` | brain/utils/memory.py:15 | Days since the user last spoke (999.0 if never), from active session buffers first, then closed-session memories. |
| `is_conversation_derived` | brain/utils/memory.py:70 | True if a memory came from a conversation buffer (conversation tag or `source_summary` starting `conversation:`), plus the legacy type. |
| `list_conversation_memories` | brain/utils/memory.py:87 | List chat-derived memories, newest first, using the conversation-derived contract. |
