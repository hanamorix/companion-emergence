# read_file: count the lines outside the window fast, and read past a cut line (#735)

Follow-ups from #590 (PR #729). Severity low.

## Why

1. Every ranged read of a text file runs each line of the whole file through a Python loop (`_stream_lines` feeding `_Window.add`), only to count lines and to call the file binary if any byte isn't UTF-8. Paging a 20 MiB file of bare `\n` costs about 3 s per read, inside the MCP server's async handler.
2. A line longer than the window budget (40,000 JSON-encoded characters) is cut, and the note says so, but no argument reaches the rest of it.

## Measured (2026-11-19, M3 Pro, a 20 MiB file)

| File | Per-line loop (today) | Decode + `str.count` per 64 KiB chunk |
|---|---|---|
| 20,971,520 bare `\n` | 2.93 s | 0.039 s (same count) |
| 230,456 lines of prose | 0.05 s | 0.041 s (same count) |

## Goals

- Only the lines that are shown go through the per-line loop. The lines before and after them are counted from whole chunks.
- `total_lines`, the window's content and its notes stay exactly as today for every existing argument, and so does the binary verdict for a bad byte anywhere in the file.
- A cut line can be read to its end: a new `char_offset` argument, and the cut note names the call that continues it.
- Memory stays bounded by the chunk and the budget (#590).

## Non-goals

- The window size and the 20 MiB cap stay; those are #595's question.
- No index or cache of line positions across calls.
- PDF extraction speed is unchanged (pypdf dominates it). PDFs only gain `char_offset`.

## Anti-goals

- A line count that differs from `str.splitlines` on any input, the other break characters and `\r\n` across a chunk boundary included.
- A file called text when a byte after the window isn't UTF-8.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Count the lines outside the window per chunk (decode, then `str.count` of the break characters); per-line only inside it. `char_offset` for the cut line | Same results and contract, near C speed, small code |
| B | Stop scanning once the window is full, and report `total_lines` as unknown | Fastest, but changes the contract and misses a bad byte past the window |
| C | Cache line offsets per (path, size, mtime) | Fast paging after the first read, but the first read is still slow, plus a new cache and its invalidation |

Hana picked A on 2026-11-19.

## Design (`brain/tools/impls/read_file.py`)

### The scan

`_stream_lines` becomes `_scan_text(fh, window)`. It reads 64 KiB chunks and decodes them incrementally, with the same `\r` carry as today. For each decoded chunk:
- **Before the window** (every line ending in it, and any unfinished line, has an index below `window.start`): the line count moves on by the chunk's breaks, `sum(text.count(c) for c in _LINE_BREAKS) - text.count("\r\n")`, and nothing is built.
- **Overlapping the window:** the chunk is split with `splitlines(keepends=True)` and fed line by line, as today.
- **After it** (the window is full, or the next index is past its end): counted like the first case.

Whether an unfinished line is open is tracked, so the end of the file adds one line exactly when `splitlines` would. `_Window` gains `skip(n)`, which adds `n` to `total` without keeping anything.

### `char_offset`

- `read_file(path, *, max_lines=None, offset=0, char_offset=0)`. A negative or missing value is 0, like `offset`.
- Line `offset` is shown from character `char_offset` of its text, line break excluded. The scan drops those characters as it streams, so they're never held. A `char_offset` at or past the line's end gives the line's break alone, or nothing for the last line.
- `_Window.add(line, ...)` takes each line as it is shown. For a PDF, `_read_pdf` slices its start line before adding it.
- A cut records where it stopped: `cut_at` is the start line's `char_offset` plus the kept characters, or just the kept characters for any other line. The note becomes `line N is longer than 40000 characters and was cut at character K; pass offset=N, char_offset=K to read on`.
- With a `char_offset` above 0, the note says `line N shown from character K`. The dedup key gains `@K`, so the default key stays as it is.
- `brain/tools/schemas.py` gains a `char_offset` property with a short description. No name changes.

## Wiring

No new organ. `read_file` is reached through `dispatch()` and the MCP server, as before.

## Tests (written first, one at a time)

- **The same result as today:** for a deep offset, the tail and the default head, the content, `total_lines` and note are identical. This covers `\r\n` split across a chunk boundary, a bare `\r`, every break character, a file with no final break, and an empty file.
- **The binary check:** a bad UTF-8 byte after the window still gives the binary note.
- **The timing guard:** a 20 MiB file of bare `\n` read at a deep offset uses well under a second of CPU.
- **`char_offset`:**
  - a 100,000-character line read at offset N, then again at the `char_offset` the note names, reaches its end;
  - a past-the-end value gives the line break alone;
  - a negative value acts as 0;
  - two different `char_offset` values in the same turn aren't deduped against each other;
  - a PDF line read with it.
- **Existing tests:** #590's and #595's tests pass, with only the cut note's wording updated.
