"""companion-emergence — framework for building emotionally aware AI companions.

Reference implementation: Nell. See docs/source-spec/ for the full design.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def _opt_out_of_huggingface_telemetry() -> None:
    """Switch off huggingface_hub's telemetry and its agent-harness detection, for this process and its children.

    The library builds a user agent that names the torch version and asks whether an AI agent is calling
    (`GET /api/agent-harnesses`, at most once a day) on EVERY request header build, including the ones of a load that
    resolves entirely from the local cache. A local, privacy-first app wants neither, online or offline (#554), and the
    library's own opt-out is `HF_HUB_DISABLE_TELEMETRY`. This runs here because every entry point imports the `brain`
    package before anything of ours can import huggingface_hub, which reads the variable once, when it is first imported.

    A value the user set is left alone, and a library imported before `brain` (a host program) gets its module constant
    set as well: the library reads that constant, not the environment, when it builds headers. An empty or blank value
    is not a choice (the library reads it as false), so it gets the default. Imports nothing from the library.
    """
    import os
    import sys

    if os.environ.get("HF_HUB_DISABLE_TELEMETRY", "").strip():
        return
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_DISABLE_TELEMETRY = True


_opt_out_of_huggingface_telemetry()
del _opt_out_of_huggingface_telemetry

# Derive version from package metadata so a future bump in pyproject.toml
# can't silently disagree with what `nell --version` and `brain.__version__`
# report. Falls back to the last-known string only when running directly
# from a source tree without an install (very rare; release CI always
# installs the wheel before the smoke step).

try:
    __version__ = _pkg_version("companion-emergence")
except PackageNotFoundError:  # pragma: no cover — only hit pre-install in dev
    __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError
