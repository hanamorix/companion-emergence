"""companion-emergence — framework for building emotionally aware AI companions.

Reference implementation: Nell. See docs/source-spec/ for the full design.
"""

# Derive version from package metadata so a future bump in pyproject.toml
# can't silently disagree with what `nell --version` and `brain.__version__`
# report. Falls back to the last-known string only when running directly
# from a source tree without an install (very rare; release CI always
# installs the wheel before the smoke step).
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("companion-emergence")
except PackageNotFoundError:  # pragma: no cover — only hit pre-install in dev
    __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError
