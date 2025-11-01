"""
Team-41 MLOps demo ─ top-level package.

Keep this file *lightweight*: do not import sub-modules here.  That avoids
circular-import headaches and makes `import src` virtually instantaneous.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    # Pick up the installed distribution version when available
    __version__ = version("mlops_MCE")
except PackageNotFoundError:
    # Fallback for editable installs while the package isn't built yet
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
