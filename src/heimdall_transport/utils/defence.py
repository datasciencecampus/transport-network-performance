"""Defensive check utility funcs. Internals only."""
import pathlib
import numpy as np
import os


def _is_path_like(pth, param_nm):
    """Handle path-like parameter values.

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.

    param_nm : str
        The name of the parameter being tested.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.PosixPath.

    Returns
    -------
    None

    """
    if not isinstance(pth, (str, pathlib.PosixPath)):
        raise TypeError(f"`{param_nm}` expected path-like, found {type(pth)}.")


def _is_gtfs_pth(pth, param_nm, check_existing=True):
    """Handle file paths that should be existing GTFS feeds.

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.
    param_nm : str
        The name of the parameter being tested. Helps with debugging.
    check_existing : bool
        Whether to check if the GTFS file already exists. Defaults to True.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.PosixPath.
    FileExistsError: `pth` does not exist on disk.
    ValueError: `pth` does not have a `.zip` file extension.

    Returns
    -------
    None

    """
    _is_path_like(pth=pth, param_nm=param_nm)

    _, ext = os.path.splitext(pth)
    if check_existing and not os.path.exists(pth):
        raise FileExistsError(f"{pth} not found on file.")
    if ext != ".zip":
        raise ValueError(
            f"`gtfs_pth` expected a zip file extension. Found {ext}"
        )

    return None


def _check_namespace_export(pkg=np, func=np.min):
    """Check that a function is exported from the specified namespace.

    Parameters
    ----------
    pkg : module
        The package to check. If imported as alias, must use alias. Defaults to
        np.

    func : function
        The function to check is exported from pkg. Defaults to np.mean.

    Returns
    -------
    bool: True if func is exported from pkg namespace.

    """
    return hasattr(pkg, func.__name__)


def _url_defence(url):
    """Defence checking. Not exported."""
    if not isinstance(url, str):
        raise TypeError(f"url {url} expected string, instead got {type(url)}")
    elif not url.startswith((r"http://", r"https://")):
        raise ValueError(f"url string expected protocol, instead found {url}")

    return None


def _bool_defence(some_bool):
    """Defence checking. Not exported."""
    if not isinstance(some_bool, bool):
        raise TypeError(
            f"`extended_schema` expected boolean. Got {type(some_bool)}"
        )

    return None
