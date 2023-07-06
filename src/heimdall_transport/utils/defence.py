"""Defensive check utility funcs. Internals only."""
import pathlib
import numpy as np


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
