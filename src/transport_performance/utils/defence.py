"""Defensive check utility funcs. Internals only."""
import pathlib
import numpy as np
import os
from typing import Union


def _handle_path_like(pth, param_nm):
    """Handle path-like parameter values.

    Checks a path for symlinks and relative paths. Converts to realpath &
    outputs pathlib.Path object (platform agnostic).

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.

    param_nm : str
        The name of the parameter being tested.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.Path.

    Returns
    -------
    pathlib.Path
        Platform agnostic representation of pth.

    """
    if not isinstance(pth, (str, pathlib.Path)):
        raise TypeError(f"`{param_nm}` expected path-like, found {type(pth)}.")

    # ensure returned path is not relative or contains symbolic links
    pth = os.path.realpath(pth)

    if not isinstance(pth, pathlib.Path):
        # coerce to Path even if user passes string
        pth = pathlib.Path(pth)

    return pth


def _check_parent_dir_exists(
    pth: Union[str, pathlib.Path], param_nm: str, create: bool = False
) -> None:
    """Check if a files parent directory exists.

    The parent directory in this case will be the second layer. If only a
    single layer is passed, the parent directory will be assumed to exist as
    it will be the current working directory.

    Parameters
    ----------
    pth : Union[str, pathlib.Path]
        The path to the file who's parent dir is being confirmed to exist. The
        function will consider the second layer of the path to be the parent
        directory.
    param_nm : str
        The name of the parameter for the path. This is used in the error
        message within is_path_like()
    create : bool, optional
        Whether or not to create the parent directory if it does not already
        exist, by default False

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        An error is raised if the parent directory could not be found and also
        the create parameter is False.

    """
    pth = _handle_path_like(pth, param_nm)
    parent = os.path.dirname(pth)
    if not os.path.exists(parent):
        if create:
            os.makedirs(parent)
            print(f"Creating parent directory: {parent}")
        else:
            raise FileNotFoundError(
                f"Parent directory {parent} not found on disk."
            )

    return None


def _is_expected_filetype(pth, param_nm, check_existing=True, exp_ext=".zip"):
    """Handle file paths that should be existing filetypes.

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.
    param_nm : str
        The name of the parameter being tested. Helps with debugging.
    check_existing : bool
        Whether to check if the filetype file already exists. Defaults to True.
    exp_ext: str
        The expected filetype.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.PosixPath.
    FileExistsError: `pth` does not exist on disk.
    ValueError: `pth` does not have the expected file extension.

    Returns
    -------
    None

    """
    pth = _handle_path_like(pth=pth, param_nm=param_nm)

    _, ext = os.path.splitext(pth)
    if check_existing and not os.path.exists(pth):
        raise FileExistsError(f"{pth} not found on file.")
    if ext != exp_ext:
        raise ValueError(
            f"`{param_nm}` expected file extension {exp_ext}. Found {ext}"
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


def _bool_defence(some_bool, param_nm):
    """Defence checking. Not exported."""
    if not isinstance(some_bool, bool):
        raise TypeError(
            f"`{param_nm}` expected boolean. Got {type(some_bool)}"
        )

    return None


def _check_list(ls, param_nm, check_elements=True, exp_type=str):
    """Check a list and its elements for type.

    Parameters
    ----------
    ls : list
        List to check.
    param_nm : str
        Name of the parameter being checked.
    check_elements : (bool, optional)
        Whether to check the list element types. Defaults to True.
    exp_type : (_type_, optional):
        The expected type of the elements. Defaults to str.

    Raises
    ------
        TypeError: `ls` is not a list.
        TypeError: Elements of `ls` are not of the expected type.

    Returns
    -------
    None

    """
    if not isinstance(ls, list):
        raise TypeError(
            f"`{param_nm}` should be a list. Instead found {type(ls)}"
        )
    if check_elements:
        for i in ls:
            if not isinstance(i, exp_type):
                raise TypeError(
                    (
                        f"`{param_nm}` must contain {str(exp_type)} only."
                        f" Found {type(i)} : {i}"
                    )
                )

    return None


def gtfs_defence(gtfs, param_nm):
    """Defence checking. not exported."""
    if gtfs.__class__.__name__ != "GtfsInstance":
        raise TypeError(
            f"'{param_nm}' expected a GtfsInstance object. "
            f"Got {type(gtfs)}"
        )
