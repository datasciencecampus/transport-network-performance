"""Defensive check utility funcs. Internals only."""
import pathlib
import numpy as np
import os
from typing import Union
import pandas as pd
import warnings


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


def _is_expected_filetype(
    pth: Union[pathlib.Path, str],
    param_nm: str,
    check_existing: bool = True,
    exp_ext: Union[str, list] = ".zip",
) -> None:
    """Handle file paths that should be existing filetypes.

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.
    param_nm : str
        The name of the parameter being tested. Helps with debugging.
    check_existing : bool
        Whether to check if the filetype file already exists. Defaults to True.
    exp_ext: (str, list)
        The expected filetype as a string. Or a list of file extension strings
        to check. If the user forgets to include the "." in `exp_ext`, one will
        be added.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.PosixPath.
    FileExistsError: `pth` does not exist on disk.
    ValueError: `pth` does not have the expected file extension(s).

    Returns
    -------
    None

    """
    typing_dict = {
        "pth": [pth, (str, pathlib.Path)],
        "param_nm": [param_nm, str],
        "check_existing": [check_existing, bool],
        "exp_ext": [exp_ext, (str, list)],
    }
    for k, v in typing_dict.items():
        _type_defence(v[0], k, v[-1])
    pth = _handle_path_like(pth=pth, param_nm=param_nm)
    _, ext = os.path.splitext(pth)
    # lower for consistency
    ext = ext.lower()
    # catch cases where directories are passed. eg no file stem.
    if ext == "":
        raise ValueError(f"No file extension was found in {pth}.")
    # treat cases where user forgot to include '.'
    if isinstance(exp_ext, list):
        # lower everything for consistency
        exp_ext = [ext.lower() for ext in exp_ext]
        for i, e in enumerate(exp_ext):
            if not e.startswith(r"."):
                warnings.warn(
                    f"'.' was prepended to `exp_ext` value '{exp_ext[i]}'."
                )
                exp_ext[i] = "." + e
        is_correct = ext in exp_ext

    else:
        exp_ext = exp_ext.lower()
        if not exp_ext.startswith(r"."):
            warnings.warn("'.' was prepended to the `exp_ext`.")
            exp_ext = "." + exp_ext
        is_correct = ext == exp_ext

    if check_existing and not os.path.exists(pth):
        raise FileNotFoundError(f"{pth} not found on file.")
    # if isinstance(exp_ext, list):
    #     is_correct = ext in exp_ext
    # elif isinstance(exp_ext, str):
    #     is_correct = ext == exp_ext

    if not is_correct:
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
    _type_defence(url, "url", str)
    if not url.startswith((r"http://", r"https://")):
        raise ValueError(f"url string expected protocol, instead found {url}")

    return None


def _type_defence(some_object, param_nm, types) -> None:
    """Defence checking utility. Can handle NoneType.

    Parameters
    ----------
    some_object : Any
        Object to test with isinstance.
    param_nm : str
        A name for the parameter. Useful when this utility is used in a wrapper
        to inherit the parent's parameter name and present in error message.
    types : type or tuple
        A type or a tuple of types to test `some_object` against.

    Raises
    ------
    TypeError
        `some_object` is not of type `types`.

    """
    if not isinstance(some_object, types):
        raise TypeError(
            f"`{param_nm}` expected {types}. Got {type(some_object)}"
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


def _check_column_in_df(df: pd.DataFrame, column_name: str) -> None:
    """Defences to check that a column exists in a df.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe to check if the specified column exists in.
    column_name : str
        The name of the column to check for

    Returns
    -------
    None

    Raises
    ------
    IndexError
        Raises an error if the column (column_name) does not exist in the
        dataframe

    """
    if column_name not in df.columns:
        raise IndexError(f"'{column_name}' is not a column in the dataframe.")

    return None


def _check_item_in_list(item: str, _list: list, param_nm: str) -> None:
    """Defence to check if an item is present in a list.

    Parameters
    ----------
    item : str
        THe item to check the list for
    _list : list
        The list to check that the item is in
    param_nm : str
        The name of the param that the item has been passed to

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Error raised when item not in the list.

    """
    if item not in _list:
        raise ValueError(
            f"'{param_nm}' expected one of the following:"
            f"{_list} Got {item}"
        )
    return None


def _check_attribute(obj, attr: str, message: str = None):
    """Test to check if an attribute exists in an object.

    Parameters
    ----------
    obj : any
        The object to check that the attr exists in
    attr : str
        The attribute to check exists in an object
    message : str, optional
        The error message to display, by default None

    Raises
    ------
    AttributeError
        An error raised if the attr does not exist

    """
    err_msg = (
        message
        if message
        else (f"{obj.__class__.__name__} has no attribute {attr}")
    )

    if attr not in obj.__dir__():
        raise AttributeError(err_msg)
