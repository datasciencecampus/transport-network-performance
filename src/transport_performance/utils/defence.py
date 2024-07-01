"""Defensive check utility funcs. Internals only."""
import os
import pathlib
import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd


def _handle_path_like(
    pth: Union[str, pathlib.Path], param_nm: str
) -> pathlib.Path:
    """Handle path-like parameter values.

    Checks a path for symlinks and relative paths. Converts to realpath &
    outputs pathlib.PosixPath or pathlib.WindowsPath object
    (platform agnostic).

    Parameters
    ----------
    pth : (str, pathlib.Path)
        The path to check.

    param_nm : str
        The name of the parameter being tested.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.Path.

    Returns
    -------
    pathlib.Path
        Platform agnostic representation of pth. On unix-like a PosixPath is
        returned. On windows a WindowsPath is returned. Both are children of
        pathlib.Path.

    """
    if not isinstance(pth, (str, pathlib.Path)):
        raise TypeError(f"`{param_nm}` expected path-like, found {type(pth)}.")

    # Convert backslashes to forward slashes for Windows paths
    pth_str = str(pth).replace("\\", "/")

    # Ensure returned path is not relative or contains symbolic links
    pth = os.path.realpath(pth_str)
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
    FileNotFoundError: `pth` does not exist on disk.
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
                    UserWarning(
                        f"'.' was prepended to `exp_ext` value '{exp_ext[i]}'."
                    )
                )
                exp_ext[i] = "." + e
        is_correct = ext in exp_ext

    else:
        exp_ext = exp_ext.lower()
        if not exp_ext.startswith(r"."):
            warnings.warn(UserWarning("'.' was prepended to the `exp_ext`."))
            exp_ext = "." + exp_ext
        is_correct = ext == exp_ext

    if check_existing and not os.path.exists(pth):
        raise FileNotFoundError(f"{pth} not found on file.")

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


def _check_iter_length(iterable: Iterable, param_nm: str, length: int) -> None:
    """Check the length of an iterable.

    Parameters
    ----------
    iterable : Iterable
        Iterable to check.
    param_nm : str
        Name of the parameter being checked.
    length: int
        Expected length of the iterable to check.

    Raises
    ------
        ValueError: length of iterable does not match `length`.

    Returns
    -------
    None

    """
    # check if iterable
    _type_defence(iterable, param_nm, Iterable)
    # check if length is int
    _type_defence(length, "length", int)

    if len(iterable) != length:
        raise ValueError(
            f"`{param_nm}` is of length {len(iterable)}. "
            f"Expected length {length}."
        )

    return None


def _check_iterable(
    iterable: Iterable,
    param_nm: str,
    iterable_type: type,
    check_elements: bool = True,
    exp_type: Union[tuple, type] = str,
    check_length: bool = False,
    length: int = 0,
) -> None:
    """Check an iterable and its elements for type.

    Parameters
    ----------
    iterable : Iterable
        Iterable to check.
    param_nm : str
        Name of the parameter being checked.
    iterable_type : type
        Expected iterable type.
    check_elements : bool, optional
        Whether to check the list element types. Defaults to True.
    exp_type : Union[tuple, type], optional:
        The expected type of the elements. If using a tuple, it should be a
        tuple of types. Defaults to str.
    check_length: bool, optional
        Wether to check the length of the iterable. Defaults to False.
    length: int, optional
        Expected length of the iterable. Defaults to 0.

    Raises
    ------
        TypeError
            `iterable` is not iterable.
        TypeError
            `exp_type` contains elements that are not type.
        TypeError
            Elements of `iterable` are not of the expected type(s).

    Returns
    -------
    None

    """
    # check if iterable
    _type_defence(iterable, param_nm, Iterable)
    # check if iterable_type is type
    _type_defence(iterable_type, "iterable_type", type)
    # check if iterable type matches expected
    _type_defence(iterable, param_nm, iterable_type)

    # check if expected types tuple includes only types
    if isinstance(exp_type, tuple):
        for i in exp_type:
            if not isinstance(i, type):
                raise TypeError(
                    (
                        f"`exp_type` must contain types only. "
                        f"Found {type(i)} : {i}"
                    )
                )
    else:
        _type_defence(exp_type, "exp_type", (type, tuple))

    # check length
    if check_length:
        _check_iter_length(iterable, param_nm, length)

    # check if elements are of the expected types
    if check_elements:
        for i in iterable:
            if not isinstance(i, exp_type):
                raise TypeError(
                    (
                        f"`{param_nm}` must contain {str(exp_type)} only."
                        f" Found {type(i)} : {i}"
                    )
                )

    return None


def _gtfs_defence(gtfs, param_nm):
    """Defence checking. not exported."""
    if gtfs.__class__.__name__ != "GtfsInstance":
        raise TypeError(
            f"'{param_nm}' expected a GtfsInstance object. "
            f"Got {type(gtfs)}"
        )


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


def _check_item_in_iter(item: str, iterable: Iterable, param_nm: str) -> None:
    """Defence to check if an item is present in an iterable.

    Parameters
    ----------
    item : str
        The item to check the list for
    iterable : Iterable
        The iterable to check that the item is in
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
    # check if iterable
    _type_defence(iterable, param_nm, Iterable)

    if item not in iterable:
        raise ValueError(
            f"'{param_nm}' expected one of the following: "
            f"{iterable}. Got {item}: {type(item)}"
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


def _enforce_file_extension(
    path: Union[str, pathlib.Path],
    exp_ext: Union[str, list],
    default_ext: str,
    param_nm: str,
    msg: str = None,
) -> pathlib.Path:
    """Defensive check for enforcing file extensions.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        The filepath to check the extension of.
    exp_ext : Union[str, list]
        The expected/implemented extension(s)
    default_ext : str
        The extensions to default to if the extension is invalid
    param_nm : str
        The name of the parameter that the path was passed to
    msg : str, optional
        Custom error message to display, by default None

    Returns
    -------
    pathlib.Path
        The path with the correct file extension

    """
    _handle_path_like(path, "path")
    root, ext = os.path.splitext(path)
    if isinstance(exp_ext, str):
        exp_ext = [exp_ext]
    # formatting assurances
    exp_ext = [x.replace(".", "").lower() for x in exp_ext]
    default_ext = default_ext.replace(".", "")
    ext = ext.replace(".", "")
    # check if path in acceptable format
    if ext.lower() not in exp_ext:
        # format warning message
        if not msg:
            msg = (
                f"Format .{ext} provided. Expected {exp_ext} for path given "
                f"to '{param_nm}'. Path defaulted to .{default_ext}"
            )
        # warn user
        warnings.warn(msg, UserWarning)
        path = os.path.normpath(root + f".{default_ext}")
    return pathlib.Path(path)


def _validate_datestring(date_text: str, form: str = "%Y%m%d") -> None:
    """Ensure a date string conforms to the expected form.

    Parameters
    ----------
    date_text : str
        The datestring to be checked.
    form : str, optional
        The expected date format. Must be a valid C-standard date code format
        compatible with datetime. See https://docs.python.org/3/library/datetim
        e.html#strftime-and-strptime-format-codes for examples. Defaults to
        "%Y%m%d".

    Returns
    -------
    None

    Raises
    ------
    ValueError
        `date_text` is not in the expected `form`.
    TypeError
        `date_text` or `form` are not string.

    """
    _type_defence(date_text, "date_text", str)
    _type_defence(form, "form", str)
    try:
        datetime.strptime(date_text, form)
    except ValueError:
        raise ValueError(
            f"Incorrect date format, {date_text} should be {form}"
        )
    return None
