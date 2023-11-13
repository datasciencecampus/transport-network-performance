"""Helper functions to handle IO operations."""

import pathlib
import pickle

from typing import Union

from transport_performance.utils.defence import (
    _check_parent_dir_exists,
    _enforce_file_extension,
    _is_expected_filetype,
)


def to_pickle(
    picklable_object: object,
    path: Union[str, pathlib.Path],
) -> None:
    """Pickle an object for later reuse.

    Parameters
    ----------
    picklable_object : Any
        Object to be saved to pickle file. If the directory does not exist one
        will be generated.
    path : Union[str, pathlib.Path]
        Path to save pickle file. Must have a ".pkl" or ".pickle" file
        extension or it will be coerced to ".pkl".

    Raises
    ------
    UserWarning
        When `path` does not have a ".pkl" or ".pickle" file extension. The
        warning will signify for coercion to a path with the ".pkl" extension.

    """
    # defensive checks
    _check_parent_dir_exists(path, "path", create=True)
    path = _enforce_file_extension(
        path,
        exp_ext=[".pkl", ".pickle"],
        default_ext=".pkl",
        param_nm="path",
    )

    # write to file
    with open(path, "wb") as f:
        pickle.dump(picklable_object, f)


def from_pickle(path: Union[str, pathlib.Path]) -> object:
    """Read a pickled object from file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path of saved pickle file.

    Returns
    -------
    Any
        Object in pickle file. Must have a ".pkl" or ".pickle" file extension.

    Raises
    ------
    TypeError
        `path` is not a string or a pathlib.Path type.
    FileNotFoundError
        `path` does not exist
    ValueError
        `path` does not have either a ".pkl" or ".pickle" file extnesion.

    """
    # defensive checks
    _is_expected_filetype(
        path, "path", check_existing=True, exp_ext=[".pkl", ".pickle"]
    )

    # read and return object
    with open(path, "rb") as f:
        return pickle.load(f)
