"""Utils to assist in the creation of a HTML report for GTFS."""
from typing import Union
import pathlib
import shutil
import os

from transport_performance.utils.defence import (
    _type_defence,
    _handle_path_like,
    _check_parent_dir_exists,
)

# Constant to remove non needed columns from repeated
# pair error information.
# This is a messy method however it is the only
# way to ensure that the error report remains
# dynamic and can adadpt to different tables
# in the GTFS file.

GTFS_UNNEEDED_COLUMNS = {
    "routes": [],
    "agency": ["agency_phone", "agency_lang"],
    "stop_times": [
        "stop_headsign",
        "pickup_type",
        "drop_off_type",
        "shape_dist_traveled",
        "timepoint",
    ],
    "stops": [
        "wheelchair_boarding",
        "location_type",
        "parent_station",
        "platform_code",
    ],
    "calendar_dates": [],
    "calendar": [],
    "trips": [
        "trip_headsign",
        "block_id",
        "shape_id",
        "wheelchair_accessible",
    ],
    "shapes": [],
}


class TemplateHTML:
    """A class for inserting HTML string into a template.

    Attributes
    ----------
    template : str
        A string containing the HTML template.

    Methods
    -------
    insert(placeholder: str, value: str, replace_multiple: bool = False)
        Insert values into the HTML template
    get_template()
        Returns the template attribute

    """

    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        """Initialise the TemplateHTML object.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            The file path of the html template

        Returns
        -------
        None

        """
        _handle_path_like(path, "path")
        with open(path, "r", encoding="utf8") as f:
            self.template = f.read()
        return None

    def _insert(
        self, placeholder: str, value: str, replace_multiple: bool = False
    ) -> None:
        """Insert values into the html template.

        Parameters
        ----------
        placeholder : str
            The placeholder name in the template. This is a string. In the
            template it should be surrounded by square brackets.
        value : str
            The value to place in the placeholder
            location.
        replace_multiple : bool, optional
            Whether or not to replace multiple placeholders that share the same
            placeholder value, by default False

        Returns
        -------
        None

        Raises
        ------
        ValueError
            A ValueError is raised if there are multiple instances of a
            place-holder but 'replace_multiple' is not True

        """
        _type_defence(placeholder, "placeholder", str)
        _type_defence(value, "value", str)
        _type_defence(replace_multiple, "replace_multiple", bool)
        occurences = len(self.template.split(f"[{placeholder}]")) - 1
        if occurences > 1 and not replace_multiple:
            raise ValueError(
                "You have selected not to replace multiple"
                "placeholders of the same value, however"
                "placeholders occur more than once. \n"
                "If you would like to allow this, set the"
                "replace_multiple param to True"
            )

        self.template = self.template.replace(f"[{placeholder}]", value)

    def _get_template(self) -> str:
        """Get the template attribute of the TemplateHTML object.

        Returns
        -------
        str
            The template attribute

        """
        return self.template


def _set_up_report_dir(
    path: Union[str, pathlib.Path] = "outputs", overwrite: bool = False
) -> None:
    """Set up the directory that will hold the report.

    Parameters
    ----------
    path : Union[str, pathlib.Path], optional
        The path to the directory,
        by default "outputs"
    overwrite : bool, optional
        Whether or not to overwrite any current reports,
        by default False

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        Raises an error if you the gtfs report directory already exists in the
        given path and overwrite=False

    """
    # defences
    _check_parent_dir_exists(path, "path", create=True)

    if os.path.exists(f"{path}/gtfs_report") and not overwrite:
        raise FileExistsError(
            "Report already exists at path: "
            f"[{path}]."
            "Consider setting overwrite=True"
            "if you'd like to overwrite this."
        )

    # make gtfs_report dir
    try:
        os.mkdir(f"{path}/gtfs_report")
    except FileExistsError:
        pass
    shutil.copy(
        src="src/transport_performance/gtfs/report/css_styles/styles.css",
        dst=f"{path}/gtfs_report",
    )
    return None
