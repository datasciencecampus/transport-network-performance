"""Cleaners & utilities specific to the calendar table."""
import calendar

import numpy as np
import pandas as pd

from transport_performance.utils.defence import (
    _type_defence,
    _check_column_in_df,
)


def create_calendar_from_dates(calendar_dates: pd.DataFrame) -> pd.DataFrame:
    """Use the calendar_dates table to populate a calendar table.

    Use this in cases where a gtfs feed has elected to use a calendar-dates
    table and no calendar. Only adds values for calendar_dates entries tagged
    as exception_type 1. Exception_type 2 is not treated. Those entries will
    are left to override calendar values.

    Parameters
    ----------
    calendar_dates : pd.DataFrame
        The calendar_dates table.

    Returns
    -------
    pd.DataFrame
        A calendar table based on values in the calendar_dates table.

    Raises
    ------
    TypeError
        `calendar_dates` is not a pd.DataFrame
    IndexError
        `calendar_dates` does not contain any of the following columns -
        service_id, date, exception_type.

    """
    # defence checking
    _type_defence(calendar_dates, "calendar_dates", pd.DataFrame)
    exp_cols = ["service_id", "date", "exception_type"]
    for nm in exp_cols:
        _check_column_in_df(calendar_dates, nm)

    # create an empty calendar df
    days = [day.lower() for day in calendar.day_name]
    new_calendar = pd.DataFrame()
    new_calendar["service_id"] = calendar_dates["service_id"].unique()
    day_df = pd.DataFrame(
        {i: np.zeros(len(new_calendar), dtype=int) for i in days},
        index=list(range(0, len(new_calendar))),
    )
    new_calendar = pd.concat([new_calendar, day_df], axis=1)
    # create the start and end_date columns
    new_calendar["start_date"] = ""
    new_calendar["end_date"] = ""
    new_calendar.set_index(new_calendar["service_id"], inplace=True)
    # update this empty calendar with values from calendar_dates
    for i, r in calendar_dates.iterrows():
        # only update if calendar_dates exception_type is 1 (adding a service)
        if r["exception_type"] == 1:
            date_affected = r["date"]
            day_affected = pd.to_datetime(date_affected).weekday()
            # update weekday column entry to show the service runs on that day
            new_calendar.loc[
                r["service_id"], new_calendar.columns[day_affected + 1]
            ] = 1
            # update the start & end date columns
            s_date = new_calendar.loc[r["service_id"], "start_date"]
            if s_date == "":
                new_calendar.loc[r["service_id"], "start_date"] = date_affected
            else:
                s_date = min(s_date, date_affected)
                new_calendar.loc[r["service_id"], "start_date"] = s_date
            e_date = new_calendar.loc[r["service_id"], "end_date"]
            if e_date == "":
                new_calendar.loc[r["service_id"], "end_date"] = date_affected
            else:
                e_date = max(e_date, date_affected)
                new_calendar.loc[r["service_id"], "end_date"] = e_date
        else:
            # Type 2 removes a service and will override the calendar
            pass

    return new_calendar.reset_index(drop=True)
