"""Cleaners & utilities specific to the calendar table."""
import calendar
from copy import deepcopy

import pandas as pd

from transport_performance.utils.defence import (
    _type_defence,
    _check_column_in_df,
)


def create_calendar_from_dates(calendar_dates: pd.DataFrame) -> pd.DataFrame:
    """Use the calendar_dates table to populate a calendar table.

    Use this in cases where a gtfs feed has elected to use a calendar-dates
    table and no calendar. Only adds values for calendar_dates entries tagged
    as exception_type 1. Exception_type 2 is not treated. Those entries are
    ignored whilst making the calendar table.

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
    days = [day.lower() for day in calendar.day_name]
    # clean calendar_dates
    cal1 = calendar_dates[calendar_dates.exception_type == 1].copy()
    cal1.drop("exception_type", axis=1, inplace=True)
    # get list of dates and convert to days of the week
    grouped = cal1.groupby("service_id").agg({"date": lambda x: list(set(x))})

    def _get_day_name(date: str) -> str:
        """Small helper function to get the named day of the week."""
        day_index = pd.to_datetime(date).weekday()
        return days[day_index]

    grouped["days"] = grouped["date"].apply(
        lambda x: list(set([_get_day_name(d) for d in x]))
    )
    # start and end date
    grouped["start_date"] = grouped["date"].apply(lambda x: min(x))
    grouped["end_date"] = grouped["date"].apply(lambda x: max(x))
    # clean up unused data
    grouped.drop("date", axis=1, inplace=True)
    grouped.reset_index(inplace=True)
    # add a column for each day
    for day in days:
        grouped[day] = (
            grouped["days"]
            .apply(lambda x: 1 if day in x else 0)
            .astype("int8")
        )
    grouped.drop("days", axis=1)
    # re-order index
    order = deepcopy(days)
    order.insert(0, "service_id")
    order.append("start_date")
    order.append("end_date")
    return grouped.loc[:, order]
