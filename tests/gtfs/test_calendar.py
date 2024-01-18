"""Test calendar utilities."""
import pytest

import pandas as pd

from transport_performance.gtfs.calendar import create_calendar_from_dates


class TestCreateCalendarFromDates(object):
    """Tests for create_calendar_from_dates."""

    def test_create_calendar_from_dates_defence(self):
        """Test defensive checks for create_calendar_from_dates."""
        with pytest.raises(TypeError, match=".DataFrame'>. Got <class 'int'>"):
            create_calendar_from_dates(calendar_dates=1)
        with pytest.raises(
            IndexError, match="service_id' is not a column in the dataframe."
        ):
            create_calendar_from_dates(
                calendar_dates=pd.DataFrame({"foo": 0}, index=[0])
            )
