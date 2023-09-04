"""A set of functions that validate the GTFS data."""

import gtfs_kit.feed as GtfsFeed


def validate_travel_between_consecutive_stops(feed: GtfsFeed):
    """Validate the travel between consecutive stops in the GTFS data.

    Ensures that a trip is valid by examining the duration and distance
    of a trip. If a vehicle is travelling at an unusual speed, the trip can
    be deemed invalid.
    """
    pass


def validate_travel_over_multiple_stops(feed: GtfsFeed):
    """Validate travel over multiple stops in the GTFS data."""
    pass
