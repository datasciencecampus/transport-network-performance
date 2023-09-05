"""A notebook containing viable code for additional stop_id validation."""
# %%
# IMPORTS
from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.gtfs_utils import _add_validation_row


# %%
# SAMPLE FUNCTION
def validate_stops(gtfs: GtfsInstance) -> None:
    """Validate stop_ids across stops and stop_times tables.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GTFS instance to validate

    Returns
    -------
    None

    """
    stops = gtfs.feed.stops.copy()
    stop_times = gtfs.feed.stop_times.copy()

    # determine which stops are parent stops and remove them
    parents = stops.parent_station.unique()
    stops = stops[~stops.stop_id.isin(parents)]

    # get unique stop_ids from both tables as dataframes

    stops_ids = stops[["stop_id"]].drop_duplicates()
    stop_t_ids = stop_times[["stop_id"]].drop_duplicates()

    stops_ids["valid"] = 1
    stop_t_ids["valid"] = 1
    merged_ids = stops_ids.merge(
        stop_t_ids, how="outer", on="stop_id", suffixes=("_s", "_st")
    )

    stop_t_count = merged_ids["valid_st"].isna().count()
    stop_t_invalid_ids = merged_ids[
        merged_ids["valid_st"].isna()
    ].stop_id.unique()

    stops_count = merged_ids["valid_s"].isna().count()
    stops_invalid_ids = merged_ids[
        merged_ids["valid_s"].isna()
    ].stop_id.unique()

    if stops_count > 0:
        impacted_rows = list(
            gtfs.feed.stop_times[
                gtfs.feed.stop_times.stop_id.isin(stops_invalid_ids)
            ].index
        )
        _add_validation_row(
            gtfs=gtfs,
            _type="warning",
            message="stop_id's exist in stop_times but not in stops",
            table="stop_times",
            rows=impacted_rows,
        )

    if stop_t_count > 0:
        impacted_rows = list(
            gtfs.feed.stops[
                gtfs.feed.stops.stop_id.isin(stop_t_invalid_ids)
            ].index
        )
        _add_validation_row(
            gtfs=gtfs,
            _type="warning",
            message="stop_id's exist in stops but not in stop_times",
            table="stops",
            rows=impacted_rows,
        )
    return None
