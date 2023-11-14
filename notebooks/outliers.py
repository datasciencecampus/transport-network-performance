"""Provisional notebook to identify isolated cells."""
# %%
import pandas as pd
import geopandas as gpd
import pickle
from pyprojroot import here


# %%
# functions
def detect_isolated(od_matrix: pd.DataFrame, threshold: int = 50) -> tuple:
    """Get list of trapped cells ids.

    Parameters
    ----------
    od_matrix: pd.DataFrame
        Default Origins-Destinations matrix from r5py, including columns
        `to_id`, `from_id` and `travel_times`.
    threshold: int
        Apply only to destinations with number of origins where travel time
        is not NaN less than threshold, i.e. destinations that are only
        reachable from a small number of places.

    Return
    ------
    tuple
        Tuple with ids of identified isolated cells.

    Notes
    -----
    This method cannot detect groups of isolated cells where part of
    them are ouside the urban centre modules. That is because the method is
    based on number of origins, and cells outside the urban centre are not used
    as destination.

    """
    od_matrix_flag = od_matrix.copy()

    # finds count of origins per destination
    count_origins = (
        od_matrix.dropna()
        .groupby("to_id")["from_id"]
        .count()
        .reset_index()
        .rename(columns={"from_id": "count"})
    )

    # keeps only destinations with fewer origins than threshold
    count_origins_filt = count_origins[count_origins["count"] < threshold]

    # inner join with od_matrix to keep only filtered destinations
    od_matrix_outliers = od_matrix_flag.merge(
        count_origins_filt, on="to_id", how="inner"
    ).dropna()

    # creates tuple with all origins to a single destination
    from_id = (
        od_matrix_outliers.groupby("to_id")["from_id"].agg(tuple).reset_index()
    )

    # creates tuple with all destinations to a single set of origins
    to_id = from_id.groupby("from_id")["to_id"].agg(tuple).reset_index()

    # keeps cases where set of origins equals set of destinations
    d = to_id[to_id["from_id"].apply(set) == to_id["to_id"].apply(set)]

    # creates set of destination ids to return
    isolated = tuple()
    for i, row in d.iterrows():
        isolated = isolated + row["from_id"]

    return isolated


# the functions below are not intended to use once we have metrics module
def get_coords_dist(
    od_matrix: pd.DataFrame, centroids: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Get centroid coordinates for OD matrix."""
    # merge on from centroid
    distance_df = (
        od_matrix.merge(
            centroids.to_crs("EPSG:27700")[["id", "centroid"]],
            left_on="from_id",
            right_on="id",
        )
        .drop(columns=["id"])
        .rename(columns={"centroid": "from_centroid"})
    )

    # merge on to centroid
    distance_df = (
        distance_df.merge(
            centroids.to_crs("EPSG:27700")[["id", "centroid"]],
            left_on="to_id",
            right_on="id",
        )
        .drop(columns=["id"])
        .rename(columns={"centroid": "to_centroid"})
    )

    # convert to geoseries and caluclate distance between from and to
    from_s = gpd.GeoSeries(distance_df.from_centroid, crs="EPSG:27700")
    to_s = gpd.GeoSeries(distance_df.to_centroid, crs="EPSG:27700")
    distance_df["centroid_distance"] = from_s.distance(to_s)

    return distance_df


def calc_transport_perf(
    distance_df: pd.DataFrame,
    pop_gdf: gpd.GeoDataFrame,
    max_dist: int,
    max_time: int,
) -> gpd.GeoDataFrame:
    """Calculate transport performance."""
    distance_df["from_population"] = distance_df.merge(
        pop_gdf[["id", "population"]],
        left_on="from_id",
        right_on="id",
        how="left",
    )["population"]
    # calculate total population reach a destination within the time and
    # distance group by to_id so that it's total population that reaches
    # the destination id
    numerator = (
        distance_df[
            (distance_df.centroid_distance <= max_dist)
            & (distance_df.travel_time <= max_time)
        ]
        .groupby("to_id")["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "reachable_population"})
    )

    # calculate total population that is nearby within the distance threshold
    # group by to_id so that it's total population nearby the destination
    denominator = (
        distance_df[(distance_df.centroid_distance <= max_dist)]
        .groupby("to_id")["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "nearby_population"})
    )

    # create a transport performance gdf, but merging dataframes first the
    # numerator - remove to_id column since it's not needed after merge
    perf_gdf = pop_gdf.merge(numerator, left_on="id", right_on="to_id").drop(
        columns=["to_id"]
    )

    # then the denominator - remove to_id again, since it's not needed after
    # merge
    perf_gdf = perf_gdf.merge(
        denominator, left_on="id", right_on="to_id"
    ).drop(columns=["to_id"])

    # calculate transport performance, as a percentage
    perf_gdf["transport_performance"] = (
        perf_gdf["reachable_population"] / perf_gdf["nearby_population"]
    ) * 100

    return perf_gdf


def calc_num_origins(
    perf_gdf: gpd.GeoDataFrame, od_matrix: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Calculate number of origins to destination."""
    origins_to_dest = (
        od_matrix.dropna()
        .groupby("to_id")["from_id"]
        .count()
        .reset_index()
        .rename(columns={"from_id": "count_origins"})
    )

    perf_gdf = perf_gdf.merge(
        origins_to_dest, left_on="id", right_on="to_id"
    ).drop(columns="to_id")

    return perf_gdf


# %%
##########
# london #
##########
# load london data
with open(here("data/processed/population/london_rp.pkl"), "rb") as f:
    london_rp = pickle.load(f)

uc_bounds_london = london_rp._uc_gdf
rasterpop_london = london_rp.pop_gdf
centroids_london = london_rp.centroid_gdf

od_london = pd.read_parquet(
    here("data/processed/od_matrix_london")
).reset_index()

# %%
# merge
london_dist = get_coords_dist(od_london, centroids_london)
london_dist_gdf = gpd.GeoDataFrame(london_dist, geometry="from_centroid")
london_perf = calc_transport_perf(london_dist, rasterpop_london, 11250, 45)
london_perf = calc_num_origins(london_perf, od_london)

# %%
# calculate and plot isolated cells
# note that this method cannot detect groups of isolated cells where part of
# them are ouside the urban centre modules. That is because the method is based
# on number of origins, and cells outside the urban centre are not used as
# destination. The only example I could find based on number of origins and
# overall performance is 24508.
london_isolated = detect_isolated(od_london)
m = london_perf.explore(column="transport_performance")
london_perf[london_perf.id.isin(london_isolated)].explore(m=m, color="red")
m.save(here("outputs/e2e/london/metrics/london_metrics_isolated.html"))

# %%
#############
# marseille #
#############
# load marseille data
with open(here("data/processed/population/marseille_rp.pkl"), "rb") as f:
    marseille_rp = pickle.load(f)

uc_bounds_marseille = marseille_rp._uc_gdf
rasterpop_marseille = marseille_rp.pop_gdf
centroids_marseille = marseille_rp.centroid_gdf

od_marseille = pd.read_pickle(
    here("outputs/e2e/marseille/analyse_network/travel_times.pkl")
)

# %%
# merge
marseille_dist = get_coords_dist(od_marseille, centroids_marseille)
marseille_dist_gdf = gpd.GeoDataFrame(marseille_dist, geometry="from_centroid")
marseille_perf = calc_transport_perf(
    marseille_dist, rasterpop_marseille, 11250, 45
)
marseille_perf = calc_num_origins(marseille_perf, od_marseille)

# %%
# calculate and plot isolated cells
marseille_isolated = detect_isolated(od_marseille)
m = marseille_perf.explore(column="transport_performance")
marseille_perf[marseille_perf.id.isin(marseille_isolated)].explore(
    m=m, color="red"
)
m.save(here("outputs/e2e/marseille/metrics/marseille_metrics_isolated.html"))


# %%
#########
# leeds #
#########
# load leeds data
with open(here("outputs/e2e/leeds/rasterpop/rasterpop.pkl"), "rb") as f:
    leeds_rp = pickle.load(f)

uc_bounds_leeds = leeds_rp._uc_gdf
rasterpop_leeds = leeds_rp.pop_gdf
centroids_leeds = leeds_rp.centroid_gdf

od_leeds = pd.read_pickle(
    here("outputs/e2e/leeds/analyse_network/travel_times.pkl")
)

# %%
# merge
leeds_dist = get_coords_dist(od_leeds, centroids_leeds)
leeds_dist_gdf = gpd.GeoDataFrame(leeds_dist, geometry="from_centroid")
leeds_perf = calc_transport_perf(leeds_dist, rasterpop_leeds, 11250, 45)
leeds_perf = calc_num_origins(leeds_perf, od_leeds)

# %%
# calculate and plot isolated cells
leeds_isolated = detect_isolated(od_leeds)
m = leeds_perf.explore(column="transport_performance")
leeds_perf[leeds_perf.id.isin(leeds_isolated)].explore(m=m, color="red")
m.save(here("outputs/e2e/leeds/metrics/leeds_metrics_isolated.html"))

# %%
