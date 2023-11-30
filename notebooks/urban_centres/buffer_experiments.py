"""Notebook to check buffer and distance to edge cells."""

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from haversine import Unit, haversine_vector
from pyprojroot import here
from shapely.geometry import box

from transport_performance.population.rasterpop import RasterPop
from transport_performance.urban_centres.raster_uc import UrbanCentre

# %%
# paths
uc_processed = here("data/processed/urban_centre/newport_uc_merged.tif")
rp_processed = here("data/processed/population/pop_merged_resampled.tif")

# uc params
bbox = [
    -415194.5798256779,
    6014542.675452075,
    -178899.95729310974,
    6239314.71054581,
]  # must in in 'ESRI:54009', this represents a BBOX around Wales
centre = [51.5773597, -2.9660008]
buffer_size = 12000
centre_crs = "epsg: 4326"

# %%
# urban centre
bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="ESRI:54009")

uc = UrbanCentre(uc_processed)
uc_gdf = uc.get_urban_centre(
    bbox_gdf,
    centre=tuple(centre),
    buffer_size=buffer_size,
    centre_crs=centre_crs,
)

# %%
# reproject to 27700 before recalculating buffer
uc_27700 = uc_gdf[uc_gdf["label"] == "vectorized_uc"].to_crs("epsg: 27700")
uc_27700_buffer = uc_27700.buffer(distance=12000)
uc_27700_buffer = uc_27700_buffer.to_crs("esri: 54009")

# %%
# compare both buffers
m = uc_27700_buffer.to_crs("esri: 54009").explore()
uc_gdf[uc_gdf["label"] == "buffer"].explore(
    m=m, color="red", style_kwds={"fill": False}
)
uc_gdf[uc_gdf["label"] == "vectorized_uc"].explore(
    m=m, color="red", style_kwds={"fill": False}
)

# %%
# replace mollweide buffer by 27700
uc_gdf.loc[1, "geometry"] = uc_27700_buffer.geometry[0]

# %%
# population
aoi_bounds = uc_gdf[uc_gdf.label == "buffer"].geometry[1]
urban_centre_bounds = uc_gdf[uc_gdf.label == "vectorized_uc"].geometry[0]

# get population data
rp = RasterPop(rp_processed)
pop_gdf, centroid_gdf = rp.get_pop(
    aoi_bounds,
    threshold=1,
    urban_centre_bounds=urban_centre_bounds,
)

# %%
# map of urban centre and population results
m = uc_gdf[uc_gdf.label == "buffer"].explore(
    style_kwds={"color": "black", "fillColor": "lightgrey"}
)
rp.pop_gdf.explore(m=m, color="green")
uc_gdf[uc_gdf.label == "vectorized_uc"].explore(
    color="red", m=m, style_kwds={"fill": False}
)
rp.centroid_gdf.explore(m=m, color="blue", style_kwds={"weight": 0.1})

m


# %%
# function to calculate haversine distances
def haversine_df(
    df: pd.DataFrame,
    orig: str,
    dest: str,
    unit: Unit = Unit.KILOMETERS,
) -> np.array:
    """Calculate haversine distance between shapely Point objects.

    Parameters
    ----------
    df : pd.DataFrame
        Cartesian product of origin and destinations geodataframes. Should
        contain a geometry for each.
    orig : str
        Name of the column containing the origin geometry.
    dest : str
        Name of the column containing the destination geometry.
    unit : Unit
        Unit to calculate distance.

    Returns
    -------
    dist_array : np.array
        Array with row-wise distances.

    """
    lat_long_orig = df[orig].apply(lambda x: (x.y, x.x))
    lat_long_dest = df[dest].apply(lambda x: (x.y, x.x))

    dist_array = haversine_vector(
        list(lat_long_orig), list(lat_long_dest), unit=unit
    )

    return dist_array


# %%
# calculate haversine and geopandas distances
# create cartesian product of origins and destinations
orig_gdf = rp.centroid_gdf.copy()
dest_gdf = rp.centroid_gdf[
    rp.centroid_gdf["within_urban_centre"] == True  # noqa: E712
].copy()

# haversine distance
full_gdf = orig_gdf.merge(dest_gdf, how="cross", suffixes=["_orig", "_dest"])
full_gdf["haversine_distance"] = haversine_df(
    full_gdf, "centroid_orig", "centroid_dest"
)

# geopandas distance (needs projection to Mollweide)
full_gdf["distance_mollweide_gpd"] = (
    gpd.GeoSeries(full_gdf.centroid_orig)
    .to_crs("esri: 54009")
    .distance(gpd.GeoSeries(full_gdf.centroid_dest).to_crs("esri: 54009"))
    / 1000
)

# geopandas distance (with epsg: 27700)
full_gdf["distance_27700_gpd"] = (
    gpd.GeoSeries(full_gdf.centroid_orig)
    .to_crs("epsg: 27700")
    .distance(gpd.GeoSeries(full_gdf.centroid_dest).to_crs("epsg: 27700"))
    / 1000
)

# difference
full_gdf["dif_mollweide"] = (
    full_gdf["haversine_distance"] - full_gdf["distance_mollweide_gpd"]
)

full_gdf["dif_27700"] = (
    full_gdf["haversine_distance"] - full_gdf["distance_27700_gpd"]
)


# %%
# distribution of differences
sns.kdeplot(data=full_gdf, x="dif_mollweide")
plt.show()

sns.kdeplot(data=full_gdf, x="dif_27700")
plt.show()


# %%
# average dif (sqared) by cell
# group by origin
full_gdf["absdif_mollweide"] = abs(full_gdf["dif_mollweide"])
full_gdf["absdif_27700"] = abs(full_gdf["dif_27700"])
avg_dif = (
    full_gdf.groupby("id_orig")[["absdif_mollweide", "absdif_27700"]]
    .mean()
    .reset_index()
)

# merge and plot
pop_dif = rp.pop_gdf.merge(avg_dif, left_on="id", right_on="id_orig")
pop_dif = gpd.GeoDataFrame(pop_dif)

# %%
pop_dif.explore(column="absdif_mollweide")
# %%
pop_dif.explore(column="absdif_27700")

# %%
# get cells outside of distance (using haversine)
within_distance = full_gdf[full_gdf["haversine_distance"] <= 11.25]
unused = set(full_gdf["id_orig"]).difference(set(within_distance["id_orig"]))

pop = rp.pop_gdf.copy()
pop["unused"] = np.where(pop["id"].isin(unused), 1, 0)
cent = rp.centroid_gdf.copy()
cent["unused"] = np.where(cent["id"].isin(unused), 1, 0)

m = uc_gdf[uc_gdf.label == "buffer"].explore(
    style_kwds={"color": "black", "fillColor": "lightgrey"}
)
pop.explore(m=m, color="green", column="unused", categorical=True)
uc_gdf[uc_gdf.label == "vectorized_uc"].explore(
    color="red", m=m, style_kwds={"fill": False}
)
m

# %%
# get cells outside of distance (using geopandas mollweide)
within_distance = full_gdf[full_gdf["distance_mollweide_gpd"] <= 11.25]
unused = set(full_gdf["id_orig"]).difference(set(within_distance["id_orig"]))

pop = rp.pop_gdf.copy()
pop["unused"] = np.where(pop["id"].isin(unused), 1, 0)
cent = rp.centroid_gdf.copy()
cent["unused"] = np.where(cent["id"].isin(unused), 1, 0)

m = uc_gdf[uc_gdf.label == "buffer"].explore(
    style_kwds={"color": "black", "fillColor": "lightgrey"}
)
pop.explore(m=m, color="green", column="unused", categorical=True)
uc_gdf[uc_gdf.label == "vectorized_uc"].explore(
    color="red", m=m, style_kwds={"fill": False}
)
m

# %%
# get cells outside of distance (using geopandas 27700)
within_distance = full_gdf[full_gdf["distance_27700_gpd"] <= 11.25]
unused = set(full_gdf["id_orig"]).difference(set(within_distance["id_orig"]))

pop = rp.pop_gdf.copy()
pop["unused"] = np.where(pop["id"].isin(unused), 1, 0)
cent = rp.centroid_gdf.copy()
cent["unused"] = np.where(cent["id"].isin(unused), 1, 0)

m = uc_gdf[uc_gdf.label == "buffer"].explore(
    style_kwds={"color": "black", "fillColor": "lightgrey"}
)
pop.explore(m=m, color="green", column="unused", categorical=True)
uc_gdf[uc_gdf.label == "vectorized_uc"].explore(
    color="red", m=m, style_kwds={"fill": False}
)
m
