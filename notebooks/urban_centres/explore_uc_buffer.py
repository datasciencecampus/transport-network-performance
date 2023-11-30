# %%
"""Notebook to explore buffer calculation in different CRSs."""

# %load_ext autoreload
# %autoreload 2

import folium
import geopandas as gpd
import pandas as pd
from geopandas.testing import assert_geodataframe_equal
from haversine import haversine_vector
from pyprojroot import here
from shapely.geometry import box

from transport_performance.population.rasterpop import RasterPop
from transport_performance.urban_centres.raster_uc import UrbanCentre

# %%
# handle area selection for different urban centres
area = "newport"

if area == "newport":
    uc_input_path = here("data/processed/urban_centre/newport_uc_merged.tif")
    uc_bbox = [
        -415194.5798256779,
        6014542.675452075,
        -178899.95729310974,
        6239314.71054581,
    ]
    uc_centre = [51.5773597, -2.9660008]
    uc_centre_crs = "EPSG:4326"
    uc_buffer_size = 12000
    updated_crs = "EPSG:27700"
elif area == "leeds":
    uc_input_path = here("data/processed/urban_centre/leeds_uc_merged.tif")
    uc_bbox = [-145000.0, 6247000.0, -93000.0, 6282000.0]
    uc_centre = [-119000.0, 6264500.0]
    uc_centre_crs = "ESRI:54009"
    uc_buffer_size = 12000
    updated_crs = "EPSG:27700"

else:
    raise ValueError(f"Unkown area name {area}.")

# common population inputs
pop_threshold = 1
pop_input_path = here("data/processed/population/pop_merged_resampled.tif")


def visualise_uc_results(gdf: gpd.GeoDataFrame) -> folium.Map:
    """Visualise uc boundary, buffer and bounding box."""
    # add a base map with no tile - then is it not on a layer control
    m = folium.Map(tiles=None, control_scale=True, zoom_control=True)

    tiles = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStre'
        'etMap</a> contributors &copy; <a href="https://carto.com/attribut'
        'ions">CARTO</a>'
    )

    # add a tile layer
    folium.TileLayer(
        tiles=tiles,
        attr=attr,
        show=False,
        control=False,
    ).add_to(m)

    m = gdf[::-1].explore("label", cmap="viridis", m=m)

    m.fit_bounds(m.get_bounds())

    return m


def wire_map(gdf: gpd.GeoDataFrame, column: str) -> folium.Map:
    """Visualise boundary extents."""
    m = folium.Map(tiles=None, control_scale=True, zoom_control=True)

    tiles = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStre'
        'etMap</a> contributors &copy; <a href="https://carto.com/attribut'
        'ions">CARTO</a>'
    )

    # add a tile layer
    folium.TileLayer(
        tiles=tiles,
        attr=attr,
        show=False,
        control=False,
    ).add_to(m)

    m = gdf.explore(
        column,
        categorical=True,
        cmap=["darkorange", "blue"],
        style_kwds={"fill": None},
        m=m,
    )

    m.fit_bounds(m.get_bounds())

    return m


def explore_sources_not_included(pop_gdf_in, uc_gdf_in, centroid_gdf_in):
    """Calcualte and visualise distances to urban centre relative to thresh."""
    # copy inputs to prevent modification
    pop_gdf = pop_gdf_in.copy()
    uc_gdf = uc_gdf_in.copy()
    centroid_gdf = centroid_gdf_in.copy()

    # get cells which touch the urban centre boundary
    # simplifies and speeds up upcoming cross join
    in_urban_centre = pop_gdf[pop_gdf.within_urban_centre]
    touches_uc = in_urban_centre.geometry.touches(
        uc_gdf.set_index("label").loc["vectorized_uc"].geometry.exterior
    )

    # build a map showing cells touching the urban centre for QA-ing
    touch_map = uc_gdf.loc[uc_gdf.label == "vectorized_uc"].explore(
        color="grey"
    )
    touch_map = in_urban_centre.explore(color="red", m=touch_map)
    touch_map = in_urban_centre[touches_uc].explore(m=touch_map)

    # take sources as all cells outside the urban centre
    # take destinations as only cells which touch the urban centre boundary
    source_ids = pop_gdf[~pop_gdf.within_urban_centre].id.values
    dest_ids = in_urban_centre[touches_uc].id.values

    # get corresponding centroid data
    sources = centroid_gdf[centroid_gdf.id.isin(source_ids)]
    dests = centroid_gdf[centroid_gdf.id.isin(dest_ids)]

    # build all source/destination paris, and calculate haversine distance
    cross_join = sources.merge(
        dests, how="cross", suffixes=["_source", "_dest"]
    )
    cross_join["lat_long_source"] = cross_join.centroid_source.apply(
        lambda x: (x.y, x.x)
    )
    cross_join["lat_long_dest"] = cross_join.centroid_dest.apply(
        lambda x: (x.y, x.x)
    )
    cross_join["distance"] = haversine_vector(
        list(cross_join["lat_long_source"]), list(cross_join["lat_long_dest"])
    )

    # get minimum distance for each source cell and highlight scenarios where
    # this minimum is greater than the desired distance cut-off
    min_dist = cross_join.groupby("id_source", as_index=False)[
        "distance"
    ].min()
    outside_max_dist = min_dist[min_dist["distance"] > 11.25].id_source.values

    # classify based on whether max distance is exceeded
    pop_gdf.loc[
        pop_gdf.id.isin(outside_max_dist), "Distance Cutoff"
    ] = "Outside"
    pop_gdf.loc[
        ~pop_gdf.id.isin(outside_max_dist), "Distance Cutoff"
    ] = "Inside"

    # build an output map to show classification
    outliers_map = folium.Map(
        tiles=None, control_scale=True, zoom_control=True
    )

    # add a tile layer
    tiles = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStre'
        'etMap</a> contributors &copy; <a href="https://carto.com/attribut'
        'ions">CARTO</a>'
    )
    folium.TileLayer(
        tiles=tiles,
        attr=attr,
        show=False,
        control=False,
    ).add_to(outliers_map)

    # add inside/outside cells
    outliers_map = pop_gdf.explore(
        "Distance Cutoff",
        categorical=True,
        cmap=["blue", "red"],
        m=outliers_map,
    )

    # add uc boundary as a thicker black line
    outliers_map = uc_gdf.loc[uc_gdf.label == "vectorized_uc"].explore(
        color="black", style_kwds={"fill": None}, m=outliers_map
    )

    # add buffer as a dashed black line
    outliers_map = uc_gdf.loc[uc_gdf.label == "buffer"].explore(
        color="black",
        style_kwds={
            "fill": None,
            "dashArray": 7,
            "weight": 1,
        },
        m=outliers_map,
    )

    outliers_map.fit_bounds(outliers_map.get_bounds())

    return touch_map, outliers_map


# %% [markdown] noqa: D212, D400, D415
# # Get Urban Centre Geometries

# %%
# put bbox into a geopandas dataframe for `get_urban_centre` input
bbox_gdf = gpd.GeoDataFrame(geometry=[box(*uc_bbox)], crs="ESRI:54009")

# detect urban centre - using same CRS as input data
uc = UrbanCentre(here(uc_input_path))
uc_original_crs = uc.get_urban_centre(
    bbox_gdf,
    centre=tuple(uc_centre),
    centre_crs=uc_centre_crs,
    buffer_size=uc_buffer_size,
    buffer_estimation_crs="ESRI:54009",
)

# visualise results
visualise_uc_results(uc_original_crs)

# %%
# put bbox into a geopandas dataframe for `get_urban_centre` input
bbox_gdf = gpd.GeoDataFrame(geometry=[box(*uc_bbox)], crs="ESRI:54009")

# detect urban centre - using alternative CRS for buffer stage
uc = UrbanCentre(here(uc_input_path))
uc_updated_crs = uc.get_urban_centre(
    bbox_gdf,
    centre=tuple(uc_centre),
    centre_crs=uc_centre_crs,
    buffer_size=uc_buffer_size,
    buffer_estimation_crs=updated_crs,
)

visualise_uc_results(uc_updated_crs)

# %% [markdown] noqa: D212, D400, D415
# # Explore Urban Centre Similarities and Differences

# %%
# ensure urban centre geometry remains unchanged
updated_uc = uc_updated_crs[uc_original_crs["label"] == "vectorized_uc"].copy()
original_uc = uc_original_crs[
    uc_original_crs["label"] == "vectorized_uc"
].copy()
assert_geodataframe_equal(updated_uc, original_uc)

# %%
# visually compare differences in buffers
updated_buffer = uc_updated_crs[uc_original_crs["label"] == "buffer"].copy()
updated_buffer.loc[:, "Buffer CRS"] = updated_crs

original_buffer = uc_original_crs[uc_original_crs["label"] == "buffer"].copy()
original_buffer.loc[:, "Buffer CRS"] = original_buffer.crs.to_string()

buffers = pd.concat([original_buffer, updated_buffer], axis=0).reset_index(
    drop=True
)

wire_map(buffers, "Buffer CRS")

# %%
# visually compare differences in bounding boxes
updated_bbox = uc_updated_crs[uc_original_crs["label"] == "bbox"].copy()
updated_bbox.loc[:, "BBOX CRS"] = updated_crs

original_bbox = uc_original_crs[uc_original_crs["label"] == "bbox"].copy()
original_bbox.loc[:, "BBOX CRS"] = uc_original_crs.crs.to_string()

bboxs = pd.concat([original_bbox, updated_bbox], axis=0).reset_index(drop=True)

wire_map(bboxs, "BBOX CRS")

# %% [markdown] noqa: D212, D400, D415
# # Get Population data for each scenario

# %%
# extract population using boundaries in original CRS

# extract geometries from urban centre detection
aoi_bounds = uc_original_crs.set_index("label").loc["buffer"].geometry
urban_centre_bounds = (
    uc_original_crs.set_index("label").loc["vectorized_uc"].geometry
)

# get population data
rp_original = RasterPop(pop_input_path)
pop_original_gdf, centroid_original_gdf = rp_original.get_pop(
    aoi_bounds,
    threshold=pop_threshold,
    urban_centre_bounds=urban_centre_bounds,
)

rp_original.plot()

# %%
# extract population using boundaries in updated CRS

# extract geometries from urban centre detection
aoi_bounds = uc_updated_crs.set_index("label").loc["buffer"].geometry
urban_centre_bounds = (
    uc_updated_crs.set_index("label").loc["vectorized_uc"].geometry
)

# get population data
rp_updated = RasterPop(pop_input_path)
pop_updated_gdf, centroid_updated_gdf = rp_updated.get_pop(
    aoi_bounds,
    threshold=pop_threshold,
    urban_centre_bounds=urban_centre_bounds,
)

rp_updated.plot()

# %% [markdown] noqa: D212, D400, D415
# # Explore cells inside/outside 11.25Km limit

# %%
original_touching_map, original_outliers_map = explore_sources_not_included(
    pop_original_gdf, uc_original_crs, centroid_original_gdf
)
original_outliers_map

# %%
updated_touching_map, updated_outliers_map = explore_sources_not_included(
    pop_updated_gdf, uc_updated_crs, centroid_updated_gdf
)
updated_outliers_map

# %%
