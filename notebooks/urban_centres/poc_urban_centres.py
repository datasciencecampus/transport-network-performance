"""[EXPERIMENTAL] Notebook to generate and explore urban centres.

Uses GHSL - Global Human Settlement Layer - for the population input.
GHS-POP-R2023A for the year 2020 and 1x1km population grids[1]. Tiles
R3-C18. R3-C19, R4-C18, R4-C19 (which include British isles and France)
are used.

Rules to calculate urban centres follow the definition proposed by
Eurostat[2], without the optional corrections proposed in section 8.2.

Urban centres created and displayed are Newport, Leeds, London and Marseille.

References
----------
- [1] Schiavina, Marcello; Freire, Sergio; Alessandra Carioli;
MacManus, Kytt (2023): GHS-POP R2023A - GHS population grid multitemporal
(1975-2030). European Commission, Joint Research Centre (JRC)
[Dataset] doi: 10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE
PID: http://data.europa.eu/89h/2ff68a52-5b5b-4a22-8f40-c41da8332cfe

-[2] European Commission, Eurostat, Applying the degree of urbanisation:
a methodological manual to define cities, towns and rural areas for
international comparisons : 2021 edition, Publications Office of the
European Union, 2021, https://data.europa.eu/doi/10.2785/706535

"""
# %%
import os

import geopandas as gpd
from shapely.geometry import box
from pyprojroot import here
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt
import numpy as np
import folium

import heimdall_transport.urban_centres.urban_centres as uc

# %%
# merge raster tiles into single file

# merge file list - in data/raw/
MERGE_FILE_LIST = [
    "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C19.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R4_C18.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R4_C19.tif",
]

# create full filepath for merged tif file
MERGED_DIR = os.path.join(here(), "data", "processed", "GHSL_2020_merged.tif")

# build merge file directories
merge_dirs = [
    os.path.join(here(), "data", "raw", name) for name in MERGE_FILE_LIST
]

# build merge dataset list
arrays = []
for merge_dir in merge_dirs:
    arrays.append(rxr.open_rasterio(merge_dir, masked=True))

# merge the datasets together
xds_merged = merge_arrays(arrays)
xds_merged.plot()

# write to GeoTIFF raster file
xds_merged.rio.to_raster(MERGED_DIR)


# %%
# get bbox of interest
BBOX_DICT = {
    "newport": (-3.206023, 51.503789, -2.726744, 51.680820),
    "leeds": (-2.212408, 53.450803, -0.862837, 54.095581),
    "london": (-1.054688, 51.134555, 0.873413, 51.835778),
    "marseille": (3.319416, 42.673743, 7.873249, 44.671964),
}

# %%
# Newport
# area of interest
AREA_OF_INTEREST = "newport"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

bbox_npt = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_npt = bbox_npt.to_crs("esri:54009")

# pop only criteria
masked_rst, aff, rst_src = uc.filter_cells(MERGED_DIR, bbox_npt)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

npt = uc.fill_gaps(urban_centres)
print(np.unique(npt))
plt.imshow(npt == 8)

gdf_npt = uc.vectorize_uc(npt, 8, aff, rst_src)
npt_buffer = uc.add_buffer(gdf_npt)

fig = plt.figure
m = gdf_npt.explore(color="red")
m = npt_buffer.explore(m=m)
folium.LayerControl().add_to(m)
m


# %%
# Leeds
# area of interest
AREA_OF_INTEREST = "leeds"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

bbox_lds = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_lds = bbox_lds.to_crs("esri:54009")

# pop only criteria
masked_rst, aff, rst_src = uc.filter_cells(MERGED_DIR, bbox_lds)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

lds = uc.fill_gaps(urban_centres)
print(np.unique(lds))
plt.imshow(lds == 29)

gdf_lds = uc.vectorize_uc(lds, 29, aff, rst_src)
lds_buffer = uc.add_buffer(gdf_lds)

fig = plt.figure
m = gdf_lds.explore(color="red")
m = lds_buffer.explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
# London
# area of interest
AREA_OF_INTEREST = "london"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

bbox_lnd = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_lnd = bbox_lnd.to_crs("esri:54009")

# pop only criteria
masked_rst, aff, rst_src = uc.filter_cells(MERGED_DIR, bbox_lnd)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

lnd = uc.fill_gaps(urban_centres)
print(np.unique(lnd))
plt.imshow(lnd == 22)

gdf_lnd = uc.vectorize_uc(lnd, 22, aff, rst_src)
lnd_buffer = uc.add_buffer(gdf_lnd)

fig = plt.figure
m = gdf_lnd.explore(color="red")
m = lnd_buffer.explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
# Marseille
# area of interest
AREA_OF_INTEREST = "marseille"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

bbox_mrs = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_mrs = bbox_mrs.to_crs("esri:54009")

# pop only criteria
masked_rst, aff, rst_src = uc.filter_cells(MERGED_DIR, bbox_mrs)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

mrs = uc.fill_gaps(urban_centres)
print(np.unique(mrs))
plt.imshow(mrs == 246)

gdf_mrs = uc.vectorize_uc(mrs, 246, aff, rst_src)
mrs_buffer = uc.add_buffer(gdf_mrs)

fig = plt.figure
m = gdf_mrs.explore(color="red")
m = mrs_buffer.explore(m=m)
folium.LayerControl().add_to(m)
m
# %%
fig = plt.figure
m = gdf_mrs.explore(color="red")
m = mrs_buffer.explore(m=m)
m = gdf_lnd.explore(color="red", m=m)
m = lnd_buffer.explore(m=m)
m = gdf_npt.explore(color="red", m=m)
m = npt_buffer.explore(m=m)
m = gdf_lds.explore(color="red", m=m)
m = lds_buffer.explore(m=m)
folium.LayerControl().add_to(m)
m
# %%
