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
import folium

import transport_performance.urban_centres.raster_uc as ucc

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
    os.path.join(here(), "data", "external", "urban_centre", name)
    for name in MERGE_FILE_LIST
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
    "marseille": (5.107508, 43.096155, 5.623020, 43.499695),
}

# %%
# Newport
# area of interest
AREA_OF_INTEREST = "newport"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

# bbox
bbox_npt = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_npt_r = bbox_npt.to_crs("esri:54009")

# bbox centroid
bbox_npt_centroid = bbox_npt_r.centroid
coords = (bbox_npt_centroid.x[0], bbox_npt_centroid.y[0])

# pop only criteria
npt = ucc.UrbanCentre(path=(MERGED_DIR))
npt_uc = npt.get_urban_centre(bbox_npt_r, coords)

fig = plt.figure
m = npt_uc[npt_uc["label"] == "vectorized_uc"].explore(color="red")
m = npt_uc[npt_uc["label"] == "buffer"].explore(m=m)
folium.LayerControl().add_to(m)
m


# %%
# Leeds
# area of interest
AREA_OF_INTEREST = "leeds"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

# bbox
bbox_lds = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_lds_r = bbox_lds.to_crs("esri:54009")

# bbox centroid
bbox_lds_centroid = bbox_lds_r.centroid
coords = (bbox_lds_centroid.x[0], bbox_lds_centroid.y[0])

# pop only criteria
lds = ucc.UrbanCentre(path=(MERGED_DIR))
lds_uc = npt.get_urban_centre(bbox_lds_r, coords)

fig = plt.figure
m = lds_uc[lds_uc["label"] == "vectorized_uc"].explore(color="red")
m = lds_uc[lds_uc["label"] == "buffer"].explore(m=m)
folium.LayerControl().add_to(m)
m


# %%
# London
# area of interest
AREA_OF_INTEREST = "london"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

# bbox
bbox_lnd = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_lnd_r = bbox_lnd.to_crs("esri:54009")

# bbox centroid
bbox_lnd_centroid = bbox_lnd_r.centroid
coords = (bbox_lnd_centroid.x[0], bbox_lnd_centroid.y[0])

# pop only criteria
lnd = ucc.UrbanCentre(path=(MERGED_DIR))
lnd_uc = npt.get_urban_centre(bbox_lnd_r, coords)

fig = plt.figure
m = lnd_uc[lnd_uc["label"] == "vectorized_uc"].explore(color="red")
m = lnd_uc[lnd_uc["label"] == "buffer"].explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
# Marseille
# area of interest
AREA_OF_INTEREST = "marseille"
BBOX = BBOX_DICT[AREA_OF_INTEREST]

# bbox
bbox_mrs = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box(*BBOX)])
bbox_mrs_r = bbox_mrs.to_crs("esri:54009")

# bbox centroid
bbox_mrs_centroid = bbox_mrs_r.centroid
coords = (bbox_mrs_centroid.x[0], bbox_mrs_centroid.y[0])

# pop only criteria
mrs = ucc.UrbanCentre(path=(MERGED_DIR))
mrs_uc = npt.get_urban_centre(bbox_mrs_r, coords)

fig = plt.figure
m = mrs_uc[mrs_uc["label"] == "vectorized_uc"].explore(color="red")
m = mrs_uc[mrs_uc["label"] == "buffer"].explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
fig = plt.figure
m = npt_uc[npt_uc["label"] == "vectorized_uc"].explore(color="red")
m = npt_uc[npt_uc["label"] == "buffer"].explore(m=m)
m = lds_uc[lds_uc["label"] == "vectorized_uc"].explore(color="red", m=m)
m = lds_uc[lds_uc["label"] == "buffer"].explore(m=m)
m = lnd_uc[lnd_uc["label"] == "vectorized_uc"].explore(color="red", m=m)
m = lnd_uc[lnd_uc["label"] == "buffer"].explore(m=m)
m = mrs_uc[mrs_uc["label"] == "vectorized_uc"].explore(color="red", m=m)
m = mrs_uc[mrs_uc["label"] == "buffer"].explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
