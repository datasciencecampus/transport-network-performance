"""[EXPERIMENTAL] Notebook to explore functions to get urban centres.

Uses GHSL - Global Human Settlement Layer - for the population input.
GHS-POP-R2023A for the year 2020 and 1x1km population grids[1]. Tile
R3-C18 (which includes Ireland and part of the UK) is used.

Rules to calculate urban centres follow the definition proposed by
Eurostat[2], without the optional corrections proposed in section 8.2.

A comparison between the above custom calculation and the published
Settlement Model layers GHS-SMOD_GLOBE_R2023A[3] for the same year and
resolution is made for validation. The GHS-SMOD uses a different methodology
where cells are also included in the urban centre based on their built-up
area.

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

-[3] Schiavina, Marcello; Melchiorri, Michele; Pesaresi, Martino (2023):
GHS-SMOD R2023A - GHS settlement layers, application of the Degree of
Urbanisation methodology (stage I) to GHS-POP R2023A and GHS-BUILT-S R2023A,
multitemporal (1975-2030). European Commission, Joint Research Centre (JRC)
[Dataset] doi: 10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA
PID: http://data.europa.eu/89h/a0df7a6f-49de-46ea-9bde-563437a6e2ba

"""
# %%
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

# import rioxarray as rxr
# from affine import Affine

import numpy as np

# from scipy.ndimage import label, generic_filter, binary_dilation
# from rasterio.mask import raster_geometry_mask
import numpy.ma as ma

import heimdall_transport.urban_centres.urban_centres as uc
import shapely.geometry

import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs

# from rasterio.mask import raster_geometry_mask
from geocube.vector import vectorize
import rioxarray
import folium

# %%
file_pop = "../../data/raw/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif"  # noqa
file_smod = "../../data/raw/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif"  # noqa

# %%
bbox = gpd.read_file("../../data/raw/bbox.geojson")
bbox_rep = bbox.to_crs("esri:54009")
xmin, ymin, xmax, ymax = bbox_rep.total_bounds

# %%
# wards = gpd.read_file('../../data/raw/Wards_Dec_2020_UK_BGC_2022_-5150753419644615544.geojson')  # noqa
# wards_rep = wards.to_crs('esri:54009')

# wards_la_lkup = pd.read_csv('../../data/raw/Ward_to_Local_Authority_District_(December_2020)_Lookup_in_the_United_Kingdom_V2.csv')  # noqa
# wards_rep = wards_rep.merge(wards_la_lkup[['WD20CD', 'LAD20NM']], on='WD20CD')  # noqa
# wards_newport = wards_rep[wards_rep['LAD20NM'] == 'Newport']

# %%
# pop only criteria
masked_rst, aff, rst_src = uc.filter_cells(file_pop, bbox_rep)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

uc_filled = uc.fill_gaps(urban_centres)
plt.imshow(uc_filled)

newport = uc_filled == 16

# %%
# try to convert yo xarray for geocube
# coords = [list(range(d)) for d in newport.shape]
"""
h = [aff[0] * d + (aff[2] - aff[0] / 2)
     for d in range(1, newport.shape[0] + 1)]
v = [aff[4] * d + (aff[5] - aff[4] / 2)
     for d in range(1, newport.shape[1] + 1)]
coords = [h, v]
dims = ['x', 'y']

xcs = (
    xr.DataArray(newport, coords=coords, dims=dims)
    .astype('int32')
    .rio.write_nodata(-1)
    .rio.set_crs(src.crs)
    .rio.write_transform(aff)
)
"""
with rio.open(file_pop) as src:
    crs_pop = src.crs

npt = (
    xr.DataArray(newport)
    .astype("int32")
    .rio.write_nodata(-1)
    .rio.write_transform(aff)
    .rio.set_crs(src.crs, inplace=True)
)

gdf = vectorize(npt)
gdf.columns = ["label", "geometry"]
gdf = gdf[gdf["label"] == 1]

fig = plt.figure

m = gdf.explore(color="red")
m = gdf.buffer(10000).explore(m=m)
folium.LayerControl().add_to(m)
m

# %%
# urban centre over map
fig = plt.figure(figsize=(20, 20))
img = cimgt.OSM()
ax = plt.axes(projection=ccrs.Mollweide())
data_crs = ccrs.Mollweide()

height, width = newport.shape
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rio.transform.xy(aff, rows, cols)

ax.add_image(img, 10)

m = ax.pcolormesh(
    xs, ys, newport, transform=data_crs, alpha=0.3, cmap="viridis"
)

plt.show()

# %%
# save Newport urban centre to tif
with rio.open(file_pop) as src:
    crs_pop = src.crs

metadata = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": -200,
    "width": newport.shape[1],
    "height": newport.shape[0],
    "count": 1,
    "crs": crs_pop,
    "transform": aff,
    "compress": "lzw",
}

with rio.open("../../data/processed/newport_uc.tif", "w", **metadata) as dst:
    dst.write(newport, 1)

# %%
# vectorize urban centre
newport_rst = rioxarray.open_rasterio("../../data/processed/newport_uc.tif")

gdf = vectorize(newport_rst)
gdf.columns = ["label", "geometry"]
gdf = gdf[gdf["label"] == 1]

fig = plt.figure

m = gdf.explore(color="red")
m = gdf.buffer(10000).explore(m=m)
folium.LayerControl().add_to(m)
m

#################################
# %%
# Below, comparison between GHS-SMOD and pop only urban centres
# SMOD dataset
masked_rst_smod, affine, src_masked_crs = uc.filter_cells(file_smod, bbox_rep)
plt.imshow(masked_rst_smod >= 30)

# %%
# pop of simple urban centre
print("total_population", ma.sum(ma.masked_where(uc_filled < 1, masked_rst)))
plt.imshow(ma.masked_where(uc_filled < 1, masked_rst))
plt.show()

# %%
# pop of smod
print(
    "total_population",
    ma.sum(ma.masked_where(masked_rst_smod < 30, masked_rst)),
)
plt.imshow(ma.masked_where(masked_rst_smod < 30, masked_rst))
plt.show()


# %%
# check whole raster for benchmark
with rio.open(file_pop) as src:
    bbox_all = src.bounds

polygon = shapely.geometry.box(*bbox_all, ccw=True)
bbox_gdf = gpd.GeoDataFrame(
    gpd.GeoSeries(polygon), geometry=0, crs="ESRI:54009"
)

# %%
# urban centres in whole raster file
masked_rst_all, affine, src__all_crs = uc.filter_cells(file_pop, bbox_gdf)
flag_array_all = uc.flag_cells(masked_rst_all)
clusters_all, n_features_all = uc.cluster_cells(flag_array_all)
urban_centres_all = uc.check_cluster_pop(
    masked_rst_all, clusters_all, n_features_all
)  # 20 s
uc_filled_all = uc.fill_gaps(urban_centres_all)  # 70 s
fig = plt.figure(figsize=(20, 20))
plt.imshow(uc_filled_all > 0)

# %%
# whole GHS-SMOD raster file
smod_rst_smod, affine, smod_src = uc.filter_cells(file_smod, bbox_gdf)
fig = plt.figure(figsize=(20, 20))
plt.imshow(smod_rst_smod == 30)

# %%
