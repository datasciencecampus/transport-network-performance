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
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

# import numpy as np
# from scipy.ndimage import label, generic_filter, binary_dilation
# from rasterio.mask import raster_geometry_mask
import numpy.ma as ma

import heimdall_transport.urban_centres.urban_centres as uc
import shapely.geometry

# %%
file_pop = "../../data/raw/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif"  # noqa
file_smod = "../../data/raw/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif"  # noqa

# %%
bbox = gpd.read_file("../../data/raw/bbox_leeds.geojson")
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
masked_rst = uc.filter_cells(file_pop, bbox_rep)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

uc_filled = uc.fill_gaps(urban_centres)
plt.imshow(uc_filled)

# %%
# SMOD dataset
masked_rst_smod = uc.filter_cells(file_smod, bbox_rep)
plt.imshow(masked_rst_smod >= 30)

# %%
# pop of simple urban centre
print(ma.sum(ma.masked_where(uc_filled < 1, masked_rst)))
plt.imshow(ma.masked_where(uc_filled < 1, masked_rst))

# %%
# pop of smod
print(ma.sum(ma.masked_where(masked_rst_smod < 30, masked_rst)))
plt.imshow(ma.masked_where(masked_rst_smod < 30, masked_rst))


# %%
# check whole raster for benchmark

with rasterio.open(file_pop) as src:
    bbox_all = src.bounds

polygon = shapely.geometry.box(*bbox_all, ccw=True)
bbox_gdf = gpd.GeoDataFrame(
    gpd.GeoSeries(polygon), geometry=0, crs="ESRI:54009"
)

# %%
masked_rst = uc.filter_cells(file_pop, bbox_gdf)
# %%
flag_array = uc.flag_cells(masked_rst)

# %%
clusters, n_features = uc.cluster_cells(flag_array)
# %%
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)  # 20 s

# %%
uc_filled = uc.fill_gaps(urban_centres)  # 70 s
fig = plt.figure(figsize=(20, 20))
plt.imshow(uc_filled)

# %%
masked_rst_smod = uc.filter_cells(file_smod, bbox_gdf)
fig = plt.figure(figsize=(20, 20))
plt.imshow(masked_rst_smod == 30)

# %%
# use binary_dilation for buffer!
# bd = binary_dilation(uc)
# mf = ma.array(uc, mask=np.invert(bd))
