# %%
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
# import numpy as np
# from scipy.ndimage import label, generic_filter, binary_dilation
# from rasterio.mask import raster_geometry_mask
import numpy.ma as ma
import pandas as pd
# from scipy.stats import mode
import heimdall_transport.urban_centres.urban_centres as uc
import shapely.geometry


# %%
# OLD FUNCS
# def flag_cells(masked_rst: ma.core.MaskedArray,
#                cell_pop_thres: int = 1500) -> np.array:
#     '''
#     Flags cells that are over the threshold
#     '''
#     # initializes numpy array
#     flag_array = np.zeros(masked_rst.shape)

#     # loops through raster array and flags >= 1,500
#     for i in range(masked_rst.shape[0]):
#         for j in range(masked_rst.shape[1]):
#             if masked_rst[i][j] >= cell_pop_thres:
#                 flag_array[i][j] = 1

#     return flag_array


# def check_cluster_pop(band: Union[np.array, ma.core.MaskedArray],
#                       labeled_array: np.array,
#                       num_features: int, pop_threshold: int = 50000):
#     '''
#     Checks what clusters have more than the threshold population
#     and changes those that don't to 0
#     '''
#     urban_centres = labeled_array.copy()
#     for n in range(1, num_features + 1):
#         pops = []
#         for i in range(band.shape[0]):
#             for j in range(band.shape[1]):
#                 if urban_centres[i][j] == n:
#                     pops.append(band[i][j])

#         if sum(pops) < pop_threshold:
#             urban_centres[urban_centres == n] = 0

#     return urban_centres


# def surrounding(i: int, j: int, maxi: int,
#                 maxj: int, diag: bool = True) -> tuple:
#     '''
#     Gets sorrounding cells based on index.
#     '''
#     i_list = [(i - 1) if i >= 1 else None,
#               i,
#               (i + 1) if i < maxi else None]
#     j_list = [(j - 1) if j >= 1 else None,
#               j,
#               (j + 1) if j < maxj else None] 

#     i_list = [i for i in i_list if i is not None]
#     j_list = [j for j in j_list if j is not None]

#     surroundings = [(ix, jx) for ix in i_list for jx in j_list]
#     surroundings.remove((i, j))

#     return surroundings


# def find_gaps(band: np.array, cell: tuple, diag: bool = True,
#               threshold: int = 5, cluster: int = 1,
#               fill: int = 1) -> np.array:
#     '''
#     Checks if surrounding value of cell is equal or higher than
#     threshold.
#     '''
#     sh = band.shape
#     ixs = surrounding(cell[0], cell[1],
#                       sh[0] - 1, sh[1] - 1,
#                       diag)

#     values = [band[ix] for ix in ixs]

#     if np.count_nonzero(values == cluster) >= threshold:
#         band[cell] = fill
#         flag = True
#     else:
#         flag = False

#     return flag


# def fill_gaps(band: np.array, diag: bool = True,
#               threshold: int = 5) -> np.array:
#     '''
#     Loops through band and fills all cells that are
#     equal or higher than threshold iteratively.
#     '''
#     uc = band.copy()
#     # iterates through clusters
#     for cluster in np.delete(np.unique(uc), 0):

#         # keeps iterating until no more changes happen
#         while True:
#             for i in range(uc.shape[0]):
#                 for j in range(uc.shape[1]):
#                     # if cell already flagged, pass
#                     if uc[(i, j)] == cluster:
#                         pass
#                     # if cell does not belong to cluster,
#                     # check if it should be filled
#                     elif uc[(i, j)] == 0:
#                         flag = find_gaps(uc, (i, j),
#                                          diag=diag,
#                                          threshold=threshold,
#                                          cluster=cluster,
#                                          fill=cluster)
#                 # restart looping through cells after change
#                     if flag is True:
#                         break
#                 if flag is True:
#                     break
#             # breaks when reaching the end of the array
#             if (i + 1, j + 1) == uc.shape:
#                 break

#     return uc

# %%
file_pop = '../../data/raw/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif'  # noqa
file_smod = '../../data/raw/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R3_C18.tif'  # noqa

# %%
bbox = gpd.read_file('../../data/raw/bbox.geojson')
bbox_rep = bbox.to_crs('esri:54009')
xmin, ymin, xmax, ymax = bbox_rep.total_bounds

# %%
wards = gpd.read_file('../../data/raw/Wards_Dec_2020_UK_BGC_2022_-5150753419644615544.geojson')  # noqa
wards_rep = wards.to_crs('esri:54009')

wards_la_lkup = pd.read_csv('../../data/raw/Ward_to_Local_Authority_District_(December_2020)_Lookup_in_the_United_Kingdom_V2.csv')  # noqa
wards_rep = wards_rep.merge(wards_la_lkup[['WD20CD', 'LAD20NM']], on='WD20CD')
wards_newport = wards_rep[wards_rep['LAD20NM'] == 'Newport']

# %%
masked_rst = uc.filter_cells(file_pop, bbox_rep)
flag_array = uc.flag_cells(masked_rst)
clusters, n_features = uc.cluster_cells(flag_array)
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)

uc_filled = uc.fill_gaps(urban_centres)
plt.imshow(uc_filled)


# %%
masked_rst_smod = filter_cells(file_smod, bbox_rep)
plt.imshow(masked_rst_smod == 30)

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
bbox_gdf = gpd.GeoDataFrame(gpd.GeoSeries(polygon), geometry=0,
                            crs='ESRI:54009')

# %%
masked_rst = uc.filter_cells(file_pop, bbox_gdf)
# %%
flag_array = uc.flag_cells(masked_rst)

# %%
clusters, n_features = uc.cluster_cells(flag_array)
# %%
urban_centres = uc.check_cluster_pop(masked_rst, clusters, n_features)  # 20 s

# %%
uc_filled = uc.fill_gaps(urban_centres)  # 40 s
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
