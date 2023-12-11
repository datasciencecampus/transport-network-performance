"""Notebook to test single vs all origin runs, comparing execution time."""
# %%
import time

from pyprojroot import here

from transport_performance.analyse_network import AnalyseNetwork
from transport_performance.utils.io import from_pickle

# %%
# inputs
PATH_OSM = here("data/processed/osm/newport_latest_filtered.osm.pbf")
PATH_GTFS = here("data/processed/gtfs/itm_newport_filtered_cleaned_gtfs.zip")
PATH_RP = here("data/processed/population/newport_rp.pkl")

newport_rp = from_pickle(PATH_RP)
centroid_gdf = newport_rp.centroid_gdf

# %%
# run od_matrix with all origins
OUT_PATH_ALL = here("outputs/analyse_network/test_all")

# initialise AnalyseNetwork object
an = AnalyseNetwork(centroid_gdf, PATH_OSM, [PATH_GTFS], out_path=OUT_PATH_ALL)

start = time.time()
an.od_matrix(batch_orig=False)
end = time.time()
print(f"All origins time: {end - start}")

# %%
# run od_matrix with single origin
OUT_PATH_SINGLE = here("outputs/analyse_network/test_single")

# initialise AnalyseNetwork object
an = AnalyseNetwork(
    centroid_gdf, PATH_OSM, [PATH_GTFS], out_path=OUT_PATH_SINGLE
)

start = time.time()
an.od_matrix(out_path=OUT_PATH_SINGLE, batch_orig=True)
end = time.time()
print(f"All origins time: {end - start}")

# %%
