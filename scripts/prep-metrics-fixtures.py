"""Script to prepare dummy metrics unit test fixtures.

Uses `tests/data/metrics/input.tiff` as an input. This input corresponds to a
mock 4x4 100m gridded cell raster over the ONS Newport Office. A dummy urban
center boundary is also contructed, that reprents the centre 4 cells of the
4x4 grid. This mock urban centre boundary is then saved as a test fixture. All
these mock inputs are then feed into `RasterPop` to generate corresponding
`pop_gdf` and `centroids_gdf` fixtures. Finally, an OD travel time matrix is
mocked and saved as a parquet test fixture.

Travel times are constructed such that, when calculating the transport
performance with travel_time_threshold=3 and distance_threshold=0.11,
for each destination cell in the urban centre:

- accessible population = current cell + left and right hand cells.
- proximity population = current cell + left, right, below and above cells.

This is done to ensure effective testing of the threshold.

Note: changes made here will need to be reflected in the corresponding unit
tests that use them.

"""

import geopandas as gpd
import numpy as np
import pandas as pd

from pyprojroot import here
from shapely.geometry import Polygon

from transport_performance.population.rasterpop import RasterPop
from transport_performance.utils.io import to_pickle

# set path constants
INPUT_FIXTURE_PATH = here("tests/data/metrics/mock_raster_input.tif")
UC_FIXTURE_PATH = here("tests/data/metrics/mock_urban_centre.pkl")
POP_GDF_FIXTURE_PATH = here("tests/data/metrics/mock_pop_gdf.pkl")
CENTROID_GDF_FIXTURE_PATH = here("tests/data/metrics/mock_centroid_gdf.pkl")
TT_FIXTURE_PATH = here("tests/data/metrics/mock_tt.parquet")

# construct mock urban centre boundary and write fixture to file
coords = (
    (-225700, 6036700),
    (-225700, 6036500),
    (-225500, 6036500),
    (-225500, 6036700),
    (-225700, 6036700),
)
uc_fixture = gpd.GeoDataFrame(
    ["vectorized_uc"],
    geometry=[Polygon(coords)],
    columns=["label"],
    crs="ESRI:54009",
)
to_pickle(uc_fixture, UC_FIXTURE_PATH)

# construct pop_gdf and centroid_fixture
rp = RasterPop(INPUT_FIXTURE_PATH)
pop_fixture, centroid_fixture = rp.get_pop(
    uc_fixture.loc[0, "geometry"].buffer(100, join_style=2),
    urban_centre_bounds=uc_fixture.loc[0, "geometry"],
)

# generate population data with a fixed random seed for reproducibility
np.random.seed(42)
pop_fixture["population"] = np.random.randint(
    1, len(pop_fixture) + 1, len(pop_fixture)
)

# save pop_gdf and centroid_gdf fixtures
to_pickle(pop_fixture, POP_GDF_FIXTURE_PATH)
to_pickle(centroid_fixture, CENTROID_GDF_FIXTURE_PATH)

# construct mock travel time data using ID differences as travel times
uc_ids = pop_fixture[pop_fixture.within_urban_centre].id.unique()
ids = np.arange(0, len(pop_fixture))

travel_times = []
for uc_id in uc_ids:
    for id in ids:
        travel_times.append([id, uc_id, abs(id - uc_id)])

# save tt fixture as a parquet file (required format)
tt_fixture = pd.DataFrame(
    travel_times, columns=["from_id", "to_id", "travel_time"]
)
tt_fixture.to_parquet(TT_FIXTURE_PATH)
