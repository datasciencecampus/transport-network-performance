"""Script used to generate a subset of all uk bus gtfs taken from dft BODS.

The subset is based in newport and has a few purposes: for use in integration
tests and validating deterministic outputs of r5py. Also in running test script
to ensure development environment is setup as expected for r5py functionality.
GTFS fixture is also filtered by date to 20230613

fixtures: Newport PBF file created with osmium extract from geofabrik download:
osmium extract --strategy complete_ways --bbox
-3.077081,51.52222,-2.925075,51.593596
"""
import os
import gtfs_kit as gk
import geopandas as gpd
from shapely.geometry import box

fix_dat = os.path.join("tests", "data")
gtfs_zip = [
    os.path.join(fix_dat, x) for x in os.listdir(fix_dat) if x.endswith(".zip")
][0]
# create box polygon around newport coords
box_poly = box(-3.077081, 51.52222, -2.925075, 51.593596)
# gtfs_kit expects gdf
gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[box_poly])
feed = gk.read_feed(gtfs_zip, dist_units="km")
# feed.describe()
newport_feed = gk.miscellany.restrict_to_area(feed=feed, area=gdf)
# newport_feed.describe()
date_today = "20230613"
newport_today = gk.miscellany.restrict_to_dates(feed=newport_feed, dates=[date_today])
# newport_today.describe()
newport_today.write(os.path.join(fix_dat, f"newport-{date_today}_gtfs.zip"))
