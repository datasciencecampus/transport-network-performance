"""Wrapper for r5py to calculate O-D matrices."""
import pathlib

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd

from haversine import Unit, haversine_vector
from r5py import TransportNetwork, TravelTimeMatrixComputer
from typing import Union
from tqdm import tqdm
from math import ceil


class AnalyseNetwork:
    """Class to calculate transport network using r5py.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Geodataframe with the population centroids.
    osm : Union[str, pathlib.Path]
        Path to the location of the open street map file.
    gtfs : list
        List including path or paths to the locations of gtfs files.

    Attributes
    ----------
    gdf : gpd.GeoDataFrame


    TODO: finish docstring.

    """

    def __init__(
        self, gdf: gpd.GeoDataFrame, osm: Union[str, pathlib.Path], gtfs: list
    ):
        """Initialise AnalyseNetwork class."""
        # TODO: add defences.
        self.gdf = gdf
        self.transport_network = TransportNetwork(osm, gtfs)

    def od_matrix(
        self,
        num_origins: int,
        out_path: Union[str, pathlib.Path],
        partition_size: int = 200,
        **kwargs,
    ) -> None:
        """Calculate full O-D matrix and saves as parquet.

        Parameters
        ----------
        num_origins : int
            Number of origins to consider when batching the transport network
            calculation. A value of 1 would loop through each origin and all
            the destinations within the distance threshold. This is recommended
            for large areas where the full O-D matrix would not fit in memory
            (e.g. London). A value of len(gdf) would run only once using all
            destinations.
        out_path : Union[str, pathlib.Path]
            Path to save the O-D matrix as parquet files.
        partition_size : int
            Maximum size of each individual parquet files. If data would
            exceed this size, it will be split in several parquet files.

        Returns
        -------
        None

        Notes
        -----
        This function will work with any number of origins between 1 and all.
        However, this is not currently optimised and performance will be poor.

        """
        for sel_orig, sel_dest in tqdm(
            self._gdf_batch_origins(self.gdf, num_origins=num_origins),
            total=ceil(len(self.gdf) / num_origins),
        ):

            origin_gdf = self.gdf[self.gdf["id"].isin(sel_orig)]
            dest_gdf = self.gdf[self.gdf["id"].isin(sel_dest)]

            od_matrix = self._calculate_transport_network(
                self.transport_network,
                origins=origin_gdf,
                destinations=dest_gdf,
                **kwargs,
            )

            partitions = self._estimate_num_partitions(
                od_matrix, partition_size
            )

            self._save_to_parquet(
                od_matrix, min(sel_orig), out_path, partitions
            )

    def _calculate_transport_network(
        self,
        transport_network: TransportNetwork,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        **kwargs: dict,
    ) -> None:
        """Calculate origin-destination matrix.

        This is a wrapper around r5py TravelTimeMatrixComputer and the
        compute_travel_times method. A TransportNetwork instance has to be
        initialised separately.

        Parameters
        ----------
        transport_network : TransportNetwork
            A TransportNetwork instance with the osm and gtfs information for
            the required region.
        origins : gpd.GeoDataFrame
            TODO: complete docstrings.
        destinations : gpd.GeoDataFrame
            TODO: complete docstrings.
        num_origins: int
            Number of origins to consider. Note that more origins will greatly
            increase the cartesian product of origins and destinations, which
            may slow down processing or exceed available memory. To use all
            possible origins, use `len(gdf)`.
            # TODO: consider replacing this by a flag to use either 1 or all.
        **kwargs: dict, optional
            Optional arguments for TravelTimeMatrixComputer

        Returns
        -------
        pd.DataFrame
            Dataframe containing the O-D matrix for the origin-destination
            combinations.

        """
        # TODO: add defences.

        travel_time_matrix_computer = TravelTimeMatrixComputer(
            transport_network,
            origins=origins,
            destinations=destinations,
            **kwargs,
        )

        return travel_time_matrix_computer.compute_travel_times()

    def _gdf_batch_origins(
        self,
        gdf: gpd.GeoDataFrame,
        destination_col: str = "within_urban_centre",
        distance: float = 11.25,
        num_origins: int = 1,
    ) -> (int, gpd.GeoDataFrame):
        """Split geopandas.DataFrame into batches of a single origin.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Geodataframe containing location ids to cross match.
        destination_col : str
            Column indicating what centroids should be considered as
            destinations. Default is "within_urban_centre".
        distance: float
            Distance to filter destinations.in km. Points further away from
            origin are removed from output. Default is 11.25 km.
        num_origins: int
            Number of origins to consider. Note that more origins will greatly
            increase the cartesian product of origins and destinations, which
            may slow down processing or exceed available memory. To use all
            possible origins, use `len(gdf)`.
            # TODO: consider replacing this by a flag to use either 1 or all.

        Yields
        ------
        sel_origins : list
            Id numbers of the origins in a given iteration.
        sel_dest : list
            Id numbers of the destinations in a given iteration.

        Notes
        -----
        This function will filter out pairs of origins and destinations that
        are over the distance threshold, and it will return a list with
        unique origins and filtered destinations. However, note that this will
        only work as intended if using a single origin. If using several
        origins per batch, the r5py `TravelTimeMatrixComputer` object will
        still create all possible combinations of origins and destinations
        provided, which may include pairs that are beyond the threshold.

        """
        # TODO: add defences.

        # get sources and destinations
        # define destinations when `destination_col` is true
        orig_gdf = gdf.copy()
        dest_gdf = (
            gdf[gdf[destination_col] == True].reset_index().copy()  # noqa
        )

        origins = np.array(orig_gdf["id"])

        # TODO: REMEMBER TO REMOVE SHUFFLE
        np.random.shuffle(origins)

        # loops through origins using selected amount of origins
        for o in range(0, len(origins), num_origins):
            sel_origins = origins[o : o + num_origins]

            # calculates cross join of origins and destinations
            full_gdf = orig_gdf[orig_gdf["id"].isin(sel_origins)].merge(
                dest_gdf, how="cross", suffixes=["_orig", "_dest"]
            )

            # calculates haversine distance between origins and destinations
            full_gdf["distance"] = self._haversine_gdf(
                full_gdf, "geometry_orig", "geometry_dest"
            )

            # filters out pairs where distance is over threshold
            sel_dest = list(
                full_gdf[full_gdf["distance"] < distance]["id_dest"].unique()
            )

            # yields lists with selected origins and destinations
            yield sel_origins, sel_dest

    def _haversine_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        orig: str,
        dest: str,
        unit: Unit = Unit.KILOMETERS,
    ) -> np.array:
        """Calculate haversine distance between shapely Point objects.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Cartesian product of origin and destinations geodataframes. Should
            contain a geometry for each.
        orig : str
            Name of the column containing the origin geometry.
        dest : str
            Name of the column containing the destination geometry.
        unit : Unit
            Unit to calculte distance.

        Returns
        -------
        dist_array : np.array
            Array with row-wise distances.

        """
        # TODO: add defences.

        lat_long_orig = gdf[orig].apply(lambda x: (x.y, x.x))
        lat_long_dest = gdf[dest].apply(lambda x: (x.y, x.x))

        dist_array = haversine_vector(
            list(lat_long_orig), list(lat_long_dest), unit=unit
        )

        return dist_array

    def _estimate_num_partitions(
        self, df: pd.DataFrame, partition_size: int = 200
    ) -> int:
        """Estimate number of parquet partitions needed.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to save as parquet file.
        partition_size: int
            Maximum size in MB for each parquet file partition. Defaults to
            200 MB.

        Returns
        -------
        int
            Required partitions for the parquet file.

        """
        # TODO: add defences.

        mem_usage = sum(df.memory_usage(deep=True, index=False))
        est_parts = np.ceil(mem_usage / (partition_size * 1e6))

        return int(est_parts)

    def _save_to_parquet(
        self,
        od_matrix: pd.DataFrame,
        out_name_func: str,
        out_path: Union[str, pathlib.Path],
        npartitions: int = 1,
    ) -> None:
        """Save O-D matrix to parquet."""
        ddf = dd.from_pandas(od_matrix, npartitions=npartitions)

        filename = f"batch-{out_name_func}"
        filefunc = lambda x: filename + f"-{x}.parquet"  # noqa: E731
        ddf.to_parquet(out_path, name_function=filefunc)
