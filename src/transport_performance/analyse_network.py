"""Wrapper for r5py to calculate O-D matrices."""
import pathlib
import warnings

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd

from haversine import Unit, haversine_vector
from math import ceil
from r5py import TransportNetwork, TravelTimeMatrixComputer
from tqdm import tqdm
from typing import Union

import transport_performance.utils.defence as d


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
    gdf
    transport_network : TransportNetwork
        R5py object that contains a transport network initialised with data
        from OpenStreetMap and GTFS.

    Methods
    -------
    od_matrix
        Method that calculates the full O-D matrix and saves it as parquet.

    """

    def __init__(
        self, gdf: gpd.GeoDataFrame, osm: Union[str, pathlib.Path], gtfs: list
    ):
        """Initialise AnalyseNetwork class."""
        # defences
        d._type_defence(gdf, "gdf", gpd.GeoDataFrame)
        d._is_expected_filetype(osm, "osm", exp_ext=".pbf")
        d._check_iterable(
            iterable=gtfs,
            param_nm="gtfs",
            iterable_type=list,
            check_elements=True,
            exp_type=(str, pathlib.Path),
        )

        for path in gtfs:
            d._is_expected_filetype(path, "gtfs", exp_ext=".zip")

        # check if gdf is in EPSG: 4326
        if gdf.crs != "EPSG: 4326":
            warnings.warn(
                f"`gdf` crs needs to be EPSG: 4326, found {gdf.crs}. "
                "`gdf` will be re-projected."
            )
            self.gdf = gdf.to_crs("EPSG: 4326")
        else:
            self.gdf = gdf
        self.transport_network = TransportNetwork(osm, gtfs)

    def od_matrix(
        self,
        out_path: Union[str, pathlib.Path],
        batch_orig: bool = False,
        partition_size: int = 200,
        destination_col: str = "within_urban_centre",
        distance: float = 11.25,
        unit: Unit = Unit.KILOMETERS,
        **kwargs,
    ) -> None:
        """Calculate full O-D matrix and save as parquet.

        Parameters
        ----------
        out_path : Union[str, pathlib.Path]
            Path to save the O-D matrix as parquet files.
        batch_orig : bool
            Flag to indicate whether to calculate the transport network
            performance using the whole dataset or batching by origin
            (iteratively calculating the transport performance of each origin
            and all destinations within range). Should be True for large urban
            centres where the whole O-D matrix may be to large to be hold in
            memory. Defaults to False.
        partition_size : int
            Maximum size of each individual parquet file. If data would
            exceed this size, it will be split in several parquet files.
        destination_col : str
            Column indicating what centroids should be considered as
            destinations. Default is "within_urban_centre".
        distance: float
            Distance to filter destinations.in km. Points further away from
            origin are removed from output. Default is 11.25 km.
        unit : Unit
            Unit to calculate distance. Default is km.
        **kwargs
            Any valid argument for r5py TravelTimeMatrixComputer object.
            E.g. `departure`, `max_time`, `transport_modes`.

        Notes
        -----
        `batch_orig` = True can be used for small datasets, however this will
        likely be slower as normal r5py behaviour uses parallel computing. We
        recommend setting this argument to True only in cases where memory
        may not be enough to handle the O-D matrix.

        """
        # defences
        d._type_defence(batch_orig, "batch_orig", bool)

        if batch_orig:
            # batches are forced to a single origin per iteration
            # (see note in `_gdf_batch_origins`)
            # may change in the future
            for sel_orig, sel_dest in tqdm(
                self._gdf_batch_origins(
                    self.gdf,
                    destination_col=destination_col,
                    distance=distance,
                    num_origins=1,
                    unit=unit,
                ),
                total=len(self.gdf),
            ):

                origin_gdf = self.gdf[self.gdf["id"].isin(sel_orig)].copy()
                dest_gdf = self.gdf[self.gdf["id"].isin(sel_dest)].copy()

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
                    od_matrix, str(min(sel_orig)), out_path, partitions
                )
        else:
            origin_gdf = self.gdf.copy()
            dest_gdf = self.gdf[self.gdf[destination_col]].copy()

            od_matrix = self._calculate_transport_network(
                self.transport_network,
                origins=origin_gdf,
                destinations=dest_gdf,
                **kwargs,
            )

            partitions = self._estimate_num_partitions(
                od_matrix, partition_size
            )

            self._save_to_parquet(od_matrix, "all", out_path, partitions)

    def _calculate_transport_network(
        self,
        transport_network: TransportNetwork,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        **kwargs: dict,
    ) -> pd.DataFrame:
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
            Geodataframe containing the point(s) to use as origin.
        destinations : gpd.GeoDataFrame
            Geodataframe contianing the point(s) to use as destination.
        **kwargs
            Any valid argument for r5py TravelTimeMatrixComputer object.
            E.g. `departure`, `max_time`, `transport_modes`.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the O-D matrix for the origin-destination
            combinations.

        Raises
        ------
        IndexError
            If the date passed as departure time is not contained in the GTFS,
            or the GTFS has not been loaded correctly into the TransportNetwork
            object. Note that if no `departure` kwarg is provided, it will
            default to datetime.now().
            Other warnings will not raise this error but should still be
            printed as normal.

        """
        # defences
        d._type_defence(
            transport_network, "transport_network", TransportNetwork
        )
        d._type_defence(origins, "origins", gpd.GeoDataFrame)
        d._type_defence(destinations, "destinations", gpd.GeoDataFrame)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                "Departure time .* is outside of the time range covered.*",
            )
            try:
                travel_time_matrix_computer = TravelTimeMatrixComputer(
                    transport_network,
                    origins=origins,
                    destinations=destinations,
                    **kwargs,
                )
            except RuntimeWarning:
                raise IndexError(
                    "Date provided is outside of the time range included in "
                    "the GTFS provided, or TransportNetwork does not contain "
                    "a valid GTFS."
                )

        return travel_time_matrix_computer.compute_travel_times()

    def _gdf_batch_origins(
        self,
        gdf: gpd.GeoDataFrame,
        destination_col: str,
        distance: float,
        num_origins: int,
        unit: Unit = Unit.KILOMETERS,
    ) -> (list, list):
        """Split geopandas.DataFrame into batches of a single origin.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Geodataframe containing location ids to cross match.
        destination_col : str
            Column indicating what centroids should be considered as
            destinations. Column dtype needs to be bool.
        distance : float
            Distance to filter destinations. Points further away from
            origin are removed from output.
        num_origins : int
            Number of origins to consider. Note that more origins will greatly
            increase the cartesian product of origins and destinations, which
            may slow down processing or exceed available memory. To use all
            possible origins, use `len(gdf)`.
        unit : Unit
            Unit to calculate distance. Default is km.

        Yields
        ------
        sel_origins : list
            Id numbers of the origins in a given iteration.
        sel_dest : list
            Id numbers of the destinations in a given iteration.

        Raises
        ------
        ValueError
            If the number of origins provided is not between 1 and the total
            number of origins in the geodataframe.

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
        # defences
        d._type_defence(gdf, "gdf", gpd.GeoDataFrame)
        d._check_column_in_df(gdf, destination_col)
        d._type_defence(distance, "distance", float)
        d._type_defence(num_origins, "num_origins", int)
        d._type_defence(unit, "unit", Unit)
        if gdf[destination_col].dtype != bool:
            raise TypeError(
                f"Column `{destination_col}` should be bool. "
                f"Got {gdf[destination_col].dtype}"
            )
        if num_origins < 1 or num_origins > len(gdf):
            raise ValueError(
                f"`num_origins` should be between 1 and {len(gdf)}, "
                f"got {num_origins}"
            )

        # get sources and destinations
        # define destinations when `destination_col` is true
        orig_gdf = gdf.copy()
        geometry_col = gdf.geometry.name
        # TODO: add option to include all rows as destinations?
        dest_gdf = gdf[gdf[destination_col]].reset_index(drop=True).copy()

        origins = np.array(orig_gdf["id"])

        # loops through origins using selected amount of origins
        for o in range(0, len(origins), num_origins):
            sel_origins = origins[o : o + num_origins]

            # calculates cross join of origins and destinations
            full_gdf = orig_gdf[orig_gdf["id"].isin(sel_origins)].merge(
                dest_gdf, how="cross", suffixes=["_orig", "_dest"]
            )

            # calculates haversine distance between origins and destinations
            full_gdf["distance"] = self._haversine_df(
                full_gdf,
                f"{geometry_col}_orig",
                f"{geometry_col}_dest",
                unit=unit,
            )

            # filters out pairs where distance is over threshold
            sel_dest = list(
                full_gdf[full_gdf["distance"] <= distance]["id_dest"].unique()
            )

            # yields lists with selected origins and destinations
            yield sel_origins, sel_dest

    def _haversine_df(
        self,
        df: pd.DataFrame,
        orig: str,
        dest: str,
        unit: Unit,
    ) -> np.array:
        """Calculate haversine distance between shapely Point objects.

        Parameters
        ----------
        df : pd.DataFrame
            Cartesian product of origin and destinations geodataframes. Should
            contain a geometry for each.
        orig : str
            Name of the column containing the origin geometry.
        dest : str
            Name of the column containing the destination geometry.
        unit : Unit
            Unit to calculate distance.

        Returns
        -------
        dist_array : np.array
            Array with row-wise distances.

        """
        # defences
        d._type_defence(df, "df", pd.DataFrame)
        d._check_column_in_df(df, orig)
        d._check_column_in_df(df, dest)
        d._type_defence(unit, "unit", Unit)

        lat_long_orig = df[orig].apply(lambda x: (x.y, x.x))
        lat_long_dest = df[dest].apply(lambda x: (x.y, x.x))

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
        # defences
        d._type_defence(df, "df", pd.DataFrame)
        d._type_defence(partition_size, "partition_size", int)

        mem_usage = sum(df.memory_usage(deep=True, index=False))
        est_parts = ceil(mem_usage / (partition_size * 1e6))

        return est_parts

    def _save_to_parquet(
        self,
        od_matrix: pd.DataFrame,
        out_name_func: str,
        out_path: Union[str, pathlib.Path],
        npartitions: int = 1,
    ) -> None:
        """Save O-D matrix to parquet.

        Parameters
        ----------
        od_matrix : pd.DataFrame
            O-D matrix, including columns `from_id`, `to_id` and `travel_time`.
        out_name_func : str
            String to include in the parquet files for each batch. Needed so
            each batch has a different name than the following so they are not
            overwritten.
        out_path : Union[str, pathlib.Path]
            Path where to save parquet files.
        npartitions : int
            Number of partitions required for the parquet file from each batch.

        """
        # defences
        d._type_defence(od_matrix, "od_matrix", pd.DataFrame)
        d._type_defence(out_name_func, "out_name_func", str)
        d._type_defence(npartitions, "npartitions", int)
        d._check_parent_dir_exists(out_path, "out_path", create=True)

        ddf = dd.from_pandas(od_matrix, npartitions=npartitions)

        filename = f"batch-{out_name_func}"
        filefunc = lambda x: filename + f"-{x}.parquet"  # noqa: E731
        ddf.to_parquet(out_path, name_function=filefunc)
