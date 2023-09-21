"""Class to handle raster population data."""

import geopandas as gpd
import os
import numpy as np
import rasterio as rio
import rioxarray
import folium
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt

from datetime import datetime
from geocube.vector import vectorize
from typing import Union, Type, Tuple
from shapely.geometry.polygon import Polygon
from matplotlib import colormaps
from cartopy.mpl.geoaxes import GeoAxes
from transport_performance.utils.defence import (
    _is_expected_filetype,
    _type_defence,
)


class RasterPop:
    """Prepare raster population inputs for trasport analysis.

    This class is suited to working with rastered population data (e.g.
    gridded population estimates).

    Parameters
    ----------
    filepath : Union[str, bytes, os.PathLike]
        File path to population data

    Attributes
    ----------
    pop_gdf : gpd.GeoDataFrame
        A geopandas dataframe of the input data, with gridded geometry. This is
        in the same CRS as the input raster data.
    centroid_gdf : gpd.GeoDataFrame
        A geopandas dataframe of grid centroids, converted to EPSG:4326 for
        transport analysis.

    Methods
    -------
    get_pop
        Read and preprocess population estimates into a geopandas dataframe.
    plot
        Build static and interactive visualisations of population data. Can
        only use this method once `get_pop` has been called.

    Raises
    ------
    FileNotFoundError
        When `filepath` is not a file or can not be found.

    """

    def __init__(self, filepath: Union[str, bytes, os.PathLike]) -> None:

        # defend against cases where input is not a file and does not exist
        _is_expected_filetype(filepath, "filepath", exp_ext=".tif")

        self.__filepath = filepath

        # record the crs of the data source without reading in data
        with rio.open(filepath) as src:
            self.__crs = src.crs.to_string()

        # set attributes to None to prevent plotting function calls
        self.pop_gdf = None
        self._uc_gdf = None

    def get_pop(
        self,
        aoi_bounds: Type[Polygon],
        aoi_crs: str = None,
        round: bool = False,
        threshold: Union[int, float] = None,
        var_name: str = "population",
        urban_centre_bounds: Type[Polygon] = None,
        urban_centre_crs: str = None,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Get population data.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.
            Assumed to be in the same CRS as the rastered population data. If
            it is different, set `aoi_crs` to the CRS of this boundary.
        aoi_crs : str, optional
            CRS string for `aoi_bounds` (e.g. "EPSG:4326"), by default None
            which means it is assumed to have the same CRS as the population
            data.
        round : bool, optional
            Round population estimates to the nearest whole integer.
        threshold : Union[int, float], optional
            Threshold population estimates, where values below the set
            threshold will be set to nan, by default None which means no
            thresholding will occur.
        var_name : str, optional
            The variable name, by default "population"
        urban_centre_bounds : Type[Polygon], optional
            Polygon defining an urban centre bounday, by default None meaning
            information concerning whether the grid resides within the urban
            centre will not be added.
        urban_centre_crs : str, optional
            The urban centre polygon CRS, by default None meaning this is the
            same CRS as the input raster data. Only used when
            `urban_centre_bounds` is set.

        Returns
        -------
        pop_gdf : gpd.GeoDataFrame
            A geopandas dataframe of the input data, with gridded geometry.
            This is in the same CRS as the input raster data.
        centroid_gdf
            A geopandas dataframe of grid centroids, converted to EPSG:4326 for
            transport analysis.

        """
        # read and clip population data to area of interest
        self._read_and_clip(aoi_bounds, aoi_crs, var_name)

        # round population estimates, if requested
        if round:
            self._round_population()

        # threshold the population data, if requested
        if threshold is not None:
            self._threshold_population(threshold)

        # convert to variable and centroid geopandas dataframes
        self._to_geopandas(round)

        # add within urban centre details
        if urban_centre_bounds is not None:
            self._within_urban_centre(
                urban_centre_bounds, urban_centre_crs=urban_centre_crs
            )

        return self.pop_gdf, self.centroid_gdf

    def plot(
        self, which: str = "folium", save: str = None, **kwargs
    ) -> Union[folium.Map, Type[GeoAxes], plt.Axes, None]:
        """Plot population data.

        Parameters
        ----------
        which : str, optional
            Package to use for plotting. Must be one of {"matplotlib",
            "cartopy", "folium"}, by default "folium".
        save : str, optional
            Filepath to save file, with the file extension, by default None
            meaning a file will not be saved.
        kwargs : dict, optional
            Extra arguments passed to plotting functions to configure the plot
            styling. See Notes for more support.

        Returns
        -------
        Union[folium.Map, Type[GeoAxes], plt.Axes, None]
            A folium map is returned when the `folium` backend is used. A
            matplotlib Axes object is returned when the `matplotlib` backend is
            used. A matplotlib/cartopy GeoAxes object is return when the
            `cartopy` backend is used. None will be returned when saving to
            file.

        Raises
        ------
        ValueError
            Unexpected value of `which`.
        NotImplementedError
            When plot is called without reading data.

        Notes
        -----
        Calling `help` as follows will provide more insights on possible kwarg
        arguments for the valid plotting backends:

        - Folium backend: `help(RasterPop._plot_folium)`
        - Matplotlib backend: `help(RasterPop._plot_matplotlib)`
        - Cartopy backend: `help(RasterPop._plot_cartopy)`

        """
        # record of valid which values
        WHICH_VALUES = {"matplotlib", "catropy", "folium"}

        # defend against case where `get_pop` hasn't been called
        if self.pop_gdf is None:
            raise NotImplementedError(
                "Unable to call `plot` without calling `get_pop`."
            )

        if which == "folium":
            m = self._plot_folium(save, **kwargs)
            return m
        elif which == "cartopy":
            ax = self._plot_cartopy(save, **kwargs)
            return ax
        elif which == "matplotlib":
            ax = self._plot_matplotlib(save, **kwargs)
            return ax
        else:
            raise ValueError(
                f"Unrecognised value for `which` {which}. Must be one of "
                f"{WHICH_VALUES}."
            )

    def _read_and_clip(
        self,
        aoi_bounds: Type[Polygon],
        aoi_crs: str = None,
        var_name: str = "population",
    ) -> None:
        """Open data and clip to the area of interest boundary.

        Read and clip raster file from disk (more performant). Mask the data
        (such that no data values are nan) and read in all grids that touch
        the aoi boundary.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.
        aoi_crs : str, optional
            CRS string for `aoi_bounds` (e.g. "EPSG:4326"), by default None
            which means it is assumed to have the same CRS as `aoi_bounds`.
        var_name : str, optional
            The variable name, by default "population".

        """
        # input type defence checks
        _type_defence(aoi_bounds, "aoi_bounds", Polygon)
        _type_defence(aoi_crs, "aoi_crs", (str, type(None)))
        _type_defence(var_name, "var_name", str)

        # convert aoi bounds CRS if needed
        if aoi_crs is not None:
            gdf = gpd.GeoDataFrame(geometry=[aoi_bounds], crs=aoi_crs)
            gdf = gdf.to_crs(self.__crs)
            aoi_bounds = gdf.loc[0, "geometry"]

        # open and clip raster data - using `all_touched` method to include any
        # cell touched and `from_disk` for improved reading speeds.
        self._xds = rioxarray.open_rasterio(
            self.__filepath, masked=True
        ).rio.clip([aoi_bounds], from_disk=True, all_touched=True)

        # set the variable name - set internal variable for reuse in class
        self._xds.name = var_name
        self.__var_name = var_name

        # build aoi bounds dataframe for plotting
        self._aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_bounds], crs=self.__crs)
        self._aoi_gdf.loc[:, "boundary"] = "AOI Bounds"

    def _round_population(self) -> None:
        """Round population data."""
        self._xds = np.rint(self._xds)

    def _threshold_population(self, threshold: Union[int, float]) -> None:
        """Threshold population data."""
        # `where()` is working differently to expectation here - it keeps
        # values above the threshold and otherwise sets them to nan.
        _type_defence(threshold, "threshold", (int, float))
        self._xds = self._xds.where(self._xds >= threshold)

    def _to_geopandas(self, round: bool = False) -> None:
        """Convert to geopandas dataframe."""
        # vectorise to geopandas dataframe, setting data type to np.float32 -
        # dtype is required by vecotrize function. Squeeze needed since shape
        # is (1xnxm) (1 band in tiff file)
        self.pop_gdf = vectorize(self._xds.squeeze(axis=0).astype(np.float32))

        # dropna to remove nodata regions and those below threshold (if set)
        self.pop_gdf = self.pop_gdf.dropna(subset=self.__var_name).reset_index(
            drop=True
        )

        # if rounding, cast to int after removing na values
        if round:
            self.pop_gdf[self.__var_name] = self.pop_gdf[
                self.__var_name
            ].astype("int")

        # add an id for each cell - needed for r5py
        self.pop_gdf["id"] = np.arange(0, len(self.pop_gdf.index))

        # re-order columns for consistency
        self.pop_gdf = self.pop_gdf[["id", self.__var_name, "geometry"]]

        # create centroid geodataframe using only necessary columns - save on
        # duplicated data, can join gdfs on id if ever needed.
        self.centroid_gdf = self.pop_gdf[["id", "geometry"]].copy()
        self.centroid_gdf["centroid"] = self.centroid_gdf.geometry.centroid
        self.centroid_gdf.set_geometry("centroid", inplace=True)
        self.centroid_gdf.drop("geometry", axis=1, inplace=True)

        # convert centroids to EPSG:4326 for use in r5py
        if self.__crs != "EPSG:4326":
            self.centroid_gdf = self.centroid_gdf.to_crs("EPSG:4326")

    def _within_urban_centre(
        self,
        urban_centre_bounds: Type[Polygon],
        urban_centre_crs: str = None,
    ) -> None:
        """Categorise grid whether within urban centre.

        Parameters
        ----------
        urban_centre_bounds : Type[Polygon]
            Polygon defining urban centre bounday
        urban_centre_crs : str, optional
            The urban centre polygon CRS, by default None meaning this is the
            same CRS as the input raster data.

        Raises
        ------
        TypeError
            When `urban_centre_bounds` is not a shapely Polygon.

        """
        # defend against case where urban_centre_bounds is not a polygon
        if not isinstance(urban_centre_bounds, Polygon):
            raise TypeError(
                f"Expected type {Polygon.__name__} for `urban_centre_bounds`, "
                f"got {type(urban_centre_bounds).__name__}."
            )

        # match the crs if is one isn't provided - default assumption if this
        # arg is not set, and a CRS is needed when building the gdf below
        if urban_centre_crs is None:
            urban_centre_crs = self.__crs

        # build urban centre dataframe - set within column to true for sjoin
        # such that nan values post join imply no within urban centre. Add an a
        # column for the boundary name - only useful for plotting.
        self.__UC_COL_NAME = "within_urban_centre"
        self._uc_gdf = gpd.GeoDataFrame(
            geometry=[urban_centre_bounds], crs=urban_centre_crs
        )
        self._uc_gdf.loc[:, self.__UC_COL_NAME] = True
        self._uc_gdf.loc[:, "boundary"] = "Urban Centre"

        # if the provided crs does not match the raster crs, then convert the
        # crs before the sjoin
        if self._uc_gdf.crs != self.__crs:
            self._uc_gdf = self._uc_gdf.to_crs(self.__crs)

        # spatial join when cell is within urban centre, filling nan to false.
        # drop index_right and boundary columns as they aren't needed.
        self.pop_gdf = self.pop_gdf.sjoin(
            self._uc_gdf, how="left", predicate="within"
        ).drop(["index_right", "boundary"], axis=1)
        self.pop_gdf[self.__UC_COL_NAME] = self.pop_gdf[
            self.__UC_COL_NAME
        ].fillna(False)

        # add within_urban_centre column to centroid data too - could be useful
        # during selecting sources/destinations, so duplication is OK.
        self.centroid_gdf = self.centroid_gdf.merge(
            self.pop_gdf[["id", self.__UC_COL_NAME]], on="id"
        )

    def _plot_folium(
        self,
        save: str = None,
        tiles: str = (
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        ),
        attr: str = (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStre'
            'etMap</a> contributors &copy; <a href="https://carto.com/attribut'
            'ions">CARTO</a>'
        ),
        cmap: str = "viridis",
        boundary_color: str = "red",
        boundary_weight: int = 2,
    ) -> folium.Map:
        """Plot data onto a folium map.

        Parameters
        ----------
        save : str, optional
            Filepath to save location, with ".html" extension, by default None
            meaning the map will not be saved to file.
        tiles : str, optional
            Tile layer for base map, by default Carto Positron tiles. As per
            folium, the base map can be generated using a built in key words
            [1]_ or a custom tileset url [2]_.
        attr : str, optional
            Tile layer attribution, by default Carto Positron attribution.
        cmap : str, optional
            A colormap string recognised by `matplotlib` [3]_, by default
            "viridis".
        boundary_color : str, optional
            Color of the boundary lines, by default "red". Could also be a
            hexstring.
        boundary_weight : float, optional
            Weight (in pixels) of the boudary lines, by default 2.

        Returns
        -------
        folium.Map
            Folium map obeject, with data and boundaries in layers. Will be
            None when writing to file (saves time when displaying map
            interactive mode).

        Notes
        -----
        1. When using tile layers, check the tile provider's terms and
        conditions and assign an attribution as needed.

        References
        ----------
        .. [1] https://python-visualization.github.io/folium/modules.html
        .. [2] http://leaflet-extras.github.io/leaflet-providers/preview/
        .. [3] https://matplotlib.org/stable/tutorials/colors/colormaps.html

        """
        # add a base map with no tile - then is it not on a layer control
        m = folium.Map(tiles=None, control_scale=True, zoom_control=True)

        # add a tile layer
        folium.TileLayer(
            tiles=tiles,
            attr=attr,
            show=False,
            control=False,
        ).add_to(m)

        # add the variable data to the map
        self.pop_gdf.explore(
            self.__var_name, cmap=cmap, name=self.__var_name.capitalize(), m=m
        )

        # add the urban centre boundary, if one was provided
        if self._uc_gdf is not None:
            self._uc_gdf.explore(
                tooltip=["boundary"],
                name="Urban Centre Boundary",
                m=m,
                style_kwds={
                    "fill": False,
                    "color": boundary_color,
                    "weight": boundary_weight,
                },
            )

        # add the area of interest boundary
        self._aoi_gdf.explore(
            tooltip=["boundary"],
            name="Area of Interest Boundary",
            m=m,
            style_kwds={
                "fill": False,
                "color": boundary_color,
                "weight": boundary_weight,
                "dashArray": 7,
            },
        )

        # add the centroids to a separate layer
        # conditionally style plot based on whether UC is provided
        if self._uc_gdf is not None:
            centroid_plot_col = self.__UC_COL_NAME
            # this dict will change the centriod color in/out the UC.
            centroid_style_dict = {
                "style_function": lambda x: {
                    "color": "#BC544B"
                    if x["properties"][self.__UC_COL_NAME] is False
                    else "#8B0000"
                }
            }
        else:
            centroid_plot_col = None
            centroid_style_dict = {
                "style_function": lambda x: {"color": "#BC544B"}
            }

        # add in the centroid layer with the conditional styling
        self.centroid_gdf.explore(
            centroid_plot_col,
            name="Centroids",
            m=m,
            show=False,
            style_kwds=centroid_style_dict,
            legend=False,
        )

        # fit bounds to the plot limits and add a layer control button
        m.fit_bounds(m.get_bounds())
        m.add_child(folium.LayerControl())

        # write to file if filepath is given
        if save is not None:
            out_df = os.path.dirname(save)
            if not os.path.exists(out_df):
                os.mkdir(out_df)
            m.save(save)
            m = None

        return m

    def _plot_cartopy(
        self,
        save: str = None,
        figsize: tuple = (10, 8),
        map_tile_zoom: int = 12,
        cmap: str = "viridis",
        cbar_fraction: float = 0.034,
        cbar_pad: float = 0.04,
        cbar_label: str = "Population count per cell",
        var_attr: str = "GHS-POP 2020 (R2023)",
        base_map_attr: str = "(C) OpenSteetMap contributors",
        attr_font_size: float = 7.5,
        attr_location: str = "bottom_left",
    ) -> Union[Type[GeoAxes], None]:
        """Plot data via cartopy (static plot with an OpenStreetMap base tile).

        Parameters
        ----------
        save : str, optional
            Filepath to save location, with ".png" extension, by default None
            meaning the map will not be saved to file.
        figsize : tuple, optional
            The matplotlib figure size width and height in inches, by default
            (10, 8).
        map_tile_zoom : int, optional
            The zoom level for the OpenSteetMap base tile layer, by default 12
            which is suitable for towns/cities. See [1]_ for more details on
            the zoom level.
        cmap : str, optional
            A colormap string recognised by `matplotlib` [2]_, by default
            "viridis".
        cbar_fraction : float, optional
            Fractional size of the colour bar. Use this value to change the
            height of the colour bar, by default 0.034.
        cbar_pad : float, optional
            Amount of padding between the plot and the colourbar, by default
            0.04.
        cbar_label : str, optional
            Colour label, by default "Population count per cell".
        var_attr : str, optional
            Attribution to assign for variable/population data, by default
            "GHS-POP 2020 (R2023)".
        base_map_attr : str, optional
            Base map tile attribution, by default "(C) OpenSteetMap
            contributors".
        attr_font_size : float, optional
            Font size of the attribution text, by default 7.5.
        attr_location : str, optional
            Location of the attribution. Must be one of {"bottom_left",
            "bottom_right", "top_right", "top_left"}, by default "bottom_left".

        Returns
        -------
        Union[Type[GeoAxes], None]
            A matplotlib/cartopy GeoAxes displaying the data ontop of an
            OpenStreetMap base tile. Returns None when writing to file.

        Raises
        ------
        ValueError
            When `attr_location` is assigned an unexpected value.

        References
        ----------
        .. [1] https://wiki.openstreetmap.org/wiki/Zoom_levels
        .. [2] https://matplotlib.org/stable/tutorials/colors/colormaps.html

        """
        # make attribution location dictionary
        ATTR_LOC = {
            "bottom_left": {
                "x": 0.01,
                "y": 0.01,
                "ha": "left",
                "va": "bottom",
            },
            "bottom_right": {
                "x": 0.99,
                "y": 0.01,
                "ha": "right",
                "va": "bottom",
            },
            "top_right": {"x": 0.99, "y": 0.99, "ha": "right", "va": "top"},
            "top_left": {"x": 0.01, "y": 0.99, "ha": "left", "va": "top"},
        }

        # get OpenStreetMap tile layer object in greyscale
        map_tile = cimgt.OSM(desired_tile_form="L")

        # build plot axis and add map tile
        ax = plt.axes(projection=map_tile.crs)
        ax.figure.set_size_inches(*figsize)
        data_crs = ccrs.Mollweide()
        ax.add_image(map_tile, map_tile_zoom, cmap="gray")

        # build a colormap and add pcolormesh plot data, setting vmin and vmax
        # to match the whole colormap range. transform to data_crs required to
        # project onto base map.
        plot_cmap = colormaps.get_cmap(cmap)
        plot_data = self._xds.squeeze(axis=0).to_numpy()
        vmin_data = np.nanmin(plot_data)
        vmax_data = np.nanmax(plot_data)
        ctf = ax.pcolormesh(
            self._xds.squeeze().x.to_numpy(),
            self._xds.squeeze().y.to_numpy(),
            plot_data,
            cmap=plot_cmap,
            vmin=vmin_data,
            vmax=vmax_data,
            transform=data_crs,
        )

        # add a colorbar - converting format of smaller numbers to exponents
        # modify the y label and reformat the tick axis to show min and max
        cbar = plt.colorbar(
            ctf,
            ax=ax,
            fraction=cbar_fraction,
            pad=cbar_pad,
            format=lambda x, _: f"{x:.0E}" if x <= 1 else f"{x:.0f}",
        )
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=20)
        cbar.set_ticks(
            np.concatenate(
                [
                    np.array([vmin_data]),
                    cbar.get_ticks()[1:-1],
                    np.array([vmax_data]),
                ]
            )
        )

        # defend against case attr_location is invalid
        if attr_location not in ATTR_LOC.keys():
            raise ValueError(
                f"Unrecognised value for `attr_location` {attr_location}."
                f"Expecting one of {list(ATTR_LOC.keys())}"
            )

        # create an attribution string and add it to the axis
        grid_res = self._xds.rio.resolution()
        attribution = (
            f"Generated on: {datetime.strftime(datetime.now(), '%Y-%m-%d')}\n"
            f"Population data: {var_attr}\n"
            f"Grid Size: {abs(grid_res[0])}m x {abs(grid_res[1])}m\n"
            f"Base map: {base_map_attr}"
        )
        ax.text(
            ATTR_LOC[attr_location]["x"],
            ATTR_LOC[attr_location]["y"],
            attribution,
            transform=ax.transAxes,
            size=attr_font_size,
            wrap=True,
            fontdict={"name": "Arial", "color": "#000000"},
            va=ATTR_LOC[attr_location]["va"],
            ha=ATTR_LOC[attr_location]["ha"],
        )

        # use tight layout to maximise axis size of axis
        plt.tight_layout()

        # write to file if filepath is given, since there is no figure, need to
        # get the current figure and resize it to match the axis before saving
        if save is not None:
            out_df = os.path.dirname(save)
            if not os.path.exists(out_df):
                os.mkdir(out_df)
            fig = plt.gcf()
            fig.set_size_inches(*figsize)
            fig.savefig(save)
            ax = None

        return ax

    def _plot_matplotlib(
        self, save: str = None, figsize: tuple = (6.4, 4.8)
    ) -> Union[plt.Axes, None]:
        """Plot raster data using matplotlib.

        A shallow wrapper around rioxarray's plot function, to display the
        population as a raster using the matplotlib backend.

        Parameters
        ----------
        save : str, optional
            Filepath to save location, with ".png" extension, by default None
            meaning the plot will not be saved to file.
        figsize : tuple, optional
            The matplotlib figursize width and height in inches, by default
            (6.4, 4.8).

        Returns
        -------
        Union[plt.Axes, None]
            Returns a QuadMesh axis object when not writing to file. When
            writing to file, None is return.

        """
        # handle matplotlib and rioxarry steps
        fig, ax = plt.subplots(figsize=figsize)
        self._xds.plot(ax=ax)
        plt.tight_layout()

        # write to file if filepath is given
        if save is not None:
            out_df = os.path.dirname(save)
            if not os.path.exists(out_df):
                os.mkdir(out_df)
            fig.savefig(save)
            ax = None

        return ax
