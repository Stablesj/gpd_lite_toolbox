# -*- coding: utf-8 -*-
"""
@author: mthh


Copy-paste from geopandas.plotting module with only minor modif. in order to
take linewidth and edgecolor on polygon plotting into account.
"""
from geopandas.plotting import (
    # _mapclassify_choro, 
    plot_series, 
    plot_point_collection, 
    plot_linestring_collection
    )
from mapclassify import classify
from geopandas import GeoSeries
import numpy as np
import geopandas as gpd

# From: https://github.com/geopandas/geopandas/pull/1737/files
def _mapclassify_choro(values, scheme, **classification_kwds):
    """
    Wrapper for choropleth schemes from mapclassify for use with plot_dataframe
    Parameters
    ----------
    values
        Series to be plotted
    scheme : str
        One of mapclassify classification schemes
        Options are BoxPlot, EqualInterval, FisherJenks,
        FisherJenksSampled, HeadTailBreaks, JenksCaspall,
        JenksCaspallForced, JenksCaspallSampled, MaxP,
        MaximumBreaks, NaturalBreaks, Quantiles, Percentiles, StdMean,
        UserDefined
    **classification_kwds : dict
        Keyword arguments for classification scheme
        For details see mapclassify documentation:
        https://pysal.org/mapclassify/api.html
    Returns
    -------
    binning
        Binning objects that holds the Series with values replaced with
        class identifier and the bins.
    """
    try:
        import mapclassify.classifiers as classifiers

    except ImportError:
        raise ImportError(
            "The 'mapclassify' >= 2.2.0 package is required to use the 'scheme' keyword"
        )
    from mapclassify import __version__ as mc_version

    if mc_version < LooseVersion("2.2.0"):
        raise ImportError(
            "The 'mapclassify' >= 2.2.0 package is required to "
            "use the 'scheme' keyword"
        )
    schemes = {}
    for classifier in classifiers.CLASSIFIERS:
        schemes[classifier.lower()] = getattr(classifiers, classifier)

    scheme = scheme.lower()

    # mapclassify < 2.1 cleaned up the scheme names (removing underscores)
    # trying both to keep compatibility with older versions and provide
    # compatibility with newer versions of mapclassify
    oldnew = {
        "Box_Plot": "BoxPlot",
        "Equal_Interval": "EqualInterval",
        "Fisher_Jenks": "FisherJenks",
        "Fisher_Jenks_Sampled": "FisherJenksSampled",
        "HeadTail_Breaks": "HeadTailBreaks",
        "Jenks_Caspall": "JenksCaspall",
        "Jenks_Caspall_Forced": "JenksCaspallForced",
        "Jenks_Caspall_Sampled": "JenksCaspallSampled",
        "Max_P_Plassifier": "MaxP",
        "Maximum_Breaks": "MaximumBreaks",
        "Natural_Breaks": "NaturalBreaks",
        "Std_Mean": "StdMean",
        "User_Defined": "UserDefined",
    }
    scheme_names_mapping = {}
    scheme_names_mapping.update(
        {old.lower(): new.lower() for old, new in oldnew.items()}
    )
    scheme_names_mapping.update(
        {new.lower(): old.lower() for old, new in oldnew.items()}
    )

    try:
        scheme_class = schemes[scheme]
    except KeyError:
        scheme = scheme_names_mapping.get(scheme, scheme)
        try:
            scheme_class = schemes[scheme]
        except KeyError:
            raise ValueError(
                "Invalid scheme. Scheme must be in the set: %r" % schemes.keys()
            )

    if classification_kwds["k"] is not None:
        from inspect import getfullargspec as getspec

        spec = getspec(scheme_class.__init__)
        if "k" not in spec.args:
            del classification_kwds["k"]
    try:
        binning = scheme_class(np.asarray(values), **classification_kwds)
    except TypeError:
        raise TypeError("Invalid keyword argument for %r " % scheme)
    return binning


def m_plot_multipolygon(ax, geom, linewidth, facecolor='red', edgecolor='grey', alpha=0.5):
    """ Can safely call with either Polygon or Multipolygon geometry
    """
    if geom.geom_type == 'Polygon':
        m_plot_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=linewidth, alpha=alpha)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            m_plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=linewidth, alpha=alpha)


def m_plot_polygon(ax, poly, facecolor='red', edgecolor='black', linewidth=0.5, alpha=0.5):
    """ Plot a single Polygon geometry """
    from descartes.patch import PolygonPatch
    # a = np.asarray(poly.exterior)
    # # without Descartes, we could make a Patch of exterior
    # ax.add_patch(PolygonPatch(poly, facecolor=facecolor,
    #                           linewidth=linewidth, alpha=alpha))
    
    a = np.asarray(poly.exterior.xy).T
    
    gpd.GeoSeries(poly).plot(ax=ax, facecolor=facecolor, linewidth=linewidth, alpha=alpha)
    
    ax.plot(a[:, 0], a[:, 1], color=edgecolor, linewidth=linewidth)
    
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth)


def m_plot_dataframe(s, column=None, colormap=None, alpha=0.5, edgecolor=None,
                     categorical=False, legend=False, axes=None, scheme=None,
                     contour_poly_width=0.5,
                     k=5):
    """ Plot a GeoDataFrame

        Generate a plot of a GeoDataFrame with matplotlib.  If a
        column is specified, the plot coloring will be based on values
        in that column.  Otherwise, a categorical plot of the
        geometries in the `geometry` column will be generated.

        Parameters
        ----------

        GeoDataFrame
            The GeoDataFrame to be plotted.  Currently Polygon,
            MultiPolygon, LineString, MultiLineString and Point
            geometries can be plotted.

        column : str (default None)
            The name of the column to be plotted.

        categorical : bool (default False)
            If False, colormap will reflect numerical values of the
            column being plotted.  For non-numerical columns (or if
            column=None), this will be set to True.

        colormap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.

        alpha : float (default 0.5)
            Alpha value for polygon fill regions.  Has no effect for
            lines or points.

        legend : bool (default False)
            Plot a legend (Experimental; currently for categorical
            plots only)

        axes : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        scheme : pysal.esda.mapclassify.Map_Classifier
            Choropleth classification schemes

        k   : int (default 5)
            Number of classes (ignored if scheme is None)


        Returns
        -------

        matplotlib axes instance
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if column is None:
        return plot_series(s.geometry, colormap=colormap, alpha=alpha, axes=axes)
    else:
        if s[column].dtype is np.dtype('O'):
            categorical = True
        if categorical:
            if colormap is None:
                colormap = 'Set1'
            categories = list(set(s[column].values))
            categories.sort()
            valuemap = dict([(key, v) for (v, key) in enumerate(categories)])
            values = [valuemap[key] for key in s[column]]
        else:
            values = s[column]
        if scheme is not None:
            values = _mapclassify_choro(values, scheme, k=k)
            # values = classify(np.asarray(values), scheme, k=k)

        norm = Normalize(vmin=values.min(), vmax=values.max())
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        if not axes:
            fig = plt.gcf()
            fig.add_subplot(111, aspect='equal')
            ax = plt.gca()
        else:
            ax = axes
        for geom, value in zip(s.geometry, values):
            if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                m_plot_multipolygon(ax, geom, facecolor=cmap.to_rgba(value),
                                    edgecolor=edgecolor,
                                    linewidth=contour_poly_width, alpha=alpha)
            elif geom.geom_type == 'LineString' or geom.geom_type == 'MultiLineString':
                plot_linestring_collection(ax, GeoSeries([geom]), colors=[cmap.to_rgba(value)])
            # TODO: color point geometries
            elif geom.geom_type == 'Point':
                plot_point_collection(ax, GeoSeries([geom]))
        if legend:
            if categorical:
                patches = []
                for value, cat in enumerate(categories):
                    patches.append(Line2D([0], [0], linestyle="none",
                                          marker="o", alpha=alpha,
                                          markersize=10,
                                          markerfacecolor=cmap.to_rgba(value)))
                ax.legend(patches, categories, numpoints=1, loc='best')
            else:
                # TODO: show a colorbar
                raise NotImplementedError
    plt.draw()
    return ax