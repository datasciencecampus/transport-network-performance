"""Helpers for working with routes.txt."""
import pandas as pd
from bs4 import BeautifulSoup
import requests
import warnings

from transport_performance.utils.defence import _url_defence, _bool_defence

warnings.filterwarnings(
    action="ignore", category=DeprecationWarning, module=".*pkg_resources"
)
# see https://github.com/datasciencecampus/transport-network-performance/
# issues/19


def _construct_extended_schema_table(some_soup, cd_list, desc_list):
    """Create the extended table from a soup object. Not exported.

    Parameters
    ----------
    some_soup : bs4.BeautifulSoup
        A bs4 soup representation of `ext_spec_url`.
    cd_list : list
        A list of schema codes scraped so far. Will append addiitonal codes to
        this list.
    desc_list : list
        A list of schema descriptions found so far. Will append additional
        descriptions to this list.

    Returns
    -------
        tuple[0]: Proposed extension to route_type codes
        tuple[1]: Proposed extension to route_type descriptions

    """
    for i in some_soup.findAll("table"):
        # target table has 'nice_table' class
        if i.get("class")[0] == "nice-table":
            target = i

    for row in target.tbody.findAll("tr"):
        # Get the table headers
        found = row.findAll("th")
        if found:
            cols = [f.text for f in found]
        else:
            # otherwise get the table data
            dat = [i.text for i in row.findAll("td")]
            # subset to the required column
            cd_list.append(dat[cols.index("Code")])
            desc_list.append(dat[cols.index("Description")])

    return (cd_list, desc_list)


def _get_response_text(url):
    """Return the response & extract the text. Not exported."""
    r = requests.get(url)
    t = r.text
    return t


def scrape_route_type_lookup(
    gtfs_url="https://gtfs.org/schedule/reference/",
    ext_spec_url=(
        "https://developers.google.com/transit/gtfs/reference/"
        "extended-route-types"
    ),
    extended_schema=True,
):
    """Scrape a lookup of GTFS route_type codes to descriptions.

    Scrapes HTML tables from `gtfs_url` to provide a lookup of `route_type`
    codes to human readable descriptions. Useful for confirming available
    modes of transport within a GTFS. If `extended_schema` is True, then also
    include the proposed extension of route_type to the GTFS.

    Parameters
    ----------
    gtfs_url : str
        The url containing the GTFS accepted route_type codes. Defaults to
        "https://gtfs.org/schedule/reference/".
    ext_spec_url : str
        The url containing the table of the proposed extension to the GTFS
        schema for route_type codes. Defaults to
        ( "https://developers.google.com/transit/gtfs/reference/"
        "extended-route-types" ).
    extended_schema : bool
        Should the extended schema table be scraped and included in the output?
        Defaults to True.

    Returns
    -------
        pd.core.frame.DataFrame: A lookup of route_type codes to descriptions.

    """
    # a little defence
    for url in [gtfs_url, ext_spec_url]:
        _url_defence(url)

    _bool_defence(extended_schema)
    # Get the basic scheme lookup
    resp_txt = _get_response_text(gtfs_url)
    soup = BeautifulSoup(resp_txt, "html.parser")
    for dat in soup.findAll("td"):
        # Look for a pattern to target, going with Tram, could go more specific
        # with regex if table format unstable.
        if "Tram" in dat.text:
            target_node = dat

    cds = list()
    txts = list()
    # the required data is in awkward little inline 'table' that's really
    # a table row, but helpfully the data is either side of some break
    # tags
    for x in target_node.findAll("br"):
        cds.append(x.nextSibling.text)
        txts.append(x.previousSibling.text)
    # strip out rubbish
    cds = [cd for cd in cds if len(cd) > 0]
    txts = [t.strip(" - ") for t in txts if t.startswith(" - ")]
    # catch the final description which is not succeeded by a break
    txts.append(target_node.text.split(" - ")[-1])
    # if interested in the extended schema, get that too. Perhaps not
    # relevant to all territories
    if extended_schema:
        resp_txt = _get_response_text(ext_spec_url)
        soup = BeautifulSoup(resp_txt, "html.parser")
        cds, txts = _construct_extended_schema_table(soup, cds, txts)

    route_lookup = pd.DataFrame(zip(cds, txts), columns=["route_type", "desc"])

    return route_lookup
