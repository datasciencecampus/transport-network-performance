"""Helpers for working with routes.txt."""
import pandas as pd
from bs4 import BeautifulSoup
import requests


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
    ext_spec_url : tuple
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
    # Get the basic scheme lookup
    resp = requests.get(gtfs_url).text
    soup = BeautifulSoup(resp, "html.parser")
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
    route_lookup = pd.DataFrame(zip(cds, txts), columns=["route_type", "desc"])
    # if interested in the extended schema, get that too. Perhaps not
    # relevant to all territories
    if extended_schema:
        resp = requests.get(ext_spec_url).text
        soup = BeautifulSoup(resp, "html.parser")
        for i in soup.findAll("table"):
            # target table has 'nice_table' class
            if i.get("class")[0] == "nice-table":
                target = i

        cds = list()
        descs = list()
        for row in target.tbody.findAll("tr"):
            # Get the table headers
            found = row.findAll("th")
            if found:
                cols = [f.text for f in found]
            else:
                # otherwise get the table data
                dat = [i.text for i in row.findAll("td")]
                # subset to the required column
                cds.append(dat[cols.index("Code")])
                descs.append(dat[cols.index("Description")])

        added_spec = pd.DataFrame(
            zip(cds, descs), columns=["route_type", "desc"]
        )
        route_lookup = pd.concat([route_lookup, added_spec]).reset_index(
            drop=True
        )

    return route_lookup
