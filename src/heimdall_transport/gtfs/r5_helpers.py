"""Utilty functions specific to setting up r5."""

import os

from r5py import TransportNetwork


def check_r5_setup(search_pth=os.path.join("tests", "data")):
    """Check development environment will cope with r5py requirements.

    Parameters
    ----------
        search_pth : str
            Path to directory containing GTFS feed. Defaults to
            `os.path.join("tests", "data")`

    Returns
    -------
        str
            path to expected mapdb file.
        str
            path to expected mapdb.p file.

    """
    # search the ext dir for pbf & gtfs
    foundf = os.listdir(search_pth)
    gtfs = [os.path.join(search_pth, x) for x in foundf if x.endswith(".zip")][
        0
    ]
    pbf = [os.path.join(search_pth, x) for x in foundf if x.endswith(".pbf")][
        0
    ]
    # needs wrapping in try but specific exception to raise unknown.
    # Examining r5py exception classes, I'll go with the below.
    try:
        TransportNetwork(pbf, [gtfs])
    except RuntimeError:
        print("RuntimeError encountered")
        pass
    except MemoryError:
        print("Memory error encountered")
        pass
    # return paths to .mapdb files
    mapdb_f = pbf + ".mapdb"
    mapdb_p_f = pbf + ".mapdb.p"
    return (mapdb_f, mapdb_p_f)
