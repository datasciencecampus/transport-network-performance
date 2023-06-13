# Allow 8 GB of memory
import sys
import os
from r5py import TransportNetwork

sys.argv.append(["--max-memory", "8G"])

"""Check development environment will cope with r5py requirements."""
# search the ext dir for pbf & gtfs
search_pth = os.path.join("tests", "data")
foundf = os.listdir(search_pth)
gtfs = [os.path.join(search_pth, x) for x in foundf if x.endswith(".zip")][0]
pbf = [os.path.join(search_pth, x) for x in foundf if x.endswith(".pbf")][0]

# needs wrapping in try but specific exception to raise unknown. Examining r5py
# exception classes, I'll go with the below.
try:
    transport_network = TransportNetwork(pbf, [gtfs])
except RuntimeError:
    print("RuntimeError encountered")
    pass
except MemoryError:
    print("Memory error encountered")
    pass

# has a .mapdb file been created in the external directory?
mapdb_f = pbf + ".mapdb"
mapdb_p_f = pbf + ".mapdb.p"
if not os.path.exists(mapdb_f):
    raise FileNotFoundError(f"r5py did not create the expected file at:/ {mapdb_f}")
elif not os.path.exists(mapdb_p_f):
    raise FileNotFoundError(f"r5py did not create the expected file at:/ {mapdb_p_f}")
else:
    print("r5py has created the expected database files.")
