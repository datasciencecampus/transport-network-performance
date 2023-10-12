"""Validation of OSM pbf files."""
import osmium

PBF_FIX_PTH = "tests/data/newport-2023-06-13.osm.pbf"
WALES_LATEST_PTH = "data/external/osm/wales-latest.osm.pbf"


def _compile_tags(osmium_feature):
    tagdict = {}
    for tag in osmium_feature.tags:
        # taglist.append(tag)
        for i, tag in enumerate(tag):
            if i % 2 == 0:
                # even tags are keys
                k = tag
            else:
                # odd tags are values
                v = tag
        tagdict[k] = v
    return tagdict


# ---- look at internals
class GenericHandler(osmium.SimpleHandler):
    """Placeholder.

    Parameters
    ----------
    osmium : class
        Inherits from osmium.SimpleHandler

    """

    def __init__(self):
        super().__init__()
        self.node_ids = []
        self.node_locs = {}
        self.node_tags = {}
        self.way_ids = []
        self.way_nodes = {}
        self.way_tags = {}
        self.relations_ids = []
        self.relations_members = {}
        self.relations_tags = {}

    # the methods you define must be called node, way, relation, area or
    # changeset

    def way(self, w):
        """Process ways.

        Parameters
        ----------
        w : osmium.osm.types.Way
            A 'way' feature.

        """
        self.way_ids.append(w.id)
        # compile the member nodes of each way
        nodelist = []
        for node in w.nodes:
            nodelist.append(node.ref)
        self.way_nodes[w.id] = nodelist
        # compile tags for each way
        tags_dict = _compile_tags(w)
        self.way_tags[w.id] = tags_dict

    def relation(self, r):
        """Process relations.

        Parameters
        ----------
        r : osmium.osm.types.Relation
            A 'relation' feature.

        """
        self.relations_ids.append(r.id)
        members_list = []
        # compile the relation members
        for member in r.members:
            members_list.append(member)
        self.relations_members[r.id] = members_list
        # compile the relation tags
        tags_dict = _compile_tags(r)
        self.relations_tags[r.id] = tags_dict

    def node(self, n):
        """Process nodes.

        Parameters
        ----------
        n : osmium.osm.types.Node
            A 'node' feature.

        """
        self.node_ids.append(n.id)
        # extract x,y
        x, y = str(n.location).split("/")
        # store representative point for each node
        self.node_locs[n.id] = {"lon": x, "lat": y}
        # get tags for each node
        tagdict = _compile_tags(n)
        self.node_tags[n.id] = tagdict


i = GenericHandler()
i.apply_file(PBF_FIX_PTH, locations=False)

# counts of features.
len(i.node_ids)
len(i.way_ids)
len(i.relations_ids)

i.node_locs[10971292662]
i.node_tags[10971292664]

i.relations_tags[15775326]
i.relations_members[15775326]  # note that internally, osmium will prepend the
# tag with the type of feature, eg r15775326 is a relation, n15775326 would be
# a node etc.
i.way_nodes[1181392035]
i.way_tags[1181392039]
