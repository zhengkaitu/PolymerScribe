import os

import pandas as pd
import time
import copy
import re

import rdkit
from rdkit import Chem
import networkx as nx

from polymersearch import topology
from polymersearch.graphs import build_atomistic, build_topology, get_objects, get_repeats, build_all_atomistic_graphs
from polymersearch.adjacency import feed_to_bonds_n


def generate_NetworkX_graphs(input_string, is_bigsmarts=False):
    atomistic_directed, no_other_objects = build_atomistic(input_string, is_bigsmarts)
    topology, topology_undir, multidigraph, descriptors, ids = build_topology(atomistic_directed)
    return {"string": input_string,
            "atomistic": atomistic_directed,
            "topology": topology,
            "top_undir": topology_undir,
            "ids": ids,
            "descriptors": descriptors,
            "multidigraph": multidigraph,
            "no_other_objects": no_other_objects
            }

def generate_ids_attribute(atomistic, elements, descriptors, desc_regex=r"\<\d*|\>\d*|\$\d*"):
    """
    This function creates the attribute "ids" of the atomistic graph nodes. Nodes with the same "ids" are from the
    same repeat unit or end group
    Args:
        atomistic: atomistic graph
        elements: symbols of the nodes
        descriptors: bonding descriptor nodes
        desc_regex: regex that identifies the bonding descriptors

    Returns: atomistic graph

    """
    def extract_atoms(graph, extracted, start_atom, descriptors):
        extracted.add(start_atom)
        neighbors = list(graph[start_atom])
        next_atoms = []
        for n in neighbors:
            if n not in descriptors and n not in extracted:
                extracted.add(n)
                next_atoms.append(n)
        for n in next_atoms:
            extracted = extract_atoms(graph, extracted, n, descriptors)
        return extracted

    # STEP 1: Create a new attribute, ids, that groups atoms in the same repeat unit or end group
    # Find the nodes connected to bonding descriptor by a bond type 1
    edge_labels = nx.get_edge_attributes(atomistic, "bond_type")
    start = []
    for i in edge_labels:
        if edge_labels[i] == "1":
            d = list(re.finditer(desc_regex, elements[i[0]]))
            if len(d) == 0:
                start.append(i[0])
            else:
                start.append(i[1])
    if len(start) == 0:
        start = [0]
    # Find nodes that are in the same repeat unit or end group as the ones found above
    extracted = []
    for i in range(len(start)):
        extracted.append(extract_atoms(atomistic, set(), start[i], descriptors))
    # Create an ID for each repeat unit or end group
    ids = dict()
    counter = 1
    for i in extracted:
        for k in i:
            ids[k] = counter
        counter += 1
    # Set the new attribute, ids
    nx.set_node_attributes(atomistic, ids, "ids")

    # Find the nodes connected to bonding descriptor by a bond type 2
    start = []
    for i in edge_labels:
        if edge_labels[i] == "2":
            d = list(re.finditer(desc_regex, elements[i[0]]))
            if len(d) == 0:
                if i[0] not in ids:
                    start.append(i[0])
            else:
                if i[1] not in ids:
                    start.append(i[1])
    # Find nodes that are in the same repeat unit or end group as the ones found above
    extracted = []
    for i in range(len(start)):
        extracted.append(extract_atoms(atomistic, set(), start[i], descriptors))
    # Create an ID for each repeat unit or end group
    for i in extracted:
        for k in i:
            ids[k] = counter
        counter += 1
    for i in descriptors:
        ids[i] = counter
        counter += 1
    # Set the new attribute, ids
    nx.set_node_attributes(atomistic, ids, "ids")

    return atomistic

def swap_bonding_descriptor_edges(atomistic, bonding_descriptor):
    """
    This function swaps all indices of edges to a bonding descriptor. Bond types 2 are converted into 1 and vice-versa.
    Args:
        atomistic: atomistic graph
        bonding_descriptor: bonding descriptor node

    Returns: None

    """

    # Get all edge indices
    edge_labels = nx.get_edge_attributes(atomistic, "bond_type")

    # Swap
    for edge, type in edge_labels.items():
        # If the bonding descriptor is not one of the nodes, skip it
        if bonding_descriptor not in edge:
            continue
        if type == "1":
            edge_labels[edge] = "2"
        elif type == "2":
            edge_labels[edge] = "1"

    # Set edge indices
    nx.set_edge_attributes(atomistic, edge_labels, "bond_type")

def traverse_and_fix_indices(atomistic, bonding_descriptor, list_of_descriptors, ids_to_node, ids_to_edges, ids_visited, bds_visited, atomistic_graph_list=[]):
    """
    For a given bonding descriptor, this function looks for all repeat units and end groups connected to it. If one of
    them only have bond indices 1, it takes each of the other bonding descriptors they are connected to and swaps all
    connections to it (type 2 becomes type 1 and vice-versa). This means that, if there is a repeat unit connected to 2
    other bonding descriptors, the function will create 2 new graphs.
    Args:
        atomistic: atomistic graph
        bonding_descriptor: bonding descriptor that is being checked
        list_of_descriptors: list of bonding descriptor nodes
        ids_to_node: dictionary that maps "ids" to nodes
        ids_to_edges: dictionary that maps "ids" to edges
        ids_visited: list of visited "ids"
        bds_visited: list of visited bonding descriptor nodes
        atomistic_graph_list: list of atomistic graphs

    Returns: list of atomistic graphs

    """

    # Find ids of RUs and end groups connected to the bonding descriptor
    rus_ids = [ids for ids, edges in ids_to_edges.items() if [1 for e in edges if (bonding_descriptor in e)]]

    # Get the RUs and end groups with more than one index 1 and no 2
    only_index_one = []
    for ids in rus_ids:
        bond_types = [atomistic.edges[e]["bond_type"] for e in ids_to_edges[ids]]
        # If there are no 2
        if_no_two = "2" not in bond_types
        # If there are more than two 1's
        if_many_ones = bond_types.count("1") > 1
        if if_no_two and if_many_ones:
            only_index_one.append(ids)

    # For each repeat unit or end group with more than one index 1 and no 2'
    atomistic_graphs = [atomistic]
    for ids in only_index_one:
        _atomistic_graphs = []
        for _atomistic in atomistic_graphs:
            # Create a new atomistic graph for each edge that is not the one connected to the actual bonding descriptor
            edges_to_other_bds = [e for e in ids_to_edges[ids] if (bonding_descriptor not in e) and (e[0] in list_of_descriptors or e[1] in list_of_descriptors)]
            # For each of these edges, create a new graph
            for edge in edges_to_other_bds:
                # Create graph
                new_atomistic_graph = copy.deepcopy(atomistic)
                # Get the bond index
                new_edge_bond_index = atomistic.edges[edge]["bond_type"]
                # Swap all indices of that bonding descriptor
                new_bonding_descriptor = edge[0] if edge[0] in list_of_descriptors else edge[1]
                swap_bonding_descriptor_edges(new_atomistic_graph, new_bonding_descriptor)
                # Execute the function at that bonding descriptor
                bds_visited.append(new_bonding_descriptor)
                new_atomistic_graphs = traverse_and_fix_indices(atomistic=new_atomistic_graph,
                                                                bonding_descriptor=new_bonding_descriptor,
                                                                list_of_descriptors=list_of_descriptors,
                                                                ids_to_node=ids_to_node,
                                                                ids_to_edges=ids_to_edges,
                                                                ids_visited=ids_visited,
                                                                bds_visited=bds_visited,
                                                                atomistic_graph_list=atomistic_graph_list)

                # Add it to list of atomistic graphs
                _atomistic_graphs += new_atomistic_graphs
        atomistic_graphs = copy.deepcopy(_atomistic_graphs)

    return atomistic_graphs

def fix_atomistic_graph_indices(atomistic, desc_regex=r"\<\d*|\>\d*|\$\d*"):
    """
    Sometimes, the indices of the atomistic graph are not consistent. For example, linkers between blocks may have only
    bonds of type 1. This function looks for end groups with only one bond type 2. For each one of them, it calls
    traverse_and_fix_indices() to fix its indices. This means that, if there is a repeat unit or linker that only have
    bonds to bonding descriptor nodes with type 1, it changes the bond indices so that there is one bond type 2. When
    fixing the indices, more graphs can be created.
    Args:
        atomistic: atomistic graph
        desc_regex: regex that identifies bonding descriptors

    Returns: list of atomistic graphs.

    """

    # List of symbols
    elements = nx.get_node_attributes(atomistic, "symbol")
    # Generate list of bonding descriptor nodes
    descriptors = []
    for key in elements:
        d = list(re.finditer(desc_regex, elements[key]))
        if len(d) != 0:
            descriptors.append(key)

    # Set attribute that groups repeat units and end groups
    atomistic = generate_ids_attribute(atomistic, elements, descriptors, desc_regex)

    # Get node ids
    node_to_ids = nx.get_node_attributes(atomistic, "ids")
    ids_to_node = {ids: [] for ids in node_to_ids.values()}
    for node, ids in node_to_ids.items():
        ids_to_node[ids].append(node)

    # Find all edges with bonding descriptors
    edge_labels = {e: label for e, label in nx.get_edge_attributes(atomistic, "bond_type").items() if (e[0] in descriptors) or (e[1] in descriptors)}
    # Create dict that stores all bonds to bonding descriptor per ids
    ids_to_edges = {ids: [] for ids in node_to_ids.values()}
    for e in edge_labels:
        if e[0] not in descriptors:
            ids_to_edges[node_to_ids[e[0]]].append(e)
        elif e[1] not in descriptors:
            ids_to_edges[node_to_ids[e[1]]].append(e)

    # Get end groups. Those are groups of nodes that are connected to only one bonding descriptor
    end_group_ids = [ids for ids, edges in ids_to_edges.items() if len(edges) == 1]
    end_group_edges = {ids: ids_to_edges[ids][0] for ids in end_group_ids}
    end_group_nodes = {ids: ids_to_node[ids] for ids in end_group_ids}
    # Find nodes in end groups that are connected to bonding descriptor
    endgroup_start_atoms = {ids: e[0] if (e[0] not in descriptors) else e[1] for ids, e in end_group_edges.items()}
    # Starting end groups (the ones whose edges are 2)
    starting_endgroups = {ids: e[0] if (e[0] not in descriptors) else e[1] for ids, e in end_group_edges.items() if edge_labels[e] == "2"}
    # If there are no end groups, use the bonding descriptors
    if not starting_endgroups:
        starting_endgroups = {node_to_ids[n]: n for n in descriptors}

    # Get atoms in end groups that are connected to bonding descriptors
    list_of_atomistic = []
    ids_visited = []
    bds_visited = []
    for ids, start in starting_endgroups.items():
        # If it already is a bonding descriptor, use it
        if start in descriptors:
            bonding_descriptor = start
        else:
            # Get bonding descriptor connected to it
            bonding_descriptor = end_group_edges[ids][0] if end_group_edges[ids][0] != start else end_group_edges[ids][1]
        ids_visited.append(ids)
        bds_visited.append(bonding_descriptor)
        list_of_atomistic += traverse_and_fix_indices(atomistic=atomistic,
                                                      bonding_descriptor=bonding_descriptor,
                                                      list_of_descriptors=descriptors,
                                                      ids_to_node=ids_to_node,
                                                      ids_to_edges=ids_to_edges,
                                                      ids_visited=ids_visited,
                                                      bds_visited=bds_visited)

    return list_of_atomistic

def remove_graph_duplicates(graph_list, filter=lambda a, b: a == b):
    """
    This function removes duplicates from a list of graphs. Two graphs are the same if they are isomorfic and their
    attributes are the same
    Args:
        graph_list: list of graphs
        filter: function that checks whether 2 graphs are the same. Given the attributes of 2 nodes, it must return True
        if the nodes are the same

    Returns: list of graphs without duplicates

    """
    seen = set()
    result = []
    for item in graph_list:
        if not any(nx.is_isomorphic(item, x, edge_match=filter) for x in seen):
            seen.add(item)
            result.append(item)
    return result

def generate_all_possible_graphs(input_string):
    """
    This function generates all possible graphs (atomistic and topology), given a BigSMILES string
    Args:
        input_string: BigSMILES
    Returns: List of dictionaries, whose keys are "atomistic" and "topology"
    """
    # Shift the bonding descriptor indices. For example > and <1 become >1 and <2
    bonding_descriptors = set(re.findall(r"\<\d*|\>\d*|\$\d*", input_string))    # List of bonding descriptors without duplicates
    for bd in bonding_descriptors:
        # Get the bonding descriptor symbol and index
        if len(bd) == 1:
            bd_symbol = bd
            bd_index = 0
        else:
            bd_symbol = bd[0]
            bd_index = int(bd[1]) + 1
        # Replace
        input_string = input_string.replace(f"[{bd}]", f"[replace{bd_symbol}{bd_index}]")
    input_string = input_string.replace("replace", "")

    # Generate all atomistic graphs
    atomistic_list = build_all_atomistic_graphs(input_string)

    _list = []
    for atomistic in atomistic_list:
        # Skip if number of edges is less than 2
        if len(atomistic.edges) < 2:
            continue
        # Fix atomistic graph
        _list += fix_atomistic_graph_indices(atomistic)
    atomistic_list = _list

    # Remove graph duplicates
    atomistic_list = remove_graph_duplicates(atomistic_list)

    # Generate all topology graphs and fix indices
    topology_list = []
    for atomistic in atomistic_list:
        # Build topology graph
        topology, _, _, _, _ = build_topology(atomistic)
        topology_list.append(topology)

    # Generate list of graphs
    graphs = []
    for atomistic, topology in zip(atomistic_list, topology_list):
        graphs.append({"atomistic": atomistic,
                       "topology": topology})

    return graphs


def identify_cycles(graphs):
    topology = graphs["topology"]

    ids = nx.get_node_attributes(topology, "ids")
    individual_cycles = []
    for i in nx.simple_cycles(topology):
        individual_cycles.append([ids[j] for j in i])

    explicit = nx.get_node_attributes(graphs["atomistic"], "explicit_atom_ids")
    ids = nx.get_node_attributes(graphs["atomistic"], "ids")
    treat_as_cycle = []
    for key in explicit:
        if explicit[key]:
            remove = -1
            for i in individual_cycles:
                if ids[key] in i:
                    remove = i
                    break
            if remove != -1:
                individual_cycles.remove(remove)
                treat_as_cycle.append(remove)

    # https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    def to_graph(l):
        G = nx.Graph()
        for part in l:
            G.add_nodes_from(part)
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        it = iter(l)
        last = next(it)
        for current in it:
            yield last, current
            last = current

    individual_clusters = []
    G = to_graph(individual_cycles)
    for cycles in nx.connected_components(G):
        individual_clusters.append(cycles)

    level = nx.get_node_attributes(graphs["top_undir"], "level")
    ids = nx.get_node_attributes(graphs["top_undir"], "ids")
    inv_map = {v: k for k, v in ids.items()}
    bonds = nx.get_edge_attributes(graphs["top_undir"], "bond_type")
    nested = []
    for i in range(len(individual_clusters)):
        count_0 = 0
        count_1 = 0
        for j in individual_clusters[i]:
            id = inv_map[j]
            if id in level and level[id] == 1:
                count_1 = 1
            if id in level and level[id] == 0:
                count_0 = 1
        if count_0 == 0 and count_1 == 1 and individual_clusters[i] not in nested:
            nested.append(individual_clusters[i])
    for i in range(len(nested)):
        if nested[i] not in individual_clusters:
            continue
        for j in range(len(individual_clusters)):
            adjacent = False
            for k in nested[i]:
                for l in individual_clusters[j]:
                    if feed_to_bonds_n(inv_map[k], inv_map[l]) in bonds:
                        adjacent = True
            if adjacent:
                individual_clusters[j].update(nested[i])
                individual_clusters.remove(nested[i])
                break

    group_cycles = []
    for i in range(len(individual_clusters)):
        group_cycles.append([])
        for j in range(len(individual_cycles)):
            if individual_cycles[j][0] in individual_clusters[i]:
                group_cycles[-1].append(individual_cycles[j])
    return group_cycles, individual_clusters, treat_as_cycle


def cycles_to_ring_SMILES(graphs, cycle):
    # https://github.com/maxhodak/keras-molecules/pull/32/files

    symbols = nx.get_node_attributes(graphs["atomistic"], "symbol")
    formal_charge = nx.get_node_attributes(graphs["atomistic"], "formal_charge")
    is_aromatic = nx.get_node_attributes(graphs["atomistic"], "is_aromatic")
    bonds = nx.get_edge_attributes(graphs["atomistic"], "bond_type_object")
    ids = nx.get_node_attributes(graphs["atomistic"], "ids")

    rings = []
    for i in range(len(cycle)):
        for j in range(len(cycle[i])):
            mol = Chem.RWMol()
            node_to_idx = {}
            for node in graphs["atomistic"].nodes():
                if ids[node] in cycle[i][j] and node not in graphs["descriptors"] and symbols[node] != "":
                    a = Chem.Atom(symbols[node])
                    a.SetFormalCharge(formal_charge[node])
                    a.SetIsAromatic(is_aromatic[node])
                    idx = mol.AddAtom(a)
                    node_to_idx[node] = idx

            d_bonds = []
            already_added = set()
            for edge in graphs["atomistic"].edges():
                first, second = edge
                if ids[first] in cycle[i][j] and ids[second] in cycle[i][j]:
                    if first in graphs["descriptors"] or second in graphs["descriptors"]:
                        d_bonds.append([(first, second), rdkit.Chem.rdchem.BondType.SINGLE])
                    else:
                        ifirst = node_to_idx[first]
                        isecond = node_to_idx[second]
                        bond_type_object = bonds[first, second]
                        if tuple(sorted([ifirst, isecond])) not in already_added:
                            mol.AddBond(ifirst, isecond, bond_type_object)
                            already_added.add(tuple(sorted([ifirst, isecond])))

            for k in range(len(d_bonds)):
                for l in range(k + 1, len(d_bonds)):
                    descriptor = set(d_bonds[k][0]).intersection(d_bonds[l][0])
                    if len(descriptor) == 1:
                        try:
                            a = set(d_bonds[k][0]).difference(descriptor).pop()
                            b = set(d_bonds[l][0]).difference(descriptor).pop()
                            ifirst = node_to_idx[a]
                            isecond = node_to_idx[b]
                            if tuple(sorted([ifirst, isecond])) not in already_added:
                                mol.AddBond(ifirst, isecond, d_bonds[k][1])
                                already_added.add(tuple(sorted([ifirst, isecond])))
                        except:
                            continue

            Chem.SanitizeMol(mol)
            rings.append(Chem.MolToSmiles(mol))

    return rings


def get_repeats_as_rings(graphs):
    cycles, clusters = identify_cycles(graphs["topology"])
    ring_smiles = cycles_to_ring_SMILES(graphs, cycles)
    return ring_smiles


def contains_substructure(bigsmarts_graphs, bigsmiles_graphs):
    q_cycles, q_clusters, q_macrocycle = identify_cycles(bigsmarts_graphs)
    t_cycles, t_clusters, t_macrocycle = identify_cycles(bigsmiles_graphs)
    search = topology.Topology_Graph_Matcher(bigsmarts_graphs, bigsmiles_graphs,
                                             q_cycles, q_clusters,
                                             t_cycles, t_clusters,
                                             q_macrocycle, t_macrocycle)
    return search.search_repeats_endgroups()


def logical_repeat_unit_search(bigsmarts):
    # Assumptions:
    # only 1 object in the query has logical operators
    # only one type of logical operation per query: "!", "or", "xor" along with "and"

    # determine all objects in the string
    objects = get_objects(bigsmarts)

    # iterate through each object
    for object in objects[0]:

        # get all repeat units in the object
        repeats = get_repeats(object)

        # map the logical operator to the repeat units
        logical = {}
        for repeat in repeats[0]:

            # group logical operator with repeat units or SMARTS
            if repeat.find("[or") == 0 and repeat[0:5] not in logical:
                logical[repeat[0:5]] = [repeat[5:]]
            elif repeat.find("[or") == 0 and repeat[0:5] in logical:
                logical[repeat[0:5]].append(repeat[5:])

            elif repeat.find("[xor") == 0 and repeat[0:6] not in logical:
                logical[repeat[0:6]] = [repeat[6:]]
            elif repeat.find("[xor") == 0 and repeat[0:6] in logical:
                logical[repeat[0:6]].append(repeat[6:])

            elif repeat.find("!") == 0 and repeat != "!*" and "!" not in logical:
                logical["!"] = [repeat[1:]]
            elif repeat.find("!") == 0 and repeat != "!*" and repeat[1:] in logical:
                logical["!"].append(repeat[1:])

            elif "and" not in logical:
                logical["and"] = [repeat]
            else:
                logical["and"].append(repeat)

        # is this the object with the logical operators?
        logic = list(logical.keys())
        logic = [i for i in logic if i != "and"]

        # if not, continue
        if len(logic) == 0:
            continue

        # list of object strings that convert logical strings into valid BigSMARTS
        objects_replaced = []
        logic_return = "and"
        for logic in logical:

            if "or" in logic or "xor" in logic:
                logic_return = logic
                for repeat in logical[logic]:
                    # delete logical operator and repeat unit
                    if logic + repeat + "," in object:
                        # this is for every other repeat unit in the stochastic object
                        replaced = object.replace(logic + repeat + ",", "")
                    else:
                        # this is for the last repeat unit in the stochastic object only
                        replaced = object.replace("," + logic + repeat, "")
                    replaced = replaced.replace(logic, "")
                    objects_replaced.append(replaced)

            elif "!" in logic:
                logic_return = logic
                for repeat in logical[logic]:
                    # delete logical operator
                    replaced = object.replace("!", "")
                    objects_replaced.append(replaced)
                # delete logical operator and repeat unit
                if logic + repeat + "," in object:
                    # this is for every other repeat unit in the stochastic object
                    replaced = object.replace(logic + repeat + ",", "")
                else:
                    # this is for the last repeat unit in the stochastic object only
                    replaced = object.replace("," + logic + repeat, "")
                replaced = replaced.replace(logic, "")
                objects_replaced.append(replaced)

        bigsmarts_replaced = []
        for o in objects_replaced:
            bigsmarts_replaced.append(bigsmarts.replace(object, o))

        return bigsmarts_replaced, logic_return

    return [bigsmarts], "and"


def search_matches(bigsmarts, bigsmiles):
    bigsmarts_list, logic = logical_repeat_unit_search(bigsmarts)
    matches = []
    for bigsmarts in bigsmarts_list:
        bigsmarts_graphs = generate_NetworkX_graphs(input_string=bigsmarts, is_bigsmarts=True)
        bigsmiles_graphs = generate_NetworkX_graphs(input_string=bigsmiles, is_bigsmarts=False)
        # visualize_NetworkX_graphs(graphs = bigsmarts_graphs, id = "bigsmarts")
        # visualize_NetworkX_graphs(graphs = bigsmiles_graphs, id = "bigsmiles")
        m = contains_substructure(bigsmarts_graphs=bigsmarts_graphs, bigsmiles_graphs=bigsmiles_graphs)
        if "or" in logic and "xor" not in logic and m:
            return True
        elif "and" in logic:
            return m
        matches.append(m)
    if "!" in logic:
        if matches[0] == False and matches[1] == True:
            return True
        return False
    if "xor" in logic:
        if matches[0] == False and matches[1] == True or matches[0] == True and matches[1] == False:
            return True
        return False
    if "or" in logic:
        return matches[1]
    return False
