import itertools
import multiprocessing
import numpy as np
import traceback
import rdkit
import rdkit.Chem as Chem
from rdkit.Geometry import Point3D
from typing import List, Tuple
from .constants import RGROUP_SYMBOLS, ABBREVIATIONS
from .utils import log_rank_0

rdkit.RDLogger.DisableLog('rdApp.*')

BOND_TYPES = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
}

def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    try:
        n = mol.GetNumAtoms()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
            # print(f"i: {i}, x: {x}, y: {y}")
        mol.AddConformer(conf)

        return mol.GetMol()

        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        Chem.SanitizeMol(mol_tmp)

        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  # List[Tuple[int, any]] -> List[int]

        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
            # print(f"i: {i}, x: {x}, y: {y}")
        mol.AddConformer(conf)

        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _add_conformer(mol, coords, debug=False):
    try:
        n = mol.GetNumAtoms()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
            # print(f"i: {i}, x: {x}, y: {y}")
        mol.AddConformer(conf)

        return mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _add_sgroup(
    mol,
    bracket_symbols: List[str],
    bracket_coords: List[Tuple[float, float]]
):
    # mol is edited in-place
    sep_indices = [i for i, coord in enumerate(bracket_coords) if coord is None]
    start_indices = [0] + [i + 1 for i in sep_indices[:-1]]

    # Atom positions from the conformer.  The conformer stores (x, 1-y, 0), matching
    # the y-flip applied to bracket coords below.
    atom_positions = []
    try:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_positions.append((pos.x, pos.y))
    except Exception:
        pass

    for start_i, end_i in zip(start_indices, sep_indices):
        symbols = bracket_symbols[start_i:end_i]
        coords = bracket_coords[start_i:end_i]

        sep_token = bracket_symbols[end_i]
        assert bracket_coords[end_i] is None

        token = sep_token.rstrip("<sep>").lstrip("<scn>")
        tokens = token.split("<smt>")
        SCN = tokens[0]
        SMT = tokens[1] if len(tokens) > 1 else None

        if SMT:
            sg = Chem.CreateMolSubstanceGroup(mol, type="SRU")
            sg.SetProp("CONNECT", SCN)
            sg.SetProp("LABEL", SMT)
        else:
            sg = Chem.CreateMolSubstanceGroup(mol, type="GEN")

        # Parse bracket line segments and register them on the SGroup.
        # Apply the same y-flip as the atom conformer: (x, 1-y).
        brackets = []   # list of ((bx, by), (kx, ky)) in conformer space
        for j in range(0, len(symbols) - 1, 2):
            assert symbols[j] == "<bra>"
            assert symbols[j+1] == "<ket>"
            assert len(coords[j]) == 2
            assert len(coords[j+1]) == 2

            bra = Point3D(coords[j][0], 1 - coords[j][1], 0.0)
            ket = Point3D(coords[j+1][0], 1 - coords[j+1][1], 0.0)
            sg.AddBracket([bra, ket, Point3D(0.0, 0.0, 0.0)])
            brackets.append(((bra.x, bra.y), (ket.x, ket.y)))

        if not atom_positions or len(brackets) < 2:
            continue

        # Interior reference: centroid of all bracket midpoints.
        # Used to identify which side of each bracket line is "inside".
        interior_ref = (
            np.mean([(b[0][0] + b[1][0]) / 2 for b in brackets]),
            np.mean([(b[0][1] + b[1][1]) / 2 for b in brackets]),
        )

        # Step 1: find bonds whose segment properly intersects a bracket segment.
        # Two line segments AB and CD intersect iff:
        #   - A and B are on opposite sides of line CD, AND
        #   - C and D are on opposite sides of line AB.
        crossing_bond_idxs = set()
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            p1, p2 = atom_positions[a1], atom_positions[a2]
            for (bx, by), (kx, ky) in brackets:
                # Condition 1: atom endpoints straddle the bracket's infinite line.
                dx, dy = kx - bx, ky - by
                nx, ny = -dy, dx
                s1 = nx * (p1[0] - bx) + ny * (p1[1] - by)
                s2 = nx * (p2[0] - bx) + ny * (p2[1] - by)
                if s1 * s2 >= 0:
                    continue
                # Condition 2: bracket endpoints straddle the bond's infinite line.
                bdx, bdy = p2[0] - p1[0], p2[1] - p1[1]
                bnx, bny = -bdy, bdx
                t1 = bnx * (bx - p1[0]) + bny * (by - p1[1])
                t2 = bnx * (kx - p1[0]) + bny * (ky - p1[1])
                if t1 * t2 < 0:
                    crossing_bond_idxs.add(bond.GetIdx())
                    break

        # Step 2: for each crossing bond, identify its "inside" endpoint — the atom on
        # the same side of the crossed bracket line as interior_ref.
        inside_seeds = set()
        for bond_idx in crossing_bond_idxs:
            bond = mol.GetBondWithIdx(bond_idx)
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            p1, p2 = atom_positions[a1], atom_positions[a2]
            for (bx, by), (kx, ky) in brackets:
                dx, dy = kx - bx, ky - by
                nx, ny = -dy, dx
                s1 = nx * (p1[0] - bx) + ny * (p1[1] - by)
                s2 = nx * (p2[0] - bx) + ny * (p2[1] - by)
                if s1 * s2 < 0:   # this is the bracket being crossed
                    s_ref = nx * (interior_ref[0] - bx) + ny * (interior_ref[1] - by)
                    inside_seeds.add(a1 if s_ref * s1 >= 0 else a2)
                    break

        # Step 3: cut the molecule at crossing bonds and enumerate connected fragments.
        # RemoveBond preserves atom indices, so GetMolFrags returns original atom indices.
        mol_cut = Chem.RWMol(mol)
        for bond_idx in crossing_bond_idxs:
            bond = mol.GetBondWithIdx(bond_idx)
            mol_cut.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        frags = Chem.GetMolFrags(mol_cut)   # tuple of tuples of original atom indices

        # Step 4: collect all fragments that contain at least one inside seed.
        # This handles atoms whose coordinates extend past the bracket lines: they are
        # included because their whole connected fragment is pulled in via the seed.
        inside_atoms = set()
        for frag in frags:
            if inside_seeds.intersection(frag):
                inside_atoms.update(frag)

        # Step 5: register inside atoms and crossing bonds on the SGroup.
        for atom_idx in sorted(inside_atoms):
            sg.AddAtomWithIdx(atom_idx)
        for bond_idx in sorted(crossing_bond_idxs):
            sg.AddBondWithIdx(bond_idx)

    return mol


def _convert_graph_to_molblock(
    symbols: List[str],
    coords: List[Tuple[float, float]],
    edges: List[List[int]],
    bracket_symbols: List[str],
    bracket_coords: List[Tuple[float, float]],
    image=None,
    debug: bool = False
):
    if image is not None:
        height, width, _ = image.shape
        ratio = width / height
        coords = [[x * ratio * 10, y * 10] for x, y in coords]
        bracket_coords = [
            [c[0] * ratio * 10, c[1] * 10] if c else None
            for c in bracket_coords
        ]

    mol = Chem.RWMol()
    n = len(symbols)
    assert len(coords) == n, f"len(coords): {len(coords)}, n: {n}"

    id_mappings = {}
    processed_coords = []
    dropped_ids = []
    for i in range(n):
        symbol = symbols[i]
        x, y = coords[i]
        if (x, y) in processed_coords:
            dropped_ids.append(i)
            continue
        else:
            processed_coords.append((x, y))

        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        # assert idx == i
        id_mappings[i] = idx

    for i in range(n):
        if i in dropped_ids:
            continue

        for j in range(i + 1, n):
            if j in dropped_ids:
                continue

            if edges[i][j] == 1:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(id_mappings[i], id_mappings[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(id_mappings[i], id_mappings[j]).SetBondDir(Chem.BondDir.BEGINDASH)
            elif edges[i][j] == 7:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.DOUBLE)
                # TODO TODO
            elif edges[i][j] == 8:
                mol.AddBond(id_mappings[i], id_mappings[j], Chem.BondType.SINGLE)
                # TODO TODO

    debug = True
    # [print(e) for e in edges]
    try:
        mol = _add_conformer(mol, processed_coords, debug)
        mol = _add_sgroup(
            mol,
            bracket_symbols=bracket_symbols,
            bracket_coords=bracket_coords
        )

        # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
        # TODO: make sure molblock has the abbreviation information
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_molblock = _postprocess_molblock(pred_molblock)

        # TURN OFF fn group expansion -- since we will convert molfile to BigSMILES *AFTER*
        # pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        success = True
    except Exception as e:
        if debug:
            log_rank_0(traceback.format_exc())
        pred_molblock = ''
        success = False

    # if debug:
    #     return pred_smiles, pred_molblock, mol, success

    return "<invalid>", pred_molblock, success


def convert_graph_to_molblock(
    node_symbols: List[List[str]],
    node_coords: List[List[Tuple[float, float]]],
    edges: List[List[List[int]]],
    bracket_symbols: List[List[str]],
    bracket_coords: List[List[Tuple[float, float]]],
    images=None,
    num_workers: int = 16
) -> Tuple[List, List, float]:
    if images is None:
        args_zip = zip(node_symbols, node_coords, edges, bracket_symbols, bracket_coords)
    else:
        args_zip = zip(node_symbols, node_coords, edges, bracket_symbols, bracket_coords, images)
    if num_workers <= 1:
        results = itertools.starmap(_convert_graph_to_molblock, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_molblock, args_zip, chunksize=128)

    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)

    return smiles_list, molblock_list, r_success


def _postprocess_molblock(molblock: str) -> str:
    lines = molblock.split("\n")
    newlines = []
    for line in lines:
        fields = line.strip().split()
        # reset double bond stereo
        if len(fields) == 4:
            if all(field.isdigit() for field in fields) and fields[2] == "2":
                newline = line[:-1] + "0"
            else:
                newline = line
        else:
            newline = line
        newlines.append(newline)

    processed_molblock = "\n".join(newlines)

    return processed_molblock
