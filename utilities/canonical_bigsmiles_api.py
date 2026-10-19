"""Client for the bigsmiles and canonicalization services.

Extracted from get_all_canonical_bigsmiles.py so that evaluate.py can score
predictions with the same client that produced the ground truth, rather than a
second implementation that would have to rediscover the same server quirks:

  * bigsmiles-server answers unparseable input with 200 and an empty "data",
    not an error, so an empty string means failure.
  * canonicalization-server *always* answers 200 and reports failure in the
    body, so dispatch on "status"; and a non-empty "error" does not mean
    failure -- an already-canonical input comes back SUCCESS with
    "Canonical BigSMILES is the same as the original, possible error!".
  * canonicalization-server is not concurrency-safe: it uses the BigSMILES
    string as a directory name and removes it afterwards, so two concurrent
    identical inputs corrupt each other. Call it sequentially.
"""
import csv
import itertools
import os
import re
import requests
import sys
import time
import traceback as tb
from typing import Optional
from rdkit import Chem, rdBase
from tqdm import tqdm


FAILED_BIGSMILES = "failed to obtain BigSMILES"

# Canonicalization outcomes, recorded per row so that a genuine failure is
# never confused with a molecule that was already canonical -- both of which
# leave the canonical column equal to its input.
STATUS_SUCCESS = "SUCCESS"                          # canonicalized, changed
STATUS_NOOP = "NOOP"                                # already canonical
STATUS_NO_STOCHASTIC_OBJECT = "NO_STOCHASTIC_OBJECT"  # plain SMILES, via RDKit
STATUS_FAIL = "FAIL"                                # canonicalization failed
STATUS_BIGSMILES_FAILED = "BIGSMILES_FAILED"        # never got a BigSMILES

# A canonical BigSMILES is usable as ground truth in these states only.
VALID_STATUSES = frozenset(
    {STATUS_SUCCESS, STATUS_NOOP, STATUS_NO_STOCHASTIC_OBJECT}
)

FIELDNAMES = ["path", "bigsmiles", "canonical_bigsmiles", "status"]

# A small polymer that both services handle, used to tell "the service is down"
# apart from "this particular input brought it down".
PROBE_BIGSMILES = "CCO{[>][<]CCO[>][<]}CCO"

# The element symbol every wildcard/R-group atom is rewritten to before a
# molblock is sent to bigsmiles-server. See normalize_wildcard_atoms.
#
# Yttrium, of all things, because the constraint is brutally narrow:
#
#   * The canonicalization server parses BigSMILES with RDKit, so the atom has
#     to be a *real element*. "R", "A", "E", "Q" and "Z" are MDL query-atom
#     codes with no SMILES meaning -- [R] is a ring-count primitive, valid only
#     in SMARTS -- and RDKit rejects all five. Measured on the corpus: an [R]
#     in a BigSMILES failed canonicalization 95 times out of 95.
#   * It must not be "*". In this corpus a "*" atom is an attachment point and
#     comes back as a bonding descriptor ([$]/[<]/[>]), so using it for a
#     substituent stub would rewrite the polymer topology.
#   * Y has no default valence in RDKit (GetDefaultValence(39) == -1), so it
#     adds no implicit hydrogens and perturbs nothing. At/Xe/Rn parse but carry
#     real valences and gave asymmetric results when probed.
#   * 3 ground-truth molfiles already use Y by hand as exactly this kind of
#     placeholder (e.g. C#C[Y]C#C), and one of them canonicalizes today.
WILDCARD_SYMBOL = "Y"

# Element symbols in a V2000 atom block that already mean "wildcard".
WILDCARD_INPUT_SYMBOLS = frozenset({"R", "R#"})

# ChemDraw writes a bare "X" (a halogen, a ring heteroatom or a linker,
# depending on the figure) as that element symbol, with no alias record, while
# MolScribe writes the same atom as an aliased R and so already gets the
# wildcard. Treating X as a wildcard symbol makes the two sides agree; as "X"
# it never canonicalizes (all 5 such ground truths failed).
BARE_X_SYMBOL = "X"

# Deuterium as drawn by ChemDraw: element "D", which the converter writes as
# [D] and RDKit rejects. Written as [2H] it parses, but the canonicalizer drops
# isotope labels, so a deuterated block collapses into its protiated twin.
# Astatine stands in for it on the way through the canonicalizer instead: a
# real, monovalent element no molfile or prediction in the corpus contains.
# The stored BigSMILES and canonical carry [2H]; [At] exists only in transit.
DEUTERIUM_SYMBOL = "D"
DEUTERIUM_STAND_IN = "At"
_STAND_IN_ATOM = f"[{DEUTERIUM_STAND_IN}]"
_DEUTERIUM_ATOM = "[2H]"

# A polymer attachment point, which must stay one on both sides. It is NOT a
# wildcard: bigsmiles-server turns it into a bonding descriptor, and turning
# it into a substituent stub would rewrite the topology.
ATTACHMENT_SYMBOL = "*"

# Columns 31-33 of a V2000 atom line hold the element symbol, left-justified.
_SYMBOL_START = 31
_SYMBOL_END = 34


class ServerUnavailableError(RuntimeError):
    """Raised when a service stops answering and does not come back."""


def normalize_wildcard_atoms(
    molblock: str,
    symbol: str = WILDCARD_SYMBOL,
    input_symbols: frozenset = WILDCARD_INPUT_SYMBOLS | {BARE_X_SYMBOL}
) -> str:
    """Give every wildcard/R-group atom the same element symbol.

    The two sides of the benchmark draw an abbreviation differently and
    bigsmiles-server keys off the element symbol while *ignoring* the alias
    record, so the same molecule used to convert two different ways:

        ground truth (ChemDraw)  carbon + "A" alias "CN"  ->  ...C...
        prediction (MolScribe)   element R + same alias   ->  ...[R]...

    and [R] never canonicalizes. Rewriting both to one real element makes the
    conversion a function of the structure instead of the exporter.

    An atom is a wildcard if it carries an alias record or its symbol is one
    of input_symbols -- except an atom whose alias text is "*", which is a polymer
    attachment point and is normalized back to "*" rather than to the
    wildcard. Attachment points become bonding descriptors, not substituent
    stubs, and the two exporters disagree about them too: the ground truth
    writes element "*", while a prediction arrives as element "R" aliased
    "*" because RDKit writes every dummy atom as "R".

    Returns the molblock unchanged when there is no V2000 counts line, so the
    empty-string probe payload still round-trips.
    """
    lines = molblock.split("\n")

    start = num_atoms = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            start = i + 1
            break

    if start is None:
        return molblock

    # "A  <idx>" on one line, the alias text on the next. Indices are 1-based.
    aliases = {}
    for i, line in enumerate(lines):
        head = line.split()
        if len(head) == 2 and head[0] == "A" and head[1].isdigit():
            text = lines[i + 1].strip() if i + 1 < len(lines) else ""
            aliases[int(head[1])] = text

    for offset in range(num_atoms):
        i = start + offset
        if i >= len(lines):
            break

        current = lines[i][_SYMBOL_START:_SYMBOL_END].strip()
        alias = aliases.get(offset + 1)

        if alias == ATTACHMENT_SYMBOL:
            # MolScribe builds an attachment point as a dummy atom aliased
            # "*", and RDKit writes any dummy atom's element as "R" -- so a
            # predicted attachment point arrives looking exactly like an
            # R-group and used to be rewritten to the wildcard, while the
            # ground truth's plain "*" was left alone. Putting "*" back in
            # the element column is what makes the two sides agree.
            replacement = ATTACHMENT_SYMBOL
        elif alias is not None or current in input_symbols:
            replacement = symbol
        else:
            continue

        lines[i] = (
            f"{lines[i][:_SYMBOL_START]}{replacement:<3}"
            f"{lines[i][_SYMBOL_END:]}"
        )

    return "\n".join(lines)


def substitute_deuterium(molblock: str) -> str:
    """Rewrite every element-D atom to the stand-in (see DEUTERIUM_STAND_IN).

    Returns the molblock unchanged when there is no V2000 counts line.
    """
    lines = molblock.split("\n")

    start = num_atoms = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            start = i + 1
            break

    if start is None:
        return molblock

    for i in range(start, min(start + num_atoms, len(lines))):
        if lines[i][_SYMBOL_START:_SYMBOL_END].strip() == DEUTERIUM_SYMBOL:
            lines[i] = (
                f"{lines[i][:_SYMBOL_START]}{DEUTERIUM_STAND_IN:<3}"
                f"{lines[i][_SYMBOL_END:]}"
            )

    return "\n".join(lines)


def fill_missing_crossing_bonds(molblock: str) -> str:
    """Add an "M  SBL" line to every S-group that has atoms but none.

    bigsmiles-server places bonding descriptors on an S-group's crossing
    bonds, and its V2000 reader takes those only from SBL -- with no SBL the
    list is empty and the conversion fails. Its V3000 reader, given no XBONDS,
    instead derives them as the bonds with exactly one end in the S-group;
    this does the same for V2000, so the two formats convert alike.

    It matters for the 12 MT ground truths drawn as Indigo V3000 without
    XBONDS: written back out as V2000 they had no SBL, and all 12 failed.
    Predictions always carry SBL and pass through unchanged.

    Returns the molblock unchanged when there is no V2000 counts line.
    """
    lines = molblock.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            num_bonds = int(line[3:6])
            start = i + 1
            break

    if start is None:
        return molblock

    # Fixed columns, not split(): two 3-wide atom indices run together once
    # one reaches 100 (" 99100").
    bonds = []
    for line in lines[start + num_atoms:start + num_atoms + num_bonds]:
        bonds.append((int(line[0:3]), int(line[3:6])))

    # "M  SAL sssn15 aaa ...": S-group index, count, then 4-wide atom indices.
    sgroup_atoms = {}
    has_sbl = set()
    end = None
    for i, line in enumerate(lines):
        if line.startswith("M  SAL"):
            sgroup = int(line[6:10])
            count = int(line[10:13])
            atoms = [int(line[13 + 4 * k:17 + 4 * k]) for k in range(count)]
            sgroup_atoms.setdefault(sgroup, set()).update(atoms)
        elif line.startswith("M  SBL"):
            has_sbl.add(int(line[6:10]))
        elif line.startswith("M  END"):
            end = i
            break

    if end is None:
        return molblock

    added = []
    for sgroup, atoms in sorted(sgroup_atoms.items()):
        if sgroup in has_sbl:
            continue

        crossing = [
            index for index, (a, b) in enumerate(bonds, start=1)
            if (a in atoms) != (b in atoms)
        ]
        # At most 15 entries per line. The server keeps only the last SBL line
        # of an S-group, so more than 15 crossing bonds would still lose some.
        for k in range(0, len(crossing), 15):
            chunk = crossing[k:k + 15]
            added.append(
                f"M  SBL{sgroup:>4}{len(chunk):>3}"
                + "".join(f"{index:>4}" for index in chunk)
            )

    if not added:
        return molblock

    return "\n".join(lines[:end] + added + lines[end:])


def stereogenic_double_bonds(molblock: str) -> Optional[set]:
    """The double bonds RDKit reads as E/Z from the molblock's 2D coordinates.

    Returned as sorted pairs of 1-based atom indices, or None when RDKit
    cannot read the molblock. Double bonds in rings smaller than 8, with two
    identical substituents on one end, or in a cumulated chain are not
    stereogenic and are left out.
    """
    # RDKit warns about every ambiguous stereocentre it meets; these are
    # ordinary in drawings and say nothing about the double bonds.
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        if mol is None:
            return None

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
            )
            Chem.DetectBondStereochemistry(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            return None

    return {
        tuple(sorted((b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1)))
        for b in mol.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    }


def clear_ring_double_bond_stereo(molblock: str) -> str:
    """Mark non-stereogenic ring double bonds as "either" (bond stereo 3).

    bigsmiles-server writes an E/Z mark on every double bond whose stereo
    field is 0 ("use the coordinates"), ring bonds included. On a benzene drawn
    in Kekule form, or in a fused or macrocyclic ring system, those marks
    describe no real stereo and stop the canonicalizer from parsing the
    result. Stereo 3 makes the server leave them out.

    A ring double bond that does carry E/Z from its coordinates (in a ring of
    8 or more, e.g. MT 202's 16-membered ring) keeps its stereo field.

    Returns the molblock unchanged when there is no V2000 counts line or RDKit
    cannot read it.
    """
    lines = molblock.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            num_bonds = int(line[3:6])
            start = i + 1
            break

    if start is None:
        return molblock

    stereogenic = stereogenic_double_bonds(molblock)
    if stereogenic is None:
        return molblock

    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        Chem.FastFindRings(mol)

    changed = False
    for i in range(start + num_atoms, start + num_atoms + num_bonds):
        line = lines[i]
        a, b = int(line[0:3]), int(line[3:6])
        order, stereo = int(line[6:9]), int(line[9:12] or 0)
        if order != 2 or stereo != 0:
            continue
        if tuple(sorted((a, b))) in stereogenic:
            continue
        bond = mol.GetBondBetweenAtoms(a - 1, b - 1)
        if bond is None or not bond.IsInRing():
            continue
        lines[i] = f"{line[:9]}  3{line[12:]}"
        changed = True

    return "\n".join(lines) if changed else molblock


def label_atoms_with_index(molblock: str) -> str:
    """Set every atom's isotope to its own 1-based index.

    bigsmiles-server writes an isotope into the atom's bracket ([12C]), so
    converting this copy tells which molfile atom each atom of the BigSMILES
    came from. The labels do not change the order the server writes atoms in
    (checked on all 1734 convertible ground truths). Attachment points ("*")
    stay unlabelled: a labelled "*" stops being read as an empty terminal.
    """
    lines = molblock.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            start = i + 1
            break

    if start is None:
        return molblock

    atoms = [
        a for a in range(1, num_atoms + 1)
        if lines[start + a - 1][_SYMBOL_START:_SYMBOL_END].strip()
        != ATTACHMENT_SYMBOL
    ]
    lines = [line for line in lines if not line.startswith("M  ISO")]
    end = next(
        (i for i, line in enumerate(lines) if line.startswith("M  END")), None
    )
    if end is None:
        return molblock

    # At most 8 (atom, isotope) pairs per line.
    added = []
    for k in range(0, len(atoms), 8):
        chunk = atoms[k:k + 8]
        added.append(
            f"M  ISO{len(chunk):>3}"
            + "".join(f"{a:>4}{a:>4}" for a in chunk)
        )

    return "\n".join(lines[:end] + added + lines[end:])


# One SMILES token: a bracket atom or descriptor, a two-letter organic atom,
# or any single character.
_TOKEN = re.compile(r"\[[^\]]*\]|Br|Cl|.")

# An E/Z mark the converter writes where SMILES does not allow one: before a
# branch, which RDKit rejects, or before a bond symbol.
_MARK_BEFORE_BRANCH = re.compile(r"[/\\]\(")
_MARK_BEFORE_BOND = re.compile(r"[/\\](?=[=#])")

# Placements tried for a mark written before "(": past the branch(es), or
# into the first branch. The molfile geometry decides between them.
AFTER_BRANCH = "after"
INTO_BRANCH = "into"

# Above this many misplaced marks, the 2^n placements are not all tried.
_MAX_MISPLACED_MARKS = 12


def _skeleton(bigsmiles: str) -> list:
    """The token sequence with every atom reduced to a placeholder."""
    return [
        "atom" if (
            (t.startswith("[") and re.match(r"\[\d*[A-Za-z*]", t))
            or re.fullmatch(r"Br|Cl|[BCNOSPFIbcnosp*]", t)
        ) else t
        for t in _TOKEN.findall(bigsmiles)
    ]


def place_direction_marks(bigsmiles: str, placements: list) -> str:
    """Rewrite each mark written before "(" as placements says, in order.

    A mark before a bond symbol ("/=") is dropped: the double bond it sits
    next to carries its own marks on the atoms around it.
    """
    bigsmiles = _MARK_BEFORE_BOND.sub("", bigsmiles)

    out, i, k = [], 0, 0
    while i < len(bigsmiles):
        c = bigsmiles[i]
        if c in "/\\" and bigsmiles[i + 1:i + 2] == "(":
            if placements[k] == INTO_BRANCH:
                out.append("(" + c)
                i += 2
            else:
                # Skip every branch that follows, then write the mark.
                j = i + 1
                while j < len(bigsmiles) and bigsmiles[j] == "(":
                    depth = 0
                    while True:
                        depth += bigsmiles[j] == "("
                        depth -= bigsmiles[j] == ")"
                        j += 1
                        if depth == 0:
                            break
                out.append(bigsmiles[i + 1:j] + c)
                i = j
            k += 1
            continue
        out.append(c)
        i += 1

    return "".join(out)


def _mapped_smiles(labeled: str) -> str:
    """A labelled BigSMILES as one SMILES whose atom map numbers are indices.

    Stochastic-object terminals are dropped, inner bonding descriptors become
    unmapped dummy atoms, and repeat-unit separators become ".", so RDKit can
    read the marks without needing the polymer structure.
    """
    s = re.sub(r"\[(\d+)([^\]]*)\]", r"[\2:\1]", labeled)
    s = re.sub(r"\{\[[^\]]*\]", "", s)
    s = re.sub(r"\[[^\]]*\]\}", "", s)
    s = re.sub(r"\[[<>$][^\]:]*\]", "[*]", s)
    return s.replace(",", ".").replace(";", ".")


def _side(p, a, b) -> float:
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)


def _mark_disagreements(
    labeled: str, conformer, stereogenic: set
) -> Optional[int]:
    """How many stereogenic double bonds the marks make cis/trans wrongly.

    Each bond's cis/trans as the marks give it is compared with the 2D
    coordinates. Returns None if RDKit cannot read the marks at all.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(_mapped_smiles(labeled), sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.SetBondStereoFromDirections(mol)

    bad = 0
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOCIS,
                          Chem.BondStereo.STEREOTRANS):
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not a.GetAtomMapNum() or not b.GetAtomMapNum():
            continue
        if tuple(sorted((a.GetAtomMapNum(), b.GetAtomMapNum()))) \
                not in stereogenic:
            continue

        # Measure against a mapped neighbour on each end. If it is not the
        # neighbour RDKit recorded the stereo against, cis and trans swap.
        cis = stereo == Chem.BondStereo.STEREOCIS
        ends = []
        for end, other, ref in zip((a, b), (b, a), bond.GetStereoAtoms()):
            nbrs = [x for x in end.GetNeighbors()
                    if x.GetIdx() != other.GetIdx() and x.GetAtomMapNum()]
            if not nbrs:
                break
            if nbrs[0].GetIdx() != ref:
                cis = not cis
            ends.append(nbrs[0])
        if len(ends) < 2:
            continue

        pos = lambda atom: conformer.GetAtomPosition(  # noqa: E731
            atom.GetAtomMapNum() - 1
        )
        pa, pb = pos(a), pos(b)
        geo_cis = _side(pos(ends[0]), pa, pb) * _side(pos(ends[1]), pa, pb) > 0
        bad += geo_cis != cis

    return bad


def repair_direction_marks(
    bigsmiles: str, labeled: str, molblock: str
) -> Optional[str]:
    """Put the converter's misplaced E/Z marks where the drawing says.

    bigsmiles-server sometimes writes a mark before a branch ("C=C/(C)C"),
    which no SMILES parser accepts. Whether it belongs past the branch
    ("C=C(C)/C") or inside it ("C=C(/C)C") differs between molecules -- no
    rule on the string alone is right for all of them -- so every placement is
    tried on the atom-labelled copy and checked against the molfile's 2D
    coordinates.

    Returns the repaired BigSMILES, or None when no placement agrees with the
    drawing on every stereogenic double bond (or the labelled copy cannot be
    lined up with the plain one). The caller then keeps the original string,
    which fails canonicalization: an honest failure rather than wrong stereo.
    """
    count = len(_MARK_BEFORE_BRANCH.findall(bigsmiles))
    if count > _MAX_MISPLACED_MARKS:
        return None
    if _skeleton(bigsmiles) != _skeleton(labeled):
        return None

    stereogenic = stereogenic_double_bonds(molblock)
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if stereogenic is None or mol is None:
        return None
    conformer = mol.GetConformer()

    # AFTER_BRANCH is right far more often, so it goes first and wins ties.
    for placements in itertools.product((AFTER_BRANCH, INTO_BRANCH),
                                        repeat=count):
        placed = place_direction_marks(labeled, list(placements))
        if _mark_disagreements(placed, conformer, stereogenic) == 0:
            return place_direction_marks(bigsmiles, list(placements))

    return None


# Linux refuses a file name longer than this many bytes. See
# split_long_name_segments.
_MAX_NAME_BYTES = 255

# Atoms that, written bare and back to back, are joined by plain single bonds.
_CHAIN_ATOMS = frozenset({"C", "N", "O", "S"})


def split_long_name_segments(bigsmiles: str) -> Optional[str]:
    """Break every '/'-free stretch longer than 255 bytes with a harmless '/'.

    canonicalization-server makes a scratch folder named after the BigSMILES
    ("Output/<bigsmiles>") for its graph drawings, so every E/Z mark is a
    directory separator and any stretch over 255 bytes without one fails with
    ENAMETOOLONG -- whatever the chemistry.

    The mark goes on the middle bond of four bare chain atoms in a row ("CCCC"):
    neither atom it joins has a double bond, so the mark carries no stereo and
    RDKit drops it. Probed on three ground truths, the canonical comes out the
    same wherever the mark goes and never contains it.

    Returns None when some stretch has no such bond to split at.
    """
    out = bigsmiles
    while True:
        tokens = _TOKEN.findall(out)
        offsets, pos = [], 0
        for t in tokens:
            offsets.append(pos)
            pos += len(t.encode())

        start, too_long = 0, None
        for t, at in zip(tokens + ["/"], offsets + [pos]):
            if t in "/\\":
                if at - start > _MAX_NAME_BYTES:
                    too_long = start
                    break
                start = at + 1
        if too_long is None:
            return out

        # The latest split point that still fits keeps the inserts to a few.
        best = None
        for j in range(1, len(tokens) - 2):
            cut = offsets[j + 1]
            if not too_long < cut <= too_long + _MAX_NAME_BYTES:
                continue
            if all(t in _CHAIN_ATOMS for t in tokens[j - 1:j + 3]):
                best = j + 1
        if best is None:
            return None
        out = "".join(tokens[:best]) + "/" + "".join(tokens[best:])


def _needs_name_split(bigsmiles: str) -> bool:
    return any(
        len(part.encode()) > _MAX_NAME_BYTES
        for part in re.split(r"[/\\]", bigsmiles)
    )


def bigsmiles_matches_molblock(labeled: str, molblock: str) -> bool:
    """Whether the converter's string keeps the molfile's graph intact.

    Read through the atom-labelled copy (see label_atoms_with_index): every
    atom written once with its element, no bond the molfile lacks, every bond
    that does not cross an S-group boundary present, both ends of every
    crossing bond carrying a bonding descriptor, and no "." in a string drawn
    as one connected molecule.

    It catches the strings the converter breaks while still returning them --
    bonds misread past atom 99, a bond between sibling S-groups dropped as a
    "." -- which would otherwise canonicalize to something else.
    """
    lines = molblock.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith("V2000"):
            num_atoms = int(line[:3])
            num_bonds = int(line[3:6])
            start = i + 1
            break
    if start is None:
        return False

    symbols = {
        a: lines[start + a - 1][_SYMBOL_START:_SYMBOL_END].strip()
        for a in range(1, num_atoms + 1)
    }
    bonds = {
        tuple(sorted((int(line[0:3]), int(line[3:6]))))
        for line in lines[start + num_atoms:start + num_atoms + num_bonds]
    }
    sgroups = {}
    for line in lines:
        if line.startswith("M  SAL"):
            count = int(line[10:13])
            sgroups.setdefault(int(line[6:10]), set()).update(
                int(line[13 + 4 * k:17 + 4 * k]) for k in range(count)
            )

    with rdBase.BlockLogs():
        drawn = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if drawn is None:
        return False
    if "." in labeled and len(Chem.GetMolFrags(drawn)) == 1:
        return False

    # Every bonding descriptor becomes an unmapped dummy atom, so a crossing
    # bond shows up as a dummy on each of its two atoms.
    s = re.sub(r"[/\\]", "", labeled)
    s = re.sub(r"\[(\d+)([^\]]*)\]", r"[\2:\1]", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\[[<>$][^\]:]*\]|\[\]", "[*]", s)
    s = s.replace(",", ".").replace(";", ".")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return False

    written = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            written.setdefault(atom.GetAtomMapNum(), []).append(atom)
    expected = {a for a, sym in symbols.items() if sym != ATTACHMENT_SYMBOL}
    if set(written) != expected or any(len(v) > 1 for v in written.values()):
        return False
    if any(v[0].GetSymbol().lower() != symbols[a].lower()
           for a, v in written.items()):
        return False

    present = {
        tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),
                      b.GetEndAtom().GetAtomMapNum())))
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomMapNum() and b.GetEndAtom().GetAtomMapNum()
    }
    if present - bonds:
        return False

    for pair in bonds - present:
        if any(symbols[a] == ATTACHMENT_SYMBOL for a in pair):
            continue
        crossing = any((pair[0] in g) != (pair[1] in g)
                       for g in sgroups.values())
        if not crossing:
            return False
        for a in pair:
            if not any(n.GetAtomicNum() == 0 and not n.GetAtomMapNum()
                       for n in written[a][0].GetNeighbors()):
                return False

    return True


class CanonicalBigSMILESAPI:
    def __init__(
        self,
        bigsmiles_url: str,
        bigsmiles_port: int,
        canonicalization_url: str,
        canonicalization_port: int,
        timeout: int,
        retries: int,
        retry_wait: int,
        payload_attempts: int,
        normalize_wildcards: bool = True,
        fill_crossing_bonds: bool = True,
        clear_ring_stereo: bool = True,
        repair_marks: bool = True,
        split_long_names: bool = True,
        x_wildcard: bool = True,
        deuterium_stand_in: bool = True,
        smiles_fallback: bool = True
    ) -> None:
        self.bigsmiles_uri = \
            f"{bigsmiles_url}:{bigsmiles_port}/api/molblock-to-bigsmiles"
        self.canonicalization_uri = \
            f"{canonicalization_url}:{canonicalization_port}" \
            f"/canonicalize-bigsmiles/"
        self.timeout = timeout
        self.retries = retries
        self.retry_wait = retry_wait
        self.payload_attempts = payload_attempts
        self.normalize_wildcards = normalize_wildcards
        self.fill_crossing_bonds = fill_crossing_bonds
        self.clear_ring_stereo = clear_ring_stereo
        self.repair_marks = repair_marks
        self.split_long_names = split_long_names
        self.x_wildcard = x_wildcard
        self.deuterium_stand_in = deuterium_stand_in
        self.smiles_fallback = smiles_fallback
        # BigSMILES checked against their molfile, and so safe to send to the
        # canonicalizer with an extra '/' (see split_long_name_segments).
        self.splittable = set()
        self.bigsmiles_session = requests.Session()
        self.canonicalization_session = requests.Session()
        self.killer_inputs = []
        self.probes = {
            "bigsmiles-server": (
                self.bigsmiles_session, self.bigsmiles_uri,
                {"molblock_string": ""}, "success"
            ),
            "canonicalization-server": (
                self.canonicalization_session, self.canonicalization_uri,
                {"bigsmiles": PROBE_BIGSMILES}, "status"
            )
        }

    def _wait_for(self, name: str) -> bool:
        """Poll a service with known-good input until it answers again."""
        session, uri, probe, key = self.probes[name]

        for attempt in range(1, self.retries + 1):
            time.sleep(self.retry_wait)
            try:
                resp = session.post(url=uri, json=probe, timeout=self.timeout)
                if resp.status_code == 200 and key in resp.json():
                    tqdm.write(f"{name} is back after {attempt} probe(s)")
                    return True
            except Exception:
                pass
            tqdm.write(f"{name} still down, probe {attempt}/{self.retries}")

        return False

    def _post(self, session, uri: str, payload: dict, name: str):
        """POST, distinguishing the ways a call can go wrong.

        Returns a Response, or None when this particular input could not be
        processed (too slow, or it crashes the service). Raises
        ServerUnavailableError when the service is gone for good, so the caller
        can stop rather than silently emit a column of fallback values.
        """
        for attempt in range(1, self.payload_attempts + 1):
            try:
                return session.post(
                    url=uri, json=payload, timeout=self.timeout
                )
            except requests.exceptions.Timeout:
                # This input is too slow; the service itself is still fine.
                return None
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.RequestException:
                tb.print_exc()
                return None

            # The connection dropped. Wait to see whether the service comes
            # back; if it never does, the run cannot produce real results.
            if not self._wait_for(name):
                raise ServerUnavailableError(
                    f"{name} at {uri} stopped responding and did not come "
                    f"back after {self.retries} probes"
                )

            # It came back, so the service is healthy and this input is what
            # brought it down. Do not keep feeding it the same payload.
            if attempt >= self.payload_attempts:
                tqdm.write(
                    f"{name} died on this input {attempt}x; "
                    f"recording it as a failure and moving on"
                )
                self.killer_inputs.append(payload)
                return None

            tqdm.write(f"{name} restarted; retrying this input once")

        return None

    def molblock_to_bigsmiles(self, molblock: str) -> str:
        """Convert a molblock into a BigSMILES via the bigsmiles-server.

        Returns FAILED_BIGSMILES when the server reports failure or hands back
        an empty string, which it does (with success=true) for input it cannot
        parse.

        This is the one place in the repo that sends a molblock anywhere, so
        it is also where wildcard normalization, crossing-bond filling and the
        E/Z repairs happen -- ground truth and predictions cannot diverge if
        they are prepared here.
        """
        if self.normalize_wildcards:
            molblock = normalize_wildcard_atoms(
                molblock,
                input_symbols=WILDCARD_INPUT_SYMBOLS | (
                    {BARE_X_SYMBOL} if self.x_wildcard else set()
                )
            )
        if self.deuterium_stand_in:
            molblock = substitute_deuterium(molblock)
        if self.fill_crossing_bonds:
            molblock = fill_missing_crossing_bonds(molblock)
        if self.clear_ring_stereo:
            molblock = clear_ring_double_bond_stereo(molblock)

        bigsmiles = self._convert(molblock)
        if bigsmiles == FAILED_BIGSMILES:
            return bigsmiles

        labeled = None
        if self.repair_marks and (_MARK_BEFORE_BRANCH.search(bigsmiles)
                                  or _MARK_BEFORE_BOND.search(bigsmiles)):
            labeled = self._convert(label_atoms_with_index(molblock))
            if labeled != FAILED_BIGSMILES:
                bigsmiles = repair_direction_marks(
                    bigsmiles, labeled, molblock
                ) or bigsmiles

        # Splitting a long name lets the string reach the canonicalizer, so
        # only a string that still matches its drawing earns it: a converter-
        # broken one would otherwise come back as a confident wrong canonical.
        if self.split_long_names and _needs_name_split(bigsmiles):
            if labeled is None:
                labeled = self._convert(label_atoms_with_index(molblock))
            if (labeled != FAILED_BIGSMILES
                    and bigsmiles_matches_molblock(labeled, molblock)):
                self.splittable.add(bigsmiles)

        # The stand-in never leaves the client: _sendable puts it back for the
        # canonicalizer only.
        if self.deuterium_stand_in and _STAND_IN_ATOM in bigsmiles:
            split = bigsmiles in self.splittable
            bigsmiles = bigsmiles.replace(_STAND_IN_ATOM, _DEUTERIUM_ATOM)
            if split:
                self.splittable.add(bigsmiles)

        return bigsmiles

    def _convert(self, molblock: str) -> str:
        """One molblock -> BigSMILES call, FAILED_BIGSMILES on any failure."""
        resp = self._post(
            self.bigsmiles_session,
            self.bigsmiles_uri,
            {"molblock_string": molblock},
            "bigsmiles-server"
        )
        if resp is None or resp.status_code != 200:
            return FAILED_BIGSMILES

        try:
            result = resp.json()
        except Exception:
            tb.print_exc()
            return FAILED_BIGSMILES

        if not result.get("success"):
            return FAILED_BIGSMILES

        return result.get("data", "") or FAILED_BIGSMILES

    def _sendable(self, bigsmiles: str) -> str:
        """The string to send the canonicalizer in place of bigsmiles.

        Deuterium goes as its stand-in. Otherwise unchanged, unless
        molblock_to_bigsmiles checked it against its drawing and it has a
        stretch too long for the server's scratch folder name.
        """
        sendable = bigsmiles
        if bigsmiles in self.splittable:
            sendable = split_long_name_segments(bigsmiles) or bigsmiles
        if self.deuterium_stand_in:
            sendable = sendable.replace(_DEUTERIUM_ATOM, _STAND_IN_ATOM)
        return sendable

    def canonicalize_with_status(self, bigsmiles: str) -> tuple[str, str]:
        """Canonicalize a BigSMILES, reporting how it went.

        Returns (canonical, status). On any failure the input is handed back
        unchanged, so the status is the only thing that distinguishes a failure
        from a molecule that was already canonical.
        """
        if bigsmiles == FAILED_BIGSMILES:
            return bigsmiles, STATUS_BIGSMILES_FAILED

        # With no stochastic object the string is plain SMILES, and the
        # canonicalizer fails on every such input (even "CCO"), so RDKit's
        # canonical SMILES is the canonical form.
        if self.smiles_fallback and "{" not in bigsmiles:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(bigsmiles)
            if mol is None:
                return bigsmiles, STATUS_FAIL
            return Chem.MolToSmiles(mol), STATUS_NO_STOCHASTIC_OBJECT

        resp = self._post(
            self.canonicalization_session,
            self.canonicalization_uri,
            {"bigsmiles": self._sendable(bigsmiles)},
            "canonicalization-server"
        )
        if resp is None or resp.status_code != 200:
            return bigsmiles, STATUS_FAIL

        try:
            result = resp.json()
        except Exception:
            tb.print_exc()
            return bigsmiles, STATUS_FAIL

        # The server always answers with HTTP 200 and reports failure in the
        # body; a non-empty "error" does not by itself mean failure.
        if result.get("status") != "SUCCESS":
            return bigsmiles, STATUS_FAIL

        canonical = result.get("result", {}).get("canonical_bigsmiles", "") \
            or bigsmiles
        if self.deuterium_stand_in:
            canonical = canonical.replace(_STAND_IN_ATOM, _DEUTERIUM_ATOM)

        if canonical != bigsmiles:
            return canonical, STATUS_SUCCESS

        # Unchanged but successful. Usually the molecule was already canonical;
        # but with no stochastic object (reached only with smiles_fallback
        # off) there is nothing to enumerate and the canonicalizer returns its
        # input, which is a different claim.
        if "{" not in bigsmiles:
            return canonical, STATUS_NO_STOCHASTIC_OBJECT

        return canonical, STATUS_NOOP

    def canonicalize(self, bigsmiles: str) -> str:
        """Canonicalize a BigSMILES, keeping only the string."""
        return self.canonicalize_with_status(bigsmiles)[0]

    def canonicalization_response(self, bigsmiles: str) -> Optional[dict]:
        """The canonicalization server's whole reply, error text included.

        canonicalize_with_status keeps only the verdict; this is for callers
        that need to know *why* an input failed. Returns None when there was
        no reply to read -- the input timed out, or crashed the service.
        """
        resp = self._post(
            self.canonicalization_session,
            self.canonicalization_uri,
            {"bigsmiles": self._sendable(bigsmiles)},
            "canonicalization-server"
        )
        if resp is None:
            return None

        try:
            return resp.json()
        except Exception:
            return {"status": STATUS_FAIL,
                    "error": f"HTTP {resp.status_code}: {resp.text}"}


def add_server_args(parser) -> None:
    """Add the service flags shared by every caller of this client."""
    parser.add_argument("--bigsmiles_url", type=str, default="http://0.0.0.0")
    parser.add_argument("--bigsmiles_port", type=int, default=3318)
    parser.add_argument("--canonicalization_url", type=str,
                        default="http://0.0.0.0")
    parser.add_argument("--canonicalization_port", type=int, default=3319)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=12,
                        help="probes before declaring a service unavailable")
    parser.add_argument("--retry_wait", type=int, default=10,
                        help="seconds between probes")
    parser.add_argument("--payload_attempts", type=int, default=2,
                        help="times to send one input before blaming it for a "
                             "service crash")
    parser.add_argument("--no_wildcard_normalization", action="store_true",
                        help="send molblocks verbatim instead of rewriting "
                             "wildcard/R-group atoms to a common element; "
                             "reproduces the pre-normalization numbers")
    parser.add_argument("--no_crossing_bond_fill", action="store_true",
                        help="send S-groups without SBL lines as they are "
                             "instead of deriving their crossing bonds; "
                             "reproduces the pre-fill numbers")
    parser.add_argument("--no_ring_stereo_clear", action="store_true",
                        help="send non-stereogenic ring double bonds with "
                             "their stereo field as drawn instead of marking "
                             "them 'either'; reproduces the pre-2026-10-16 "
                             "numbers")
    parser.add_argument("--no_mark_repair", action="store_true",
                        help="keep E/Z marks the converter writes before a "
                             "branch or bond symbol instead of placing them "
                             "by the molfile geometry; reproduces the "
                             "pre-2026-10-16 numbers")
    parser.add_argument("--no_long_name_split", action="store_true",
                        help="send BigSMILES with a '/'-free stretch over 255 "
                             "bytes as they are, which the canonicalizer's "
                             "scratch-folder name cannot hold; reproduces the "
                             "pre-2026-10-19 numbers")
    parser.add_argument("--no_x_wildcard", action="store_true",
                        help="leave a bare X atom as element X instead of "
                             "normalizing it to the wildcard; reproduces the "
                             "1572-usable ground-truth table")
    parser.add_argument("--no_deuterium_stand_in", action="store_true",
                        help="send deuterium as the converter writes it ([D]) "
                             "instead of through a stand-in atom; reproduces "
                             "the 1572-usable ground-truth table")
    parser.add_argument("--no_smiles_fallback", action="store_true",
                        help="send BigSMILES without a stochastic object to "
                             "the canonicalizer, which fails on all of them, "
                             "instead of canonicalizing them with RDKit; "
                             "reproduces the 1572-usable ground-truth table")


def api_from_args(args) -> CanonicalBigSMILESAPI:
    """Build the client from the flags added by add_server_args."""
    return CanonicalBigSMILESAPI(
        bigsmiles_url=args.bigsmiles_url,
        bigsmiles_port=args.bigsmiles_port,
        canonicalization_url=args.canonicalization_url,
        canonicalization_port=args.canonicalization_port,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        payload_attempts=args.payload_attempts,
        normalize_wildcards=not getattr(
            args, "no_wildcard_normalization", False
        ),
        fill_crossing_bonds=not getattr(
            args, "no_crossing_bond_fill", False
        ),
        clear_ring_stereo=not getattr(
            args, "no_ring_stereo_clear", False
        ),
        repair_marks=not getattr(args, "no_mark_repair", False),
        split_long_names=not getattr(args, "no_long_name_split", False),
        x_wildcard=not getattr(args, "no_x_wildcard", False),
        deuterium_stand_in=not getattr(args, "no_deuterium_stand_in", False),
        smiles_fallback=not getattr(args, "no_smiles_fallback", False)
    )


def check_servers(api: CanonicalBigSMILESAPI) -> None:
    """Fail fast with a clear remedy if either service is not up.

    This probes reachability only: it checks that each endpoint answers with a
    well-formed response, not that any particular conversion succeeds.
    """
    remedies = {
        "bigsmiles-server":
            "cd bigsmiles-server && make start-bigsmiles-service",
        "canonicalization-server":
            "cd canonicalization-server && make start-canonicalization-service"
    }

    for name, (session, uri, probe, key) in api.probes.items():
        try:
            resp = session.post(url=uri, json=probe, timeout=api.timeout)
            reachable = resp.status_code == 200 and key in resp.json()
        except Exception:
            reachable = False

        if not reachable:
            print(
                f"{name} is not reachable at {uri}\n"
                f"Start it with:\n"
                f"    {remedies[name]}"
            )
            sys.exit(1)

    print("Both servers are up.")


def sanitize(field: str) -> str:
    """Keep stray whitespace from a server response out of the TSV columns."""
    return field.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def load_existing_rows(output_file: str) -> dict:
    """Read an already-written TSV so an interrupted run can be continued."""
    if not os.path.exists(output_file):
        return {}

    with open(output_file, "r", newline="") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        return {row["path"]: row for row in reader}


def write_rows(rows: list, output_file: str, fieldnames: list = None) -> None:
    output_path = os.path.dirname(output_file)
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    with open(output_file, "w", newline="") as tsvfile:
        writer = csv.DictWriter(
            tsvfile,
            fieldnames=fieldnames or FIELDNAMES,
            delimiter="\t",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
