"""How a ground-truth molecule is bucketed by size.

`evaluate.py` reports its metrics per size bucket, and
`create_filelists_for_all_splits.py` stratifies the test draw over the same
buckets. They have to agree: a molecule sampled into one bucket and reported in
another makes a stratified split come out uneven for no visible reason. So the
definition lives here, imported by both, rather than being spelled twice.

Not in `utilities/paths.py`, which is deliberately stdlib-only so `predict.py`
can import it on the GPU box without pulling anything heavy in.
"""
from typing import List, Optional

from rdkit import Chem

from utilities.paths import gt_molfile_for_image

BUCKET_WIDTH = 10

# Everything at or above this lands in one bucket. The tail is long and thin --
# a handful of molecules span 50 to well over 100 atoms -- so splitting it
# further would leave buckets too small to say anything about.
BUCKET_MAX = 50


def atom_count(mol) -> int:
    """Non-hydrogen atoms, *including* R-groups and attachment points.

    Not GetNumHeavyAtoms(), which skips dummy atoms: an R-group and a polymer
    attachment point are both dummies, so a molecule drawn with eight carbons
    and four R-groups counted as an eight-atom molecule. Those atoms are part
    of what the model has to read, and part of what makes the drawing complex,
    so they count.
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)


def bucket_of(count: int) -> int:
    """The bucket label for an atom count: 0, 10, ... BUCKET_MAX."""
    return min(count // BUCKET_WIDTH * BUCKET_WIDTH, BUCKET_MAX)


def bucket_of_molblock(molblock: str) -> Optional[int]:
    """Bucket a molecule from its molblock, or None if RDKit cannot read it."""
    mol = Chem.MolFromMolBlock(
        molblock, sanitize=False, removeHs=False, strictParsing=True
    )

    return None if mol is None else bucket_of(atom_count(mol))


def bucket_of_image(image_path: str) -> Optional[int]:
    """Bucket a molecule from its image path, via the molblock beside it."""
    with open(gt_molfile_for_image(image_path), "r") as f:
        return bucket_of_molblock(f.read())


def buckets() -> List[int]:
    """Every bucket label, in order."""
    return list(range(0, BUCKET_MAX + 1, BUCKET_WIDTH))
