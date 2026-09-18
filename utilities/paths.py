"""Shared path conventions for the PolymerScribe pipeline.

Every stage keys off the same mapping: an image `X.png` has its ground truth in
`X.corrected.mol` beside it, and its prediction at the mirrored location under a
prediction root. Before this module that mapping was spelled three different
ways (`preprocess.py`, `evaluate.py`, `create_filelists_for_all_splits.py`), and
the prediction half was encoded independently in `predict.py` and `evaluate.py`
-- which is why they disagreed for any subset not exactly two directories deep.

Deliberately stdlib-only: `predict.py` imports this on the GPU box and should
not pull in `requests`/`tqdm` to resolve a filename.
"""
import glob
import os
from typing import Iterator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_SUFFIX = ".png"
GT_SUFFIX = ".corrected.mol"
PRED_SUFFIX = ".predicted.mol"
PRED_FIGURE_SUFFIX = ".predicted.png"

DEFAULT_DATA_ROOT = "data/PolymerLit"


def normalize_image_path(path: str) -> str:
    """Return an image path as a clean repo-root-relative path.

    Accepts "./a/b.png", "a/b.png" and absolute paths inside the repo, and
    always returns "a/b.png". Filelists are written repo-root-relative, so this
    is the canonical form every stage agrees on.
    """
    if not path.endswith(IMAGE_SUFFIX):
        raise ValueError(f"Not a {IMAGE_SUFFIX} path: {path}")

    if os.path.isabs(path):
        path = os.path.relpath(path, REPO_ROOT)

    return os.path.normpath(path).replace(os.sep, "/")


def stem_path(image_path: str) -> str:
    """Drop exactly the trailing ".png".

    Not str.replace (which would also rewrite ".png" inside a directory name)
    and not str.rstrip (which strips any trailing "."/"p"/"n"/"g" character).
    """
    if not image_path.endswith(IMAGE_SUFFIX):
        raise ValueError(f"Not a {IMAGE_SUFFIX} path: {image_path}")

    return image_path[:-len(IMAGE_SUFFIX)]


def gt_molfile_for_image(image_path: str) -> str:
    """The manually corrected molblock sitting beside the image."""
    return f"{stem_path(normalize_image_path(image_path))}{GT_SUFFIX}"


def pred_molfile_for_image(image_path: str, pred_root: str) -> str:
    """Where a prediction for this image lives.

    The image's repo-root-relative path is mirrored under pred_root, so the
    mapping is a pure function of the path and needs no data-root argument that
    two separately invoked scripts would have to agree on.
    """
    rel = f"{stem_path(normalize_image_path(image_path))}{PRED_SUFFIX}"

    return os.path.join(pred_root, rel)


def pred_figure_for_image(image_path: str, pred_root: str) -> str:
    """Where the side-by-side comparison figure for this image lives."""
    rel = f"{stem_path(normalize_image_path(image_path))}{PRED_FIGURE_SUFFIX}"

    return os.path.join(pred_root, rel)


def gt_tsv_key(image_path: str, data_root: str = DEFAULT_DATA_ROOT) -> str:
    """The canonical_bigsmiles.tsv key for an image.

    The TSV is keyed by *.corrected.mol paths relative to the data root, while
    filelists hold *.png paths relative to the repo root.
    """
    molfile = gt_molfile_for_image(image_path)
    data_root = os.path.normpath(data_root).replace(os.sep, "/")

    return os.path.relpath(molfile, data_root).replace(os.sep, "/")


def iter_filelist(filelist_path: str) -> Iterator[str]:
    """Yield the images named by a filelist, repo-root-relative.

    A filelist has two kinds of line: one ending in "/" is a directory whose
    *.png files are all included, anything else is a single image. This mirrors
    what preprocess.py does, so every stage expands a filelist identically.
    """
    with open(filelist_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith("/"):
                for png_fn in sorted(glob.glob(os.path.join(line, "*.png"))):
                    yield normalize_image_path(png_fn)
            else:
                yield normalize_image_path(line)
