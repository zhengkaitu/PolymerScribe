"""Generate train/val/test filelists for the PolymerLit experiments.

Run from the repo root as a module:

    python -m utilities.create_filelists_for_all_splits

Writes straight into the per-experiment layout that preprocess.py and the
submit_*.sh scripts expect, so no renaming step is needed afterwards:

    experiments/<exp>_<split>_<count>/<exp>_<split>_<count>_{train,val,test}.filelist.txt

A filelist has two kinds of line: one ending in "/" is a directory whose *.png
files are all included, anything else is a single image. The synthetic sources
are emitted as directory lines, so their trailing slashes are load-bearing.

Val and test are drawn only from images with a valid canonical BigSMILES, so
that predictions can be scored on canonical BigSMILES. Everything else stays in
train, where it is still useful: every png/mol pair was manually corrected.
The ladder family is exempt -- essentially no ladder structure canonicalizes,
so filtering it would empty the ladder holdout entirely.

The test set is *stratified by molecule size* (see TEST_BUCKET_QUOTA). Drawn
uniformly at random it followed the corpus, which put only ~13 of 100 test
images above 40 atoms -- too few to resolve anything, and the sweep's whole
question is how accuracy scales with size. Stratifying trades precision in the
crowded small buckets, which had a surplus, for precision in the large ones,
which had almost none.

Only test is stratified. Val is drawn at random from what is left, because it
selects checkpoints rather than being reported per bucket, and balancing it
would consume scarce large molecules twice over.

Note that a stratified test set is no longer a simple random sample of the
corpus: large molecules are deliberately over-represented (~30% of test
against ~5.5% of the pool), so the unweighted overall metric reads a few
points below the corpus rate and is not comparable to a number from an
unstratified split. The per-bucket columns are what this split is for.
"""
import argparse
import csv
import glob
import os
import random

from collections import Counter

from utilities.binning import bucket_of_image, buckets
from utilities.canonical_bigsmiles_api import VALID_STATUSES
from utilities.paths import gt_tsv_key, normalize_image_path, stem_path


GENERIC_VAL = 95
LADDER_VAL = 5
LADDER_TEST = 5

# Test images per size bucket, over generic *and* ladder together. The ladder
# holdout is drawn first and counts against these, so the totals hold and the
# ladder images still appear in whichever buckets they fall in.
TEST_BUCKET_QUOTA = {0: 20, 10: 20, 20: 15, 30: 15, 40: 15, 50: 15}

TEST_TOTAL = sum(TEST_BUCKET_QUOTA.values())
GENERIC_TEST = TEST_TOTAL - LADDER_TEST


def verify_labels(sources: list[str]) -> None:
    """Check every image has its manually corrected molblock beside it."""
    for source in sources:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        missing = [
            fn for fn in fl
            if not os.path.exists(f"{stem_path(fn)}.corrected.mol")
        ]
        assert not missing, f"No .corrected.mol for: {missing[:5]}"
        print(f"All corrected for {source}. Number of images: {len(fl)}")


def load_valid_images(canonical_tsv: str, data_root: str) -> set[str]:
    """Image paths whose ground-truth canonical BigSMILES is usable.

    Keyed the same way the TSV is: by the *.corrected.mol path relative to the
    data root. Rows whose status says the canonicalization failed, or that
    never produced a BigSMILES at all, are not valid.
    """
    valid_keys = set()
    with open(canonical_tsv, "r", newline="") as tsvfile:
        for row in csv.DictReader(tsvfile, delimiter="\t"):
            if (row.get("status") or "").strip() in VALID_STATUSES:
                valid_keys.add(row["path"])

    print(f"{len(valid_keys)} rows in {canonical_tsv} have a valid canonical")

    return valid_keys


def collect_images(sources: list[str]) -> list[str]:
    images = []
    for source in sources:
        images.extend(sorted(glob.glob(os.path.join(source, "*.png"))))

    return [normalize_image_path(fn) for fn in images]


def split_generic(images: list[str], valid_keys: set[str], data_root: str,
                  ladder_test: list[str]):
    """Split the generic pool, drawing val/test from valid images only.

    Test is filled per size bucket to TEST_BUCKET_QUOTA, minus whatever the
    ladder holdout already contributes to each bucket. Val is then drawn at
    random from everything still unused.
    """
    valid = [
        fn for fn in images if gt_tsv_key(fn, data_root) in valid_keys
    ]
    invalid = [
        fn for fn in images if gt_tsv_key(fn, data_root) not in valid_keys
    ]
    print(
        f"Generic: {len(valid)} with a valid canonical, "
        f"{len(invalid)} without (all of which go to train)"
    )

    # The ladder holdout is already chosen, so it eats into the quota. Quotas
    # are far larger than LADDER_TEST, so no bucket can be overdrawn -- but
    # assert it rather than silently taking a negative slice.
    ladder_per_bucket = Counter(bucket_of_image(fn) for fn in ladder_test)
    remaining = {}
    for bucket in buckets():
        want = TEST_BUCKET_QUOTA[bucket] - ladder_per_bucket[bucket]
        assert want >= 0, (
            f"bucket {bucket}: ladder contributed "
            f"{ladder_per_bucket[bucket]} test images but the quota is only "
            f"{TEST_BUCKET_QUOTA[bucket]}"
        )
        remaining[bucket] = want

    by_bucket = {bucket: [] for bucket in buckets()}
    for fn in valid:
        by_bucket[bucket_of_image(fn)].append(fn)

    test = []
    for bucket in buckets():
        pool = by_bucket[bucket]
        want = remaining[bucket]
        assert len(pool) >= want, (
            f"bucket {bucket}: only {len(pool)} generic images have a valid "
            f"canonical, need {want} for test"
        )
        random.shuffle(pool)
        test.extend(pool[:want])
        by_bucket[bucket] = pool[want:]

    print(
        "Test buckets (generic + ladder): "
        + ", ".join(
            f"{bucket}: {remaining[bucket]}+{ladder_per_bucket[bucket]}"
            for bucket in buckets()
        )
    )

    # Whatever the quotas did not take, pooled back together for val/train.
    rest = [fn for bucket in buckets() for fn in by_bucket[bucket]]
    assert len(rest) >= GENERIC_VAL, (
        f"Only {len(rest)} valid generic images left after the test draw, "
        f"need {GENERIC_VAL} for val"
    )
    random.shuffle(rest)
    val = rest[:GENERIC_VAL]
    train = rest[GENERIC_VAL:] + invalid

    return train, val, test


def split_ladder(images: list[str]):
    """Split the ladder pool. No validity filter -- see the module docstring."""
    random.shuffle(images)
    val = images[:LADDER_VAL]
    test = images[LADDER_VAL:LADDER_VAL + LADDER_TEST]
    train = images[LADDER_VAL + LADDER_TEST:]

    return train, val, test


def get_filelist_fully_random(
    sources_generic: list[str],
    sources_ladder: list[str],
    valid_keys: set[str],
    data_root: str
):
    """Image-level random split: images from one document may straddle sets."""
    # Ladder first: it is drawn at random and unfiltered, and the generic test
    # draw fills the size buckets around whatever it happened to take.
    ladder_train, ladder_val, ladder_test = split_ladder(
        collect_images(sources_ladder)
    )
    generic_train, generic_val, generic_test = split_generic(
        collect_images(sources_generic), valid_keys, data_root, ladder_test
    )

    assert len(generic_val) == GENERIC_VAL
    assert len(generic_test) == GENERIC_TEST
    assert len(ladder_val) == LADDER_VAL
    assert len(ladder_test) == LADDER_TEST
    assert len(generic_test) + len(ladder_test) == TEST_TOTAL

    filelist_train_realistic = generic_train + ladder_train
    filelist_val = generic_val + ladder_val
    filelist_test = generic_test + ladder_test

    random.shuffle(filelist_train_realistic)

    return filelist_train_realistic, filelist_val, filelist_test


def write_filelist(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as of:
        for line in lines:
            of.write(f"{line}\n")


def create_splits(args, split: str, sources_synthetic, sources_generic,
                  sources_ladder, valid_keys) -> None:
    if split == "fully_random":
        filelist_train_realistic, filelist_val, filelist_test = \
            get_filelist_fully_random(
                sources_generic, sources_ladder, valid_keys, args.data_root
            )
    else:
        raise NotImplementedError(f"Split {split} not implemented")

    for realistic_count in args.counts:
        expt_id = f"{args.exp_no}_{split}_{realistic_count}"
        output_path = os.path.join(args.output_root, expt_id)

        # Synthetic data is always entirely in train, as directory lines.
        train_lines = list(sources_synthetic) + \
            sorted(filelist_train_realistic[:realistic_count])

        write_filelist(
            os.path.join(output_path, f"{expt_id}_train.filelist.txt"),
            train_lines
        )
        write_filelist(
            os.path.join(output_path, f"{expt_id}_val.filelist.txt"),
            sorted(filelist_val)
        )
        write_filelist(
            os.path.join(output_path, f"{expt_id}_test.filelist.txt"),
            sorted(filelist_test)
        )
        print(
            f"{expt_id}: train {len(train_lines)} lines "
            f"({len(sources_synthetic)} dirs + {realistic_count} images), "
            f"val {len(filelist_val)}, test {len(filelist_test)}"
        )


def main(args):
    random.seed(args.seed)

    olsen = os.path.join(args.data_root, "PolymerLit-Olsen_processed")
    oa = os.path.join(args.data_root, "PolymerLit-OA_processed")

    sources_synthetic = [
        f"{os.path.join(args.data_root, 'PolymerLit-MT_processed')}/",
        f"{os.path.join(olsen, 'bigsmiles_manuscript')}/",
        f"{os.path.join(olsen, 'bigsmiles_si')}/",
        f"{os.path.join(olsen, 'canonicalization_manuscript')}/",
        f"{os.path.join(olsen, 'canonicalization_si')}/",
        f"{os.path.join(olsen, 'non-covalent_manuscript')}/",
        f"{os.path.join(olsen, 'non-covalent_si')}/"
    ]

    sources_generic = [
        os.path.join(oa, "generic", journal)
        for journal in ["acspolymersau", "acsmacrolett", "macromolecules"]
    ]

    sources_ladder = [
        os.path.join(oa, "ladder", journal)
        for journal in [
            "acsmacrolett", "angewchemie", "chemengjournal", "chemicalscience",
            "digitaldiscovery", "faradaydiscussions", "macromolecules",
            "polymer"
        ]
    ]

    verify_labels(sources=sources_synthetic)
    verify_labels(sources=sources_generic)
    verify_labels(sources=sources_ladder)

    valid_keys = load_valid_images(args.canonical_tsv, args.data_root)

    create_splits(args, args.split, sources_synthetic, sources_generic,
                  sources_ladder, valid_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test filelists for PolymerLit"
    )
    parser.add_argument("--data_root", type=str, default="data/PolymerLit")
    parser.add_argument("--canonical_tsv", type=str,
                        default="data/PolymerLit/canonical_bigsmiles.tsv")
    parser.add_argument("--output_root", type=str, default="experiments")
    parser.add_argument("--exp_no", type=str, default="full_rerun",
                        help="experiment name prefix, e.g. full_rerun")
    parser.add_argument("--split", type=str, default="fully_random")
    parser.add_argument("--counts", type=int, nargs="+",
                        default=[0, 200, 400, 600, 800],
                        help="numbers of realistic training images to include")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
