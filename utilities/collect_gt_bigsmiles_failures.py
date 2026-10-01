"""Collect the ground truths that fail BigSMILES conversion or canonicalization.

`canonical_bigsmiles.tsv` records only *that* a ground truth failed, as
BIGSMILES_FAILED (bigsmiles-server could not turn the molfile into a BigSMILES)
or FAIL (canonicalization-server could not canonicalize the BigSMILES). This
works out *why*, and copies each failure's image and molfile into a directory
named for the reason, so the failures can be browsed by cause:

    <out_root>/
        README.md           what each category means, with counts
        summary.tsv         one row per failure
        1_conversion/<category>/<name>.png
                                   /<name>.corrected.mol
                                   /<name>.txt      BigSMILES and error text
        2_canonicalization/<category>/...

where <name> is the molfile's path under the data root with "/" flattened to
"__", so every file traces back to exactly one source.

Categories are tried in order and the first that applies wins. Conversion
failures are classified from the molfile alone; bigsmiles-server only ever
answers "Failed to process data", so there is no error text worth asking for.
Canonicalization failures are classified from the molfile and the BigSMILES
first, and otherwise by re-sending the BigSMILES and reading the traceback.
Every structural rule below was checked against the whole corpus, and none of
them holds for a usable ground truth -- except overlapping S-groups, which 3
usable ground truths also have (see USABLE_WITH_OVERLAP_NOTE).

Needs canonicalization-server (not bigsmiles-server). Run from the repo root:

    python -m utilities.collect_gt_bigsmiles_failures
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter

from rdkit import Chem, RDLogger
from tqdm import tqdm

from utilities.canonical_bigsmiles_api import (
    STATUS_BIGSMILES_FAILED,
    STATUS_FAIL,
    ServerUnavailableError,
    add_server_args,
    api_from_args,
    load_existing_rows,
)
from utilities.paths import GT_SUFFIX, IMAGE_SUFFIX

RDLogger.DisableLog("rdApp.*")

STAGE_DIRS = {
    STATUS_BIGSMILES_FAILED: "1_conversion",
    STATUS_FAIL: "2_canonicalization",
}

# (directory, one-line meaning), in the order they are tried.
CONVERSION_CATEGORIES = [
    ("100_or_more_atoms",
     "The molecule has 100 or more atoms. Not a problem with the drawing: "
     "@cript-web/bigsmiles-toolkit splits V2000 bond lines on whitespace, but "
     "the format is fixed-width, so once an atom index reaches 100 the two "
     "3-character atom fields run together (\" 99100\") and every such bond "
     "points at a nonexistent atom. Every ground truth this large fails, and "
     "all of them convert once bond lines are read by column."),
    ("sru_without_crossing_bonds",
     "An SRU bracket lists no crossing bonds (no `M  SBL` line), so the "
     "converter cannot tell where the repeat unit connects. No convertible "
     "ground truth lacks it."),
    ("branched_repeat_unit",
     "An SRU bracket is crossed by more than two bonds: a branched or network "
     "repeat unit, which the converter cannot express."),
    ("overlapping_sgroups",
     "Two S-groups share atoms without one containing the other. Not always "
     "fatal: 3 usable ground truths have overlapping S-groups too (see the "
     "note at the end)."),
    ("other",
     "None of the above."),
]

CANONICALIZATION_CATEGORIES = [
    ("bigsmiles_too_long_for_a_filename",
     "Not a problem with the molecule: the canonicalizer creates a working "
     "directory named after the BigSMILES, and this one has a stretch "
     "between \"/\" characters longer than the 255-byte filename limit, so "
     "it fails before doing any chemistry."),
    ("branched_or_ladder_repeat_unit",
     "An SRU bracket is crossed by more than two bonds, or the BigSMILES "
     "carries ladder bonding descriptors such as [<[<]1]. The canonicalizer "
     "supports neither; this includes every ladder polymer in the corpus."),
    ("disconnected_fragments",
     "The BigSMILES has more than one component (\".\"), almost always a "
     "counter-ion or salt drawn as a separate fragment. The canonicalizer "
     "needs one connected graph. No usable ground truth has a \".\"."),
    ("nonstandard_atom_label",
     "The BigSMILES has an atom RDKit does not accept as an element: [X] (a "
     "placeholder -- wildcard normalization rewrites R and aliased atoms to "
     "Y, but not a bare X), [D] (deuterium written as D), or [K] (here an "
     "amino-acid abbreviation, read as potassium). No usable ground truth "
     "has any of them."),
    ("cis_trans_stereo",
     "The BigSMILES has E/Z double-bond stereo (\"/\" or \"\\\"), and the "
     "same string with those marks removed canonicalizes. The stereo alone "
     "is what the canonicalizer cannot handle."),
    ("no_reply",
     "The canonicalizer timed out or crashed on this input."),
    ("kekulization_error",
     "RDKit could not kekulize a fragment the canonicalizer built -- "
     "typically an aromatic ring whose N-H or charge does not survive being "
     "cut out of the repeat unit."),
    ("valence_error",
     "RDKit rejected a fragment the canonicalizer built for exceeding an "
     "atom's valence."),
    ("rdkit_cannot_parse_fragment",
     "RDKit returned no molecule for a fragment the canonicalizer built "
     "(the canonicalizer then fails on None). Includes borane-phosphine "
     "repeat units and charged pyridinium chains."),
    ("graph_error",
     "The canonicalizer's own graph construction failed (NetworkXNoPath, "
     "IndexError), e.g. a stochastic object inside a ring, or a star "
     "polymer whose first arm opens the string."),
    ("other",
     "None of the above; see the error text."),
]

CATEGORY_DESCRIPTIONS = {
    STATUS_BIGSMILES_FAILED: CONVERSION_CATEGORIES,
    STATUS_FAIL: CANONICALIZATION_CATEGORIES,
}

USABLE_WITH_OVERLAP_NOTE = (
    "**Note on overlapping S-groups.** Three ground truths with partially "
    "overlapping brackets *do* get a canonical BigSMILES, and count as "
    "usable: `OA generic/acsmacrolett/4c00400_f1_1`, "
    "`OA generic/macromolecules/5b02710_s1_1` and `5b02710_s1_3`. Their "
    "BigSMILES look malformed rather than right -- e.g. "
    "`{[>][<]NC(=O)[C@@]N([H])C[<]}` closes on `[<]` where `[>]` belongs -- "
    "so they may be silent failures worth a look."
)

SUMMARY_FIELDNAMES = [
    "stage",
    "category",
    "name",
    "family",
    "molfile",
    "bigsmiles",
    "error",
]

# A bonding descriptor that is itself nested, as in [<[<]1] -- only ever
# written for ladder connectivity.
LADDER_DESCRIPTOR = re.compile(r"\[[<>$]\[")
EZ_MARK = re.compile(r"[/\\]")
NONSTANDARD_ATOM = re.compile(r"\[(?:X|D|K)\]")

# NAME_MAX on Linux. canonicalization-server does os.makedirs("Output/" +
# bigsmiles), so each "/"-separated piece of the string is one path component.
FILENAME_MAX_BYTES = 255


def get_args():
    parser = argparse.ArgumentParser(
        description="Copy out the ground truths that fail BigSMILES "
                    "conversion or canonicalization, grouped by cause"
    )
    parser.add_argument("--gt_canonical_tsv", type=str,
                        default="data/PolymerLit/canonical_bigsmiles.tsv")
    parser.add_argument("--data_root", type=str, default="data/PolymerLit")
    parser.add_argument("--out_root", type=str,
                        default="bigsmiles_analysis/gt_failures")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove an existing --out_root first")
    add_server_args(parser)

    return parser.parse_args()


def family(rel_path: str) -> str:
    """MT / Olsen / OA generic / OA ladder, from the path under the data root."""
    parts = rel_path.split("/")
    name = parts[0].replace("PolymerLit-", "").replace("_processed", "")
    if name == "OA" and len(parts) > 1:
        name = f"OA {parts[1]}"

    return name


def flat_name(rel_path: str) -> str:
    return rel_path[:-len(GT_SUFFIX)].replace("/", "__")


def v2000_counts(molblock: str) -> tuple:
    """(atoms, bonds) from the counts line, read by column as the spec says."""
    line = molblock.split("\n")[3]

    return int(line[0:3]), int(line[3:6])


def has_merged_bond_fields(molblock: str) -> bool:
    """True if some bond line cannot be split on whitespace.

    Each bond line starts with two 3-character atom indices. When the second
    reaches 100 it fills its field and abuts the first -- " 99100" -- which a
    whitespace-splitting reader takes as one number.
    """
    num_atoms, num_bonds = v2000_counts(molblock)
    lines = molblock.split("\n")[4 + num_atoms:4 + num_atoms + num_bonds]

    return any(len(line) > 3 and line[3] != " " for line in lines)


def sgroup_profile(molblock: str) -> dict:
    """The S-group facts the conversion and canonicalization rules look at."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    profile = {"no_crossing_bonds": False, "branched": False,
               "overlapping": False}
    if mol is None:
        return profile

    atom_sets = []
    for sgroup in Chem.GetMolSubstanceGroups(mol):
        atoms = set(sgroup.GetAtoms())
        atom_sets.append(atoms)

        crossing = sum(
            1 for bond in mol.GetBonds()
            if (bond.GetBeginAtomIdx() in atoms) != (bond.GetEndAtomIdx() in atoms)
        )
        if crossing > 2:
            profile["branched"] = True
        if len(sgroup.GetBonds()) == 0:
            profile["no_crossing_bonds"] = True

    for i, a in enumerate(atom_sets):
        for b in atom_sets[i + 1:]:
            if a & b and not (a <= b or b <= a):
                profile["overlapping"] = True

    return profile


def classify_conversion(molblock: str) -> str:
    if has_merged_bond_fields(molblock):
        return "100_or_more_atoms"

    profile = sgroup_profile(molblock)
    if profile["no_crossing_bonds"]:
        return "sru_without_crossing_bonds"
    if profile["branched"]:
        return "branched_repeat_unit"
    if profile["overlapping"]:
        return "overlapping_sgroups"

    return "other"


def last_error_line(error: str) -> str:
    lines = [line.strip() for line in error.strip().split("\n") if line.strip()]

    return lines[-1] if lines else ""


def classify_error(error: str) -> str:
    """Map a canonicalization traceback to a category."""
    last = last_error_line(error)
    if "KekulizeException" in last:
        return "kekulization_error"
    if "AtomValenceException" in last:
        return "valence_error"
    if "'NoneType' object has no attribute" in last:
        return "rdkit_cannot_parse_fragment"
    if "NetworkXNoPath" in last or last.startswith("IndexError"):
        return "graph_error"

    return "other"


def classify_canonicalization(api, molblock: str, bigsmiles: str) -> tuple:
    """Return (category, error text) for one FAIL row."""
    if max(len(p.encode()) for p in bigsmiles.split("/")) > FILENAME_MAX_BYTES:
        return "bigsmiles_too_long_for_a_filename", ""

    profile = sgroup_profile(molblock)
    if profile["branched"] or LADDER_DESCRIPTOR.search(bigsmiles):
        return "branched_or_ladder_repeat_unit", ""
    if "." in bigsmiles:
        return "disconnected_fragments", ""
    if NONSTANDARD_ATOM.search(bigsmiles):
        return "nonstandard_atom_label", ""

    response = api.canonicalization_response(bigsmiles)
    if response is None:
        return "no_reply", "no reply: timed out or crashed the service"
    if response.get("status") == "SUCCESS":
        # Canonicalizes today although the TSV says FAIL; say so rather than
        # inventing a reason.
        return "other", "canonicalizes successfully when re-sent"
    error = response.get("error", "")

    if EZ_MARK.search(bigsmiles):
        stripped = api.canonicalization_response(EZ_MARK.sub("", bigsmiles))
        if stripped is not None and stripped.get("status") == "SUCCESS":
            return "cis_trans_stereo", error

    return classify_error(error), error


def write_info(path: str, row: dict, stage: str, category: str,
               error: str) -> None:
    lines = [
        f"molfile    : {row['path']}",
        f"stage      : {stage}",
        f"category   : {category}",
        f"status     : {row['status']}",
        "",
        f"bigsmiles  : {row['bigsmiles']}",
        "",
    ]
    if error:
        lines += ["error:", error.rstrip(), ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_readme(path: str, summary: list, counts: Counter, total: int) -> None:
    by_stage = Counter(row["stage"] for row in summary)
    lines = [
        "# Ground truths that fail BigSMILES conversion or canonicalization",
        "",
        f"{len(summary)} of the {total} PolymerLit ground truths have no "
        "usable canonical BigSMILES. Each failure's image (`.png`) and "
        "manually corrected molfile (`.corrected.mol`) are here, grouped by "
        "why it fails, with a `.txt` beside them holding the BigSMILES and, "
        "for canonicalization, the canonicalizer's traceback. "
        "`summary.tsv` lists them all.",
        "",
        "The pipeline has two stages, and a ground truth fails at the first "
        "one it cannot pass:",
        "",
        "1. **Conversion** -- molfile to BigSMILES, by bigsmiles-server "
        "(`@cript-web/bigsmiles-toolkit` 1.1.1).",
        "2. **Canonicalization** -- BigSMILES to canonical BigSMILES, by "
        "canonicalization-server.",
        "",
        "Within a stage, categories are tried in the order listed and the "
        "first that applies wins.",
        "",
    ]

    for status, stage_dir in STAGE_DIRS.items():
        title = "1. Conversion" if status == STATUS_BIGSMILES_FAILED \
            else "2. Canonicalization"
        lines += [f"## {title} failures ({by_stage[stage_dir]})", "",
                  "| Folder | Count | Meaning |", "|---|---:|---|"]
        for category, meaning in CATEGORY_DESCRIPTIONS[status]:
            n = counts[(stage_dir, category)]
            if n:
                lines.append(f"| `{stage_dir}/{category}/` | {n} | {meaning} |")
        lines.append("")

    families = sorted({row["family"] for row in summary})
    lines += ["## By source", "",
              "| Source | Conversion | Canonicalization |", "|---|---:|---:|"]
    for fam in families:
        conv = sum(1 for r in summary
                   if r["family"] == fam and r["stage"] == "1_conversion")
        canon = sum(1 for r in summary
                    if r["family"] == fam and r["stage"] == "2_canonicalization")
        lines.append(f"| {fam} | {conv} | {canon} |")
    lines += [
        "",
        USABLE_WITH_OVERLAP_NOTE,
        "",
        "Regenerate with `python -m utilities.collect_gt_bigsmiles_failures "
        "--overwrite` from the PolymerScribe repo root, with "
        "canonicalization-server running.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    if os.path.exists(args.out_root):
        if not args.overwrite:
            sys.exit(f"{args.out_root} already exists; "
                     f"pass --overwrite to replace it")
        shutil.rmtree(args.out_root)

    rows = load_existing_rows(args.gt_canonical_tsv)
    failed = [row for row in rows.values() if row["status"] in STAGE_DIRS]
    print(f"{len(failed)} of {len(rows)} ground truths failed")

    api = api_from_args(args)
    summary = []
    counts = Counter()

    for row in tqdm(sorted(failed, key=lambda r: r["path"])):
        molfile = os.path.join(args.data_root, row["path"])
        with open(molfile, "r") as f:
            molblock = f.read()

        stage = STAGE_DIRS[row["status"]]
        if row["status"] == STATUS_BIGSMILES_FAILED:
            category, error = classify_conversion(molblock), ""
        else:
            try:
                category, error = classify_canonicalization(
                    api, molblock, row["bigsmiles"]
                )
            except ServerUnavailableError as e:
                sys.exit(f"Aborting: {e}")

        name = flat_name(row["path"])
        case_dir = os.path.join(args.out_root, stage, category)
        os.makedirs(case_dir, exist_ok=True)
        shutil.copyfile(molfile, os.path.join(case_dir, f"{name}{GT_SUFFIX}"))
        image = f"{molfile[:-len(GT_SUFFIX)]}{IMAGE_SUFFIX}"
        if os.path.exists(image):
            shutil.copyfile(image, os.path.join(case_dir, f"{name}{IMAGE_SUFFIX}"))
        write_info(os.path.join(case_dir, f"{name}.txt"),
                   row, stage, category, error)

        counts[(stage, category)] += 1
        summary.append({
            "stage": stage,
            "category": category,
            "name": name,
            "family": family(row["path"]),
            "molfile": row["path"],
            "bigsmiles": row["bigsmiles"],
            "error": last_error_line(error),
        })

    summary.sort(key=lambda r: (r["stage"], r["category"], r["name"]))
    with open(os.path.join(args.out_root, "summary.tsv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    write_readme(os.path.join(args.out_root, "README.md"),
                 summary, counts, len(rows))

    print(f"\nWrote {len(summary)} failures to {args.out_root}")
    for (stage, category), n in sorted(counts.items()):
        print(f"  {stage}/{category}: {n}")


if __name__ == "__main__":
    args = get_args()
    main(args)
