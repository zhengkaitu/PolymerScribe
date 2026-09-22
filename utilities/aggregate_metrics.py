"""Aggregate every `evaluate.py` log of a sweep into one CSV.

`evaluate.py` prints its results one run at a time, as a per-molecule line per
test image followed by a per-bucket summary. Reading a training-size sweep off
six such logs by eye is how the last round of per-bucket analysis went wrong,
so this collects them into a single table:

    run, bucket, n, <every metric>

with one row per size bucket plus a `pooled` row -- the plain average over all
100 test samples, not an average of the six bucket averages, which would
silently reweight the unequal quotas.

The per-molecule lines carry the full metric dict (precision and recall as
well as F1) but no bucket; the summary lines carry the bucket but only the F1s,
rounded. So this re-derives each molecule's bucket from its ground-truth
molblock, with `utilities.binning` -- the same function `evaluate.py` and the
splitter use -- and then *checks* the result against the summary lines the log
already printed. A drift between the two means the bucket definition moved
since the run, which would quietly reshuffle every row here, so it aborts
rather than emitting a plausible-looking table.

Canonical match is the one metric with no per-molecule line, so it is read
back off the summary lines. It is a proportion of a known denominator, so the
4-decimal print recovers the underlying count exactly.

Hits no servers and reads no predictions -- only the logs and the ground-truth
molfiles.

Run as a module from the repo root:

    python -m utilities.aggregate_metrics --exp_no full_rerun
"""
import argparse
import ast
import csv
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

from utilities.binning import bucket_of_molblock, buckets

# The metrics every per-molecule line carries, in the order they are reported.
GEOMETRY_METRICS = [
    "exact_match",
    "atom_precision", "atom_recall", "atom_f1",
    "bond_precision", "bond_recall", "bond_f1",
    "sgroup_precision", "sgroup_recall", "sgroup_f1",
]

# Added later, so a log written before the split has neither. Kept separate
# from GEOMETRY_METRICS so an older log still aggregates, with these columns
# left empty rather than the run dropped.
NOLABEL_METRICS = ["sgroup_f1_nolabel", "exact_match_nolabel"]

POOLED = "pooled"

SAMPLE_RE = re.compile(
    r"molfile_gt: (?P<path>\S+), metrics: (?P<metrics>\{.*\})\s*$"
)
BUCKET_RE = re.compile(
    r"count: (?P<count>\d+), occurrences: (?P<n>\d+), "
    r"Exact matches:\s*(?P<exact>[\d.]+), "
    r"Atom F1:\s*(?P<atom_f1>[\d.]+), "
    r"Bond F1:\s*(?P<bond_f1>[\d.]+), "
    r"Sgroup F1:\s*(?P<sgroup_f1>[\d.]+)"
    r"(?:, Sgroup F1 nolabel:\s*(?P<sgroup_f1_nolabel>[\d.]+))?"
    r"(?:, Exact nolabel:\s*(?P<exact_match_nolabel>[\d.]+))?"
    r"(?:, Canon match:\s*(?P<canon>[\d.]+))?"
)
OVERALL_CANON_RE = re.compile(
    r"Canonical match \(all (?P<n>\d+) samples\):\s*(?P<value>[\d.]+)"
)
UNCANONICALIZABLE_RE = re.compile(
    r"of which (?P<n>\d+) have no usable ground-truth canonical"
)
ABORTED_RE = re.compile(r"Canonical matching (?:aborted|did not finish)")


def get_args():
    parser = argparse.ArgumentParser(
        description="Aggregate a sweep's evaluate logs into one metrics CSV"
    )
    parser.add_argument("--exp_no", type=str, default="full_rerun")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="defaults to logs/<exp_no>")
    parser.add_argument("--out", type=str, default=None,
                        help="defaults to predictions/<exp_no>.metrics.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing --out")

    return parser.parse_args()


def run_sort_key(run: str):
    """cold start first, then the warm starts by training-set size."""
    match = re.search(r"(\d+)$", run)

    return (1, int(match.group(1))) if match else (0, 0)


def read_lines(path: str) -> List[str]:
    """Universal newlines split tqdm's \\r-joined lines back apart."""
    with open(path, "r") as f:
        return f.read().splitlines()


def parse_log(path: str) -> dict:
    """One evaluate log -> its per-molecule samples and its own summary."""
    samples = []
    summary = OrderedDict()
    overall_canon = None
    overall_n = None
    uncanonicalizable = None
    aborted = False

    for line in read_lines(path):
        match = SAMPLE_RE.search(line)
        if match:
            samples.append((
                match.group("path"), ast.literal_eval(match.group("metrics"))
            ))
            continue

        match = BUCKET_RE.search(line)
        if match:
            canon = match.group("canon")
            row = {
                "n": int(match.group("n")),
                "exact_match": float(match.group("exact")),
                "atom_f1": float(match.group("atom_f1")),
                "bond_f1": float(match.group("bond_f1")),
                "sgroup_f1": float(match.group("sgroup_f1")),
                "canonical_match": None if canon is None else float(canon),
            }
            for key in NOLABEL_METRICS:
                value = match.group(key)
                if value is not None:
                    row[key] = float(value)
            summary[int(match.group("count"))] = row
            continue

        match = OVERALL_CANON_RE.search(line)
        if match:
            overall_n = int(match.group("n"))
            overall_canon = float(match.group("value"))
            continue

        match = UNCANONICALIZABLE_RE.search(line)
        if match:
            uncanonicalizable = int(match.group("n"))
            continue

        if ABORTED_RE.search(line):
            aborted = True

    return {
        "samples": samples,
        "summary": summary,
        "overall_canon": overall_canon,
        "overall_n": overall_n,
        "uncanonicalizable": uncanonicalizable,
        "aborted": aborted,
    }


def bucket_for(molfile_gt: str, cache: Dict[str, Optional[int]]) -> int:
    """The bucket `evaluate.py` would have put this molecule in."""
    if molfile_gt not in cache:
        with open(molfile_gt, "r") as f:
            cache[molfile_gt] = bucket_of_molblock(f.read())

    bucket = cache[molfile_gt]
    assert bucket is not None, f"RDKit could not read {molfile_gt}"

    return bucket


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def check_against_log(run: str, grouped: dict, summary: dict) -> None:
    """Abort unless the re-derived buckets reproduce the log's own summary.

    The log printed to 2 or 4 decimals, so the comparison is at that
    tolerance; anything coarser than a rounding difference means the bucket
    definition drifted and the rows below would be quietly wrong.
    """
    if set(grouped) != set(summary):
        sys.exit(
            f"{run}: re-derived buckets {sorted(grouped)} do not match the "
            f"log's {sorted(summary)}. utilities/binning.py has changed since "
            f"the run -- rerun evaluate.py rather than trusting this table."
        )

    for bucket, rows in sorted(grouped.items()):
        printed = summary[bucket]
        if len(rows) != printed["n"]:
            sys.exit(
                f"{run}: bucket {bucket} holds {len(rows)} samples here but "
                f"{printed['n']} in the log. The bucket definition drifted."
            )

        for key, places in [("exact_match", 2), ("atom_f1", 4),
                            ("bond_f1", 4), ("sgroup_f1", 4),
                            ("sgroup_f1_nolabel", 4),
                            ("exact_match_nolabel", 2)]:
            if key not in printed or key not in rows[0]:
                continue
            ours = round(mean([float(r[key]) for r in rows]), places)
            if abs(ours - printed[key]) > 10 ** -places:
                sys.exit(
                    f"{run}: bucket {bucket} {key} is {ours} here but "
                    f"{printed[key]} in the log."
                )


def canonical_counts(run: str, summary: dict, parsed: dict) -> Dict[int, int]:
    """Per-bucket canonical matches, recovered as counts from the printed rate.

    A rate is always k/n for a known n, so rounding the 4-decimal print back
    up to a count is exact -- and summing those counts has to land on the
    overall figure the log printed independently, which is the check.
    """
    counts = {}
    for bucket, printed in summary.items():
        if printed["canonical_match"] is None:
            return {}
        counts[bucket] = round(printed["canonical_match"] * printed["n"])

    if parsed["overall_canon"] is not None:
        total = sum(counts.values())
        expected = round(parsed["overall_canon"] * parsed["overall_n"])
        if total != expected:
            sys.exit(
                f"{run}: per-bucket canonical matches sum to {total} but the "
                f"log's overall figure implies {expected}."
            )

    return counts


def rows_for_run(run: str, parsed: dict, cache: dict) -> List[dict]:
    """One row per bucket, plus the pooled row, for a single run."""
    grouped = OrderedDict((bucket, []) for bucket in buckets())
    for molfile_gt, metrics in parsed["samples"]:
        grouped[bucket_for(molfile_gt, cache)].append(metrics)

    grouped = OrderedDict((b, rows) for b, rows in grouped.items() if rows)
    check_against_log(run, grouped, parsed["summary"])

    # A log from before the subscript split has neither field. Those columns
    # come out empty for that run rather than failing the whole table.
    keys = list(GEOMETRY_METRICS)
    if all(k in parsed["samples"][0][1] for k in NOLABEL_METRICS):
        keys += NOLABEL_METRICS
        for molfile_gt, m in parsed["samples"]:
            # exact_match_nolabel only forgives the subscript text, so it can
            # never be the stricter of the two. If it is, the second set of
            # S-group arrays in evaluate.py was scored off a different
            # assignment than the first.
            if float(m["exact_match_nolabel"]) < float(m["exact_match"]):
                sys.exit(
                    f"{run}: exact_match_nolabel is stricter than exact_match "
                    f"on {molfile_gt}. The relaxed metric is mis-wired."
                )

    canon = canonical_counts(run, parsed["summary"], parsed)
    rows = []

    for bucket, metrics in grouped.items():
        row = {"run": run, "bucket": bucket, "n": len(metrics)}
        for key in keys:
            row[key] = mean([float(m[key]) for m in metrics])
        row["canonical_match"] = (
            canon[bucket] / len(metrics) if canon else None
        )
        row["uncanonicalizable_gt"] = None
        row["canonical_ceiling"] = None
        rows.append(row)

    pooled = [m for metrics in grouped.values() for m in metrics]
    row = {"run": run, "bucket": POOLED, "n": len(pooled)}
    for key in keys:
        row[key] = mean([float(m[key]) for m in pooled])
    row["canonical_match"] = (
        sum(canon.values()) / len(pooled) if canon else None
    )
    # Reported only here: evaluate.py tallies the unusable ground truth over
    # the whole test set, never per bucket.
    row["uncanonicalizable_gt"] = parsed["uncanonicalizable"]
    row["canonical_ceiling"] = (
        None if parsed["uncanonicalizable"] is None
        else 1 - parsed["uncanonicalizable"] / len(pooled)
    )
    rows.append(row)

    return rows


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    log_dir = args.log_dir or os.path.join("logs", args.exp_no)
    out_path = args.out or os.path.join(
        "predictions", f"{args.exp_no}.metrics.csv"
    )

    if not os.path.isdir(log_dir):
        sys.exit(f"No such log directory: {log_dir}")

    if os.path.exists(out_path) and not args.overwrite:
        sys.exit(f"{out_path} exists. Pass --overwrite to replace it.")

    logs = {}
    for name in os.listdir(log_dir):
        if not name.endswith(".evaluate.log"):
            continue
        run = name[:-len(".evaluate.log")]
        # Strip the sweep prefix so the run column reads "cold_start",
        # "fully_random_800" rather than repeating the experiment name.
        if run.startswith(args.exp_no + "_"):
            run = run[len(args.exp_no) + 1:]
        logs[run] = os.path.join(log_dir, name)

    if not logs:
        sys.exit(f"No *.evaluate.log under {log_dir}")

    rows = []
    cache = {}
    for run in sorted(logs, key=run_sort_key):
        parsed = parse_log(logs[run])
        if not parsed["samples"]:
            sys.exit(f"{run}: no per-molecule metrics in {logs[run]}")
        if parsed["aborted"]:
            # The geometry numbers are still complete, but a partial canonical
            # column in a table nobody re-reads is exactly how a bad number
            # gets published.
            sys.exit(
                f"{run}: canonical matching did not finish in {logs[run]}. "
                f"Rerun evaluate.py (cached results are reused) before "
                f"aggregating."
            )
        rows.extend(rows_for_run(run, parsed, cache))
        print(f"{run}: {len(parsed['samples'])} samples from {logs[run]}")

    fieldnames = (
        ["run", "bucket", "n"] + GEOMETRY_METRICS + NOLABEL_METRICS
        + ["canonical_match", "uncanonicalizable_gt", "canonical_ceiling"]
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if row.get(key) is None else
                      f"{row[key]:.6f}" if isinstance(row[key], float)
                      else row[key])
                for key in fieldnames
            })

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main(get_args())
