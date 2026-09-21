"""Obtain canonical BigSMILES for every *.corrected.mol file under a data root.

Run from the repo root as a module, so that `utilities.*` imports resolve:

    python -m utilities.get_all_canonical_bigsmiles
"""
import argparse
import glob
import os
import sys
import traceback as tb
from collections import Counter
from tqdm import tqdm

from utilities.canonical_bigsmiles_api import (
    FAILED_BIGSMILES,
    FIELDNAMES,
    STATUS_BIGSMILES_FAILED,
    STATUS_SUCCESS,
    ServerUnavailableError,
    add_server_args,
    api_from_args,
    check_servers,
    load_existing_rows,
    sanitize,
    write_rows,
)


def status_from_existing(row: dict) -> str:
    """Infer a row's status from a TSV written before the status column.

    Two of the three cases are decidable from the strings alone. The third --
    canonical equal to its input -- is exactly the ambiguous one: it means
    either a genuine failure or a molecule that was already canonical, and only
    the server can say which. Those rows come back "" and need --reclassify.
    """
    recorded = (row.get("status") or "").strip()
    if recorded:
        return recorded

    if row["bigsmiles"] == FAILED_BIGSMILES:
        return STATUS_BIGSMILES_FAILED

    if row["canonical_bigsmiles"] != row["bigsmiles"]:
        return STATUS_SUCCESS

    return ""


def main(args):
    # The split is a pure function of this file's statuses, so a half-finished
    # run must not be able to leave a truncated TSV in its place. Everything
    # is written to a sibling .partial and promoted only once the run
    # completes; --resume reads the .partial back.
    final_file = args.output_file
    partial_file = f"{final_file}.partial"
    if os.path.exists(final_file) and not (
        args.resume or args.reclassify or args.overwrite
    ):
        print(
            f"{final_file} already exists. Pass --overwrite to replace it, "
            f"--resume to continue it, or --output_file to write elsewhere."
        )
        sys.exit(1)

    args.output_file = partial_file

    api = api_from_args(args)
    check_servers(api)

    pattern = os.path.join(args.data_dir, "**", "*.corrected.mol")
    mol_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(mol_files)} *.corrected.mol files under {args.data_dir}")

    resume_from = partial_file if os.path.exists(partial_file) else final_file
    done = load_existing_rows(resume_from) \
        if (args.resume or args.reclassify) else {}
    if done:
        print(f"Read {len(done)} existing rows from {resume_from}")
    elif args.reclassify:
        print(f"--reclassify needs an existing {resume_from}")
        sys.exit(1)

    rows = []
    reclassified = 0
    aborted = False

    for mol_file in tqdm(mol_files):
        rel_path = sanitize(os.path.relpath(mol_file, args.data_dir))
        existing = done.get(rel_path)

        if existing:
            status = status_from_existing(existing)

            # Settled already: keep the row untouched.
            if status:
                row = dict(existing)
                row["status"] = status
                rows.append(row)
                done[rel_path] = row
                continue

            # Ambiguous. Only --reclassify pays to resolve it, and it can reuse
            # the stored BigSMILES instead of redoing the molblock conversion.
            if not args.reclassify:
                rows.append(dict(existing, status=status))
                continue

            try:
                canonical, status = api.canonicalize_with_status(
                    existing["bigsmiles"]
                )
            except ServerUnavailableError as e:
                tqdm.write(f"Aborting: {e}")
                aborted = True
                break

            row = {
                "path": rel_path,
                "bigsmiles": existing["bigsmiles"],
                "canonical_bigsmiles": sanitize(canonical),
                "status": status
            }
            rows.append(row)
            done[rel_path] = row
            reclassified += 1
            write_rows(rows, args.output_file)
            continue

        try:
            with open(mol_file, "r") as f:
                molblock = f.read()
            bigsmiles = api.molblock_to_bigsmiles(molblock)
            canonical_bigsmiles, status = \
                api.canonicalize_with_status(bigsmiles)
        except ServerUnavailableError as e:
            tqdm.write(f"Aborting: {e}")
            aborted = True
            break
        except Exception:
            tb.print_exc()
            bigsmiles = FAILED_BIGSMILES
            canonical_bigsmiles = FAILED_BIGSMILES
            status = STATUS_BIGSMILES_FAILED

        row = {
            "path": rel_path,
            "bigsmiles": sanitize(bigsmiles),
            "canonical_bigsmiles": sanitize(canonical_bigsmiles),
            "status": status
        }
        rows.append(row)
        done[rel_path] = row
        write_rows(rows, args.output_file)

    write_rows(rows, args.output_file)

    if aborted:
        print(
            f"Run was incomplete ({len(rows)}/{len(mol_files)} files). "
            f"{final_file} was left untouched; progress is in {partial_file}. "
            f"Restart the server, then rerun with --resume to continue."
        )
        sys.exit(1)

    os.replace(partial_file, final_file)

    counts = Counter(row["status"] or "UNKNOWN" for row in rows)
    print(f"Wrote {len(rows)} rows to {final_file}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    if reclassified:
        print(f"Reclassified {reclassified} previously ambiguous row(s)")
    if api.killer_inputs:
        print(
            f"{len(api.killer_inputs)} input(s) crashed a service and were "
            f"recorded as failures"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain canonical BigSMILES for all *.corrected.mol files"
    )
    parser.add_argument("--data_dir", type=str, default="./data/PolymerLit")
    parser.add_argument("--output_file", type=str,
                        default="./data/PolymerLit/canonical_bigsmiles.tsv")
    parser.add_argument("--resume", action="store_true",
                        help="reuse rows already present in --output_file")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing --output_file; "
                             "without it an existing file is left alone, "
                             "because the train/val/test split is a function "
                             "of this file and is not regenerable once it "
                             "changes")
    parser.add_argument("--reclassify", action="store_true",
                        help="re-probe only the rows whose canonical equals "
                             "its input, to tell a genuine failure apart from "
                             "an already-canonical molecule")
    add_server_args(parser)

    args = parser.parse_args()
    main(args)
