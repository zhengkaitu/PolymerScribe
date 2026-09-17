import argparse
import csv
import glob
import os
import requests
import sys
import time
import traceback as tb
from tqdm import tqdm


FAILED_BIGSMILES = "failed to obtain BigSMILES"

FIELDNAMES = ["path", "bigsmiles", "canonical_bigsmiles"]

# A small polymer that both services handle, used to tell "the service is down"
# apart from "this particular input brought it down".
PROBE_BIGSMILES = "CCO{[>][<]CCO[>][<]}CCO"


class ServerUnavailableError(RuntimeError):
    """Raised when a service stops answering and does not come back."""


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
        payload_attempts: int
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
        """
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

    def canonicalize(self, bigsmiles: str) -> str:
        """Canonicalize a BigSMILES via the canonicalization-server.

        On a per-molecule failure the input bigsmiles is returned unchanged, so
        a failed canonicalization leaves the two columns identical (and a failed
        conversion carries FAILED_BIGSMILES through to both).
        """
        if bigsmiles == FAILED_BIGSMILES:
            return bigsmiles

        resp = self._post(
            self.canonicalization_session,
            self.canonicalization_uri,
            {"bigsmiles": bigsmiles},
            "canonicalization-server"
        )
        if resp is None or resp.status_code != 200:
            return bigsmiles

        try:
            result = resp.json()
        except Exception:
            tb.print_exc()
            return bigsmiles

        # The server always answers with HTTP 200 and reports failure in the
        # body; a non-empty "error" does not by itself mean failure.
        if result.get("status") != "SUCCESS":
            return bigsmiles

        return result.get("result", {}).get("canonical_bigsmiles", "") \
            or bigsmiles


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


def write_rows(rows: list, output_file: str) -> None:
    output_path = os.path.dirname(output_file)
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    with open(output_file, "w", newline="") as tsvfile:
        writer = csv.DictWriter(
            tsvfile,
            fieldnames=FIELDNAMES,
            delimiter="\t",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    api = CanonicalBigSMILESAPI(
        bigsmiles_url=args.bigsmiles_url,
        bigsmiles_port=args.bigsmiles_port,
        canonicalization_url=args.canonicalization_url,
        canonicalization_port=args.canonicalization_port,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        payload_attempts=args.payload_attempts
    )
    check_servers(api)

    pattern = os.path.join(args.data_dir, "**", "*.corrected.mol")
    mol_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(mol_files)} *.corrected.mol files under {args.data_dir}")

    done = load_existing_rows(args.output_file) if args.resume else {}
    if done:
        print(f"Resuming: {len(done)} rows already in {args.output_file}")

    rows = []
    aborted = False

    for mol_file in tqdm(mol_files):
        rel_path = sanitize(os.path.relpath(mol_file, args.data_dir))

        if rel_path in done:
            rows.append(done[rel_path])
            continue

        try:
            with open(mol_file, "r") as f:
                molblock = f.read()
            bigsmiles = api.molblock_to_bigsmiles(molblock)
            canonical_bigsmiles = api.canonicalize(bigsmiles)
        except ServerUnavailableError as e:
            tqdm.write(f"Aborting: {e}")
            aborted = True
            break
        except Exception:
            tb.print_exc()
            bigsmiles = FAILED_BIGSMILES
            canonical_bigsmiles = FAILED_BIGSMILES

        row = {
            "path": rel_path,
            "bigsmiles": sanitize(bigsmiles),
            "canonical_bigsmiles": sanitize(canonical_bigsmiles)
        }
        rows.append(row)
        done[rel_path] = row
        write_rows(rows, args.output_file)

    write_rows(rows, args.output_file)

    failure_count = sum(
        1 for row in rows if row["bigsmiles"] == FAILED_BIGSMILES
    )
    print(
        f"Wrote {len(rows)} rows to {args.output_file} "
        f"({failure_count} failed conversions)"
    )
    if api.killer_inputs:
        print(
            f"{len(api.killer_inputs)} input(s) crashed a service and were "
            f"recorded as failures"
        )
    if aborted:
        print(
            f"Run was incomplete ({len(rows)}/{len(mol_files)} files). "
            f"Restart the server, then rerun with --resume to continue."
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain canonical BigSMILES for all *.corrected.mol files"
    )
    parser.add_argument("--data_dir", type=str, default="./data/PolymerLit")
    parser.add_argument("--output_file", type=str,
                        default="./data/PolymerLit/canonical_bigsmiles.tsv")
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
    parser.add_argument("--resume", action="store_true",
                        help="reuse rows already present in --output_file")

    args = parser.parse_args()
    main(args)
