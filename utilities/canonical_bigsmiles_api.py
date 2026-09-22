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
import os
import requests
import sys
import time
import traceback as tb
from tqdm import tqdm


FAILED_BIGSMILES = "failed to obtain BigSMILES"

# Canonicalization outcomes, recorded per row so that a genuine failure is
# never confused with a molecule that was already canonical -- both of which
# leave the canonical column equal to its input.
STATUS_SUCCESS = "SUCCESS"                          # canonicalized, changed
STATUS_NOOP = "NOOP"                                # already canonical
STATUS_NO_STOCHASTIC_OBJECT = "NO_STOCHASTIC_OBJECT"  # nothing to enumerate
STATUS_FAIL = "FAIL"                                # canonicalization failed
STATUS_BIGSMILES_FAILED = "BIGSMILES_FAILED"        # never got a BigSMILES

# A canonical BigSMILES is usable as ground truth in these states only.
VALID_STATUSES = frozenset({STATUS_SUCCESS, STATUS_NOOP})

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
    symbol: str = WILDCARD_SYMBOL
) -> str:
    """Give every wildcard/R-group atom the same element symbol.

    The two sides of the benchmark draw an abbreviation differently and
    bigsmiles-server keys off the element symbol while *ignoring* the alias
    record, so the same molecule used to convert two different ways:

        ground truth (ChemDraw)  carbon + "A" alias "CN"  ->  ...C...
        prediction (MolScribe)   element R + same alias   ->  ...[R]...

    and [R] never canonicalizes. Rewriting both to one real element makes the
    conversion a function of the structure instead of the exporter.

    An atom is a wildcard if it carries an alias record or its symbol already
    says so -- except an atom whose alias text is "*", which is a polymer
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
        elif alias is not None or current in WILDCARD_INPUT_SYMBOLS:
            replacement = symbol
        else:
            continue

        lines[i] = (
            f"{lines[i][:_SYMBOL_START]}{replacement:<3}"
            f"{lines[i][_SYMBOL_END:]}"
        )

    return "\n".join(lines)


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
        normalize_wildcards: bool = True
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
        it is also where wildcard normalization happens -- ground truth and
        predictions cannot diverge if they are normalized here.
        """
        if self.normalize_wildcards:
            molblock = normalize_wildcard_atoms(molblock)

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

    def canonicalize_with_status(self, bigsmiles: str) -> tuple[str, str]:
        """Canonicalize a BigSMILES, reporting how it went.

        Returns (canonical, status). On any failure the input is handed back
        unchanged, so the status is the only thing that distinguishes a failure
        from a molecule that was already canonical.
        """
        if bigsmiles == FAILED_BIGSMILES:
            return bigsmiles, STATUS_BIGSMILES_FAILED

        resp = self._post(
            self.canonicalization_session,
            self.canonicalization_uri,
            {"bigsmiles": bigsmiles},
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

        if canonical != bigsmiles:
            return canonical, STATUS_SUCCESS

        # Unchanged but successful. Usually the molecule was already canonical;
        # but with no stochastic object there is nothing to enumerate and the
        # canonicalizer returns its input, which is a different claim.
        if "{" not in bigsmiles:
            return canonical, STATUS_NO_STOCHASTIC_OBJECT

        return canonical, STATUS_NOOP

    def canonicalize(self, bigsmiles: str) -> str:
        """Canonicalize a BigSMILES, keeping only the string."""
        return self.canonicalize_with_status(bigsmiles)[0]


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
        )
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
