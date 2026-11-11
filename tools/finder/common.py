"""Shared helpers for the weekly finder (tools/finder): polite HTTP, identifier normalization, the index of what is
already listed, and the finder's state file.

Nothing here talks to an AI model and nothing here needs a secret (OPENALEX_API_KEY is optional)."""
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
WORK = HERE / "work" / "finder"
STATE = HERE / "source" / "finder" / "state.json"
AUTHORS = HERE / "source" / "finder" / "authors.json"
UA = "OpenMotor-curation/1.0 (research data index; https://github.com/physicalintelligencelab/openmotor)"
MAILTO = "physical.intelligence.lab@gmail.com"

TASKS = ["reaching", "walking", "speech", "sequence learning", "finger tapping and timing", "eye movements", "tracking",
         "grasping and lifting", "sports and games", "finger movements", "typing", "balance and posture",
         "handwriting and drawing", "hand gestures and grasps", "robot-assisted training", "infant movements", "running", "other"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ HTTP
HOST_GAP = {"zenodo.org": 1.2, "api.osf.io": 0.6, "api.github.com": 1.0, "api.figshare.com": 0.5, "api.openalex.org": 0.2,
            "www.biorxiv.org": 6.0, "www.medrxiv.org": 6.0, "api.biorxiv.org": 1.0, "www.ebi.ac.uk": 0.5}


class Http:
    """GET/POST with a fixed User-Agent, at most `rate` requests per second per host, backoff on 429/5xx/network errors.
    Returns parsed JSON (or bytes with raw=True), or None on 400/404/410 or after the last retry."""

    def __init__(self, rate=3.0, tries=4, timeout=45, give_up_after=8):
        self.min_gap = 1.0 / rate
        self.tries = tries
        self.timeout = timeout
        self.give_up_after = give_up_after  # consecutive failed requests after which a host is left alone for this run
        self.last = {}
        self.count = {}
        self.errors = {}
        self.failing = {}
        self.down = set()
        self.deadline = None  # time.time() after which no request is sent (a run's time budget)
        self.out_of_time = False

    def _wait(self, host):
        gap = max(self.min_gap, HOST_GAP.get(host, 0))
        dt = time.time() - self.last.get(host, 0)
        if dt < gap:
            time.sleep(gap - dt)
        self.last[host] = time.time()

    def request(self, url, data=None, headers=None, raw=False, method=None, quiet404=True, key=None, tries=None, timeout=None):
        """key: name of the circuit breaker (default: the host); tries: attempts for this request (default self.tries)."""
        host = urllib.parse.urlparse(url).hostname or ""
        key = key or host
        tries = tries or self.tries
        if key in self.down:
            return None
        if self.deadline and time.time() > self.deadline:
            self.out_of_time = True
            return None
        h = {"User-Agent": UA, "Accept": "*/*" if raw else "application/json"}
        if host == "api.crossref.org":
            h["User-Agent"] = UA + " mailto:" + MAILTO
        h.update(headers or {})
        body = json.dumps(data).encode() if isinstance(data, (dict, list)) else data
        if isinstance(data, (dict, list)):
            h["Content-Type"] = "application/json"
        last_err = ""
        for attempt in range(tries):
            self._wait(host)
            self.count[host] = self.count.get(host, 0) + 1
            try:
                req = urllib.request.Request(url, data=body, headers=h, method=method)
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    b = r.read()
                self.failing[key] = 0
                return b if raw else json.loads(b.decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404, 410, 422):
                    if not (quiet404 and e.code in (404, 410)):
                        self.errors[host] = self.errors.get(host, 0) + 1
                    return None
                wait = 4 * 2 ** attempt
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.isdigit():
                    wait = max(wait, min(int(ra), 120))
                if e.code == 429:
                    wait = max(wait, 30)
                    self.last[host] = time.time() + (min(int(ra), 120) if ra and ra.isdigit() else 30)  # the next request waits too
                    log(f"  429 from {host}; waiting {wait}s" if attempt < tries - 1 else f"  429 from {host}; skipped")
                last_err = f"HTTP {e.code}"
            except json.JSONDecodeError:
                self.errors[host] = self.errors.get(host, 0) + 1
                return None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                wait = 3 * 2 ** attempt
                last_err = type(e).__name__
            if attempt < tries - 1:
                time.sleep(wait)
        self.errors[host] = self.errors.get(host, 0) + 1
        self.failing[key] = self.failing.get(key, 0) + 1
        log(f"  failed after {tries} tries ({last_err}): {url.split('?')[0][:120]}")
        if self.failing[key] >= self.give_up_after:
            self.down.add(key)
            log(f"  {key}: {self.failing[key]} requests in a row failed; not asking it again in this run")
        return None

    def get(self, url, params=None, **kw):
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        return self.request(url, **kw)


# ------------------------------------------------------------------ identifiers
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>(){}\[\],;]+)", re.I)
DATA_DOI_PREFIXES = ("10.5281/zenodo", "10.6084/m9.figshare", "10.5061/dryad", "10.7910/dvn", "10.17632/", "10.17605/osf.io",
                     "10.12751/g-node", "10.13026/", "10.18112/openneuro", "10.34973/", "10.4121/", "10.25452/", "10.48324/dandi",
                     "10.3886/", "10.5683/sp", "10.18710/", "10.34894/", "10.17026/", "10.25446/", "10.26180/", "10.6084/")


def norm_doi(s):
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", s, flags=re.I)
    s = s.rstrip(".,;)]}>").lower()
    s = re.sub(r"/full$|/abstract$|/pdf$", "", s)
    return s if s.startswith("10.") else ""


def is_data_doi(d):
    return any(d.startswith(p) for p in DATA_DOI_PREFIXES)


def dois_in(text):
    out = []
    for m in DOI_RE.finditer(text or ""):
        d = norm_doi(m.group(1))
        if d and d not in out:
            out.append(d)
    return out


REPO_URL_RE = re.compile(
    r"(https?://(?:www\.)?(?:osf\.io/[A-Za-z0-9]{5}\b|(?:sandbox\.)?zenodo\.org/(?:records?|doi)/[0-9/.a-z]+|"
    r"(?:[a-z0-9-]+\.)?figshare\.com/[^\s\"'<>)]+|github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|"
    r"[a-z0-9.-]*dataverse[a-z0-9.-]*/[^\s\"'<>)]+|datadryad\.org/[^\s\"'<>)]+|physionet\.org/content/[^\s\"'<>)]+|"
    r"openneuro\.org/datasets/ds\d+|data\.mendeley\.com/datasets/[a-z0-9]+|gin\.g-node\.org/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|"
    r"borealisdata\.ca/[^\s\"'<>)]+|data\.4tu\.nl/[^\s\"'<>)]+|dandiarchive\.org/dandiset/\d+)"
    r"|(?:https?://)?(?:dx\.)?doi\.org/10\.\d{4,9}/[^\s\"'<>)]+"
    r"|\b(?:osf\.io/[A-Za-z0-9]{5}\b|zenodo\.org/records?/\d+|github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|openneuro\.org/datasets/ds\d+))",
    re.I)


def repo_links(text):
    """Repository links (and data DOIs) in free text, as URLs, in order of appearance."""
    out = []
    t = html.unescape(text or "")
    for m in REPO_URL_RE.finditer(t):
        u = m.group(0).rstrip(".,;:)]}>\"'")
        if not u.lower().startswith("http"):
            u = "https://" + u
        if "doi.org/" in u.lower():
            d = norm_doi(u)
            if not (d and is_data_doi(d)):
                continue
            u = "https://doi.org/" + d
        if u not in out:
            out.append(u)
    for d in dois_in(t):
        if is_data_doi(d) and "https://doi.org/" + d not in out:
            out.append("https://doi.org/" + d)
    return out


def data_key(url):
    """A normalized key for a data location, so the same deposit written different ways compares equal."""
    if not url:
        return ""
    u = html.unescape(str(url)).strip()
    low = u.lower()
    d = norm_doi(u) if ("doi.org/" in low or low.startswith("10.") or low.startswith("doi:")) else ""
    if d:
        m = re.match(r"10\.5281/zenodo\.(\d+)", d)
        if m:
            return "zenodo:" + m.group(1)
        m = re.match(r"10\.6084/m9\.figshare\.(?:c\.)?(\d+)", d)
        if m:
            return "figshare:" + m.group(1)
        m = re.match(r"10\.17605/osf\.io/([a-z0-9]{5})", d)
        if m:
            return "osf:" + m.group(1)
        m = re.match(r"10\.18112/openneuro\.(ds\d+)", d)
        if m:
            return "openneuro:" + m.group(1)
        m = re.match(r"10\.17632/([a-z0-9]+)", d)
        if m:
            return "mendeley:" + m.group(1)
        return "doi:" + re.sub(r"\.v\d+$", "", d)
    m = re.search(r"osf\.io/(?:preprints/[a-z]+/)?([a-z0-9]{5})(?:_v\d+)?\b", low)
    if m:
        return "osf:" + m.group(1)
    m = re.search(r"zenodo\.org/(?:records?|doi/10\.5281/zenodo\.)/?(\d+)", low)
    if m:
        return "zenodo:" + m.group(1)
    m = re.search(r"figshare\.com/.*?/(\d{5,})(?:/\d+)?/?(?:$|[?#])", low) or re.search(r"figshare\.com/.*?/(\d{5,})", low)
    if m:
        return "figshare:" + m.group(1)
    m = re.search(r"datadryad\.org/.*?(10\.5061/dryad\.[a-z0-9]+)", low)
    if m:
        return "doi:" + m.group(1)
    m = re.search(r"persistentid=doi:(10\.[^&#]+)", low)
    if m:
        return "doi:" + m.group(1)
    m = re.search(r"openneuro\.org/datasets/(ds\d+)", low)
    if m:
        return "openneuro:" + m.group(1)
    m = re.search(r"physionet\.org/content/([a-z0-9-]+)", low)
    if m:
        return "physionet:" + m.group(1)
    m = re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)", low)
    if m:
        return "mendeley:" + m.group(1)
    m = re.search(r"(github\.com|gin\.g-node\.org)/([a-z0-9_.-]+)/([a-z0-9_.-]+)", low)
    if m:
        return ("github:" if "github" in m.group(1) else "gin:") + m.group(2) + "/" + re.sub(r"\.git$", "", m.group(3))
    m = re.search(r"dandiarchive\.org/dandiset/(\d+)", low)
    if m:
        return "dandi:" + m.group(1)
    return "url:" + re.sub(r"^https?://(www\.)?", "", low).rstrip("/")


HOST_LABEL = [("osf", "OSF"), ("zenodo", "Zenodo"), ("figshare", "Figshare"), ("github", "GitHub"), ("openneuro", "OpenNeuro"),
              ("physionet", "PhysioNet"), ("mendeley", "Mendeley Data"), ("gin", "G-Node GIN"), ("dandi", "DANDI")]


def host_label(url):
    k = data_key(url)
    for pre, lab in HOST_LABEL:
        if k.startswith(pre + ":"):
            return lab
    if k.startswith("doi:10.5061/dryad"):
        return "Dryad"
    if k.startswith("doi:10.7910/dvn"):
        return "Harvard Dataverse"
    if "dataverse" in k or k.startswith("doi:10.34894/") or k.startswith("doi:10.5683/") or k.startswith("doi:10.18710/"):
        return "Dataverse"
    if k.startswith("doi:10.4121/"):
        return "4TU.ResearchData"
    return "other"


def canonical_url(url):
    """A stable public URL for a data location."""
    k = data_key(url)
    kind, _, v = k.partition(":")
    return {"osf": f"https://osf.io/{v}/", "zenodo": f"https://zenodo.org/records/{v}", "openneuro": f"https://openneuro.org/datasets/{v}",
            "doi": f"https://doi.org/{v}", "github": f"https://github.com/{v}", "gin": f"https://gin.g-node.org/{v}",
            "physionet": f"https://physionet.org/content/{v}/", "dandi": f"https://dandiarchive.org/dandiset/{v}"}.get(kind, url)


def norm_title(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t or "")).lower()
    t = re.sub(r"^(data|dataset|datasets|data and code|code and data|raw data|supplementary data|data set)\s*"
               r"(for|from|of|supporting|accompanying|associated with|underlying|related to)?\s*[:\-]?\s*", "", t)
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()


def title_sim(a, b):
    a, b = " ".join(norm_title(a)), " ".join(norm_title(b))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------ what is already listed
def listed_index():
    """DOIs, data keys, titles and ids of every study already on the site (source/extra_studies.json + site/catalog.json)."""
    extra = json.load(open(HERE / "source/extra_studies.json", encoding="utf-8"))
    cat = json.load(open(HERE / "site/catalog.json", encoding="utf-8"))["studies"]
    dois, keys, titles, ids = set(), set(), [], set()
    for e in extra:
        ids.add(e["id"])
        keys.add(data_key(e.get("data_url")))
        for p in e.get("papers") or []:
            if norm_doi(p.get("doi")):
                dois.add(norm_doi(p.get("doi")))
            if p.get("title"):
                titles.append((" ".join(norm_title(p["title"])), e["id"]))
    for s in cat:
        ids.add(s["id"])
        for l in s.get("links") or []:
            keys.add(data_key(l.get("url")))
        for p in s.get("papers") or []:
            if norm_doi(p.get("doi")):
                dois.add(norm_doi(p.get("doi")))
            if p.get("title"):
                titles.append((" ".join(norm_title(p["title"])), s["id"]))
    for p in (HERE / "source/release_extra_records").glob("*.json"):
        ids.add(p.name.split(".")[0])
    for p in (HERE / "tools/std/convert").glob("*.py"):
        ids.add(p.stem)
    keys.discard("")
    return {"dois": dois, "keys": keys, "titles": titles, "ids": ids, "extra": extra, "catalog": cat}


def listed_title_match(title, idx, thr=0.9):
    t = " ".join(norm_title(title))
    if len(t) < 25:
        return None
    for lt, sid in idx["titles"]:
        if abs(len(lt) - len(t)) < 0.3 * len(t) and SequenceMatcher(None, t, lt).ratio() >= thr:
            return sid
    return None


# ------------------------------------------------------------------ state
def load_json(path, default):
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    tmp.replace(p)


def empty_state():
    return {"last_run": None, "seen": {}, "held": {}, "watch": {}, "pending": {}, "history": []}


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_text(t, limit=None):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t or ""))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit] if limit else t


def openalex_params(p=None):
    p = dict(p or {})
    if os.environ.get("OPENALEX_API_KEY"):
        p["api_key"] = os.environ["OPENALEX_API_KEY"]
    else:
        p["mailto"] = MAILTO
    return p
