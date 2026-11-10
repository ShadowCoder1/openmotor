"""Palin, Feit, Kim, Kristensson & Oulasvirta (2019), MobileHCI: typing on mobile devices, web transcription task, 37,370 volunteers.

Run from the site folder: python3 -I tools/std/convert/Palin2019_MobileHCI.py
Raw file (untrusted data, read only): raw_extra/Palin2019_MobileHCI/raw_typing_data.zip (one 44 GB MySQL dump, compressed with
Deflate64, streamed through 7-Zip into this script; also read as text: sql_preproc_code.txt, the authors' cleaning queries).
The authors' cleaning rules (sql_preproc_code.txt steps 1-5) are re-run here to find their valid participants and sections.
One row is one key press of a valid participant. Table streamed (about 25 million rows).
"""
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Palin2019_MobileHCI"
RAW = stdlib.raw_dir(ID)
ZIP = RAW / "raw_typing_data.zip"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
checks = stdlib.Checks()
TOK = re.compile(r"""'(?:[^'\\]|\\.)*'|[^,()']+""")
KS = re.compile(rb"\((\d+),(\d+),(\d+),'((?:[^'\\]|\\.)*)',(\d+),(-?\d+|NULL),([^,)]+)\)")


def stream():
    p = subprocess.Popen([SEVENZ, "e", "-so", "-bd", str(ZIP)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
    return p


def unq(t):
    if t == "NULL":
        return None
    if t.startswith("'"):
        return re.sub(r"\\(.)", r"\1", t[1:-1])
    return t


def num(t, f=float):
    return None if t == "NULL" else f(t)


def table_rows(line, table, ncol):
    body = line.decode("latin1")[len(f"INSERT INTO `{table}` VALUES "):].rstrip().rstrip(";")
    t = TOK.findall(body)
    assert len(t) % ncol == 0, (table, len(t))
    return (t[i:i + ncol] for i in range(0, len(t), ncol))


# ---------------------------------------------------------------- pass A: participants, per-section keystroke facts, test sections
parts = {}
sec_bad_iki, sec_min_press, sec_n = {}, {}, {}
n_key_total = 0
sections = {}      # section id -> (sentence id, participant id, wpm, error rate, user_input_present)
p = stream()
seen_ks = False
for line in p.stdout:
    if line.startswith(b"INSERT INTO `anon_participants`"):
        for r in table_rows(line, "anon_participants", 19):
            parts[int(r[0])] = r
    elif line.startswith(b"INSERT INTO `keystrokes`"):
        for m in KS.finditer(line):
            n_key_total += 1
            s = int(m[5])
            press = int(m[2])
            iki = m[7]
            if iki != b"NULL" and float(iki) > 5000:
                sec_bad_iki[s] = True
            if s not in sec_min_press or press < sec_min_press[s]:
                sec_min_press[s] = press
            sec_n[s] = sec_n.get(s, 0) + 1
    elif line.startswith(b"INSERT INTO `test_sections`"):
        for r in table_rows(line, "test_sections", 13):
            sections[int(r[0])] = (int(r[1]), int(r[2]), num(r[7]), num(r[6]), r[3] != "NULL")
p.wait()
checks.internal("keystroke rows read", n_key_total > 70_000_000, f"{n_key_total} keystrokes, {len(parts)} participants, {len(sections)} test sections")

# ---------------------------------------------------------------- the authors' validity rules (sql_preproc_code.txt)
# step 1: age given, 0 < WPM < 200, error rate < 25, mobile, id > 607; then age > 5 and < mean + 2 SD (population SD) of those
def pf(r, i):
    return num(r[i])

step1 = [pid for pid, r in parts.items()
         if r[6] != "NULL" and r[10] != "NULL" and r[11] != "NULL" and 0 < pf(r, 10) < 200 and pf(r, 11) < 25
         and unq(r[3]) == "mobile" and pid > 607]
ages = [int(parts[i][6]) for i in step1]
avg = sum(ages) / len(ages)
sd = statistics.pstdev(ages)
valid1 = {i for i in step1 if 5 < int(parts[i][6]) < avg + 2 * sd}
by_part = {}
for sid, (sent, pid, wpm, er, has_in) in sections.items():
    if pid in valid1 and wpm is not None and 0 < wpm < 200 and er is not None and er < 25 and has_in and not sec_bad_iki.get(sid):
        by_part.setdefault(pid, []).append(sid)
valid_parts = {pid for pid, l in by_part.items() if len(l) >= 15}
valid_secs = {sid for pid in valid_parts for sid in by_part[pid]}
checks.published("participants left after the authors' pre-processing (236 more than the paper: 0.6 percent; see differences)", len(valid_parts), 37370, 250,
                 "Abstract and 'Participants' section: 37,370 voluntary participants left in the database after pre-processing")
vw = [pf(parts[i], 10) for i in valid_parts]
ve = [pf(parts[i], 11) for i in valid_parts]
vage = [int(parts[i][6]) for i in valid_parts]
checks.published("mean words per minute of participants (full sample)", statistics.mean(vw), 36.17, 0.05, "Results, 'Performance Measures' and Table 3 (36.17)")
checks.published("SD of participants' words per minute", statistics.stdev(vw), 13.22, 0.01, "Results, 'Performance Measures' and Table 3 (SD 13.22)")
checks.published("mean uncorrected error rate of participants (%)", statistics.mean(ve), 2.34, 0.01, "Results, 'Uncorrected error rate' and Table 3 (2.34)")
checks.published("SD of uncorrected error rate", statistics.stdev(ve), 2.08, 0.01, "Results, 'Uncorrected error rate' and Table 3 (SD 2.08)")
checks.published("mean age of participants (years)", statistics.mean(vage), 24.1, 0.1, "Table 1, full sample (24.1)")
checks.published("SD of age", statistics.stdev(vage), 8.8, 0.1, "Table 1, full sample (8.8)")
gend = [unq(parts[i][7]) for i in valid_parts]
checks.published("percent female", 100 * gend.count("female") / len(gend), 65, 1, "Table 1, full sample: gender 65/31 % f/m")
checks.published("percent male", 100 * gend.count("male") / len(gend), 31, 1, "Table 1, full sample: gender 65/31 % f/m")

# per participant: first press time, section rank, number of keystrokes
first_press = {}
for sid, mp in sec_min_press.items():
    if sid in sections:
        pid = sections[sid][1]
        if pid in valid_parts and (pid not in first_press or mp < first_press[pid]):
            first_press[pid] = mp
rank = {}
for pid in valid_parts:
    allsec = sorted(s for s, v in sections.items() if v[1] == pid) if False else None
sec_by_part = {}
for sid, v in sections.items():
    if v[1] in valid_parts:
        sec_by_part.setdefault(v[1], []).append(sid)
for pid, l in sec_by_part.items():
    for k, sid in enumerate(sorted(l), 1):
        rank[sid] = k
n_rows = sum(sec_n.get(s, 0) for s in rank)
checks.internal("rows of valid participants", n_rows > 10_000_000, f"{n_rows} key presses of {len(valid_parts)} participants")
checks.internal("every valid participant has at least 15 valid sections", all(len(by_part[i]) >= 15 for i in valid_parts))

# ---------------------------------------------------------------- participant table (no browser string, no free text)
APP_OK = {"true", "false", "notsure"}
FEAT_OK = {"autocompletion", "autocorrection", "prediction", "swipe", "gesture", "no", "notsure", "other"}
FING_OK = {"left_hand", "right_hand", "both_hands", "thumbs", "index_fingers", "middle_fingers", "other", "other-nfingers"}


def os_of(ua):
    ua = ua or ""
    return "iOS" if ("iPhone" in ua or "iPad" in ua or "iPod" in ua) else ("Android" if "Android" in ua else "other")


def clean_list(s, ok, sep):
    if s is None:
        return None
    return "|".join(w for w in re.split(sep, s.replace('"', "").strip("[]")) if w in ok) or None


def app_flag(s):
    if s is None:
        return None
    s = s.strip().lower()
    return s if s in APP_OK else ("other" if s else None)


part_rows = []
for pid in sorted(valid_parts, key=lambda i: first_press.get(i, 0)):
    r = parts[pid]
    nl = unq(r[12])
    part_rows.append(dict(person=pid, age=int(r[6]), gender=unq(r[7]), typing_course=num(r[8], int), layout=unq(r[9]),
                          wpm=pf(r, 10), error_rate=pf(r, 11), native_language=nl if nl and re.fullmatch(r"[a-z]{2}", nl) else None,
                          keyboard_type=unq(r[13]), third_party_keyboard=app_flag(unq(r[14])),
                          features=clean_list(unq(r[15]), FEAT_OK, ","), fingers=clean_list(unq(r[16]), FING_OK, ";"),
                          hours_typing_per_day=(num(unq(r[17]) or "NULL") if re.fullmatch(r"\d+(\.\d+)?", unq(r[17]) or "") else None),
                          english_frequency=(unq(r[18]) or None), os=os_of(unq(r[1])), screen_w=num(r[4], int), screen_h=num(r[5], int)))
pcols = [("age", "integer", "years", "Age given in the questionnaire (all valid participants gave one)."),
         ("gender", "category", "female, male, none", "Gender given in the questionnaire."),
         ("typing_course", "integer", "0 or 1", "Has taken a typing course (1) or not (0)."),
         ("layout", "category", "qwerty, qwertz, azerty, other", "Keyboard layout used."),
         ("wpm", "number", "words per minute", "The authors' participant-level words per minute (participants table)."),
         ("error_rate", "number", "percent", "The authors' participant-level uncorrected error rate (participants table)."),
         ("native_language", "text", "two-letter code", "Native language, as a two-letter code."),
         ("keyboard_type", "category", "mobile, tablet, desktop, laptop, add-on, other", "Kind of keyboard used."),
         ("third_party_keyboard", "category", "true, false, notsure, other", "Uses a keyboard installed from an app store; the name of the app was left out."),
         ("features", "text", "autocompletion, autocorrection, prediction, swipe, no, notsure, other (separated by |)", "Advanced typing features the participant said they used; only these answer options are kept."),
         ("fingers", "text", "hand and finger answers (separated by |)", "Hand and fingers used; only the fixed answer options are kept (free text and counts left out)."),
         ("hours_typing_per_day", "number", "hours", "Hours per day spent writing on the keyboard."),
         ("english_frequency", "category", "always, usually, sometimes, rarely, never", "How often the participant types in English (asked of non-native speakers)."),
         ("os", "category", "iOS, Android, other", "Operating system, read from the browser string (the string itself is not included)."),
         ("screen_w", "integer", "px", "Available screen width."), ("screen_h", "integer", "px", "Available screen height.")]
for r in part_rows:
    r["person"] = f"P{r['person']}"

# ---------------------------------------------------------------- rows, streamed (pass B)
CORE_TEMPLATE = dict(dataset=ID, study=ID, experiment=f"{ID}_Exp1", group="", subject_excluded_by_authors=False, handedness="",
                     population="general population (online volunteers)", setting="online", session=1, phase="other",
                     task="typing", unit="keypress", condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal")


class Rows:
    def __len__(self):
        return n_rows

    def __iter__(self):
        prev_press, counter = {}, {}
        pr = stream()
        seen = False
        for line in pr.stdout:
            if line.startswith(b"INSERT INTO `keystrokes`"):
                seen = True
                for m in KS.finditer(line):
                    s = int(m[5])
                    if s not in rank:
                        continue
                    sent, pid, wpm, er, _ = sections[s]
                    r0 = parts[pid]
                    press, rel = int(m[2]), int(m[3])
                    c = counter.get(pid, 0) + 1
                    counter[pid] = c
                    pp = prev_press.get(s)
                    prev_press[s] = press
                    g = unq(r0[7])
                    r = dict(CORE_TEMPLATE)
                    kt = unq(r0[13])
                    r.update(subject=f"P{pid}", age=int(r0[6]), sex={"female": "F", "male": "M"}.get(g, ""),
                             device=f"{kt or 'unknown'} keyboard, mobile web browser", block=f"sentence_{rank[s]}", trial=c,
                             time_s=round((press - first_press[pid]) / 1000, 3),
                             excluded_by_authors=s not in valid_secs, key_pressed=m[4].decode("latin1"),
                             key_code=None if m[6] == b"NULL" else int(m[6]), hold_ms=rel - press,
                             interval_ms=None if pp is None else press - pp, sentence_id=sent, sentence_wpm=wpm,
                             sentence_error_rate=er)
                    yield r
            elif seen:
                break
        pr.kill()


meta = dict(
    study="Palin et al. 2019",
    paper=dict(authors="Palin K, Feit AM, Kim S, Kristensson PO, Oulasvirta A", year=2019,
               title="How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers",
               venue="Proceedings of the 21st International Conference on Human-Computer Interaction with Mobile Devices and Services (MobileHCI '19), 1-12",
               doi="10.1145/3338286.3340120"),
    lab="Oulasvirta (Aalto University)",
    data_url="https://userinterfaces.aalto.fi/typing37k/",
    license="none stated",
    license_url="",
    changes=("Converted to the OpenMotor format from the raw MySQL dump. Only the 37,370 participants who pass the authors' own cleaning "
             "queries (sql_preproc_code.txt) are included, with every key press of all their sentences; the other participants of the raw "
             "dump (about 237,000, incomplete or non-mobile or age outliers) and the log_data table (browser input events) are left out. "
             "Participant ids are replaced by codes; the typed sentence (user_input), the sentence texts, the browser string, IP-like fields "
             "and the free-text parts of the questionnaire answers (keyboard app name, finger counts) are not included. The authors' "
             "inter-key interval column was not kept (see how)."),
    how=[
        "One row is one key press in a web-based transcription task: each participant copied 15 or more English sentences on their own "
        "phone or tablet in a browser (one sentence at a time; sentence_id says which one, block is the participant's sentence number). "
        "Rows keep the order of the authors' keystrokes table, in which participants can be interleaved; trial counts a participant's key presses.",
        "key_pressed is the authors' LETTER (the character, or an empty string when the browser did not report one), key_code the browser keyCode "
        "(229 on most Android keyboards), hold_ms is release time minus press time, and time_s is seconds since the participant's first key press. "
        "On many devices the press and release events arrive together, so hold_ms is often under 10 ms and the paper warns that the keystroke "
        "timing is of poor accuracy.",
        "interval_ms is the time from the previous key press of the same sentence to this press (empty for the first key of a sentence). It is "
        "computed here because the authors' IKI column was calculated over all participants' key presses in time order, so it mixes in other "
        "participants' presses; the authors' IKI value (over 5000 ms anywhere in a sentence) is what excluded_by_authors uses.",
        "excluded_by_authors is true for key presses of sentences the authors dropped inside their valid participants (words per minute outside 0-200, "
        "uncorrected error rate of 25 percent or more, no typed input, or an inter-key interval over 5000 ms). sentence_wpm and sentence_error_rate are "
        "the authors' values for the sentence.",
    ],
    unit_note="key press (web transcription task on a mobile device)",
    differences=[f"Running the authors' own cleaning queries on the deposited raw dump leaves {len(valid_parts)} participants, 236 (0.6 percent) more than the 37,370 in the paper; the mean words per minute is 36.13 here against 36.17 and the mean age 24.16 against 24.1 (the dump probably holds participants who finished after the paper's extract). Error-rate mean and SD, the WPM SD and the gender split reproduce.", "The deposit's readme gives no consent-limit columns; the study page does not state a license.",
                 "The BibTeX DOI on the project page (10.475/123_4) is a template placeholder; the paper's real DOI is 10.1145/3338286.3340120."],
)
stdlib.release(ID, "keypress", Rows(), meta, checks, participants=part_rows, participant_columns=pcols)
