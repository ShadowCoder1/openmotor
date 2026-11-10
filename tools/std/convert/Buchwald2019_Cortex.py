"""Buchwald, Calhoun, Rimikis, Lowe, Wellner & Edwards (2019), Cortex 111:274-285: single-session transcranial direct current
stimulation (tDCS) and speech motor learning. 80 adults practise nonwords that start with eight non-native consonant clusters
(fn, fm, vn, vm, zb, zg, pt, gd), receiving sham or active tDCS (1 mA, 20 min) before or during practice, and are tested about
30 minutes (R1) and two days (R2) later.

Raw files, raw_extra/Buchwald2019_Cortex/ (OSF dpsyz, fetched with tools/std/fetch.py; Latin-1 text, no license stated):
  alldata_new.csv              66,068 rows, one per spoken item: subj, updown, timing, #_in_session, traintype (trained, untrained,
        gen, na), word, session (1-6: recording chunks), response_IPA, wordacc, cluster, cluster_acc, complete, clusacc. The rows of
        each participant are in scrambled recording-session order. 17 rows are placeholders of participants who did not complete
        the study (all cells na).
  alldata_new_withorder.csv    the 12,800 practice productions of the analysed participants (cluster items only) with order (1-160),
        block (1-10) and quarts
  all_gd_data.csv, gd_practice.csv   the /gd/ tokens (5,439; the practice part is gd_practice.csv, 1,600) that two coders re-scored,
        with the first coder's code (original coding), the final code (accuracy), the inserted vowel's start, duration and end
        (s) and its final duration in ms (duration)
  osf_*.R                      the authors' analysis code, read as text. They name sham as updown "up" (shamdata <- subset(data,
        updown=="up")) and the two active groups downbefore and downduring, test epenthesis with accuracy=="e", and cut practice
        into quintiles by pairs of blocks.
What the values are: paper (PMC6358507) Methods "Participants", "Speech stimuli", "Speech motor learning task procedure",
"tDCS procedure", "Data analysis"; see also how/changes in the meta below.
Run from the site folder: python3 -I tools/std/convert/Buchwald2019_Cortex.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Buchwald2019_Cortex"
SRC = "Buchwald et al. 2019, Cortex"
RAW = stdlib.raw_dir(ID)
CL = ["fn", "fm", "vn", "vm", "zb", "zg", "pt", "gd"]
DEVICE = ("Shure BETA 58A microphone and Marantz PMD660 recorder; Soterix 1x1 tDCS stimulator (1 mA for 20 min, anode over left "
          "motor cortex C3, cathode over right supraorbital area; sham: current ramped up and turned off)")


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def clean(v):
    return str(v).replace("\xa0", " ").strip()


def rd(name):
    return pd.read_csv(RAW / name, dtype=str, keep_default_na=False, encoding="latin-1")


def code3(c):
    c = clean(c)
    return c if c in ("c", "e") else "o"


def main():
    A, W, G = rd("alldata_new.csv"), rd("alldata_new_withorder.csv"), rd("all_gd_data.csv")
    ck = stdlib.Checks()
    s80 = set(W.subj)
    A["row"] = np.arange(len(A))
    placeholder = ~A.session.isin(list("123456"))
    ph = A[placeholder]
    ck.internal("the 17 placeholder rows belong to participants who are not in the analysed set and hold no item",
                len(ph) == 17 and not (set(ph.subj) & s80) and ph.word.isin(["na", ""]).all(), f"{len(ph)} rows")
    d = A[~placeholder].copy()
    ck.internal("the other rows are exactly the 80 participants of the practice file", set(d.subj) == s80 and len(s80) == 80)
    d["sess_i"] = d.session.astype(int)
    d["n_i"] = d["#_in_session"].astype(int)
    d["pos"] = np.arange(len(d))  # position in the authors' file (without the placeholders)

    # chronological order inside each participant: recording session, then the file's run of that session, then the authors' number
    chg = (d.subj != d.subj.shift()) | (d.sess_i != d.sess_i.shift())
    d["run_id"] = chg.cumsum()
    d["run_rank"] = d.groupby(["subj", "sess_i"]).run_id.rank(method="dense")
    d["subj_i"] = d.subj.map({s: i for i, s in enumerate(dict.fromkeys(d.subj))})
    order = d.sort_values(["subj_i", "sess_i", "run_rank", "n_i", "pos"], kind="stable")
    d["trial"] = order.groupby("subj_i").cumcount().reindex(d.index) + 1
    n_dup_runs = int((d.groupby(["subj", "sess_i"]).run_rank.max() > 1).sum())

    # practice block of the authors (alldata_new_withorder.csv): key (participant, recording session, number in session)
    W["sess_i"] = W.session.astype(int)
    W["n_i"] = W["#_in_session"].astype(int)
    assert not W.duplicated(["subj", "sess_i", "n_i"]).any()
    m = W.merge(d, on=["subj", "sess_i", "n_i"], suffixes=("_w", "_a"))
    same = ((m.word_w == m.word_a) & (m.response_IPA_w == m.response_IPA_a) & (m.cluster_acc_w == m.cluster_acc_a)
            & (m.wordacc_w == m.wordacc_a) & (m.clusacc_w == m.clusacc_a) & (m.traintype_w == m.traintype_a)
            & (m.updown_w == m.updown_a) & (m.timing_w == m.timing_a))
    ck.internal("all 12,800 practice rows of the authors' withorder file are in the table with the same word, response, codes and group",
                len(m) == 12800 and bool(same.all()), f"{len(m)} rows matched, {int(same.sum())} identical")
    wblock = {(r.subj, r.sess_i, r.n_i): (int(r.block), r.word) for r in W.itertuples(index=False)}

    # the /gd/ re-scoring table onto the matching tokens (participant, session, number in session, word)
    G["sess_i"] = G.session.astype(int)
    G["n_i"] = G["#_in_session"].astype(int)
    assert not G.duplicated(["Participant", "sess_i", "n_i", "word"]).any()
    gd_pos = {}
    for r in d[d.cluster == "gd"].itertuples():
        gd_pos.setdefault((r.subj, r.sess_i, r.n_i, r.word), []).append(r.Index)
    link, unlinked = {}, []
    for g in G.to_dict("records"):
        cands = gd_pos.get((g["Participant"], g["sess_i"], g["n_i"], g["word"]), [])
        if len(cands) == 1:
            link[cands[0]] = g
        elif len(cands) > 1:  # the same recording session twice in the file: take the token whose coarse code is the first coder's
            mt = [i for i in cands if code3(d.at[i, "cluster_acc"]) == code3(g["original coding"])]
            if len(mt) == 1:
                link[mt[0]] = g
            else:
                unlinked.append(g)
        else:
            unlinked.append(g)

    # ---- rows ------------------------------------------------------------------------------------------------------------
    rows, helper = [], []
    for r in d.itertuples():
        target = r.cluster in CL
        cacc = clean(r.cluster_acc)
        used = target and r.traintype in ("trained", "untrained")
        word = clean(r.word)
        word = "" if word in ("#na", "#N/A", "na") else word
        practice = r.sess_i in (1, 2)
        w = wblock.get((r.subj, r.sess_i, r.n_i))
        in_w = w is not None and w[1] == r.word
        if practice:
            blk = (f"Practice block {w[0]} of 10 (quintile {(w[0] + 1) // 2}), recording session {r.sess_i}" if in_w
                   else f"Practice, recording session {r.sess_i}")
        else:
            blk = f"{'R1' if r.sess_i in (3, 4) else 'R2'}, recording session {r.sess_i}"
        if not word:
            seg = ""
        elif target:
            seg = f"word-initial {r.cluster} cluster"
        elif r.traintype == "gen":
            seg = f"word onset {r.cluster} (cluster outside the eight target clusters)"
        else:
            seg = f"word onset {r.cluster} (filler item without a target cluster)"
        g = link.get(r.Index)
        dur = ""
        if g is not None:
            dv = pd.to_numeric(g["duration"], errors="coerce")
            dur = "" if pd.isna(dv) else float(dv)
        rows.append(dict(
            dataset=ID, study=SRC, experiment="",
            group=f"{'sham' if r.updown == 'up' else 'active'} tDCS, {r.timing} practice [updown={r.updown}]",
            subject=r.subj, person=r.subj, subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy",
            setting="lab", device=DEVICE, session=1 if r.sess_i <= 4 else 2, block=blk, trial=int(r.trial),
            phase="learning" if practice else "test", time_s="", task="speech", unit="utterance", condition=r.traintype,
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=not used,
            stimulus=word, segment=seg, response_transcription=clean(r.response_IPA),
            word_correct=r.wordacc if r.wordacc in ("0", "1") else "", cluster_correct=r.clusacc, response_code=cacc,
            epenthetic_vowel_ms=dur,
            response_code_first_coder=clean(g["original coding"]) if g is not None else "",
            response_code_final=clean(g["accuracy"]) if g is not None else "",
            authors_trial_number=int(r.n_i)))
        helper.append(dict(subj=r.subj, sess_i=r.sess_i, practice=practice, target=target, used=used, cluster=r.cluster,
                           traintype=r.traintype, word=word, in_w=in_w, cacc_raw=r.cluster_acc, clusacc=r.clusacc,
                           linked=g is not None))
    h = pd.DataFrame(helper)
    df = pd.DataFrame(rows)

    # ---- internal checks ---------------------------------------------------------------------------------------------------
    ck.internal("every non-placeholder row of alldata_new.csv kept, in the file's order", len(df) == len(A) - 17 == 66051, f"{len(df)} rows")
    ck.internal("the practice rows the authors analysed (cluster items, trained) are exactly their 12,800 withorder rows",
                bool(((h.practice & h.used) == h.in_w).all()) and int((h.practice & h.used).sum()) == 12800,
                f"{int((h.practice & h.used).sum())} practice cluster rows")
    ck.internal("trial runs 1..n without gaps in each participant",
                all(sorted(g.trial) == list(range(1, len(g) + 1)) for _, g in df.groupby("subject")))
    ck.internal("the 5,439 /gd/ re-scored rows: every one attached to a token except the few that cannot be told apart",
                int(h.linked.sum()) + len(unlinked) == len(G) and len(unlinked) <= 5, f"{int(h.linked.sum())} linked, {len(unlinked)} not")
    ck.internal("re-scored rows only on cluster gd", bool((h[h.linked].cluster == "gd").all()))
    tg = h[h.target]
    ck.internal("among the eight target clusters, cluster_correct is 1 exactly on the rows coded c as typed",
                bool((tg.clusacc.eq("1") == tg.cacc_raw.eq("c")).all()))
    ev = df[df.epenthetic_vowel_ms != ""]
    ev_n = ev.epenthetic_vowel_ms.astype(float)
    ck.internal("inserted-vowel durations: 0 for final code c, positive for e, all within 0-101 ms",
                bool(ev_n[ev.response_code_final == "c"].eq(0).all() and ev_n[ev.response_code_final == "e"].gt(0).all()
                     and ev_n.between(0, 101).all()), f"{len(ev)} durations")

    # ---- published numbers -------------------------------------------------------------------------------------------------
    PAPER = "Buchwald et al. 2019, Cortex"
    per = df.drop_duplicates("subject")
    grp = per.groupby("group").size()
    ck.published("participants who completed the study", per.person.nunique(), 80, 0, f"{PAPER}, Methods, Participants")
    ck.published("participants in the smallest of the four stimulation groups", grp.min(), 20, 0,
                 f"{PAPER}, Methods, Participants (20 participants in each group)")
    ck.published("participants in the largest of the four stimulation groups", grp.max(), 20, 0,
                 f"{PAPER}, Methods, Participants (20 participants in each group)")
    pr = h[h.practice & h.target & (h.traintype == "trained")]
    ck.published("practice productions of cluster nonwords per participant (32 nonwords x 5), mean",
                 pr.groupby("subj").size().mean(), 160, 0, f"{PAPER}, Methods, Practice")
    ck.published("different cluster nonwords practised per participant, mean", pr[pr.word != ""].groupby("subj").word.nunique().mean(), 32, 0,
                 f"{PAPER}, Methods, Speech stimuli (4 nonwords per cluster x 8 clusters)")
    ck.published("practice filler items per participant, median",
                 h[h.practice & ~(h.target & (h.traintype == "trained"))].groupby("subj").size().median(), 62, 0,
                 f"{PAPER}, Methods, Practice (62 filler stimuli)")
    for lab, ss in (("R1", (3, 4)), ("R2", (5, 6))):
        t = h[h.sess_i.isin(ss) & h.target & h.traintype.isin(["trained", "untrained"])].groupby("subj").size()
        ck.published(f"{lab} cluster productions per participant, median", t.median(), 192, 0,
                     f"{PAPER}, Methods, Retention (192 total cluster stimuli)")
        f = h[h.sess_i.isin(ss) & ~(h.target & h.traintype.isin(["trained", "untrained"]))].groupby("subj").size()
        ck.published(f"{lab} filler items per participant, median", f.median(), 110, 0, f"{PAPER}, Methods, Retention (110 filler items)")
    ep = df[(df.phase == "learning") & df.segment.eq("word-initial gd cluster") & df.response_code_final.eq("e")]
    ck.published("epenthesis errors in practice /gd/ tokens, tDCS Before group", int(ep.group.str.startswith("active tDCS, before").sum()), 259, 0,
                 f"{PAPER}, Results, Epenthesis duration (Before, N=259)")
    ck.published("epenthesis errors in practice /gd/ tokens, tDCS During group", int(ep.group.str.startswith("active tDCS, during").sum()), 258, 0,
                 f"{PAPER}, Results, Epenthesis duration (During, N=258)")
    sham_n = int(ep.group.str.startswith("sham").sum())
    ck.published("epenthesis errors in practice /gd/ tokens, both sham groups together", sham_n, 485, 1,
                 f"{PAPER}, Results, Epenthesis duration (Sham, N=485)")
    pg = df[(df.phase == "learning") & df.response_code_final.ne("")]
    agree = ((pg.response_code_first_coder == "e") == (pg.response_code_final == "e")).mean() * 100
    allg = df[df.response_code_final.ne("")]
    agree_all = ((allg.response_code_first_coder == "e") == (allg.response_code_final == "e")).mean() * 100
    ck.published("two coders agree on whether a practice /gd/ token has an inserted vowel (%)", agree, 85.7, 0.05,
                 f"{PAPER}, Methods, Data analysis (85.7%)")
    for x in ck.items:
        if not x["ok"]:
            print("FAILED:", x)
    c_space = int((h.target & h.cacc_raw.str.strip().eq("c") & h.cacc_raw.ne("c")).sum())
    c_space_zero = int((h.target & h.cacc_raw.str.strip().eq("c") & h.cacc_raw.ne("c") & h.clusacc.eq("0")).sum())
    print(f"rows {len(df)}, persons {per.person.nunique()}, sessions with a second run {n_dup_runs}, gd linked {int(h.linked.sum())}, "
          f"unlinked {len(unlinked)}; sham epenthesis {sham_n}; agreement practice {agree:.4f}, all {agree_all:.4f}; "
          f"c codes with white space {c_space} (clusacc 0 on {c_space_zero})")

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Dropped the 17 placeholder rows of participants who did not complete the study (every cell na except their "
                 "stimulation group), joined the authors' /gd/ re-scoring table (all_gd_data.csv) onto the matching tokens, gave "
                 "practice items the block the authors list in their withorder file, removed white space from the production "
                 "codes, and replaced participant numbers with codes."),
        how=[
            "One row is one spoken item of one of the 80 participants in the authors' file alldata_new.csv, in the file's order: "
            "a nonword that starts with one of eight non-native consonant clusters (160 in practice, 192 in each retention test), "
            "or a filler item (62 in practice, 110 in each retention test). Practice (recording sessions 1 and 2 of the file) and "
            "the first retention test R1 (sessions 3 and 4), about 30 minutes later, took place on day 1 (session 1 here); the "
            "second retention test R2 (sessions 5 and 6) took place two days later (session 2 here).",
            "stimulus is the nonword and response_transcription what the participant said, in the authors' ASCII phonetic "
            "coding (@ is a schwa). word_correct and cluster_correct are the authors' scores for the whole nonword and for the "
            "first two consonants; response_code is their code for the cluster (c correct, e vowel inserted between the "
            "consonants, p vowel added before the cluster, s_c1 and s_c2 first or second consonant substituted, d_c1 and d_c2 "
            "deleted, o other, na not scored). For the /gd/ tokens that two coders re-scored, response_code_first_coder and "
            "response_code_final are the first and final three-way codes (c, e, o) and epenthetic_vowel_ms is the final "
            "duration of the inserted vowel in ms (0 when none was inserted, empty for other errors).",
            "group says whether the participant had sham or active tDCS and whether it came before or during practice; the "
            "authors' code is updown (up = sham, down = active, from their analysis code and confirmed by their counts of "
            "epenthesis errors per group) and timing. condition is the authors' traintype: trained and untrained cluster "
            "nonwords, gen for items with other clusters (ft, fp) and na for fillers without a cluster.",
            "trial is the chronological order within the participant (recording session, then the authors' number in the "
            "session; authors_trial_number), not the position in the file, which lists each participant's six recording sessions "
            "in scrambled order; for the two participants whose file holds some recording sessions twice, the run that comes "
            "first in the file is placed first. excluded_by_authors marks the items outside the authors' analyses (fillers and "
            "items with other clusters); practice items carry the authors' block (1-10) and quintile in block.",
        ],
        unit_note="spoken item (utterance) of one participant",
        differences=[
            "The paper's 485 epenthesis errors in the sham groups' practice /gd/ tokens counts only final codes typed exactly e; "
            "one token's final code is typed with trailing spaces (e plus spaces), which the authors' code does not count and "
            f"the cleaned code does, so {sham_n} here (Before 259 and During 258 agree).",
            f"The paper's 85.7% agreement between the two coders on whether a /gd/ token has an inserted vowel reproduces on the "
            f"1,600 practice /gd/ tokens ({agree:.2f}%); over all 5,439 re-scored /gd/ tokens it is {agree_all:.1f}%.",
            f"{c_space} cluster codes are typed with white space (c plus a space); the authors' cluster_correct is 0 on "
            f"{c_space_zero} of them, so those tokens are coded correct but scored incorrect in the shared file (kept as shared).",
            f"{len(unlinked)} of the 5,439 re-scored /gd/ rows could not be attached to a single token (two tokens of the same "
            "recording session with the same word and code) and are not in the table; the authors' R scripts name two files "
            "(alldata_new_withorder_new.csv, gd_new_compquint.csv) that are not in the deposit, so the retention analyses "
            "of the paper could not be recomputed.",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "speech", rows, meta, ck)


if __name__ == "__main__":
    main()
