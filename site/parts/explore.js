/* Explore (site/parts/explore.*): behaviour pooled across the standardized studies that match a term.
   Data: explore/index.json and explore/<family>.json, written by tools/explore_build.py (its first lines describe the format).
   Everything here is computed in the browser from group means. A visitor's own file is read in the browser and never sent. */
(function () {
  "use strict";
  var PARTS = window.OM_PARTS = window.OM_PARTS || {};
  /* the nav tab: added here, so it exists only on pages that include this part */
  (function () {
    var nav = document.querySelector(".site-nav");
    if (!nav || nav.querySelector('a[data-view="explore"]')) return;
    var a = document.createElement("a"); a.href = "#explore"; a.setAttribute("data-view", "explore"); a.textContent = "Explore";
    var std = nav.querySelector('a[data-view="standardization"]');
    nav.insertBefore(a, std ? std.nextSibling : null);
  })();

  /* STATS-BEGIN: plain statistics, checked against metafor 4.6 (rma method "DL", robust(), see the build notes) */
  var STATS = (function () {
    function gammln(x) {
      var c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
      var y = x, tmp = x + 5.5, ser = 1.000000000190015;
      tmp -= (x + 0.5) * Math.log(tmp);
      for (var j = 0; j < 6; j++) ser += c[j] / ++y;
      return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
    function gammq(a, x) {  /* regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), accurate for tiny values */
      if (x <= 0) return 1;
      var gln = gammln(a), n, ap, sum, del;
      if (x < a + 1) {
        ap = a; sum = del = 1 / a;
        for (n = 0; n < 500; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
        return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
      }
      var b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d, an;
      for (n = 1; n < 500; n++) {
        an = -n * (n - a); b += 2;
        d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d; del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break;
      }
      return Math.exp(-x + a * Math.log(x) - gln) * h;
    }
    function chi2p(q, df) { return df > 0 && q >= 0 ? gammq(df / 2, q / 2) : NaN; }  /* upper tail */
    function betacf(a, b, x) {
      var qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap, m, m2, aa, del, h;
      if (Math.abs(d) < 1e-300) d = 1e-300;
      d = 1 / d; h = d;
      for (m = 1; m <= 500; m++) {
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d; del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break;
      }
      return h;
    }
    function betai(a, b, x) {  /* regularized incomplete beta I_x(a, b) */
      if (x <= 0) return 0; if (x >= 1) return 1;
      var bt = Math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * Math.log(x) + b * Math.log(1 - x));
      return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
    }
    function tcdf(t, df) { var p = 0.5 * betai(df / 2, 0.5, df / (df + t * t)); return t > 0 ? 1 - p : p; }
    function tq975(df) {  /* 97.5% quantile of Student's t */
      if (!(df > 0)) return NaN;
      var lo = 0, hi = 1000;
      for (var i = 0; i < 200; i++) { var mid = (lo + hi) / 2; if (tcdf(mid, df) < 0.975) lo = mid; else hi = mid; }
      return (lo + hi) / 2;
    }
    var Z975 = 1.959963984540054;
    function normp2(z) { return chi2p(z * z, 1); }  /* two-sided normal p-value */
    /* DerSimonian-Laird random-effects meta-analysis. y: estimates, v: sampling variances, cl: cluster (study) of each.
       Returns the pooled estimate with its 95% CI, tau^2, I^2, Q, and a CI clustered by study (robust variance with the
       small-sample factor m/(m-1) and t on m-1 df, as metafor's robust(..., adjust = TRUE)). */
    function dl(y, v, cl) {
      var k = y.length, i;
      if (k < 1) return null;
      var w = v.map(function (x) { return 1 / x; });
      var sw = 0, sw2 = 0, swy = 0;
      for (i = 0; i < k; i++) { sw += w[i]; sw2 += w[i] * w[i]; swy += w[i] * y[i]; }
      var fe = swy / sw, Q = 0;
      for (i = 0; i < k; i++) Q += w[i] * (y[i] - fe) * (y[i] - fe);
      var df = k - 1, C = sw - sw2 / sw, tau2 = k > 1 ? Math.max(0, (Q - df) / C) : 0;
      var ws = v.map(function (x) { return 1 / (x + tau2); });
      var s = 0, sy = 0;
      for (i = 0; i < k; i++) { s += ws[i]; sy += ws[i] * y[i]; }
      var mu = sy / s, se = Math.sqrt(1 / s);
      var out = { k: k, est: mu, se: se, lo: mu - Z975 * se, hi: mu + Z975 * se, p: normp2(mu / se), tau2: tau2, Q: Q, df: df,
                  Qp: df > 0 ? chi2p(Q, df) : NaN, I2: df > 0 && Q > 0 ? Math.max(0, (Q - df) / Q) * 100 : 0 };
      if (cl) {
        var u = {}, m = 0;
        for (i = 0; i < k; i++) { if (!(cl[i] in u)) { u[cl[i]] = 0; m++; } u[cl[i]] += ws[i] * (y[i] - mu); }
        out.m = m;
        if (m >= 2) {
          var ss = 0; for (var c in u) ss += u[c] * u[c];
          var seR = Math.sqrt(m / (m - 1) * ss) / s, tq = tq975(m - 1);
          out.seR = seR; out.loR = mu - tq * seR; out.hiR = mu + tq * seR; out.dfR = m - 1;
          out.pR = 2 * (1 - tcdf(Math.abs(mu / seR), m - 1));
        }
      }
      return out;
    }
    /* difference between two separately pooled groups (subgroup test, Q-between on 1 df) */
    function diff(a, b) {
      if (!a || !b) return null;
      var d = a.est - b.est, se = Math.sqrt(a.se * a.se + b.se * b.se), Qb = d * d / (se * se);
      return { est: d, se: se, lo: d - Z975 * se, hi: d + Z975 * se, Q: Qb, p: chi2p(Qb, 1) };
    }
    /* equal-weight mean of lines at one trial, with a 95% CI clustered by study (same sandwich as above with equal weights) */
    function meanCl(y, cl) {
      var k = y.length, i, s = 0;
      if (!k) return null;
      for (i = 0; i < k; i++) s += y[i];
      var mu = s / k, u = {}, m = 0;
      for (i = 0; i < k; i++) { if (!(cl[i] in u)) { u[cl[i]] = 0; m++; } u[cl[i]] += y[i] - mu; }
      var out = { est: mu, k: k, m: m };
      if (m >= 2) { var ss = 0; for (var c in u) ss += u[c] * u[c]; var se = Math.sqrt(m / (m - 1) * ss) / k, tq = tq975(m - 1); out.lo = mu - tq * se; out.hi = mu + tq * se; }
      return out;
    }
    function percentile(vals, x) {  /* share of values below x, ties counted half, in percent */
      if (!vals.length) return NaN;
      var below = 0;
      vals.forEach(function (v) { below += v < x ? 1 : v === x ? 0.5 : 0; });
      return 100 * below / vals.length;
    }
    return { dl: dl, diff: diff, meanCl: meanCl, percentile: percentile, tq975: tq975, tcdf: tcdf, chi2p: chi2p };
  })();
  /* STATS-END */

  /* COMPARE-BEGIN: a visitor's file, read in the browser and turned into one line exactly as tools/explore_build.py
     (with tools/beta_build.py) turns a group of a standardized dataset into a line. Nothing here touches the page. */
  var CMP = (function () {
    var P = { pre: 20, maxX: 400, minFb: 30, lateN: 20, early: [1, 11], aeN: 10, smooth: 5, boot: 1000, halfSeMin: 0.29, minPeople: 3 };
    function splitLine(line, sep) {
      if (line.indexOf("\"") < 0) return line.split(sep);
      var out = [], cur = "", q = false;
      for (var i = 0; i < line.length; i++) {
        var ch = line.charAt(i);
        if (q) { if (ch === "\"") { if (line.charAt(i + 1) === "\"") { cur += "\""; i++; } else q = false; } else cur += ch; }
        else if (ch === "\"") q = true;
        else if (ch === sep) { out.push(cur); cur = ""; }
        else cur += ch;
      }
      out.push(cur);
      return out;
    }
    function readTable(text) {  /* {cols, rows}: strings, header in the first line; comma, semicolon or tab separated */
      if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
      var lines = text.split(/\r?\n/), head = lines[0] || "";
      var sep = [",", "\t", ";"].map(function (s) { return [s, head.split(s).length]; }).sort(function (a, b) { return b[1] - a[1]; })[0][0];
      var cols = splitLine(head, sep).map(function (c) { return c.trim(); }), rows = [];
      for (var i = 1; i < lines.length; i++) { if (lines[i]) rows.push(splitLine(lines[i], sep)); }
      return { cols: cols, rows: rows };
    }
    function unzipCsv(buf) {  /* the data CSV of a ZIP (stored or deflated entries; needs DecompressionStream) */
      var dv = new DataView(buf), u8 = new Uint8Array(buf), eocd = -1;
      for (var i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 65557); i--) { if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; } }
      if (eocd < 0) return Promise.reject(new Error("not a ZIP file"));
      var n = dv.getUint16(eocd + 10, true), p = dv.getUint32(eocd + 16, true), best = null, dec = new TextDecoder();
      for (var k = 0; k < n; k++) {
        if (dv.getUint32(p, true) !== 0x02014b50) break;
        var method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true);
        var nl = dv.getUint16(p + 28, true), xl = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true), off = dv.getUint32(p + 42, true);
        var name = dec.decode(u8.subarray(p + 46, p + 46 + nl));
        var mainCsv = /^([^\/]+)\/\1\.csv$/.test(name);
        if (/\.(csv|tsv|txt)$/i.test(name) && !/(^|\/)(columns\.csv|README\.txt)$/i.test(name) && (!best || (mainCsv && !best.main) || (mainCsv === !!best.main && usize > best.usize))) best = { name: name, method: method, csize: csize, usize: usize, off: off, main: mainCsv };
        p += 46 + nl + xl + cl;
      }
      if (!best) return Promise.reject(new Error("no CSV file inside the ZIP"));
      var lo = best.off, start = lo + 30 + dv.getUint16(lo + 26, true) + dv.getUint16(lo + 28, true), data = u8.subarray(start, start + best.csize);
      if (best.method === 0) return Promise.resolve({ name: best.name, text: dec.decode(data) });
      if (best.method !== 8) return Promise.reject(new Error("unsupported compression in the ZIP"));
      if (typeof DecompressionStream === "undefined") return Promise.reject(new Error("this browser cannot unpack ZIP files; unzip it and add the CSV"));
      var stream = new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
      return new Response(stream).text().then(function (t) { return { name: best.name, text: t }; });
    }
    var OM = ["subject", "trial", "hand_angle", "rotation", "clamp", "fb_mode", "fb_visible"];
    function isOpenMotor(cols) { return OM.every(function (c) { return cols.indexOf(c) >= 0; }); }
    function guess(cols) {
      var low = cols.map(function (c) { return c.toLowerCase().replace(/[^a-z0-9]/g, ""); });
      function find(names) { for (var i = 0; i < names.length; i++) { var j = low.indexOf(names[i]); if (j >= 0) return String(j); } return ""; }
      return { subj: find(["subject", "participant", "participantid", "subj", "sub", "sn", "pid", "ppt", "id", "subjectid"]) || "0",
               trial: find(["trial", "trialnum", "trialnumber", "trialn", "tn", "trialno"]) || "1",
               hand: find(["handangle", "hand", "handdeg", "angle", "reachangle", "handdirection", "handdir", "error", "handerror"]) || "2",
               pert: find(["rotation", "rot", "perturbation", "pert", "clamp", "clampangle", "rotationdeg"]),
               fb: find(["fbvisible", "feedback", "fb", "cursorvisible", "visible"]), f1: "", sign: "om" };
    }
    function numv(s) { if (s == null) return NaN; s = String(s).trim(); if (s === "" || /^(na|nan|null|none)$/i.test(s)) return NaN; var v = +s; return isFinite(v) ? v : NaN; }
    function istrue(s) { s = String(s == null ? "" : s).trim().toLowerCase(); return s === "true" || s === "1" || s === "yes" || s === "t"; }
    function isNoFb(s) { s = String(s == null ? "" : s).trim().toLowerCase(); return s === "none" || s === "0" || s === "false" || s === "no" || s === "off" || s === "nofb" || s === "no feedback"; }
    function r1(v) { return isFinite(v) ? parseFloat(v.toFixed(1)) : null; }
    function round1even(x) {  /* numpy's round(x, 1): half to even */
      var y = x * 10, r = Math.round(y);
      if (Math.abs(y % 1) === 0.5) r = 2 * Math.round(y / 2);
      return r / 10 + 0;
    }
    function median(a) { if (!a.length) return NaN; a = a.slice().sort(function (p, q) { return p - q; }); var m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; }
    function meanOf(a) { var s = 0, n = 0; for (var i = 0; i < a.length; i++) if (isFinite(a[i])) { s += a[i]; n++; } return n ? s / n : NaN; }
    function stat(v) {
      v = v.filter(isFinite);
      if (v.length < P.minPeople) return null;
      var m = meanOf(v), ss = 0;
      v.forEach(function (x) { ss += (x - m) * (x - m); });
      return [+m.toFixed(2), +(Math.sqrt(ss / (v.length - 1)) / Math.sqrt(v.length)).toFixed(3), v.length];
    }
    function movavg(y) {
      var h = (P.smooth - 1) / 2, out = [];
      for (var i = 0; i < y.length; i++) {
        var s = 0, c = 0;
        for (var j = i - h; j <= i + h; j++) if (j >= 0 && j < y.length && isFinite(y[j])) { s += y[j]; c++; }
        out.push(c >= 3 ? s / c : NaN);
      }
      return out;
    }
    function half(curve, xs) {
      var fin = curve.filter(isFinite);
      if (fin.length < P.minFb) return NaN;
      var late = meanOf(fin.slice(-P.lateN));
      if (!(late > 0)) return NaN;
      var sm = movavg(curve);
      for (var i = 0; i < sm.length; i++) if (sm[i] >= late / 2) return xs[i] + 1;
      return NaN;
    }
    function rng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
    function mode(counts) { var best = null, bn = 0; Object.keys(counts).forEach(function (k) { if (counts[k] > bn) { bn = counts[k]; best = k; } }); return best; }

    /* recs: [{s: sequence, p: person, t: trial (integer), h: hand angle or NaN when not usable, pert: perturbation or NaN,
       none: true on no-feedback trials, mode: "rotated" | "clamped" | "veridical" | "none" | other}] of one group */
    function computeLine(recs) {
      var subs = [], bySub = {};
      recs.forEach(function (r) { if (!bySub[r.s]) { bySub[r.s] = []; subs.push(r.s); } bySub[r.s].push(r); });
      if (!subs.length) return { error: "No rows to use." };
      subs.forEach(function (s) {
        var rs = bySub[s].sort(function (a, b) { return a.t - b.t; }), first = null;
        for (var i = 0; i < rs.length; i++) if (isFinite(rs[i].pert) && rs[i].pert !== 0) { first = rs[i].pert; break; }
        var f = first == null ? 1 : -Math.sign(first);
        rs.forEach(function (r) { r.hn = f * r.h; r.pn = isFinite(r.pert) ? round1even(-f * r.pert) : NaN; });
      });
      var T = 0;
      recs.forEach(function (r) { if (r.t + 1 > T) T = r.t + 1; });
      if (T > 200000) return { error: "Trial numbers go above 200,000; check the trial column." };
      var byT = [];
      for (var t = 0; t < T; t++) byT.push([]);
      recs.forEach(function (r) { if (r.t >= 0) byT[r.t].push(r); });
      var sched = [], modes = [], nf = [], has = [];
      for (t = 0; t < T; t++) {
        var rs = byT[t], ps = [], mc = {}, nn = 0, hc = 0;
        rs.forEach(function (r) { if (isFinite(r.pn)) ps.push(r.pn); if (r.mode) mc[r.mode] = (mc[r.mode] || 0) + 1; if (r.none) nn++; if (isFinite(r.hn)) hc++; });
        var md = median(ps);
        sched.push(isFinite(md) ? round1even(md) : NaN); modes.push(mode(mc)); nf.push(rs.length > 0 && nn / rs.length >= 0.5); has.push(hc);
      }
      var start = -1;
      for (t = 0; t < T; t++) if (isFinite(sched[t]) && sched[t] !== 0) { start = t; break; }
      if (start < 0) return { error: "No perturbed trial was found. Check the perturbation column or give the perturbed trials." };
      var mode0 = modes[start], family = mode0 === "rotated" ? "rotation" : mode0 === "clamped" ? "clamp" : null;
      if (!family) return { error: "The first perturbed trial is neither a rotation nor a clamp (feedback mode \"" + mode0 + "\")." };
      var Pz = sched[start], end = start;
      for (t = start + 1; t < T; t++) {
        var v = sched[t];
        if (isFinite(v)) { if (v === Pz) { end = t; continue; } break; }
        if (nf[t] || has[t] === 0) continue;
        break;
      }
      var isFb = [], nfb = 0;
      for (t = 0; t < T; t++) isFb.push(isFinite(sched[t]) && sched[t] === Pz && !nf[t] && modes[t] === mode0);
      for (t = start; t <= end; t++) if (isFb[t]) nfb++;
      /* participants analysed: own first perturbation at the onset, same size through the block */
      var cons = subs.filter(function (s) {
        var rs = bySub[s], own = null, ok = true;
        rs.forEach(function (r) { if (isFinite(r.pn) && r.pn !== 0 && own === null) own = r.t; if (r.t >= start && r.t <= end && isFinite(r.pn) && r.pn !== Pz) ok = false; });
        return own === start && ok;
      });
      var inCons = {}, personOf = {};
      cons.forEach(function (s) { inCons[s] = 1; personOf[s] = bySub[s][0].p; });
      var hasA = [], hand = [];
      for (t = 0; t < T; t++) {
        var hs = byT[t].filter(function (r) { return inCons[r.s] && isFinite(r.hn); }).map(function (r) { return r.hn; });
        hasA.push(hs.length); hand.push(meanOf(hs));
      }
      var x0 = -Math.min(P.pre, start), y = [];
      for (var x = x0; x <= Math.min(end - start, P.maxX); x++) {
        t = start + x;
        if (nf[t] || (x >= 0 && !isFb[t]) || (x < 0 && isFinite(sched[t]) && sched[t] !== 0)) y.push(null);
        else y.push(isFinite(hand[t]) ? r1(hand[t]) : null);
      }
      /* feedback trials of the block, per sequence */
      var seqTr = {};
      cons.forEach(function (s) {
        seqTr[s] = bySub[s].filter(function (r) { return r.t >= start && r.t <= end && r.mode === mode0 && r.pn === Pz && isFinite(r.hn); });
      });
      function perPerson(valOf) {
        var acc = {}, order = [];
        cons.forEach(function (s) { var v = valOf(s); if (!isFinite(v)) return; var p = personOf[s]; if (!acc[p]) { acc[p] = []; order.push(p); } acc[p].push(v); });
        return order.map(function (p) { return meanOf(acc[p]); });
      }
      var m = {};
      if (nfb >= P.minFb) {
        var e = stat(perPerson(function (s) { var a = seqTr[s]; return a.length >= P.early[1] ? meanOf(a.slice(P.early[0], P.early[1]).map(function (r) { return r.hn; })) : NaN; }));
        var l = stat(perPerson(function (s) { var a = seqTr[s]; return a.length >= P.lateN ? meanOf(a.slice(-P.lateN).map(function (r) { return r.hn; })) : NaN; }));
        if (e) m.early = e;
        if (l) m.late = l;
      }
      t = end + 1;
      while (t < T && hasA[t] === 0 && t <= end + 2) t++;
      if (t < T && nf[t] && nfb >= P.minFb) {
        var run = [];
        while (t < T && nf[t] && run.length < P.aeN) { if (hasA[t] > 0) run.push(t); t++; }
        if (run.length >= 3) {
          var inRun = {}; run.forEach(function (q) { inRun[q] = 1; });
          var ae = stat(perPerson(function (s) { return meanOf(bySub[s].filter(function (r) { return inRun[r.t] && isFinite(r.hn) && r.none; }).map(function (r) { return r.hn; })); }));
          if (ae) m.ae = ae;
        }
      }
      if (m.late && m.late[0] > 2 * m.late[1]) {
        var cols = {}, persons = [], cell = {};
        cons.forEach(function (s) {
          var p = personOf[s];
          if (!cell[p]) { cell[p] = {}; persons.push(p); }
          seqTr[s].forEach(function (r) { cols[r.t] = 1; (cell[p][r.t] = cell[p][r.t] || []).push(r.hn); });
        });
        var tr = Object.keys(cols).map(Number).sort(function (a, b) { return a - b; }), xs = tr.map(function (q) { return q - start; });
        var Mx = persons.filter(function (p) { return Object.keys(cell[p]).length; }).map(function (p) { return tr.map(function (q) { return cell[p][q] ? meanOf(cell[p][q]) : NaN; }); });
        var curveOf = function (rowsIdx) { return tr.map(function (_, j) { return meanOf(rowsIdx.map(function (i) { return Mx[i][j]; })); }); };
        var all = Mx.map(function (_, i) { return i; }), h0 = half(curveOf(all), xs);
        if (isFinite(h0)) {
          var rand = rng(12345), hb = [];
          for (var b = 0; b < P.boot; b++) { var idx = all.map(function () { return Math.floor(rand() * Mx.length); }); var hv = half(curveOf(idx), xs); if (isFinite(hv)) hb.push(hv); }
          var se = null;
          if (hb.length >= 0.8 * P.boot) { var mu = meanOf(hb), ss = 0; hb.forEach(function (q) { ss += (q - mu) * (q - mu); }); se = +Math.max(Math.sqrt(ss / (hb.length - 1)), P.halfSeMin).toFixed(2); }
          m.half = [h0, se, Mx.length];
        }
      }
      var nAll = {}, nCons = {};
      subs.forEach(function (s) { nAll[bySub[s][0].p] = 1; });
      cons.forEach(function (s) { nCons[personOf[s]] = 1; });
      var n = Object.keys(nCons).length, why = null, word = family;
      if (cons.length < 0.9 * subs.length) why = "participants followed different " + word + " schedules";
      else if (nfb < P.minFb) why = "the first " + word + " changed after " + nfb + " feedback trial" + (nfb === 1 ? "" : "s") + " (fewer than " + P.minFb + ")";
      else if (n < P.minPeople) why = "fewer than " + P.minPeople + " participants";
      else if (!m.late) why = "fewer than " + P.minPeople + " participants with " + P.lateN + " usable feedback trials";
      return { family: family, size: +Pz.toFixed(2), x0: x0, y: y, m: m, n: n, nall: Object.keys(nAll).length, blk: end - start + 1, nfb: nfb, start: start, why: why };
    }

    /* the visitor's table and choices -> computeLine */
    function fromTable(c, s) {
      var cols = c.cols, recs = [];
      var ix = function (name) { return cols.indexOf(name); };
      if (c.om) {
        var I = {}; ["subject", "person", "trial", "hand_angle", "rotation", "clamp", "fb_mode", "fb_visible", "trial_kind", "excluded_by_authors", "subject_excluded_by_authors", "experiment", "group"].forEach(function (k) { I[k] = ix(k); });
        var e1 = I.experiment >= 0 ? (s.v1 != null ? s.v1 : null) : null, g2 = I.group >= 0 && s.v2 != null && s.v2 !== "\u0000all" ? (s.v2 === "(no group)" ? "" : s.v2) : null;
        var rows = c.rows.filter(function (r) { return (e1 == null || r[I.experiment] === e1) && (g2 == null || r[I.group] === g2); });
        var dropSub = {};
        if (I.subject_excluded_by_authors >= 0) rows.forEach(function (r) { if (istrue(r[I.subject_excluded_by_authors])) dropSub[r[I.subject]] = 1; });
        rows.forEach(function (r) {
          if (dropSub[r[I.subject]]) return;
          var t = numv(r[I.trial]); if (!isFinite(t)) return;
          var ex = I.excluded_by_authors >= 0 ? String(r[I.excluded_by_authors]).trim() : "False";
          var usable = (I.trial_kind < 0 || r[I.trial_kind] === "reach") && !(ex === "" || istrue(ex));
          var md = r[I.fb_mode] || null;
          recs.push({ s: r[I.subject], p: I.person >= 0 && r[I.person] !== "" ? r[I.person] : r[I.subject], t: Math.round(t), h: usable ? numv(r[I.hand_angle]) : NaN,
                      pert: md === "clamped" ? numv(r[I.clamp]) : numv(r[I.rotation]), none: r[I.fb_visible] === "none", mode: md });
        });
        var subset = [e1, g2 == null ? null : g2 === "" ? "no group" : "group " + g2].filter(function (v) { return v != null; }).join(", ");
        var res = computeLine(recs);
        if (!res.error) res.subset = subset;
        return res;
      }
      var si = +s.subj, ti = +s.trial, hi = +s.hand, pi = s.pert === "" || s.pert == null ? -1 : +s.pert, fi = s.fb === "" || s.fb == null ? -1 : +s.fb;
      var f1 = s.f1 === "" || s.f1 == null ? -1 : +s.f1, sign = s.sign === "pos" ? -1 : 1, typeMode = s.type === "clamp" ? "clamped" : "rotated";
      var size = numv(s.size), a = numv(s.a), b = numv(s.b);
      if (pi < 0 && !(isFinite(size) && size !== 0 && isFinite(a) && isFinite(b) && b >= a)) return { error: "Give the perturbation size and the first and last perturbed trials, or choose a perturbation column." };
      var tmin = Infinity;
      c.rows.forEach(function (r) { var t = numv(r[ti]); if (isFinite(t) && t < tmin) tmin = t; });
      var off = tmin < 0 ? -tmin : 0;
      c.rows.forEach(function (r) {
        if (f1 >= 0 && r[f1] !== s.v1) return;
        var t = numv(r[ti]); if (!isFinite(t)) return;
        var none = fi >= 0 && isNoFb(r[fi]);
        var pert = pi >= 0 ? numv(r[pi]) : (t >= a && t <= b ? size : 0);
        if (none) pert = NaN;
        recs.push({ s: r[si], p: r[si], t: Math.round(t + off), h: sign * numv(r[hi]), pert: pert, none: none,
                    mode: none ? "none" : isFinite(pert) && pert !== 0 ? typeMode : "veridical" });
      });
      var out = computeLine(recs);
      if (!out.error) { out.subset = f1 >= 0 ? cols[f1] + " = " + s.v1 : ""; out.start -= off; }
      return out;
    }

    /* families made by series_lines() in tools/explore_build.py (walking, saccades, force fields, speech): the same steps.
       recs: [{s: sequence, p: person, t: order, v: value or NaN, st: "base" | "pert" | "post" | "other", size}];
       o: {meta: the family entry, base: subtract the baseline mean, baseN: baseline units used (0 = all), scale} */
    function winMean(y, x, full, post, w, L) {
      var v = [], need, i;
      if (w[0] === "minus") return winMean(y, x, full, post, w[1], L) - winMean(y, x, full, post, w[2], L);
      if (w[0] === "x") { for (i = 0; i < y.length; i++) if (x[i] >= w[1] && x[i] < w[2]) v.push(y[i]); need = w.length > 3 ? w[3] : Math.max(1, Math.floor((w[2] - w[1]) / 2)); }
      else if (w[0] === "last") { for (i = 0; i < y.length; i++) if (x[i] >= L - w[1]) v.push(y[i]); need = w.length > 2 ? w[2] : Math.max(1, Math.floor(w[1] / 2)); }
      else if (w[0] === "full_first" || w[0] === "full_last") {
        if (w[0] === "full_first" && full.length && !full[0]) return NaN;  /* a ramp came first: no "early" at full size */
        var f = []; for (i = 0; i < y.length; i++) if (full[i] && isFinite(y[i])) f.push(y[i]);
        v = w[0] === "full_first" ? f.slice(0, w[1]) : f.slice(-w[1]); need = w.length > 2 ? w[2] : Math.max(1, Math.floor(w[1] / 2));
      } else if (w[0] === "post") { v = post.filter(isFinite).slice(0, w[1]); need = w.length > 2 ? w[2] : Math.max(1, Math.floor(w[1] / 2)); }
      v = v.filter(isFinite);
      return v.length >= need ? meanOf(v) : NaN;
    }
    function computeSeries(recs, o) {
      var M = o.meta, pre = M.pre || 0, maxX = M.maxX || 400, subs = [], by = {}, per = [];
      recs.forEach(function (r) { if (!by[r.s]) { by[r.s] = []; subs.push(r.s); } by[r.s].push(r); });
      if (!subs.length) return { error: "No rows to use." };
      subs.forEach(function (sq) {
        var rs = by[sq].sort(function (a, b) { return a.t - b.t; }), on = -1, i;
        for (i = 0; i < rs.length; i++) if (rs[i].st === "pert") { on = i; break; }
        if (on < 0) return;
        var end = on; while (end + 1 < rs.length && (rs[end + 1].st === "pert" || rs[end + 1].st === "probe")) end++;
        var base = rs.slice(0, on).filter(function (r) { return r.st === "base"; }).map(function (r) { return r.v; });
        var bw = o.baseN ? base.slice(-o.baseN) : base, b0 = o.ref === "first" ? rs[on].v : o.base ? meanOf(bw) : 0;
        if (!isFinite(b0)) return;
        var f = function (v) { return isFinite(v) ? (v - b0) * o.scale : NaN; };
        var blk = rs.slice(on, end + 1), size = 0;
        blk.forEach(function (r) { if (isFinite(r.size) && Math.abs(r.size) > size) size = Math.abs(r.size); });
        var k = end + 1; while (k < rs.length && rs[k].st !== "post" && rs[k].st !== "pert" && rs[k].st !== "base" && k <= end + 3) k++;
        var q = k; while (q < rs.length && rs[q].st === "post") q++;
        per.push({ p: rs[0].p, on: rs[on].t, y: blk.map(function (r) { return r.st === "pert" ? f(r.v) : NaN; }), full: blk.map(function (r) { return Math.abs(Math.abs(r.size) - size) <= 0.006; }),
                   post: rs.slice(k, q).map(function (r) { return f(r.v); }), base: pre ? bw.map(f).slice(-pre) : [] });
      });
      if (!per.length) return { error: "No participant has perturbed " + (M.xunit || "trial") + "s" + (o.base ? " after baseline " + (M.xunit || "trial") + "s" : "") + ". Check the perturbation choices." };
      var lens = per.map(function (r) { return r.y.length; }), L = median(lens), end = Math.min(L, maxX + 1), y = [], x, i;
      var bmean = [];
      for (x = 0; x < pre; x++) bmean.push(meanOf(per.map(function (r) { var j = x - (pre - r.base.length); return j >= 0 ? r.base[j] : NaN; })));
      var first = 0; while (first < pre && !isFinite(bmean[first])) first++;
      for (x = first; x < pre; x++) y.push(isFinite(bmean[x]) ? +bmean[x].toFixed(M.yround) : null);
      for (x = 0; x < end; x++) { var mv = meanOf(per.map(function (r) { return x < r.y.length ? r.y[x] : NaN; })); y.push(isFinite(mv) ? +mv.toFixed(M.yround) : null); }
      var m = {};
      M.measures.forEach(function (ms) {
        var acc = {}, order = [];
        per.forEach(function (r) {
          var xs = r.y.map(function (_, j) { return j; }), v = winMean(r.y, xs, r.full, r.post, ms.w, r.y.length);
          if (!isFinite(v)) return;
          if (!acc[r.p]) { acc[r.p] = []; order.push(r.p); } acc[r.p].push(v);
        });
        var v = order.map(function (p) { return meanOf(acc[p]); }).filter(isFinite);  /* as stat() above, not rounded */
        if (v.length < P.minPeople) return;
        var mu = meanOf(v), ss = 0; v.forEach(function (q) { ss += (q - mu) * (q - mu); });
        m[ms.key] = [mu, Math.sqrt(ss / (v.length - 1)) / Math.sqrt(v.length), v.length];
      });
      var people = {}, all = {};
      per.forEach(function (r) { people[r.p] = 1; });
      recs.forEach(function (r) { all[r.p] = 1; });
      var n = Object.keys(people).length, why = null, unit = M.xunit || "trial", need = (M.measures.filter(function (q) { return q.key === "late"; })[0] || M.measures[0]);
      if (L < (M.minBlock || 1)) why = "the perturbation block has " + L + " " + unit + "s (fewer than " + M.minBlock + ")";
      else if (n < P.minPeople) why = "fewer than " + P.minPeople + " participants";
      else if (!m[need.key]) why = "fewer than " + P.minPeople + " participants with enough usable " + unit + "s for the " + need.label.toLowerCase();
      var nfb = median(per.map(function (r) { return r.y.filter(isFinite).length; }));
      return { family: M.id, size: o.size, x0: -(pre - first), y: y, m: m, n: n, nall: Object.keys(all).length, blk: L, nfb: nfb, start: per[0].on, why: why, series: true };
    }
    function guessSeries(cols, meta) {
      var low = cols.map(function (c) { return c.toLowerCase().replace(/[^a-z0-9]/g, ""); });
      function find(names) { for (var i = 0; i < names.length; i++) { var j = low.indexOf(names[i].replace(/[^a-z0-9]/g, "")); if (j >= 0) return String(j); } return ""; }
      var g = guess(cols);
      return { subj: g.subj, trial: find(["trial", "stride", "stridenumber", "trialnumber", "trialnum", "utterance", "order", "n"]) || g.trial,
               val: find((meta.cmp && meta.cmp.guess) || []) || "2", pert: find(["pert_value", "perturbation", "pert", "ratio", "step", "field", "shift", "size"]),
               f1: "", size: "", a: "", b: "", dir: "back", shift: "" };
    }
    /* the visitor's table and choices -> computeSeries */
    function fromSeries(c, s, meta) {
      var C = meta.cmp || {}, cols = c.cols, recs = [];
      var si = +s.subj, ti = +s.trial, vi = +s.val, pi = s.pert === "" || s.pert == null ? -1 : +s.pert, f1 = s.f1 === "" || s.f1 == null ? -1 : +s.f1;
      var ex = cols.indexOf("excluded_by_authors"), sx = cols.indexOf("subject_excluded_by_authors"), pc = cols.indexOf("person");
      var size = numv(s.size), a = numv(s.a), b = numv(s.b), scale = C.scale || 1;
      if (pi < 0 && !(isFinite(a) && isFinite(b) && b >= a)) return { error: "Give the first and last perturbed " + (meta.xunit || "trial") + ", or choose a perturbation column." };
      if (C.ask === "dir") scale *= s.dir === "back" ? -1 : 1;
      if (C.ask === "shift") { var sh = numv(s.shift); if (!isFinite(sh) || sh === 0) return { error: "Give the shift, signed, in the unit of the formant column." }; scale = -100 / sh; }
      var dropSub = {};
      if (sx >= 0) c.rows.forEach(function (r) { if (istrue(r[sx])) dropSub[r[si]] = 1; });
      var rows = c.rows.filter(function (r) { return !dropSub[r[si]] && (f1 < 0 || r[f1] === s.v1) && isFinite(numv(r[ti])); });
      var by = {};
      rows.forEach(function (r) { (by[r[si]] = by[r[si]] || []).push(r); });
      Object.keys(by).forEach(function (sq) {
        var rs = by[sq].sort(function (p, q) { return numv(p[ti]) - numv(q[ti]); });
        var pv = rs.map(function (r) { var t = numv(r[ti]), q = pi >= 0 ? numv(r[pi]) : (t >= a && t <= b ? (isFinite(size) ? size : 1) : 0); return isFinite(q) ? q : 0; });
        /* the first block: from the first perturbed row to the last one before a run of more than 15 unperturbed rows
           (shorter runs inside it, such as catch or no-feedback trials, are probes without a value) */
        var on = pv.findIndex(function (q) { return q !== 0; }), end = on;
        if (on >= 0) for (var i = on + 1; i < pv.length; i++) { if (pv[i] !== 0) { if (i - end > 16) break; end = i; } }
        var again = false;  /* a later perturbation: the rows after it are neither baseline nor aftereffect */
        rs.forEach(function (r, i) {
          if (on >= 0 && i > end && pv[i] !== 0) again = true;
          var st = on < 0 || i < on ? "base" : i <= end ? (pv[i] !== 0 ? "pert" : "probe") : again ? "other" : "post";
          var v = numv(r[vi]);
          if (ex >= 0 && istrue(r[ex])) v = NaN;
          recs.push({ s: sq, p: pc >= 0 && r[pc] !== "" ? r[pc] : sq, t: numv(r[ti]), v: v, st: st, size: pv[i] });
        });
      });
      var out = computeSeries(recs, { meta: meta, base: !!C.base, baseN: C.baseN || 0, ref: C.ref, scale: scale, size: isFinite(size) ? Math.abs(size) : null });
      if (!out.error) { if (out.size == null) out.size = NaN; out.subset = f1 >= 0 ? cols[f1] + " = " + s.v1 : ""; }
      return out;
    }
    return { readTable: readTable, unzipCsv: unzipCsv, isOpenMotor: isOpenMotor, guess: guess, computeLine: computeLine, fromTable: fromTable,
             computeSeries: computeSeries, guessSeries: guessSeries, fromSeries: fromSeries };
  })();
  /* COMPARE-END */

  var BASE = "explore/";
  var MINK = 3;  /* lines needed to draw the pooled curve at a trial (from at least 2 studies) */
  var MINM_BAND = 3;  /* studies needed to draw its confidence band (with fewer, t on m-1 df makes it meaningless) */
  function drawn(r) { return r && r.k >= MINK && r.m >= 2; }
  var TAG_FACETS = ["manipulation", "population", "topic", "setting", "device", "measure", "design"];
  var FACET_NAME = { manipulation: "method", population: "population", topic: "topic", setting: "setting", device: "device", measure: "measure", design: "design" };
  /* tags whose groups are coded in the data: only the matching groups are linked */
  function attrRule(key, val, text) {
    return { known: function (l) { return l.a[key] != null && l.a[key] !== "unknown"; }, test: function (l) { return l.a[key] === val; }, text: text };
  }
  /* group names, for studies whose data have no ages: "older", "OA", "older_65_69", "elderly" against "young", "YA" */
  function nameHas(l, re) { return re.test(" " + String(l.label || "").replace(/[_\-]/g, " ").toLowerCase() + " "); }
  var OLD_NAME = /[^a-z](older|elderly|oa|old)(?![a-z])/, YOUNG_NAME = /[^a-z](young|younger|ya|adult|adults)(?![a-z])/, CHILD_NAME = /[^a-z](child|children|kids?|adolescents?)(?![a-z])/;
  var LINE_RULES = {
    "population-older-adults": { known: function (l) { return l.a.age != null; }, test: function (l) { return l.a.age >= 60; }, text: "groups with a median age of 60 or more",
      byName: function (l) { return nameHas(l, CHILD_NAME) ? false : nameHas(l, OLD_NAME) ? true : nameHas(l, YOUNG_NAME) ? false : null; }, nameText: "groups named as older adults (the data have no ages)" },
    "population-children": { known: function (l) { return l.a.age != null; }, test: function (l) { return l.a.age < 18; }, text: "groups with a median age under 18",
      byName: function (l) { return nameHas(l, CHILD_NAME) ? true : nameHas(l, OLD_NAME) || nameHas(l, YOUNG_NAME) ? false : null; }, nameText: "groups named as children (the data have no ages)" },
    "population-parkinsons-disease": attrRule("pop", "parkinsons", "groups coded as Parkinson's disease"),
    "population-cerebellar-degeneration": attrRule("pop", "cerebellar_degeneration", "groups coded as cerebellar degeneration"),
    "population-low-vision": attrRule("pop", "low_vision", "groups coded as low vision"),
    "population-deafferentation": attrRule("pop", "deafferented", "groups coded as deafferented"),
    "setting-online": attrRule("setting", "online", "groups tested online"),
    "setting-lab": attrRule("setting", "lab", "groups tested in the lab"),
    "device-mouse": attrRule("device", "mouse", "groups who used a mouse"),
    "device-trackpad": attrRule("device", "trackpad", "groups who used a trackpad"),
    "device-mouse-or-trackpad": attrRule("device", "mouse_or_trackpad", "groups who used a mouse or trackpad"),
    "device-touchscreen": attrRule("device", "touchscreen", "groups who used a touchscreen"),
    "device-digitizing-tablet": attrRule("device", "tablet", "groups who used a digitizing tablet"),
    "device-kinarm": attrRule("device", "kinarm", "groups who used a KINARM"),
    "device-robot": attrRule("device", "robot", "groups who used a robot")
  };
  var ICON_DL = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>';
  var ICON_UP = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 13v8"/><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="m8 17 4-4 4 4"/></svg>';

  var YD = 1;  /* decimals of y values in the family on screen (dec in its entry) */
  function td(step) { return step >= 1 ? 0 : Math.max(1, Math.ceil(-Math.log10(step) - 1e-9)); }  /* tick decimals */
  var S = { fam: "", term: "", size: "all", unit: "", measure: "late", win: 0, q: "", sort: "count" };
  var IDX = null, FAMS = {}, PAGE = null, USER = null, CMPS = null, CUR = null, active = false, renderToken = 0;

  function $(s) { return document.querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function fmtInt(n) { return Math.round(n).toLocaleString("en-US"); }
  function mb(b) { return b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1e3)) + " kB"; }
  function num(v, d) { if (v == null || !isFinite(v)) return "-"; var s = v.toFixed(d); return s === "-0" || /^-0\.0*$/.test(s) ? s.slice(1) : s; }
  function plural(n, w, ws) { return fmtInt(n) + " " + (n === 1 ? w : (ws || w + "s")); }
  function pText(p) { return !isFinite(p) ? "-" : p < 0.001 ? "p < 0.001" : "p = " + (p < 0.01 ? p.toFixed(3) : p.toFixed(2)); }

  /* ---------- page data (studies, files) ---------- */
  function page() {
    if (PAGE) return PAGE;
    var D = JSON.parse(document.getElementById("catalog-data").textContent);
    PAGE = { study: {}, file: {} };
    D.studies.forEach(function (s) { PAGE.study[s.id] = s; });
    D.files.forEach(function (f) { PAGE.file[f.name] = f; });
    return PAGE;
  }
  function studyLabel(sid) { var s = page().study[sid]; return s ? s.label : sid; }
  function studyName(sid) { return studyLabel(sid).replace(/ (\d{4})\b/g, " ($1)"); }
  function shortName(l) {  /* "Clayton 2024, Exp 1, Kids" for narrow screens */
    var sl = studyLabel(l.study).replace(/ et al\.| & [^0-9]+?(?= \d{4})/g, "");
    return sl + ", " + lineName(l, true).replace(/\bExperiment\b/g, "Exp");
  }
  function lineName(l, short) {
    var lab = l.label && l.label !== "All participants" ? l.label : "";
    var el = l.el && l.el !== "Experiment" ? l.el : "";
    return (short ? "" : studyLabel(l.study) + ", ") + [el, lab].filter(Boolean).join(", ") || (short ? "All participants" : studyLabel(l.study));
  }

  /* ---------- loading ---------- */
  function getJSON(u) { return fetch(u).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }); }
  function loadIndex() { return IDX ? Promise.resolve(IDX) : getJSON(BASE + "index.json").then(function (j) { IDX = j; return j; }); }
  function famMeta(id) { return (IDX.families || []).filter(function (f) { return f.id === id; })[0]; }
  function loadFam(id) {
    if (FAMS[id]) return Promise.resolve(FAMS[id]);
    var meta = famMeta(id);
    return getJSON(BASE + meta.file).then(function (j) { FAMS[id] = prep(meta, j); return FAMS[id]; });
  }

  /* ---------- a family: lines and terms ---------- */
  function prep(meta, j) {
    var F = { meta: meta, lines: j.lines, studies: j.studies, mined: j.terms || [] };
    F.byStudy = {};
    F.lines.forEach(function (l) { (F.byStudy[l.study] = F.byStudy[l.study] || []).push(l); });
    F.pooled = F.lines.filter(function (l) { return l.ok; });
    F.terms = buildTerms(F);
    F.termBy = {};
    F.terms.forEach(function (t) { F.termBy[t.slug] = t; });
    return F;
  }
  function tagText(t) { return t.f === "topic" ? t.v : t.f + ": " + t.v; }
  function mentionText(sid, v, F) {
    var s = page().study[sid], title = s ? s.papers.map(function (p) { return p.title; }).join(" ").toLowerCase() : "";
    var inTitle = v[2].some(function (w) { return title.indexOf(w) >= 0; });
    var where = !F.studies[sid] || !F.studies[sid].abstract ? "the title" : inTitle ? "the title and abstract" : "the abstract";
    return "Mentioned " + plural(v[1], "time") + " in " + where + " (" + v[2].map(function (w) { return "\"" + w + "\""; }).join(", ") + "); tf-idf " + v[0].toFixed(3);
  }
  function buildTerms(F) {
    var P = page(), tags = {}, out = [];
    Object.keys(F.byStudy).forEach(function (sid) {
      var s = P.study[sid];
      if (!s) return;
      (s.tags || []).forEach(function (t) {
        if (TAG_FACETS.indexOf(t.f) < 0) return;
        var T = tags[t.slug] || (tags[t.slug] = { slug: t.slug, label: t.v, facet: t.f, kind: "tag", lines: [], how: {} });
        var ls = F.byStudy[sid], rule = LINE_RULES[t.slug], pick = ls, how = "Tag \"" + tagText(t) + "\"";
        if (rule) {
          var dec = ls.map(function (l) { return rule.known(l) ? rule.test(l) : rule.byName ? rule.byName(l) : null; });
          var byName = ls.some(function (l, i) { return !rule.known(l) && dec[i] !== null; });
          if (dec.some(function (d) { return d !== null; })) {
            pick = ls.filter(function (l, i) { return dec[i] === true; });
            how += ", only " + (byName && !ls.some(rule.known) ? rule.nameText : rule.text);
          } else how += "; the data do not code its groups, so every group is included";
        }
        if (!pick.length) return;
        T.lines = T.lines.concat(pick);
        T.how[sid] = how;
      });
    });
    Object.keys(tags).forEach(function (k) { out.push(tags[k]); });
    F.mined.forEach(function (m) {
      var T = { slug: tags[m.slug] ? m.slug + "-words" : m.slug, label: m.label, kind: "text", lines: [], how: {}, df: m.df };
      Object.keys(m.studies).forEach(function (sid) {
        if (!F.byStudy[sid]) return;
        T.lines = T.lines.concat(F.byStudy[sid]);
        T.how[sid] = mentionText(sid, m.studies[sid], F);
      });
      out.push(T);
    });
    var minS = (IDX.rules && IDX.rules.min_studies) || 3;
    out.forEach(function (T) {
      var st = {}, k = 0, n = 0;
      T.lines.forEach(function (l) { if (l.ok) { st[l.study] = 1; k++; n += l.n; } });
      T.nStudies = Object.keys(st).length; T.nLines = k; T.nPeople = n;
      T.ids = {};
      T.lines.forEach(function (l) { T.ids[l.id] = 1; });
    });
    return out.filter(function (T) { return T.nStudies >= minS; });
  }

  /* ---------- units and sizes ---------- */
  function unitOf(F) { return F.meta.pct && S.unit !== "deg" ? "pct" : "y"; }
  function scale(l, v) { return unitOf(CUR.F) === "pct" ? v / l.size * 100 : v; }
  function dec(F) { return F.meta.dec == null ? 1 : F.meta.dec; }  /* decimals of y values */
  function xu(F) { return F.meta.xunit || "trial"; }  /* what x counts: trial, stride */
  function Xu(F) { var w = xu(F); return w.charAt(0).toUpperCase() + w.slice(1); }
  function gapOf(F) { return F.meta.gap || GAP; }
  function unitWords(F) { return F.meta.unit === "deg" ? "degrees" : F.meta.unitName || F.meta.unit; }
  function suffix(F, kind) {
    if (kind === "trials") return "";
    if (unitOf(F) === "pct") return "%";
    return F.meta.unit === "deg" ? "°" : F.meta.unit === "%" ? "%" : F.meta.unit ? " " + F.meta.unit : "";
  }
  function yLabel(F) {
    return F.meta.y + (unitOf(F) === "pct" ? " (% of " + F.meta.size.toLowerCase() + ")" : F.meta.unit === "deg" ? " (degrees)" : F.meta.unit ? " (" + unitWords(F) + ")" : "");
  }
  function fv(F, v, kind, d) { return num(v, d == null ? (kind === "trials" ? 0 : dec(F)) : d) + suffix(F, kind); }
  function sizeTxt(F, v) { if (v == null) return "-"; var u = F.meta.sizeUnit; return num(v, v % 1 ? 1 : 0) + (u === "deg" ? "°" : u === "%" || u.charAt(0) === ":" ? u : " " + u); }
  function sizeOptions(F) {
    var opts = [["all", "All sizes", F.pooled.length]], cnt = {};
    (F.meta.sizeBins || []).forEach(function (b, i) {
      var n = F.pooled.filter(function (l) { return l.size >= b[0] && l.size < b[1]; }).length;
      var lab = b[0] === 0 ? "Under " + sizeTxt(F, b[1]) : i === F.meta.sizeBins.length - 1 ? sizeTxt(F, b[0]) + " or more" : sizeTxt(F, b[0]) + " to under " + sizeTxt(F, b[1]);
      if (n) opts.push(["r" + i, lab, n]);
    });
    F.pooled.forEach(function (l) { cnt[l.size] = (cnt[l.size] || 0) + 1; });
    var exact = Object.keys(cnt).map(Number).sort(function (a, b) { return a - b; }).map(function (v) { return ["s" + v, sizeTxt(F, v), cnt[v]]; });
    return { ranges: opts, exact: exact };
  }
  function sizeTest(F) {
    var v = S.size;
    if (v.charAt(0) === "r") { var b = (F.meta.sizeBins || [])[+v.slice(1)]; if (b) return function (l) { return l.size >= b[0] && l.size < b[1]; }; }
    if (v.charAt(0) === "s") { var x = +v.slice(1); return function (l) { return l.size === x; }; }
    return function () { return true; };
  }
  function sizeLabel(F) {
    var o = sizeOptions(F), all = o.ranges.concat(o.exact);
    for (var i = 0; i < all.length; i++) if (all[i][0] === S.size) return all[i][1];
    return "All sizes";
  }

  /* ---------- routing ---------- */
  window.addEventListener("hashchange", function () { if (!/^#explore(\/|$)/.test(location.hash)) active = false; });
  function go(fam, term) { location.hash = "#explore/" + fam + (term ? "/" + term : ""); }
  PARTS.explore = {
    title: "Explore", keepScroll: true,
    onShow: function (arg) {
      var was = active;
      active = true;
      var parts = (arg || "").split("/");
      var famId = parts[0] || "", term = parts.slice(1).join("/");
      if (!was) window.scrollTo(0, 0);
      var token = ++renderToken;
      loadIndex().then(function () {
        if (!famMeta(famId)) famId = IDX.families[0].id;
        return loadFam(famId);
      }).then(function (F) {
        if (token !== renderToken) return;
        if (S.fam !== F.meta.id) { S.size = "all"; S.win = 0; }
        S.fam = F.meta.id; S.term = F.termBy[term] ? term : "";
        if (!S.win) S.win = F.meta.window[1];
        render(F, was);
      }).catch(function (err) {
        $("#ex-main").innerHTML = "<p>The Explore data could not be loaded (" + esc(err.message || err) + "). The page needs the files in <code>explore/</code> next to it; open it on the public OpenMotor site.</p>";
      });
      return "Explore";
    }
  };

  /* ---------- rendering ---------- */
  function render(F, scrollToMain) {
    var T = S.term ? F.termBy[S.term] : null;
    var inSize = sizeTest(F);
    var A = (T ? T.lines : F.lines).filter(function (l) { return l.ok && inSize(l); });
    var B = T ? F.pooled.filter(function (l) { return inSize(l) && !T.ids[l.id]; }) : null;
    CUR = { F: F, T: T, A: A, B: B };
    YD = dec(F);
    document.title = (T ? T.label + " | " : "") + "Explore " + F.meta.label.toLowerCase() + " | OpenMotor";
    renderSide(F);
    var main = $("#ex-main");
    main.innerHTML = headHtml(F, T, A, B) + filtersHtml(F) + curveHtml(F, T) + metaHtml(F, T) + forestHtml(F) + studiesHtml(F, T) + downloadHtml(F, T) + compareHtml(F) + methodsHtml(F);
    bindMain(F);
    drawCurve();
    drawForest();
    renderUser();
    $("#ex-status").textContent = (T ? T.label + ": " : F.meta.label + ": ") + plural(A.length, "line") + " pooled.";
    if (scrollToMain) {  /* a term or family was picked on this page: bring its results into view (below the term list on phones) */
      var narrow = window.innerWidth <= 1000, top = (narrow ? main : $(".ex-layout")).getBoundingClientRect().top;
      if (top < 0 || narrow) window.scrollTo(0, Math.max(0, window.scrollY + top - 12));
    }
  }

  function renderSide(F) {
    $("#ex-fams").innerHTML = IDX.families.map(function (f) {
      var on = f.id === F.meta.id;
      return "<a href=\"#explore/" + esc(f.id) + "\"" + (on ? " class=\"active\" aria-current=\"true\"" : "") + ">" + esc(f.label) +
        " <span title=\"" + f.pooledStudies + " studies with pooled lines\">" + f.pooledStudies + "</span></a>";
    }).join("");
    renderTerms(F);
  }
  function termMatches(t, words) {
    var hay = (t.label + " " + (t.facet ? FACET_NAME[t.facet] : "words")).toLowerCase();
    return words.every(function (w) { return hay.indexOf(w) >= 0; });
  }
  function renderTerms(F) {
    var words = S.q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    var sorter = S.sort === "name" ? function (a, b) { return a.label.localeCompare(b.label); }
                                  : function (a, b) { return b.nStudies - a.nStudies || b.nLines - a.nLines || a.label.localeCompare(b.label); };
    function group(kind, title) {
      var list = F.terms.filter(function (t) { return t.kind === kind && termMatches(t, words); }).sort(sorter);
      if (!list.length) return "";
      return "<p class=\"ex-th\"><span>" + title + "</span><span>Studies</span><span>Lines</span></p><ul class=\"ex-tlist\">" + list.map(function (t) {
        var on = t.slug === S.term;
        return "<li><a href=\"#explore/" + esc(F.meta.id) + "/" + esc(t.slug) + "\"" + (on ? " class=\"active\" aria-current=\"true\"" : "") +
          " title=\"" + esc(t.label + ": " + t.nStudies + " studies, " + t.nLines + " pooled lines") + "\"><span class=\"ex-tl\">" + esc(t.label) +
          (t.facet ? " <small>" + esc(FACET_NAME[t.facet]) + "</small>" : "") + "</span><span class=\"ex-tn\">" + t.nStudies + "</span><span class=\"ex-tn\">" + t.nLines + "</span></a></li>";
      }).join("") + "</ul>";
    }
    var html = (S.term || words.length ? "" : "") + group("tag", "Tags") + group("text", "Words in titles and abstracts");
    $("#ex-terms").innerHTML = (S.term ? "<p class=\"ex-all\"><a href=\"#explore/" + esc(F.meta.id) + "\">All " + esc(F.meta.label.toLowerCase()) + " lines</a></p>" : "") +
      (html || "<p class=\"muted ex-none\">No term matches. Terms are listed when they link at least " + ((IDX.rules && IDX.rules.min_studies) || 3) + " studies with pooled lines.</p>");
    document.querySelectorAll("#ex-sort [data-sort]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.getAttribute("data-sort") === S.sort)); });
  }

  function headHtml(F, T, A, B) {
    var studiesA = {}, nA = 0;
    A.forEach(function (l) { studiesA[l.study] = 1; nA += l.n; });
    var sA = Object.keys(studiesA).length;
    var sizeNote = S.size === "all" ? "" : " (" + F.meta.size.toLowerCase() + " size: " + sizeLabel(F).toLowerCase() + ")";
    var h = "<nav class=\"crumbs\" aria-label=\"Breadcrumb\"><a href=\"#explore\">Explore</a><span aria-hidden=\"true\">/</span>" +
      (T ? "<a href=\"#explore/" + esc(F.meta.id) + "\">" + esc(F.meta.label) + "</a><span aria-hidden=\"true\">/</span><span>" + esc(T.label) + "</span>" : "<span>" + esc(F.meta.label) + "</span>") + "</nav>";
    if (!T) {
      h += "<h2 class=\"ex-h\">" + esc(F.meta.label) + "</h2><p class=\"ex-kind\">All lines of this kind of experiment</p>" +
        "<p class=\"ex-sum\">" + plural(A.length, "line") + " from " + plural(sA, "study", "studies") + ", " + plural(nA, "participant") + sizeNote + ". " +
        esc(F.meta.about) + " Choose a term to compare the studies it links with all the others.</p>";
    } else {
      var studiesB = {};
      B.forEach(function (l) { studiesB[l.study] = 1; });
      var notPooled = T.lines.filter(function (l) { return !l.ok; }).length;
      h += "<h2 class=\"ex-h\">" + esc(T.label) + "</h2><p class=\"ex-kind\">" + (T.kind === "tag" ? "Tag (" + esc(FACET_NAME[T.facet]) + ")" : "Term found in titles and abstracts") + "</p>" +
        "<p class=\"ex-sum\"><b>" + plural(A.length, "line") + "</b> from <b>" + plural(sA, "study", "studies") + "</b> (" + plural(nA, "participant") + ") of " + esc(F.meta.label.toLowerCase()) + sizeNote +
        ", compared with the other <b>" + plural(B.length, "line") + "</b> from " + plural(Object.keys(studiesB).length, "study", "studies") + ". " +
        (notPooled ? plural(notPooled, "more line") + " of these studies " + (notPooled === 1 ? "is" : "are") + " listed below but not pooled, with the reason. " : "") +
        (T.kind === "tag" ? "Linked through the study tags." : "Linked when the term's tf-idf weight in a study's titles and abstracts is at least " + IDX.rules.tfidf_min + ".") + "</p>";
    }
    return h;
  }

  function filtersHtml(F) {
    var o = sizeOptions(F);
    var sel = function (v) { return v === S.size ? " selected" : ""; };
    var h = "<div class=\"ex-filters\" role=\"group\" aria-label=\"Filters for everything below\">" + (F.meta.noSize ? "" :
      "<label class=\"st-filter\">" + esc(F.meta.sizeLabel || F.meta.size + " size") + " <select id=\"ex-size\">" +
      o.ranges.map(function (r) { return "<option value=\"" + r[0] + "\"" + sel(r[0]) + ">" + esc(r[1]) + " (" + r[2] + ")</option>"; }).join("") +
      "<optgroup label=\"One size\">" + o.exact.map(function (r) { return "<option value=\"" + r[0] + "\"" + sel(r[0]) + ">" + esc(r[1]) + " (" + r[2] + ")</option>"; }).join("") + "</optgroup></select></label>");
    if (F.meta.pct) h += "<label class=\"st-filter\">Units <select id=\"ex-unit\"><option value=\"pct\"" + (unitOf(F) === "pct" ? " selected" : "") + ">% of " + esc(F.meta.size.toLowerCase()) + "</option><option value=\"deg\"" + (unitOf(F) === "pct" ? "" : " selected") + ">" + esc(unitWords(F)) + "</option></select></label>";
    var maxX = F.meta.maxX || (IDX.rules && IDX.rules.max_x) || 400;
    h += "<label class=\"st-filter\">" + esc(F.meta.xname || "Trials") + " <select id=\"ex-win\">" + [F.meta.window[1], maxX].filter(function (v, i, a) { return a.indexOf(v) === i; }).map(function (v) {
      return "<option value=\"" + v + "\"" + (v === S.win ? " selected" : "") + ">" + F.meta.window[0] + " to " + v + "</option>";
    }).join("") + "</select></label></div>";
    return h;
  }

  function curveHtml(F, T) {
    return "<figure class=\"ex-fig\" id=\"ex-curve-fig\"><figcaption class=\"ex-figh\"><b>Learning curves</b><span>" + esc(F.meta.x) + "; " + xu(F) + "s before 0 are baseline</span></figcaption>" +
      "<div class=\"ex-legend\" id=\"ex-legend\"></div><div class=\"ex-chart\" id=\"ex-curve\"></div>" +
      "<p class=\"ex-note\">Thin lines: one group each. Thick line: the mean of the lines at each " + xu(F) + ", every line weighted equally, drawn where at least " + MINK + " lines from 2 or more studies contribute; the band is its 95% CI with lines from the same study treated as one cluster, drawn where " + MINM_BAND + " or more studies contribute. The strip under the curve counts the lines in the mean, which falls as shorter blocks end. Gaps in thin lines are " + (F.meta.xunit ? xu(F) + "s without a value" : "no-feedback trials") + "; for the mean, gaps of up to " + gapOf(F) + " " + xu(F) + "s are bridged by linear interpolation.</p>" +
      "<details class=\"ex-details\"><summary>Curve values as a table</summary><div class=\"table-wrap\" id=\"ex-curve-table\"></div></details></figure>";
  }

  /* ---------- meta-analysis ---------- */
  function metaFor(lines, key, F) {
    var y = [], v = [], cl = [], used = [];
    lines.forEach(function (l) {
      var m = l.m[key];
      if (!m || m[1] == null || !(m[1] > 0)) return;
      var f = key === "half" ? 1 : scale(l, 1);
      y.push(m[0] * f); v.push(Math.pow(m[1] * f, 2)); cl.push(l.study); used.push(l);
    });
    var r = y.length >= 2 ? STATS.dl(y, v, cl) : null;
    if (r) { r.lines = used; r.y = y; r.v = v; r.N = used.reduce(function (s, l) { return s + l.m[key][2]; }, 0); }
    return r;
  }
  function metaCell(F, r, kind, label) {
    var dl = " data-label=\"" + esc(label) + "\"";
    if (!r) return "<td class=\"ex-mc\"" + dl + "><span class=\"muted\">Fewer than 2 lines</span></td>";
    return "<td class=\"ex-mc\"" + dl + "><b>" + fv(F, r.est, kind, YD) + "</b> <span class=\"ex-ci\">[" + num(r.lo, YD) + ", " + num(r.hi, YD) + "]</span>" +
      "<small>" + (r.loR != null ? "clustered by study [" + num(r.loR, YD) + ", " + num(r.hiR, YD) + "]; " : "") +
      "k = " + r.k + " lines, " + plural(r.m, "study", "studies") + ", N = " + fmtInt(r.N) + "; τ² = " + num(r.tau2, YD > 1 ? 2 * YD : r.tau2 < 10 ? 2 : 1) + ", I² = " + num(r.I2, 0) + "%</small></td>";
  }
  function metaHtml(F, T) {
    var rows = F.meta.measures.map(function (ms) {
      var a = metaFor(CUR.A, ms.key, F), b = T ? metaFor(CUR.B, ms.key, F) : null;
      var kind = ms.unit === "trials" ? "trials" : "y";
      var h = "<tr><th scope=\"row\">" + esc(ms.label) + "<small>" + esc(ms.text) + "</small></th>" + metaCell(F, a, kind, T ? "This term" : "All lines");
      if (T) {
        h += metaCell(F, b, kind, "Other lines");
        var d = a && b && a.m >= 2 && b.m >= 2 ? STATS.diff(a, b) : null;
        if (!d && a && b) h += "<td class=\"ex-mc\" data-label=\"Difference\"><span class=\"muted\">Not tested: the lines of " + (a.m < 2 ? "this term" : "the other side") + " come from one study</span></td>";
        else h += d ? "<td class=\"ex-mc\" data-label=\"Difference\"><b>" + (d.est > 0 ? "+" : "") + fv(F, d.est, kind, 1) + "</b> <span class=\"ex-ci\">[" + num(d.lo, 1) + ", " + num(d.hi, 1) + "]</span><small>Q = " + num(d.Q, 2) + " on 1 df, " + pText(d.p) + "</small></td>"
               : "<td class=\"ex-mc\" data-label=\"Difference\"><span class=\"muted\">-</span></td>";
      }
      return h + "</tr>";
    }).join("");
    return "<h3 class=\"ex-h3\">Summary measures</h3>" +
      "<p class=\"ex-note ex-note-top\">Random-effects meta-analysis (DerSimonian-Laird) of the lines' means, each with its standard error from its participants. Brackets: 95% CI; the second interval treats lines from the same study as one cluster. " +
      (T ? "Difference: this term minus the other lines (subgroup test, Q-between on 1 df)." : "") + "</p>" +
      "<div class=\"table-card\"><div class=\"table-wrap\"><table class=\"data ex-meta\"><thead><tr><th scope=\"col\">Measure</th><th scope=\"col\">" + (T ? "This term" : "All lines") + "</th>" +
      (T ? "<th scope=\"col\">Other lines</th><th scope=\"col\">Difference</th>" : "") + "</tr></thead><tbody>" + rows + "</tbody></table></div></div>";
  }

  function forestHtml(F) {
    var tabs = F.meta.measures.map(function (ms) {
      return "<button type=\"button\" role=\"tab\" class=\"tab\" data-measure=\"" + esc(ms.key) + "\" aria-selected=\"" + (ms.key === S.measure) + "\">" + esc(ms.label) + "</button>";
    }).join("");
    return "<h3 class=\"ex-h3\">Forest plot</h3><div class=\"tabs ex-tabs\" role=\"tablist\" aria-label=\"Measure\">" + tabs + "</div>" +
      "<figure class=\"ex-fig\"><div class=\"ex-chart\" id=\"ex-forest\"></div>" +
      "<p class=\"ex-note\">Each square is one line's mean with its 95% CI (mean ± 1.96 SE); its area follows the line's weight in the random-effects model. Diamonds are the pooled estimates. Lines without this measure are left out. With more than 50 lines the rows are thin and sorted by value; point at a row to name it, and the study table lists every value.</p></figure>";
  }

  /* ---------- studies table ---------- */
  function studiesHtml(F, T) {
    var lines = T ? T.lines : F.lines, inSize = sizeTest(F);
    var by = {}, order = [];
    lines.forEach(function (l) { if (!by[l.study]) { by[l.study] = []; order.push(l.study); } by[l.study].push(l); });
    order.sort(function (a, b) { return studyLabel(a).localeCompare(studyLabel(b)); });
    var M = F.meta.measures;
    var body = order.map(function (sid) {
      var files = (F.studies[sid] || { ds: [] }).ds.map(function (d) { return page().file[d]; }).filter(Boolean);
      var dl = files.map(function (f) {
        return f.url ? "<a class=\"ex-zip\" href=\"" + esc(f.url) + "\">" + ICON_DL + "ZIP" + (files.length > 1 ? " " + esc(f.name.replace(sid, "").replace(/^_/, "") || f.name) : "") + " <small>" + mb(f.bytes) + "</small></a>" : "";
      }).join(" ");
      var head = "<tr class=\"ex-srow\"><th colspan=\"" + (M.length + (F.meta.noSize ? 2 : 3)) + "\" scope=\"rowgroup\"><a href=\"#study/" + encodeURIComponent(sid) + "\">" + esc(studyName(sid)) + "</a>" +
        (T ? "<span class=\"ex-how\">" + esc(T.how[sid] || "") + "</span>" : "") + "<span class=\"ex-dls\">" + dl + "</span></th></tr>";
      return "<tbody>" + head + by[sid].map(function (l) {
        var used = l.ok && inSize(l);
        var why = !l.ok ? "Not pooled: " + l.why : !inSize(l) ? "Outside the size filter" : "";
        return "<tr" + (used ? "" : " class=\"ex-off\"") + "><td>" + esc(lineName(l, true)) + (l.nall && l.ok ? "<small>" + l.n + " of the group's " + l.nall + " people had this schedule; the others are left out</small>" : "") + (why ? "<small>" + esc(why) + "</small>" : "") + "</td>" +
          "<td class=\"num\" data-label=\"n\">" + (l.ok ? l.n : l.nall || l.n) + "</td>" + (F.meta.noSize ? "" : "<td class=\"num\" data-label=\"" + esc(F.meta.size) + "\">" + sizeTxt(F, l.size) + "</td>") + M.map(function (ms) {
            var m = l.m[ms.key];
            if (!m) return "<td class=\"num ex-nil\" data-label=\"" + esc(ms.label) + "\">-</td>";
            var f = ms.unit === "trials" ? 1 : unitOf(F) === "pct" ? 100 / l.size : 1;
            return "<td class=\"num\" data-label=\"" + esc(ms.label) + "\">" + num(m[0] * f, ms.unit === "trials" ? 0 : YD) + (m[1] != null ? " <span class=\"ex-se\">± " + num(m[1] * f, YD) + "</span>" : "") + "</td>";
          }).join("") + "</tr>";
      }).join("") + "</tbody>";
    }).join("");
    var units = unitOf(F) === "pct" ? "% of the " + F.meta.size.toLowerCase() : unitWords(F);
    return "<h3 class=\"ex-h3\">Studies</h3><p class=\"ex-note ex-note-top\">" + plural(order.length, "study", "studies") + ", every line with its mean ± SE (in " + esc(units) + (M.some(function (ms) { return ms.unit === "trials"; }) ? "; trials to half in trials" : "") + "). Grey rows are not in the pooled results." +
      (T ? " Under each study: how it was linked to this term." : "") + "</p>" +
      "<div class=\"table-card\"><div class=\"table-wrap\"><table class=\"data ex-studies\"><thead><tr><th scope=\"col\">Line</th><th scope=\"col\" class=\"num\">n</th>" + (F.meta.noSize ? "" : "<th scope=\"col\" class=\"num\">" + esc(F.meta.size) + "</th>") +
      M.map(function (ms) { return "<th scope=\"col\" class=\"num\">" + esc(ms.label) + "</th>"; }).join("") + "</tr></thead>" + body + "</table></div></div>";
  }

  /* ---------- downloads ---------- */
  function subsetFiles(F, T) {
    var seen = {}, out = [];
    (T ? T.lines : F.lines).forEach(function (l) { var f = page().file[l.ds]; if (f && f.url && !seen[f.name]) { seen[f.name] = 1; out.push(f); } });
    return out.sort(function (a, b) { return a.name.localeCompare(b.name); });
  }
  function baseName(F, T) { return "openmotor-explore-" + F.meta.id + (T ? "-" + T.slug : ""); }
  function downloadHtml(F, T) {
    var files = subsetFiles(F, T), bytes = files.reduce(function (s, f) { return s + (f.bytes || 0); }, 0);
    var cmd = "curl -L --remote-name-all " + files.map(function (f) { return f.url; }).join(" ");
    return "<h3 class=\"ex-h3\">Download these data</h3><p class=\"ex-dltext\">The " + plural(files.length, "standardized dataset") + " of these studies (" + mb(bytes) + " in all), one ZIP each with the trial table, a README and columns.csv.</p>" +
      "<ul class=\"ex-dlist\"><li><button type=\"button\" class=\"linkbutton\" data-get=\"urls\">" + ICON_DL + "ZIP links as a text file</button> <span class=\"muted\">then <code>wget -i " + esc(baseName(F, T)) + "-zips.txt</code></span></li>" +
      "<li><button type=\"button\" class=\"linkbutton\" data-get=\"csv\">" + ICON_DL + "Line summaries as CSV</button> <span class=\"muted\">every line above with its measures, in " + esc(unitWords(F)) + (F.meta.pct ? " and in % of the " + esc(F.meta.size.toLowerCase()) : "") + "</span></li></ul>" +
      "<div class=\"ex-cmdbox\"><pre class=\"ex-cmd\" id=\"ex-cmd\" tabindex=\"0\" aria-label=\"curl command\">" + esc(cmd) + "</pre><button type=\"button\" class=\"btn btn-outline ex-copy\" data-copy=\"ex-cmd\">Copy command</button></div>" +
      "<p class=\"ex-note\">GitHub serves these files without the headers a web page needs to read them, so this page cannot pack them into one ZIP for you; the list and the command fetch them all.</p>";
  }
  function saveText(name, text, type) {
    var blob = new Blob([text], { type: type }), a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
  }
  function csvCell(v) { v = v == null ? "" : String(v); return /[",\n]/.test(v) ? "\"" + v.replace(/"/g, "\"\"") + "\"" : v; }
  /* CSV column names: deg stays deg (as before); % is pct, a ratio unit like ":1" is ratio, no unit is value */
  function slugU(u) { return u === "%" ? "pct" : u.charAt(0) === ":" ? "ratio" : u ? u.replace(/[^A-Za-z0-9]+/g, "_") : "value"; }
  function pctCol(F) { return F.meta.unit === "%" ? "_pct_of_" + slugU(F.meta.size.toLowerCase()) : "_pct"; }
  function linesCsv(F, T) {
    var M = F.meta.measures, inSize = sizeTest(F);
    var head = ["family", "term", "study", "dataset", "experiment", "line", "n", "size_" + slugU(F.meta.sizeUnit), "pooled", "in_size_filter", "reason", "linked_by"];
    M.forEach(function (ms) {
      var u = ms.unit === "trials" ? "trials" : slugU(F.meta.unit);
      head.push(ms.key + "_" + u, ms.key + "_se_" + u, ms.key + "_n");
      if (F.meta.pct && ms.unit !== "trials") head.push(ms.key + pctCol(F), ms.key + "_se" + pctCol(F));
    });
    head.push("zip_url");
    var rows = (T ? T.lines : F.lines).map(function (l) {
      var f = page().file[l.ds];
      var r = [F.meta.id, T ? T.label : "", l.study, l.ds, l.exp, lineName(l, true), l.n, l.size, l.ok ? "yes" : "no", inSize(l) ? "yes" : "no", l.why || "", T ? T.how[l.study] || "" : ""];
      M.forEach(function (ms) {
        var m = l.m[ms.key];
        r.push(m ? m[0] : "", m && m[1] != null ? m[1] : "", m ? m[2] : "");
        if (F.meta.pct && ms.unit !== "trials") r.push(m ? +(m[0] / l.size * 100).toFixed(2) : "", m && m[1] != null ? +(m[1] / l.size * 100).toFixed(3) : "");
      });
      r.push(f && f.url || "");
      return r.map(csvCell).join(",");
    });
    return head.join(",") + "\n" + rows.join("\n") + "\n";
  }

  /* ---------- compare your data (markup; the computation is CMP above) ---------- */
  function compareHtml(F) {
    return "<h3 class=\"ex-h3\" id=\"ex-cmp-h\">Compare your data</h3><div class=\"ex-cmp\">" +
      "<p class=\"ex-cmptext\">Add a CSV file with one row per " + xu(F) + ", or a standardized OpenMotor ZIP. Your group's mean curve is drawn on the learning curves and its early, late and aftereffect values are placed among this term's lines. <b>The file is read by this page in your browser; nothing is uploaded.</b></p>" +
      "<div class=\"ex-drop\" id=\"ex-drop\" tabindex=\"0\" role=\"button\" aria-describedby=\"ex-drop-note\">" + ICON_UP + "<b>Drop a CSV or ZIP file here</b><span>or click to choose one</span></div>" +
      "<input type=\"file\" id=\"ex-file\" accept=\".csv,.txt,.tsv,.zip,text/csv\" hidden>" +
      "<p class=\"ex-note\" id=\"ex-drop-note\">" + (F.meta.cmp ? "The file needs a participant, a " + xu(F) + " number and a " + esc(F.meta.cmp.what) + " column; you choose them, and the perturbed " + xu(F) + "s, below." :
      "OpenMotor's columns (subject, trial, hand_angle, rotation, clamp, fb_mode, fb_visible, ...) are recognised. Any other file needs a participant, a trial number and a hand angle column; you choose them, and the perturbation, below.") + "</p>" +
      "<div id=\"ex-map\"></div><div id=\"ex-user\"></div></div>";
  }

  function methodsHtml(F) {
    var R = IDX.rules || {};
    var early = R.early ? R.early[0] + " to " + R.early[1] : "2 to 11";
    if (F.meta.xunit) {  /* the series families: their own first paragraphs, the shared ones from Meta-analysis on */
      var u = xu(F), full = methodsHtml({ meta: {} });
      return "<h3 class=\"ex-h3\" id=\"ex-methods\">How the numbers are computed</h3><div class=\"ex-methods\">" +
        "<p><b>Lines.</b> Each line is one group of participants from one experiment who " + esc(F.meta.lineWord || "got the same perturbation") + ", and " + u + " 0 is the first " + esc(F.meta.onsetWord || "perturbed " + u) + ". " + esc(F.meta.methods || "") + "</p>" +
        "<p><b>Which lines are pooled.</b> A line is pooled when its " + esc(F.meta.blockWord || "first perturbation block") + " has at least " + F.meta.minBlock + " " + u + "s and at least " + R.min_people + " participants. Designs that do not fit, as listed above, are shown but not pooled; the study table says why.</p>" +
        "<p><b>Pooled curve.</b> At each " + u + " we average the lines, each line counting once, with a 95% CI that treats lines from the same study as one cluster. The measures and their windows are listed in the table above; error bars are SEs across participants.</p>" +
        full.slice(full.indexOf("<p><b>Meta-analysis.</b>"));
    }
    return "<h3 class=\"ex-h3\" id=\"ex-methods\">How the numbers are computed</h3><div class=\"ex-methods\">" +
      "<p><b>Lines.</b> Each line is one group of participants from one experiment who got the same perturbation schedule. Trials and participants the authors excluded are left out. Hand angles are flipped for each participant so that compensating for the first perturbation is positive, and trial 0 is the first perturbed trial.</p>" +
      "<p><b>Which lines are pooled.</b> A line is pooled when its first perturbation block has at least " + R.min_fb + " feedback trials, it has at least " + R.min_people + " participants, and nobody was told where to aim. Gradual and trial-by-trial schedules are listed but not pooled; the study table says why.</p>" +
      "<p><b>Pooled curve.</b> At each trial we average the lines, each line counting once so a few big online groups dont set the curve, with a 95% CI that treats lines from the same study as one cluster. Rotation lines can be shown as % of the rotation or in degrees; clamp lines are in degrees.</p>" +
      "<p><b>Measures.</b> Early is the mean of feedback trials " + early + " of the block, late the mean of the last " + R.late_n + " feedback trials, the aftereffect the first " + R.ae_n + " no-feedback trials right after the block, and trials to half the trial where the smoothed curve first reaches half of the late value. Error bars are SEs accross participants.</p>" +
      "<p><b>Meta-analysis.</b> DerSimonian-Laird random effects across lines, plus a second CI clustered by study. The difference column compares the term's lines with all the others (Q-between test).</p>" +
      "<p><b>Terms.</b> Tags are the ones on the Studies page. Words come from a list of " + R.vocabulary + " motor learning terms searched in titles and abstracts; a study is linked to a word when its tf-idf weight is at least " + R.tfidf_min + ".</p>" +
      "<p><b>Limits.</b> These studies differ in a lot of ways (instructions, feedback, devices, ages), so a difference between a term and everything else is something to follow up, not an effect of the term. Built " + esc(IDX.built) + ".</p></div>";
  }

  /* ---------- events ---------- */
  function bindMain(F) {
    var re = function () { render(F, false); };
    var sz = $("#ex-size"); if (sz) sz.addEventListener("change", function () { S.size = sz.value; re(); });
    var un = $("#ex-unit"); if (un) un.addEventListener("change", function () { S.unit = un.value; re(); });
    var wn = $("#ex-win"); if (wn) wn.addEventListener("change", function () { S.win = +wn.value; drawCurve(); });
    document.querySelectorAll(".ex-tabs .tab").forEach(function (b) {
      b.addEventListener("click", function () {
        S.measure = b.getAttribute("data-measure");
        document.querySelectorAll(".ex-tabs .tab").forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
        drawForest(); renderUser();
      });
    });
    $("#ex-main").addEventListener("click", function (e) {
      var g = e.target.closest("[data-get]");
      if (g) {
        var T = CUR.T;
        if (g.getAttribute("data-get") === "urls") saveText(baseName(F, T) + "-zips.txt", subsetFiles(F, T).map(function (f) { return f.url; }).join("\n") + "\n", "text/plain");
        else saveText(baseName(F, T) + "-lines.csv", linesCsv(F, T), "text/csv");
        return;
      }
      var c = e.target.closest("[data-copy]");
      if (c) {
        var txt = document.getElementById(c.getAttribute("data-copy")).textContent;
        var done = function (msg) { c.textContent = msg; setTimeout(function () { c.textContent = "Copy command"; }, 2000); };
        if (navigator.clipboard) navigator.clipboard.writeText(txt).then(function () { done("Copied"); }, function () { selectText(c.getAttribute("data-copy")); done("Selected, press copy"); });
        else { selectText(c.getAttribute("data-copy")); done("Selected, press copy"); }
      }
    });
    bindCompare();
  }
  function selectText(id) { var r = document.createRange(); r.selectNodeContents(document.getElementById(id)); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
  (function bindSide() {
    var q = document.getElementById("ex-q");
    if (!q) return;
    q.addEventListener("input", function () { S.q = q.value; if (S.fam && FAMS[S.fam]) renderTerms(FAMS[S.fam]); });
    q.addEventListener("keydown", function (e) {
      var first = document.querySelector("#ex-terms .ex-tlist a");
      if (e.key === "Enter" && first) { e.preventDefault(); location.hash = first.getAttribute("href"); }
      if (e.key === "ArrowDown" && first) { e.preventDefault(); first.focus(); }
    });
    document.getElementById("ex-terms").addEventListener("keydown", function (e) {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      var all = Array.prototype.slice.call(document.querySelectorAll("#ex-terms .ex-tlist a")), i = all.indexOf(document.activeElement);
      if (i < 0) return;
      e.preventDefault();
      if (e.key === "ArrowUp" && i === 0) { q.focus(); return; }
      var n = all[Math.max(0, Math.min(all.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))]; if (n) n.focus();
    });
    document.getElementById("ex-sort").addEventListener("click", function (e) {
      var b = e.target.closest("[data-sort]"); if (!b) return;
      S.sort = b.getAttribute("data-sort"); if (S.fam && FAMS[S.fam]) renderTerms(FAMS[S.fam]);
    });
    document.querySelector(".ex-lead [data-ex-goto]").addEventListener("click", function (e) {
      var t = document.getElementById("ex-methods");
      if (t) { e.preventDefault(); t.scrollIntoView({ block: "start" }); }
    });
  })();
  var rsz;
  window.addEventListener("resize", function () {
    clearTimeout(rsz);
    rsz = setTimeout(function () { if (active && CUR && $("#ex-curve")) { drawCurve(); drawForest(); } }, 150);
  });

  /* ---------- SVG helpers ---------- */
  var SVGNS = "http://www.w3.org/2000/svg";
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-9; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toFixed(10));
    return { ticks: t, step: step };
  }
  function pathOf(pts) {  /* pts: [[x, y] or null]; breaks at null */
    var d = "", pen = false;
    pts.forEach(function (p) { if (!p) { pen = false; return; } d += (pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); pen = true; });
    return d;
  }
  function textW(s, px, weight) {
    var c = textW.c || (textW.c = document.createElement("canvas").getContext("2d"));
    c.font = (weight || 400) + " " + px + "px \"Source Sans 3\", \"Source Sans Pro\", Helvetica, Arial, sans-serif";
    return c.measureText(s).width;
  }
  function fit(s, px, w) {
    if (textW(s, px) <= w) return s;
    while (s.length > 1 && textW(s + "...", px) > w) s = s.slice(0, -1);
    return s.replace(/[\s,]+$/, "") + "...";
  }
  function tipShow(box, html, px, py) {
    var tip = box.querySelector(".ex-tip");
    if (!tip) { tip = document.createElement("div"); tip.className = "ex-tip"; tip.setAttribute("role", "presentation"); box.appendChild(tip); }
    tip.innerHTML = html;
    tip.hidden = false;
    var bw = box.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    var left = px + 14 + tw > bw ? Math.max(0, px - 14 - tw) : px + 14;
    tip.style.left = left + "px";
    tip.style.top = Math.max(0, py - th / 2) + "px";
  }
  function tipHide(box) { var t = box && box.querySelector(".ex-tip"); if (t) t.hidden = true; }

  /* ---------- learning curves ---------- */
  function valAt(l, x) { var i = x - l.x0; if (i < 0 || i >= l.y.length) return null; var v = l.y[i]; return v == null ? null : scale(l, v); }
  /* for the pooled mean only: gaps of up to GAP trials inside the baseline or inside the block (interleaved no-feedback or
     report trials) are bridged by linear interpolation, so such lines do not drop in and out of the mean from trial to trial */
  var GAP = 5;
  function filled(l) {
    if (l._yi) return l._yi;
    var y = l.y.slice(), z = -l.x0;
    function fill(a, b) {
      for (var i = a; i < b;) {
        if (y[i] != null) { i++; continue; }
        var j = i; while (j < b && y[j] == null) j++;
        if (i > a && j < b && j - i <= gapOf(CUR.F)) for (var k = i; k < j; k++) y[k] = y[i - 1] + (y[j] - y[i - 1]) * (k - i + 1) / (j - i + 1);
        i = j;
      }
    }
    fill(0, z); fill(z, y.length);
    return (l._yi = y);
  }
  function valFilled(l, x) { var y = filled(l), i = x - l.x0; if (i < 0 || i >= y.length || y[i] == null) return null; return scale(l, y[i]); }
  function pooledCurve(lines, x0, x1) {
    var out = [];
    for (var x = x0; x <= x1; x++) {
      var ys = [], cl = [];
      for (var i = 0; i < lines.length; i++) { var v = valFilled(lines[i], x); if (v != null) { ys.push(v); cl.push(lines[i].study); } }
      var r = ys.length ? STATS.meanCl(ys, cl) : { k: 0, m: 0 };
      r.x = x; out.push(r);
    }
    return out;
  }
  function userAt(x) {
    var U = USER;
    if (!U || U.family !== CUR.F.meta.id) return null;
    var i = x - U.x0; if (i < 0 || i >= U.y.length || U.y[i] == null) return null;
    return unitOf(CUR.F) === "pct" ? U.y[i] / U.size * 100 : U.y[i];
  }
  function drawCurve() {
    var box = $("#ex-curve");
    if (!box) return;
    var F = CUR.F, T = CUR.T, A = CUR.A, B = CUR.B, W = Math.max(280, box.clientWidth), phone = W < 600;
    var x0 = F.meta.window[0], x1 = S.win || F.meta.window[1];
    var pA = pooledCurve(A, x0, x1), pB = B ? pooledCurve(B, x0, x1) : null;
    var ml = phone ? 46 : 62, mr = phone ? 10 : 18, mt = 14, plotH = phone ? 250 : 360, stripH = 30, gap = 34, H = mt + plotH + gap + stripH + 26;
    var px = function (x) { return ml + (x - x0) / (x1 - x0) * (W - ml - mr); };
    /* y range: pooled curves and their bands, the middle 96% of the thin lines' values, 0, full compensation, the visitor's curve */
    var vals = [0];
    if (unitOf(F) === "pct") vals.push(100);
    [pA, pB].forEach(function (p) { if (p) p.forEach(function (r) { if (drawn(r)) { vals.push(r.est); if (r.lo != null && r.m >= MINM_BAND) vals.push(r.lo, r.hi); } }); });
    var thin = [];
    A.forEach(function (l) { for (var x = x0; x <= x1; x++) { var v = valAt(l, x); if (v != null) thin.push(v); } });
    thin.sort(function (a, b) { return a - b; });
    if (thin.length) vals.push(thin[Math.floor(0.02 * (thin.length - 1))], thin[Math.ceil(0.98 * (thin.length - 1))]);
    for (var ux = x0; ux <= x1; ux++) { var uv = userAt(ux); if (uv != null) vals.push(uv); }
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo) * 0.06 || 1;
    var yt = niceTicks(lo - pad, hi + pad, phone ? 5 : 7);
    var y0 = Math.min(yt.ticks[0], lo - pad), y1 = Math.max(yt.ticks[yt.ticks.length - 1], hi + pad);
    var py = function (v) { return mt + (y1 - v) / (y1 - y0) * plotH; };
    var su = suffix(F, "y");

    var s = "<svg class=\"ex-svg\" width=\"" + W + "\" height=\"" + H + "\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" aria-labelledby=\"ex-curve-title ex-curve-desc\" tabindex=\"0\">";
    var nA = A.length, sA = {}; A.forEach(function (l) { sA[l.study] = 1; });
    s += "<title id=\"ex-curve-title\">Learning curves: " + esc(T ? T.label : F.meta.label) + "</title><desc id=\"ex-curve-desc\">" + esc(curveDesc(F, T, pA, pB, x1)) + "</desc>";
    s += "<defs><clipPath id=\"ex-clip\"><rect x=\"" + ml + "\" y=\"" + mt + "\" width=\"" + (W - ml - mr) + "\" height=\"" + plotH + "\"/></clipPath></defs>";
    yt.ticks.forEach(function (t) {
      if (t < y0 || t > y1) return;
      s += "<line class=\"ex-grid" + (t === 0 ? " ex-zero" : "") + "\" x1=\"" + ml + "\" x2=\"" + (W - mr) + "\" y1=\"" + py(t).toFixed(1) + "\" y2=\"" + py(t).toFixed(1) + "\"/>";
      s += "<text class=\"ex-tick\" x=\"" + (ml - 8) + "\" y=\"" + (py(t) + 4).toFixed(1) + "\" text-anchor=\"end\">" + num(t, td(yt.step)) + "</text>";
    });
    if (unitOf(F) === "pct" && 100 <= y1) s += "<text class=\"ex-ref\" x=\"" + (W - mr - 4) + "\" y=\"" + (py(100) - 5).toFixed(1) + "\" text-anchor=\"end\">full compensation</text>";
    var xt = niceTicks(x0, x1, phone ? 5 : 9).ticks;
    if (xt.indexOf(0) < 0) xt.push(0);
    xt.forEach(function (t) {
      s += "<text class=\"ex-tick\" x=\"" + px(t).toFixed(1) + "\" y=\"" + (mt + plotH + 18) + "\" text-anchor=\"middle\">" + t + "</text>";
    });
    s += "<line class=\"ex-onset\" x1=\"" + px(0).toFixed(1) + "\" x2=\"" + px(0).toFixed(1) + "\" y1=\"" + mt + "\" y2=\"" + (mt + plotH) + "\"/>";
    s += "<text class=\"ex-ref\" x=\"" + (px(0) + 4).toFixed(1) + "\" y=\"" + (mt + 12) + "\">onset</text>";
    s += "<text class=\"ex-axis\" x=\"" + ((ml + W - mr) / 2).toFixed(1) + "\" y=\"" + (mt + plotH + gap - 2) + "\" text-anchor=\"middle\">" + esc(F.meta.x) + "</text>";
    s += "<text class=\"ex-axis\" transform=\"translate(" + (phone ? 12 : 16) + " " + (mt + plotH / 2) + ") rotate(-90)\" text-anchor=\"middle\">" + esc(yLabel(F)) + "</text>";

    s += "<g clip-path=\"url(#ex-clip)\">";
    function band(p, cls) {
      var seg = [], out = "";
      function flush() {
        if (seg.length > 1) out += "<path class=\"" + cls + "\" d=\"M" + seg.map(function (r) { return px(r.x).toFixed(1) + " " + py(r.hi).toFixed(1); }).join("L") + "L" + seg.slice().reverse().map(function (r) { return px(r.x).toFixed(1) + " " + py(r.lo).toFixed(1); }).join("L") + "Z\"/>";
        seg = [];
      }
      p.forEach(function (r) { if (drawn(r) && r.m >= MINM_BAND && r.lo != null) seg.push(r); else flush(); });
      flush();
      return out;
    }
    function meanPath(p) { return pathOf(p.map(function (r) { return drawn(r) ? [px(r.x), py(r.est)] : null; })); }
    if (pB) s += band(pB, "ex-band-o");
    s += "<g style=\"stroke-opacity:" + Math.min(1, 2.5 / Math.sqrt(Math.max(1, A.length))).toFixed(2) + "\">";
    A.forEach(function (l, i) {
      var pts = [];
      for (var x = x0; x <= x1; x++) { var v = valAt(l, x); pts.push(v == null ? null : [px(x), py(v)]); }
      s += "<path class=\"ex-thin\" data-i=\"" + i + "\" d=\"" + pathOf(pts) + "\"/>";
    });
    s += "</g>";
    s += band(pA, "ex-band-t");
    if (pB) s += "<path class=\"ex-ring\" d=\"" + meanPath(pB) + "\"/><path class=\"ex-mean-o\" d=\"" + meanPath(pB) + "\"/>";
    s += "<path class=\"ex-ring\" d=\"" + meanPath(pA) + "\"/><path class=\"ex-mean-t\" d=\"" + meanPath(pA) + "\"/>";
    var upts = [], hasU = false;
    for (var x = x0; x <= x1; x++) { var u = userAt(x); if (u != null) hasU = true; upts.push(u == null ? null : [px(x), py(u)]); }
    if (hasU) s += "<path class=\"ex-ring\" d=\"" + pathOf(upts) + "\"/><path class=\"ex-user\" d=\"" + pathOf(upts) + "\"/>";
    s += "<path class=\"ex-hl\" id=\"ex-hl\" d=\"\"/></g>";

    /* lines in the mean */
    var kmax = Math.max(1, Math.max.apply(null, pA.map(function (r) { return r.k; })));
    var sy0 = mt + plotH + gap, bw = (W - ml - mr) / (x1 - x0 + 1);
    s += "<text class=\"ex-tick\" x=\"" + (ml - 8) + "\" y=\"" + (sy0 + 10) + "\" text-anchor=\"end\">" + kmax + "</text><text class=\"ex-tick\" x=\"" + (ml - 8) + "\" y=\"" + (sy0 + stripH) + "\" text-anchor=\"end\">0</text>";
    s += "<path class=\"ex-strip\" d=\"" + pA.map(function (r) {
      var h = r.k / kmax * stripH;
      return h > 0 ? "M" + (px(r.x) - bw / 2).toFixed(1) + " " + (sy0 + stripH) + "h" + bw.toFixed(2) + "v" + (-h).toFixed(1) + "h" + (-bw).toFixed(2) + "Z" : "";
    }).join("") + "\"/>";
    s += "<line class=\"ex-grid\" x1=\"" + ml + "\" x2=\"" + (W - mr) + "\" y1=\"" + (sy0 + stripH) + "\" y2=\"" + (sy0 + stripH) + "\"/>";
    s += "<text class=\"ex-ref\" x=\"" + ml + "\" y=\"" + (sy0 + stripH + 16) + "\">lines in the " + (T ? "term's " : "") + "mean</text>";
    s += "<line class=\"ex-cross\" id=\"ex-cross\" x1=\"0\" x2=\"0\" y1=\"" + mt + "\" y2=\"" + (mt + plotH) + "\" visibility=\"hidden\"/>";
    s += "<rect class=\"ex-hit\" x=\"" + ml + "\" y=\"" + mt + "\" width=\"" + (W - ml - mr) + "\" height=\"" + (plotH + gap + stripH) + "\"/></svg>";
    box.innerHTML = s;

    /* legend */
    var lg = [["ex-k-thin", "One group (" + nA + " lines)"], ["ex-k-t", (T ? "This term" : "All lines") + ": mean and 95% CI"]];
    if (pB) lg.push(["ex-k-o", "Other lines (" + B.length + "): mean and 95% CI"]);
    if (hasU) lg.push(["ex-k-u", "Your data"]);
    $("#ex-legend").innerHTML = lg.map(function (g) { return "<span><i class=\"ex-key " + g[0] + "\" aria-hidden=\"true\"></i>" + esc(g[1]) + "</span>"; }).join("");

    /* hover and keyboard: crosshair snaps to a trial and names the nearest line */
    var svg = box.querySelector("svg"), cross = svg.querySelector("#ex-cross"), hl = svg.querySelector("#ex-hl"), cx = null;
    function show(x, pointerY) {
      x = Math.max(x0, Math.min(x1, x)); cx = x;
      var X = px(x);
      cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.setAttribute("visibility", "visible");
      var ra = pA[x - x0], rb = pB ? pB[x - x0] : null, near = null, nd = Infinity;
      A.forEach(function (l, i) {
        var v = valAt(l, x); if (v == null) return;
        var d = pointerY == null ? Math.abs(v - (ra && ra.k ? ra.est : 0)) : Math.abs(py(v) - pointerY);
        if (d < nd) { nd = d; near = { l: l, v: v, i: i }; }
      });
      if (near) {
        var pts = [];
        for (var xx = x0; xx <= x1; xx++) { var v = valAt(near.l, xx); pts.push(v == null ? null : [px(xx), py(v)]); }
        hl.setAttribute("d", pathOf(pts));
      } else hl.setAttribute("d", "");
      function row(cls, val, label) { return "<div class=\"ex-tr\"><i class=\"ex-key " + cls + "\"></i><b>" + val + "</b> " + label + "</div>"; }
      function pr(r, label) {
        if (!drawn(r)) return row(label[0], "-", esc(label[1]) + " (" + (r ? plural(r.k, "line") + " from " + plural(r.m || 0, "study", "studies") : "no lines") + ")");
        return row(label[0], num(r.est, YD) + su, (r.lo != null && r.m >= MINM_BAND ? "[" + num(r.lo, YD) + ", " + num(r.hi, YD) + "] " : "") + esc(label[1]) + ", " + r.k + " lines from " + plural(r.m, "study", "studies"));
      }
      var html = "<div class=\"ex-tt\">" + Xu(F) + " " + x + (x < 0 ? " (baseline)" : x === 0 ? " (onset)" : "") + "</div>" + pr(ra, ["ex-k-t", T ? "this term" : "all lines"]);
      if (pB) html += pr(rb, ["ex-k-o", "other lines"]);
      var u = userAt(x); if (u != null) html += row("ex-k-u", num(u, 1) + su, "your data");
      if (near) html += "<div class=\"ex-tn2\">Nearest line: " + esc(lineName(near.l)) + ", " + num(near.v, 1) + su + " (n = " + near.l.n + ")</div>";
      var rect = svg.getBoundingClientRect();
      tipShow(box, html, X, pointerY == null ? py(ra && ra.k ? ra.est : 0) : pointerY);
      void rect;
    }
    function hide() { cross.setAttribute("visibility", "hidden"); hl.setAttribute("d", ""); tipHide(box); }
    var hit = svg.querySelector(".ex-hit");
    hit.addEventListener("pointermove", function (e) {
      var r = svg.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      show(Math.round(x0 + (mx - ml) / (W - ml - mr) * (x1 - x0)), Math.min(my, mt + plotH));
    });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); show((cx == null ? 0 : cx) + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1), null); }
      else if (e.key === "Escape") hide();
    });
    svg.addEventListener("blur", hide);

    /* the same numbers as a table, every 10 trials */
    var rows = [];
    for (var tx = x0; tx <= x1; tx += (tx < 0 ? 5 : x1 > 600 ? 20 : 10)) {
      var ra = pA[tx - x0], rb = pB ? pB[tx - x0] : null;
      var cells = function (r) { return "<td class=\"num\">" + (drawn(r) ? num(r.est, YD) : "-") + "</td><td class=\"num\">" + (drawn(r) && r.m >= MINM_BAND && r.lo != null ? num(r.lo, YD) + " to " + num(r.hi, YD) : "-") + "</td><td class=\"num\">" + r.k + "</td><td class=\"num\">" + (r.m || 0) + "</td>"; };
      rows.push("<tr><td class=\"num\">" + tx + "</td>" + cells(ra) + (pB ? cells(rb) : "") + "</tr>");
    }
    $("#ex-curve-table").innerHTML = "<table class=\"data compact ex-ctable\"><thead><tr><th scope=\"col\" class=\"num\">" + Xu(F) + "</th><th scope=\"col\" class=\"num\">" + (T ? "Term" : "All") + " mean (" + esc(su.trim() || F.meta.unit) + ")</th><th scope=\"col\" class=\"num\">95% CI</th><th scope=\"col\" class=\"num\">Lines</th><th scope=\"col\" class=\"num\">Studies</th>" +
      (pB ? "<th scope=\"col\" class=\"num\">Other mean</th><th scope=\"col\" class=\"num\">95% CI</th><th scope=\"col\" class=\"num\">Lines</th><th scope=\"col\" class=\"num\">Studies</th>" : "") + "</tr></thead><tbody>" + rows.join("") + "</tbody></table>";
  }
  function curveDesc(F, T, pA, pB, x1) {
    function at(p, x) { var r = p && p[x - F.meta.window[0]]; return drawn(r) ? num(r.est, YD) + suffix(F, "y") + " (" + r.k + " lines)" : "too few lines"; }
    var marks = [10, 50, 100, x1].filter(function (x, i, a) { return x <= x1 && a.indexOf(x) === i; });
    return "Mean " + F.meta.y.toLowerCase() + " across lines, by " + xu(F) + " from perturbation onset. " + (T ? "This term: " : "All lines: ") + marks.map(function (x) { return xu(F) + " " + x + ", " + at(pA, x); }).join("; ") + "." +
      (pB ? " Other lines: " + marks.map(function (x) { return xu(F) + " " + x + ", " + at(pB, x); }).join("; ") + "." : "") + " The table below the figure lists the values.";
  }

  /* ---------- forest plot ---------- */
  function drawForest() {
    var box = $("#ex-forest");
    if (!box) return;
    var F = CUR.F, T = CUR.T, key = S.measure, ms = F.meta.measures.filter(function (m) { return m.key === key; })[0] || F.meta.measures[0];
    key = ms.key;
    var kind = ms.unit === "trials" ? "trials" : "y", su = suffix(F, kind);
    var a = metaFor(CUR.A, key, F), b = T ? metaFor(CUR.B, key, F) : null;
    var rows = CUR.A.filter(function (l) { var m = l.m[key]; return m && m[1] != null && m[1] > 0; }).map(function (l) {
      var f = key === "half" ? 1 : scale(l, 1), m = l.m[key];
      return { l: l, est: m[0] * f, se: m[1] * f, n: m[2] };
    });
    var dense = rows.length > 50;  /* many lines: thin rows sorted by value, named on hover (a caterpillar plot) */
    rows.sort(dense ? function (p, q) { return p.est - q.est; } : function (p, q) { return studyLabel(p.l.study).localeCompare(studyLabel(q.l.study)) || p.l.id.localeCompare(q.l.id, undefined, { numeric: true }); });
    if (!rows.length) { box.innerHTML = "<p class=\"muted\">No line of this term has this measure.</p>"; return; }
    var W = Math.max(280, box.clientWidth), phone = W < 600;
    var labelW = dense ? (phone ? 96 : 150) : phone ? Math.min(150, W * 0.42) : Math.min(340, W * 0.34), textCol = phone || dense ? 0 : 160;
    var pl = labelW + 10, pr = W - textCol - (phone ? 10 : 16);
    var rowH = dense ? Math.max(4, Math.min(8, 640 / rows.length)) : rows.length > 30 ? 17 : 21, top = 26, nPool = b ? 2 : 1;
    var H = top + rows.length * rowH + 12 + nPool * 24 + 40;
    var tau2 = a ? a.tau2 : 0, ws = rows.map(function (r) { return 1 / (r.se * r.se + tau2); }), wmax = Math.max.apply(null, ws), wsum = ws.reduce(function (s, w) { return s + w; }, 0);
    var los = rows.map(function (r) { return r.est - 1.959964 * r.se; }).sort(function (p, q) { return p - q; });
    var his = rows.map(function (r) { return r.est + 1.959964 * r.se; }).sort(function (p, q) { return p - q; });
    var dom = [los[Math.floor(0.03 * (los.length - 1))], his[Math.ceil(0.97 * (his.length - 1))]];
    if (kind !== "trials") dom.push(0);
    [a, b].forEach(function (r) { if (r) dom.push(r.lo, r.hi); });
    var uval = userMeasure(key);
    if (uval != null) dom.push(uval);
    var lo = Math.min.apply(null, dom), hi = Math.max.apply(null, dom), pad = (hi - lo) * 0.04 || 1;
    var xt = niceTicks(lo - pad, hi + pad, phone ? 4 : 7), d0 = Math.min(xt.ticks[0], lo - pad), d1 = Math.max(xt.ticks[xt.ticks.length - 1], hi + pad);
    var px = function (v) { return pl + (v - d0) / (d1 - d0) * (pr - pl); };
    var s = "<svg class=\"ex-svg\" width=\"" + W + "\" height=\"" + H + "\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" aria-label=\"" + esc("Forest plot of " + ms.label.toLowerCase() + ": " + rows.length + " lines" + (a ? ", pooled " + num(a.est, 1) + su + " [" + num(a.lo, 1) + ", " + num(a.hi, 1) + "]" : "") + (b ? "; other lines " + num(b.est, 1) + su + " [" + num(b.lo, 1) + ", " + num(b.hi, 1) + "]" : "") + ". The study table lists every value.") + "\">";
    s += "<text class=\"ex-colh\" x=\"0\" y=\"14\">" + (dense ? rows.length + " lines, by value" : "Line") + "</text>";
    if (!phone && !dense) s += "<text class=\"ex-colh\" x=\"" + W + "\" y=\"14\" text-anchor=\"end\">" + esc(ms.label) + " [95% CI]</text>";
    var bottom = top + rows.length * rowH + 12 + nPool * 24;
    xt.ticks.forEach(function (t) {
      if (t < d0 || t > d1) return;
      s += "<line class=\"ex-grid" + (t === 0 ? " ex-zero" : "") + "\" x1=\"" + px(t).toFixed(1) + "\" x2=\"" + px(t).toFixed(1) + "\" y1=\"" + (top - 4) + "\" y2=\"" + bottom + "\"/>";
      s += "<text class=\"ex-tick\" x=\"" + px(t).toFixed(1) + "\" y=\"" + (bottom + 16) + "\" text-anchor=\"middle\">" + num(t, td(xt.step)) + "</text>";
    });
    s += "<text class=\"ex-axis\" x=\"" + ((pl + pr) / 2).toFixed(1) + "\" y=\"" + (bottom + 34) + "\" text-anchor=\"middle\">" + esc(ms.label + (kind === "trials" ? " (trials)" : unitOf(F) === "pct" ? " (% of " + F.meta.size.toLowerCase() + ")" : F.meta.unit === "deg" ? " (degrees)" : F.meta.unit ? " (" + unitWords(F) + ")" : "")) + "</text>";
    if (a) s += "<line class=\"ex-pooled-ref\" x1=\"" + px(a.est).toFixed(1) + "\" x2=\"" + px(a.est).toFixed(1) + "\" y1=\"" + (top - 4) + "\" y2=\"" + bottom + "\"/>";
    var lfont = phone ? 12 : rowH < 20 ? 12.5 : 13.5;
    rows.forEach(function (r, i) {
      var cy = top + i * rowH + rowH / 2, l = r.l, lo = r.est - 1.959964 * r.se, hi = r.est + 1.959964 * r.se;
      var full = lineName(l), side = dense ? Math.max(3, rowH - 1) : 4 + 7 * Math.sqrt(ws[i] / wmax);
      s += "<g class=\"ex-frow\" data-i=\"" + i + "\"><rect class=\"ex-fhit\" x=\"0\" y=\"" + (cy - rowH / 2) + "\" width=\"" + W + "\" height=\"" + rowH + "\"/>";
      if (!dense) s += "<text class=\"ex-flab\" x=\"0\" y=\"" + (cy + 4.5).toFixed(1) + "\" style=\"font-size:" + lfont + "px\">" + esc(fit(phone ? shortName(l) : full, lfont, labelW)) + "</text>";
      var cl = Math.max(pl, px(lo)), ch = Math.min(pr, px(hi));
      s += "<line class=\"ex-ci-line\" x1=\"" + cl.toFixed(1) + "\" x2=\"" + ch.toFixed(1) + "\" y1=\"" + cy + "\" y2=\"" + cy + "\"/>";
      if (px(lo) < pl) s += "<path class=\"ex-arrow\" d=\"M" + (pl + 5) + " " + (cy - 3.5) + "L" + pl + " " + cy + "L" + (pl + 5) + " " + (cy + 3.5) + "\"/>";
      if (px(hi) > pr) s += "<path class=\"ex-arrow\" d=\"M" + (pr - 5) + " " + (cy - 3.5) + "L" + pr + " " + cy + "L" + (pr - 5) + " " + (cy + 3.5) + "\"/>";
      if (r.est >= d0 && r.est <= d1) s += "<rect class=\"ex-sq\" x=\"" + (px(r.est) - side / 2).toFixed(1) + "\" y=\"" + (cy - side / 2).toFixed(1) + "\" width=\"" + side.toFixed(1) + "\" height=\"" + side.toFixed(1) + "\"/>";
      if (!phone && !dense) s += "<text class=\"ex-fval\" x=\"" + W + "\" y=\"" + (cy + 4.5).toFixed(1) + "\" text-anchor=\"end\" style=\"font-size:" + lfont + "px\">" + num(r.est, kind === "trials" ? 0 : YD) + " [" + num(lo, kind === "trials" ? 0 : YD) + ", " + num(hi, kind === "trials" ? 0 : YD) + "]</text>";
      s += "</g>";
    });
    function diamond(r, cy, cls, label) {
      var lab = label + " (" + r.k + " lines)" + (dense && !phone ? ": " + num(r.est, YD) + " [" + num(r.lo, YD) + ", " + num(r.hi, YD) + "]" : "");
      s += "<text class=\"ex-flab ex-fpool\" x=\"" + (dense && !phone ? pr : 0) + "\" y=\"" + (cy + 4.5) + "\"" + (dense && !phone ? " text-anchor=\"end\"" : "") + ">" + esc(dense && !phone ? lab : fit(lab, 13.5, labelW)) + "</text>";
      s += "<path class=\"" + cls + "\" d=\"M" + px(r.lo).toFixed(1) + " " + cy + "L" + px(r.est).toFixed(1) + " " + (cy - 7) + "L" + px(r.hi).toFixed(1) + " " + cy + "L" + px(r.est).toFixed(1) + " " + (cy + 7) + "Z\"/>";
      if (!phone && !dense) s += "<text class=\"ex-fval ex-fpool\" x=\"" + W + "\" y=\"" + (cy + 4.5) + "\" text-anchor=\"end\">" + num(r.est, YD) + " [" + num(r.lo, YD) + ", " + num(r.hi, YD) + "]</text>";
    }
    var dy = top + rows.length * rowH + 12 + 12;
    s += "<line class=\"ex-grid\" x1=\"0\" x2=\"" + W + "\" y1=\"" + (dy - 12) + "\" y2=\"" + (dy - 12) + "\"/>";
    if (a) diamond(a, dy, "ex-dia-t", T ? "This term, pooled" : "All lines, pooled");
    if (b) diamond(b, dy + 24, "ex-dia-o", "Other lines, pooled");
    if (uval != null && uval >= d0 && uval <= d1) {
      s += "<line class=\"ex-user-ref\" x1=\"" + px(uval).toFixed(1) + "\" x2=\"" + px(uval).toFixed(1) + "\" y1=\"" + (top - 4) + "\" y2=\"" + bottom + "\"/>";
      s += "<text class=\"ex-ref\" x=\"" + (px(uval) + 4).toFixed(1) + "\" y=\"" + (top - 8) + "\">your data</text>";
    }
    box.innerHTML = s + "</svg>";
    var svg = box.querySelector("svg");
    svg.addEventListener("pointermove", function (e) {
      var g = e.target.closest(".ex-frow");
      if (!g) { tipHide(box); return; }
      var r = rows[+g.getAttribute("data-i")], lo = r.est - 1.959964 * r.se, hi = r.est + 1.959964 * r.se, rc = svg.getBoundingClientRect();
      tipShow(box, "<div class=\"ex-tr\"><b>" + num(r.est, kind === "trials" ? 0 : YD) + su + "</b> [" + num(lo, YD) + ", " + num(hi, YD) + "]</div><div class=\"ex-tt\">" + esc(lineName(r.l)) + "</div><div class=\"ex-tn2\">n = " + r.n + ", " + (F.meta.noSize ? "" : esc(F.meta.size.toLowerCase()) + " " + sizeTxt(F, r.l.size) + ", ") + "weight " + num(100 * ws[+g.getAttribute("data-i")] / wsum, 1) + "%</div>",
        e.clientX - rc.left, e.clientY - rc.top);
    });
    svg.addEventListener("pointerleave", function () { tipHide(box); });
  }

  /* ---------- the visitor's data ---------- */
  function userMeasure(key) {
    var U = USER;
    if (!U || U.family !== CUR.F.meta.id || !U.m[key]) return null;
    return key === "half" ? U.m[key][0] : unitOf(CUR.F) === "pct" ? U.m[key][0] / U.size * 100 : U.m[key][0];
  }
  function renderUser() {
    var box = $("#ex-user");
    if (!box) return;
    var U = USER, F = CUR.F;
    if (!U) { box.innerHTML = ""; return; }
    var fam = famMeta(U.family);
    var h = "<div class=\"ex-ures\"><p><b>Your data</b>: " + esc(U.name) + (U.subset ? " (" + esc(U.subset) + ")" : "") + ". " + plural(U.n, "participant") + ", " +
      (U.series ? (fam && !fam.noSize ? esc(fam.size.toLowerCase()) + " " + (isFinite(U.size) ? sizeTxt(fam ? { meta: fam } : F, U.size) : "not given") + ", " : "") + "perturbed from " + xu(F) + " " + U.start + ", first block " + plural(U.blk, xu(F)) + " (" + U.nfb + " with a value)."
       : (fam ? esc(fam.size.toLowerCase()) : "perturbation") + " of " + num(U.size, U.size % 1 ? 1 : 0) + "° from trial " + U.start + ", first block " + plural(U.blk, "trial") + " (" + U.nfb + " with feedback).") +
      (U.nall && U.nall !== U.n ? " " + U.n + " of " + U.nall + " participants had the group's schedule; the others are left out." : "") + "</p>";
    if (U.family !== F.meta.id) {
      h += "<p>Your file is " + (fam ? "a " + esc(fam.label.toLowerCase()) + " experiment" : "not a rotation or clamp experiment") + ". " + (fam ? "<a href=\"#explore/" + esc(fam.id) + (FAMS[fam.id] && FAMS[fam.id].termBy[S.term] ? "/" + esc(S.term) : "") + "\">Compare it with " + esc(fam.label.toLowerCase()) + " lines</a>." : "") + "</p>";
    } else {
      if (U.why) h += "<p class=\"ex-warn\">A line like yours would not be pooled here: " + esc(U.why) + ".</p>";
      h += "<div class=\"table-card\"><div class=\"table-wrap\"><table class=\"data compact ex-utable\"><thead><tr><th scope=\"col\">Measure</th><th scope=\"col\" class=\"num\">Your value</th><th scope=\"col\" class=\"num\">" + (CUR.T ? "This term, pooled" : "All lines, pooled") + "</th><th scope=\"col\">Among " + (CUR.T ? "this term's" : "these") + " lines</th></tr></thead><tbody>" +
        F.meta.measures.map(function (ms) {
          var kind = ms.unit === "trials" ? "trials" : "y", uv = userMeasure(ms.key), r = metaFor(CUR.A, ms.key, F);
          var vals = CUR.A.filter(function (l) { return l.m[ms.key]; }).map(function (l) { return ms.key === "half" ? l.m.half[0] : scale(l, l.m[ms.key][0]); });
          var pc = uv == null ? NaN : STATS.percentile(vals, uv);
          var se = U.m[ms.key] && U.m[ms.key][1] != null ? (ms.key === "half" ? U.m[ms.key][1] : unitOf(F) === "pct" ? U.m[ms.key][1] / U.size * 100 : U.m[ms.key][1]) : null;
          return "<tr><th scope=\"row\">" + esc(ms.label) + "</th><td class=\"num\" data-label=\"Yours\">" + (uv == null ? "-" : fv(F, uv, kind, kind === "trials" ? 0 : YD) + (se != null ? " ± " + num(se, YD) : "")) + "</td><td class=\"num\" data-label=\"Pooled\">" + (r ? fv(F, r.est, kind, YD) : "-") + "</td><td data-label=\"\">" +
            (isFinite(pc) ? "percentile " + num(pc, 0) + " of " + vals.length + " lines" : "-") + "</td></tr>";
        }).join("") + "</tbody></table></div></div>";
      h += "<p class=\"ex-note\">Your mean curve is the blue line in the learning curves, and the forest plot marks your value. Percentile: the share of lines with a lower value (ties count half). Values are computed as for every line here (see How the numbers are computed).</p>";
      if (!U.series && U.m.late && U.m.late[0] < 0) h += "<p class=\"ex-warn\">Your late value is negative. If your hand angles are positive when people compensate, change the sign setting above and compute again.</p>";
    }
    h += "<p><button type=\"button\" class=\"btn btn-outline\" id=\"ex-uclear\">Remove your data</button></p></div>";
    box.innerHTML = h;
    $("#ex-uclear").addEventListener("click", function () { USER = null; render(CUR.F, false); setTimeout(function () { var d = $("#ex-drop"); if (d) d.focus(); }, 0); });
  }

  function bindCompare() {
    var drop = $("#ex-drop"), input = $("#ex-file");
    if (!drop) return;
    drop.addEventListener("click", function () { input.click(); });
    drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", function () { if (input.files[0]) readFile(input.files[0]); input.value = ""; });
    ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (e) { var f = e.dataTransfer && e.dataTransfer.files[0]; if (f) readFile(f); });
    if (CMPS) renderMap();
  }
  function setMapStatus(msg, bad) { var m = $("#ex-map"); if (m) m.innerHTML = "<p class=\"" + (bad ? "ex-warn" : "muted") + "\">" + esc(msg) + "</p>"; }
  function readFile(file) {
    setMapStatus("Reading " + file.name + "...");
    var isZip = /\.zip$/i.test(file.name);
    var p = isZip ? file.arrayBuffer().then(CMP.unzipCsv) : file.text().then(function (t) { return { name: file.name, text: t }; });
    p.then(function (got) {
      var tab = CMP.readTable(got.text);
      if (!tab.cols.length || !tab.rows.length) throw new Error("no rows found");
      if (CUR.F.meta.cmp) CMPS = { file: file.name, inner: got.name, cols: tab.cols, rows: tab.rows, series: CUR.F.meta.id, om: false, sel: CMP.guessSeries(tab.cols, CUR.F.meta) };
      else {
        CMPS = { file: file.name, inner: got.name, cols: tab.cols, rows: tab.rows, om: CMP.isOpenMotor(tab.cols), sel: CMP.guess(tab.cols) };
        CMPS.sel.type = CUR.F.meta.id === "clamp" ? "clamp" : "rotation";
      }
      renderMap();
    }).catch(function (err) { setMapStatus("This file could not be read: " + (err && err.message || err) + ".", true); });
  }
  function colOpts(cur, none) {
    return (none ? "<option value=\"\">" + esc(none) + "</option>" : "") + CMPS.cols.map(function (c, i) { return "<option value=\"" + i + "\"" + (String(cur) === String(i) ? " selected" : "") + ">" + esc(c) + "</option>"; }).join("");
  }
  function distinct(ci, filt) {
    var seen = {}, out = [];
    CMPS.rows.forEach(function (r) { if (filt && !filt(r)) return; var v = r[ci]; if (!(v in seen)) { seen[v] = 1; out.push(v); } });
    return out.sort(function (a, b) { return String(a).localeCompare(String(b), undefined, { numeric: true }); });
  }
  function renderMap() {
    var box = $("#ex-map");
    if (!box || !CMPS) return;
    if (CMPS.series) { if (CMPS.series !== CUR.F.meta.id) { CMPS.series = CUR.F.meta.id; CMPS.sel = CMP.guessSeries(CMPS.cols, CUR.F.meta); } return renderMapSeries(box); }
    var c = CMPS, s = c.sel, h = "<form class=\"ex-mapf\" id=\"ex-mapf\"><p class=\"ex-maph\"><b>" + esc(c.file) + "</b>: " + fmtInt(c.rows.length) + " rows, " + c.cols.length + " columns." + (c.om ? " The OpenMotor format was recognised." : "") + "</p><div class=\"ex-mgrid\">";
    function sel(id, label, opts) { return "<label class=\"ex-ff\"><span>" + esc(label) + "</span><select id=\"" + id + "\">" + opts + "</select></label>"; }
    if (!c.om) {
      h += sel("ex-c-subj", "Participant column", colOpts(s.subj)) + sel("ex-c-trial", "Trial column", colOpts(s.trial)) + sel("ex-c-hand", "Hand angle column (degrees)", colOpts(s.hand)) +
        sel("ex-c-pert", "Perturbation column (degrees)", colOpts(s.pert, "(none: give it below)")) + sel("ex-c-fb", "No-feedback column (optional)", colOpts(s.fb, "(every trial had feedback)")) +
        sel("ex-c-type", "Perturbation", "<option value=\"rotation\"" + (s.type === "rotation" ? " selected" : "") + ">rotation</option><option value=\"clamp\"" + (s.type === "clamp" ? " selected" : "") + ">error clamp</option>") +
        sel("ex-c-sign", "Hand angles when people compensate a positive perturbation", "<option value=\"om\"" + (s.sign !== "pos" ? " selected" : "") + ">negative (as in OpenMotor)</option><option value=\"pos\"" + (s.sign === "pos" ? " selected" : "") + ">positive</option>");
      if (s.pert === "" || s.pert == null) {
        h += "<label class=\"ex-ff\"><span>Perturbation size (degrees)</span><input id=\"ex-c-size\" type=\"number\" step=\"any\" value=\"" + esc(s.size || "") + "\"></label>" +
          "<label class=\"ex-ff\"><span>First perturbed trial (your numbering)</span><input id=\"ex-c-a\" type=\"number\" step=\"1\" value=\"" + esc(s.a || "") + "\"></label>" +
          "<label class=\"ex-ff\"><span>Last perturbed trial</span><input id=\"ex-c-b\" type=\"number\" step=\"1\" value=\"" + esc(s.b || "") + "\"></label>";
      }
    }
    var fc = c.om ? c.cols.indexOf("experiment") : s.f1;
    if (c.om) {
      if (fc >= 0) h += sel("ex-c-v1", "Experiment", distinct(fc).map(function (v) { return "<option" + (v === s.v1 ? " selected" : "") + ">" + esc(v) + "</option>"; }).join(""));
      var gc = c.cols.indexOf("group");
      if (gc >= 0) {
        var gv = distinct(gc, fc >= 0 ? function (r) { return r[fc] === (s.v1 != null ? s.v1 : distinct(fc)[0]); } : null);
        h += sel("ex-c-v2", "Group", "<option value=\"\u0000all\">all groups</option>" + gv.map(function (v) { return "<option" + (v === s.v2 ? " selected" : "") + ">" + esc(v === "" ? "(no group)" : v) + "</option>"; }).join(""));
      }
    } else {
      h += sel("ex-c-f1", "Use only rows where", colOpts(s.f1, "(use every row)"));
      if (s.f1 !== "" && s.f1 != null) h += sel("ex-c-v1", "equals", distinct(+s.f1).map(function (v) { return "<option" + (v === s.v1 ? " selected" : "") + ">" + esc(v) + "</option>"; }).join(""));
    }
    h += "</div><p class=\"ex-mact\"><button type=\"submit\" class=\"btn btn-primary\">Compute and compare</button> <button type=\"button\" class=\"linkbutton\" id=\"ex-c-forget\">Choose another file</button></p></form>";
    box.innerHTML = h;
    function read() {
      var g = function (id) { var e = document.getElementById(id); return e ? e.value : undefined; };
      if (!c.om) {
        s.subj = g("ex-c-subj"); s.trial = g("ex-c-trial"); s.hand = g("ex-c-hand"); s.pert = g("ex-c-pert"); s.fb = g("ex-c-fb"); s.type = g("ex-c-type"); s.sign = g("ex-c-sign");
        if (g("ex-c-size") !== undefined) { s.size = g("ex-c-size"); s.a = g("ex-c-a"); s.b = g("ex-c-b"); }
        s.f1 = g("ex-c-f1");
      }
      if (g("ex-c-v1") !== undefined) s.v1 = g("ex-c-v1");
      if (g("ex-c-v2") !== undefined) s.v2 = g("ex-c-v2");
    }
    box.querySelectorAll("select").forEach(function (el) {
      el.addEventListener("change", function () {
        var before = [s.pert, s.f1, s.v1];
        read();
        if (el.id === "ex-c-v1" && c.om) s.v2 = undefined;
        if (before[0] !== s.pert || before[1] !== s.f1 || (c.om && before[2] !== s.v1)) renderMap();
      });
    });
    $("#ex-c-forget").addEventListener("click", function () { CMPS = null; box.innerHTML = ""; $("#ex-drop").focus(); });
    $("#ex-mapf").addEventListener("submit", function (e) {
      e.preventDefault();
      read();
      var res;
      try { res = CMP.fromTable(c, s); } catch (err) { res = { error: err.message || String(err) }; }
      if (res.error) { var p = box.querySelector(".ex-merr") || box.appendChild(document.createElement("p")); p.className = "ex-warn ex-merr"; p.textContent = res.error; return; }
      res.name = c.file;
      USER = res;
      render(CUR.F, false);
      var u = $("#ex-user"); if (u) u.scrollIntoView({ block: "nearest" });
      $("#ex-status").textContent = "Your data were added to the curves and the forest plot.";
    });
  }

  /* column choices for the series families (one row per stride, trial or utterance) */
  function renderMapSeries(box) {
    var c = CMPS, s = c.sel, F = CUR.F, C = F.meta.cmp || {}, u = xu(F);
    function sel(id, label, opts) { return "<label class=\"ex-ff\"><span>" + esc(label) + "</span><select id=\"" + id + "\">" + opts + "</select></label>"; }
    function inp(id, label, v, step) { return "<label class=\"ex-ff\"><span>" + esc(label) + "</span><input id=\"" + id + "\" type=\"number\" step=\"" + (step || "any") + "\" value=\"" + esc(v || "") + "\"></label>"; }
    var h = "<form class=\"ex-mapf\" id=\"ex-mapf\"><p class=\"ex-maph\"><b>" + esc(c.file) + "</b>: " + fmtInt(c.rows.length) + " rows, " + c.cols.length + " columns.</p><div class=\"ex-mgrid\">" +
      sel("ex-c-subj", "Participant column", colOpts(s.subj)) + sel("ex-c-trial", Xu(F) + " number column", colOpts(s.trial)) + sel("ex-c-val", C.value || "Value column", colOpts(s.val)) +
      sel("ex-c-pert", "Perturbation column (0 when off)", colOpts(s.pert, "(none: give the perturbed " + u + "s below)")) +
      inp("ex-c-size", F.meta.size + " (" + F.meta.sizeUnit.replace(/^:/, "x:") + ")", s.size);
    if (s.pert === "" || s.pert == null) h += inp("ex-c-a", "First perturbed " + u + " (your numbering)", s.a, 1) + inp("ex-c-b", "Last perturbed " + u, s.b, 1);
    if (C.ask === "dir") h += sel("ex-c-dir", "Step direction", "<option value=\"back\"" + (s.dir !== "fwd" ? " selected" : "") + ">backward (gain down)</option><option value=\"fwd\"" + (s.dir === "fwd" ? " selected" : "") + ">forward (gain up)</option>");
    if (C.ask === "shift") h += inp("ex-c-shift", "Shift of that formant, signed, in the column's unit", s.shift);
    h += sel("ex-c-f1", "Use only rows where", colOpts(s.f1, "(use every row)"));
    if (s.f1 !== "" && s.f1 != null) h += sel("ex-c-v1", "equals", distinct(+s.f1).map(function (v) { return "<option" + (v === s.v1 ? " selected" : "") + ">" + esc(v) + "</option>"; }).join(""));
    h += "</div><p class=\"ex-note\">" + esc(C.note || "") + " " + Xu(F) + "s before the first perturbed " + u + " are the baseline and the ones after the block the aftereffect; runs of up to 15 unperturbed " + u + "s inside the block (catch " + u + "s) are left out of it.</p>" +
      "<p class=\"ex-mact\"><button type=\"submit\" class=\"btn btn-primary\">Compute and compare</button> <button type=\"button\" class=\"linkbutton\" id=\"ex-c-forget\">Choose another file</button></p></form>";
    box.innerHTML = h;
    function read() {
      var g = function (id) { var e = document.getElementById(id); return e ? e.value : undefined; };
      ["subj", "trial", "val", "pert", "size", "a", "b", "dir", "shift", "f1", "v1"].forEach(function (k) { var v = g("ex-c-" + k); if (v !== undefined) s[k] = v; });
      if (g("ex-c-f1") === "") s.v1 = undefined;
    }
    box.querySelectorAll("select").forEach(function (el) {
      el.addEventListener("change", function () { var before = [s.pert, s.f1]; read(); if (before[0] !== s.pert || before[1] !== s.f1) renderMapSeries(box); });
    });
    $("#ex-c-forget").addEventListener("click", function () { CMPS = null; box.innerHTML = ""; $("#ex-drop").focus(); });
    $("#ex-mapf").addEventListener("submit", function (e) {
      e.preventDefault();
      read();
      var res;
      try { res = CMP.fromSeries(c, s, F.meta); } catch (err) { res = { error: err.message || String(err) }; }
      if (res.error) { var p = box.querySelector(".ex-merr") || box.appendChild(document.createElement("p")); p.className = "ex-warn ex-merr"; p.textContent = res.error; return; }
      res.name = c.file;
      USER = res;
      render(CUR.F, false);
      var ub = $("#ex-user"); if (ub) ub.scrollIntoView({ block: "nearest" });
      $("#ex-status").textContent = "Your data were added to the curves and the forest plot.";
    });
  }
})();
