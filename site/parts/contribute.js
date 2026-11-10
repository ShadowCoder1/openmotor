/* Add your dataset (site/parts/contribute.js, 2026-11-10): the automatic submission flow.
   form -> POST /api/submissions -> PUT each file to its upload URL -> POST .../start -> poll GET /api/submissions/<id>
   while queued, running or publishing. The worker may ask questions (needs_answers); it then shows a preview that the
   researcher approves (POST .../approve) before the dataset is published. The API base is meta.api
   (api_base in source/release_config.json).
   Private link: #contribute/<id>/<key>. When the page opens with it, the key is kept in this browser and the address
   bar becomes #contribute/<id> before the page-view counter reads location.hash, so the key is never counted.
   Files that already follow the OpenMotor format (reach90, or Core plus one task module) are recognised and checked
   here before upload; the submission then goes in as kind "standardized" with that report as client_check. */
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("catalog-data").textContent);
  var API = String((DATA.meta && DATA.meta.api) || "").replace(/\/+$/, "");
  var CONTACT = "physical.intelligence.lab@gmail.com";
  var MAX_FILES = 60, MAX_BYTES = 500e6, SAMPLE = 2000;
  var STORE = "om-submissions", DRAFT = "openmotor-draft";
  var FIELDS = { "om-name": "name", "om-title": "title", "om-author": "author", "om-lab": "lab", "om-year": "year", "om-email": "email",
                 "om-license": "license", "om-task": "task", "om-repo": "repo", "om-doi": "doi", "om-authors": "authors", "om-notes": "notes" };
  var REQUIRED = ["om-name", "om-title", "om-author", "om-lab", "om-email", "om-license"];
  var ICON_X = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';

  function $(s, r) { return (r || document).querySelector(s); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function fmt(n) { return Number(n).toLocaleString("en-US"); }
  function mb(b) { return b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1e3)) + " kB"; }
  function count(n, one, many) { return fmt(n) + " " + (n === 1 ? one : (many || one + "s")); }
  function extOf(name) { var m = /\.([A-Za-z0-9]+)$/.exec(name || ""); return m ? m[1].toLowerCase() : ""; }
  function names(a, max) { max = max || 6; var s = a.slice(0, max).join(", "); return a.length > max ? s + " and " + (a.length - max) + " more" : s; }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function when(iso) { var d = new Date(iso); return isNaN(d) ? "" : d.getDate() + " " + MON[d.getMonth()] + " " + d.getFullYear() + ", " + pad(d.getHours()) + ":" + pad(d.getMinutes()); }
  function clock() { var d = new Date(); return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()); }

  /* ---------- API ---------- */
  function sp(id) { return "/api/submissions/" + encodeURIComponent(id); }
  function api(path, opt) {
    opt = opt || {};
    var headers = {}, init = { method: opt.body !== undefined ? "POST" : "GET", headers: headers, cache: "no-store" };
    if (opt.key) headers["X-Submission-Key"] = opt.key;
    if (opt.body !== undefined) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opt.body); }
    return fetch(API + path, init).then(function (r) {
      return r.text().then(function (t) {
        var d = null; try { d = t ? JSON.parse(t) : null; } catch (e) {}
        if (!r.ok) { var err = new Error((d && d.error) || "The server answered with an error (" + r.status + ")."); err.status = r.status; err.data = d; throw err; }
        return d;
      });
    }, function () { var err = new Error("Could not reach the OpenMotor server. Check your connection."); err.status = 0; throw err; });
  }
  var HEALTH = null;
  function health() { if (!HEALTH) HEALTH = api("/api/health").catch(function () { HEALTH = null; return null; }); return HEALTH; }

  /* ---------- this browser's submissions ---------- */
  var MEM = {};
  function loadMine() { try { var a = JSON.parse(localStorage.getItem(STORE) || "[]"); return Array.isArray(a) ? a.filter(function (s) { return s && s.id && s.key; }) : []; } catch (e) { return []; } }
  function saveMine(list) { try { localStorage.setItem(STORE, JSON.stringify(list.slice(0, 30))); } catch (e) {} }
  function remember(id, key, info) {
    MEM[id] = key;
    var list = loadMine(), i = -1;
    list.forEach(function (s, k) { if (s.id === id) i = k; });
    var row = i >= 0 ? list[i] : { id: id, added: new Date().toISOString() };
    Object.keys(info || {}).forEach(function (k) { if (info[k] != null) row[k] = info[k]; });
    row.key = key;
    if (i >= 0) list[i] = row; else list.unshift(row);
    saveMine(list);
  }
  function keyOf(id) { if (MEM[id]) return MEM[id]; var k = null; loadMine().forEach(function (s) { if (s.id === id) k = s.key; }); return k; }
  function privateLink(id, key) { return location.origin + location.pathname + "#contribute/" + encodeURIComponent(id) + "/" + encodeURIComponent(key); }

  /* ---------- OpenMotor format check, in the browser ---------- */
  var REACH = [];
  (DATA.columns || []).forEach(function (g) { (g.columns || []).forEach(function (c) { REACH.push(c); }); });
  var TF = DATA.taskFormats || [];
  var CORE = TF.length && TF[0].name === "Core" ? TF[0].columns : [];
  function fmtLabel(n) { return n.replace(/_/g, " ").toLowerCase().replace(/\b(3d|vr|rt|emg)\b/g, function (w) { return w.toUpperCase(); }); }
  var FORMATS = [{ id: "reach90", label: "reaching", cols: REACH }].concat(CORE.length ? TF.filter(function (t) { return t.name !== "Core"; }).map(function (t) {
    return { id: t.name, label: fmtLabel(t.name), cols: CORE.concat(t.columns) };
  }) : []).filter(function (f) { return f.cols.length; });

  function matchFormat(header) {
    var have = {}; header.forEach(function (h) { have[h] = true; });
    function missingOf(cols) { return cols.filter(function (c) { return !have[c.name]; }).map(function (c) { return c.name; }); }
    var best = null;
    FORMATS.forEach(function (f) {
      var miss = missingOf(f.cols);
      if (!miss.length && (!best || f.cols.length > best.cols.length)) best = { id: f.id, label: f.label, cols: f.cols, missing: [] };
    });
    var near = null;
    if (!best) {
      var mr = missingOf(REACH);
      if (REACH.length && mr.length / REACH.length <= 0.2) near = { id: "reach90", label: "reaching", cols: REACH, missing: mr };
      else if (CORE.length && !missingOf(CORE).length) near = { id: "core", label: "core", cols: CORE, missing: [] };
      if (!near) return null;
      best = near;
    }
    var known = {}; best.cols.forEach(function (c) { known[c.name] = true; });
    best.match = near ? "near" : "full";
    best.extra = header.filter(function (h) { return h && !known[h]; });
    return best;
  }

  /* column names that usually hold identifying information (format columns are exempt) */
  var PERSONAL = /^(names?|participant_?name|subject_?name|patient_?name|first_?name|last_?name|given_?name|family_?name|full_?name|surname|initials|e_?mail(_?address)?|mail|phone(_?number)?|telephone|mobile|address|street|dob|birth_?date|date_?of_?birth|birthday|ssn|zip(_?code)?|post_?code|postal_?code|ip(_?address)?|prolific(_?p?id)?|worker_?id|mturk(_?id)?)$/;
  function personalColumn(h) { var n = h.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, ""); return PERSONAL.test(n) || /e_?mail|phone|birth|surname|first_?name|last_?name|ip_?addr/.test(n); }
  var RX_NUM = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/, RX_INT = /^[-+]?\d+(\.0*)?$/, RX_BOOL = /^(true|false|TRUE|FALSE|True|False|0|1)$/,
      RX_NAN = /^[-+]?(nan|NaN|NAN|inf|Inf|INF|infinity|Infinity)$/, RX_EMAIL = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/,
      RX_PLATFORM = /^([0-9a-f]{24}|A[0-9A-Z]{11,14})$/;

  /* a small streaming CSV/TSV reader: quotes, doubled quotes, CRLF, chunk boundaries */
  function csvParser(onRow) {
    var D = -1, field = "", row = [], inQ = false, carry = "";
    function detect(t) {
      var nl = t.indexOf("\n"), h = nl < 0 ? t : t.slice(0, nl), c = h.split(",").length, tb = h.split("\t").length, sc = h.split(";").length;
      D = tb > c && tb >= sc ? 9 : (sc > c ? 59 : 44);
    }
    function emit() { row.push(field); field = ""; var r = row; row = []; onRow(r); }
    function run(t) {
      var n = t.length, i = 0, ch, j, q;
      while (i < n) {
        if (inQ) {
          q = t.indexOf('"', i);
          if (q < 0) { field += t.slice(i); break; }
          field += t.slice(i, q);
          if (q + 1 < n && t.charCodeAt(q + 1) === 34) { field += '"'; i = q + 2; } else { inQ = false; i = q + 1; }
          continue;
        }
        if (t.charCodeAt(i) === 34 && field === "") { inQ = true; i++; continue; }
        j = i;
        while (j < n) { ch = t.charCodeAt(j); if (ch === D || ch === 10 || ch === 13) break; j++; }
        if (j > i) field += t.slice(i, j);
        if (j >= n) break;
        if (ch === D) { row.push(field); field = ""; i = j + 1; }
        else { emit(); i = j + 1; if (ch === 13 && i < n && t.charCodeAt(i) === 10) i++; }
      }
    }
    return {
      push: function (t) {
        t = carry + t; carry = "";
        if (D < 0) { if (t.indexOf("\n") < 0 && t.length < 4e6) { carry = t; return; } detect(t); }
        var lc = t.charCodeAt(t.length - 1);
        if (lc === 34 || lc === 13) { carry = t.slice(-1); t = t.slice(0, -1); }
        run(t);
      },
      end: function () { var t = carry; carry = ""; if (D < 0) detect(t); if (t) run(t); if (field !== "" || row.length) emit(); }
    };
  }

  function setupChecks(st) {
    var f = st.fmt, idx = {};
    st.header.forEach(function (h, i) { if (!(h in idx)) idx[h] = i; });
    st.iSub = "subject" in idx ? idx.subject : -1;
    st.iPer = "person" in idx ? idx.person : -1;
    st.iExp = "experiment" in idx ? idx.experiment : -1;
    st.checks = [];
    f.cols.forEach(function (c) {
      if (!(c.name in idx)) return;
      var t = String(c.type || "").toLowerCase(), allowed = null;
      if (t === "category" && c.values) {
        var vals = String(c.values).split(/\s*,\s*/);
        if (vals.length && vals.every(function (v) { return /^[A-Za-z0-9_.+-]+$/.test(v); })) { allowed = {}; vals.forEach(function (v) { allowed[v] = true; }); }
      }
      if (t === "number" || t === "integer" || t === "true/false" || allowed) st.checks.push({ i: idx[c.name], name: c.name, type: t, allowed: allowed, values: c.values });
    });
    st.subs = new Set(); st.pers = new Set(); st.exps = new Set();
    st.bad = {}; st.cat = {}; st.emptySub = 0; st.badLen = 0; st.badLenAt = 0; st.platform = 0; st.email = null;
  }
  function onRow(st, row) {
    if (row.length === 1 && row[0] === "") return;
    if (!st.header) {
      st.header = row.map(function (h) { return h.replace(/^﻿/, "").trim(); });
      st.fmt = matchFormat(st.header);
      var inFmt = {};
      if (st.fmt) st.fmt.cols.forEach(function (c) { inFmt[c.name] = true; });
      st.personal = st.header.filter(function (h) { return h && !inFmt[h] && personalColumn(h); });
      if (!st.fmt || st.fmt.match !== "full") st.stop = true; else { setupChecks(st); if (st.onFormat) st.onFormat(st.fmt); }
      return;
    }
    var n = ++st.rows;
    if (row.length !== st.header.length) { st.badLen++; if (!st.badLenAt) st.badLenAt = n; }
    if (st.iSub >= 0) { var sv = row[st.iSub]; if (!sv) st.emptySub++; else { st.subs.add(sv); if (n <= SAMPLE && RX_PLATFORM.test(sv)) st.platform++; } }
    if (st.iPer >= 0 && row[st.iPer]) { st.pers.add(row[st.iPer]); if (n <= SAMPLE && RX_PLATFORM.test(row[st.iPer])) st.platform++; }
    if (st.iExp >= 0 && row[st.iExp]) st.exps.add(row[st.iExp]);
    if (n > SAMPLE) return;
    for (var k = 0; k < st.checks.length; k++) {
      var c = st.checks[k], v = row[c.i];
      if (v === undefined || v === "") continue;
      var ok = c.type === "number" ? (RX_NUM.test(v) || RX_NAN.test(v)) : c.type === "integer" ? RX_INT.test(v) : c.type === "true/false" ? RX_BOOL.test(v) : true;
      if (!ok && !st.bad[c.name]) st.bad[c.name] = { type: c.type, value: v, row: n };
      if (ok && c.allowed && !c.allowed[v] && !st.cat[c.name]) st.cat[c.name] = { value: v, values: c.values };
    }
    if (!st.email) for (var j = 0; j < row.length; j++) if (row[j].indexOf("@") >= 0 && RX_EMAIL.test(row[j])) { st.email = { col: st.header[j] || "column " + (j + 1), row: n }; break; }
  }
  function finish(st) {
    var out = { columns: st.header ? st.header.length : 0, personal: st.personal || [] };
    if (!st.fmt) return out;
    out.format = st.fmt.id; out.format_label = st.fmt.label; out.match = st.fmt.match;
    out.missing = st.fmt.missing.slice(0, 60); out.extra = st.fmt.extra.slice(0, 60);
    if (st.fmt.match !== "full") return out;
    out.rows = st.rows; out.rows_complete = !!st.complete; out.sampled_rows = Math.min(st.rows, SAMPLE);
    out.subjects = st.subs.size; out.participants = st.iPer >= 0 ? st.pers.size : st.subs.size; out.experiments = st.exps.size;
    var P = [], W = [];
    Object.keys(st.bad).forEach(function (name) {
      var b = st.bad[name], what = b.type === "integer" ? "should hold whole numbers" : b.type === "true/false" ? "should hold true or false" : "has values that are not numbers";
      P.push('Column "' + name + '" ' + what + ', for example "' + String(b.value).slice(0, 40) + '" in row ' + fmt(b.row) + ".");
    });
    if (st.iSub >= 0 && st.emptySub) P.push(count(st.emptySub, "row has", "rows have") + " no subject code.");
    if (st.badLen) P.push(count(st.badLen, "row has", "rows have") + " a different number of cells than the header, first at row " + fmt(st.badLenAt) + ".");
    if (st.email) P.push('Some cells look like email addresses, for example in column "' + st.email.col + '", row ' + fmt(st.email.row) + ". Remove them.");
    if (st.platform) P.push("Some subject or person codes look like Prolific or MTurk IDs. Replace them with codes such as s01.");
    st.personal.forEach(function (h) { P.push('Column "' + h + '" may hold personal information. Remove it if it does.'); });
    if (st.fmt.extra.length) W.push("Columns that are not part of the format: " + names(st.fmt.extra) + ".");
    Object.keys(st.cat).slice(0, 4).forEach(function (name) { var c = st.cat[name]; W.push('Column "' + name + '" has values outside the format\'s list, for example "' + String(c.value).slice(0, 30) + '" (the list: ' + c.values + ")."); });
    if (st.error) W.push("The file could not be read to the end here, so the counts are partial.");
    out.problems = P; out.warnings = W;
    return out;
  }
  async function scanStream(stream, onFormat) {
    var st = { header: null, fmt: null, rows: 0, stop: false, onFormat: onFormat }, reader = stream.pipeThrough(new TextDecoderStream()).getReader();
    var parser = csvParser(function (row) { onRow(st, row); });
    try {
      while (!st.stop) {
        var r = await reader.read();
        if (r.done) { parser.end(); st.complete = true; break; }
        parser.push(r.value);
      }
    } catch (e) { st.error = true; }
    try { reader.cancel(); } catch (e) {}
    return finish(st);
  }

  /* ZIP: read the central directory and stream single entries; nothing else of the archive is loaded */
  async function zipList(file) {
    var tailLen = Math.min(file.size, 65557), tail = new DataView(await file.slice(file.size - tailLen).arrayBuffer()), e = -1;
    for (var i = tailLen - 22; i >= 0; i--) if (tail.getUint32(i, true) === 0x06054b50) { e = i; break; }
    if (e < 0) throw new Error("not a zip");
    var n = tail.getUint16(e + 10, true), cdSize = tail.getUint32(e + 12, true), cdOff = tail.getUint32(e + 16, true);
    if (cdOff === 0xffffffff || cdSize === 0xffffffff || n === 0xffff) {
      var loc = e - 20;
      if (loc < 0 || tail.getUint32(loc, true) !== 0x07064b50) throw new Error("zip64");
      var zo = Number(tail.getBigUint64(loc + 8, true)), z = new DataView(await file.slice(zo, zo + 56).arrayBuffer());
      if (z.getUint32(0, true) !== 0x06064b50) throw new Error("zip64");
      cdSize = Number(z.getBigUint64(40, true)); cdOff = Number(z.getBigUint64(48, true));
    }
    if (cdSize > 64e6) throw new Error("directory too large");
    var cd = new DataView(await file.slice(cdOff, cdOff + cdSize).arrayBuffer()), u8 = new Uint8Array(cd.buffer), dec = new TextDecoder(), out = [], p = 0;
    while (p + 46 <= cd.byteLength && cd.getUint32(p, true) === 0x02014b50) {
      var flags = cd.getUint16(p + 8, true), method = cd.getUint16(p + 10, true), csize = cd.getUint32(p + 20, true), usize = cd.getUint32(p + 24, true);
      var nl = cd.getUint16(p + 28, true), xl = cd.getUint16(p + 30, true), cl = cd.getUint16(p + 32, true), lho = cd.getUint32(p + 42, true);
      var name = dec.decode(u8.subarray(p + 46, p + 46 + nl));
      if (usize === 0xffffffff || csize === 0xffffffff || lho === 0xffffffff) {
        for (var x = p + 46 + nl, xe = x + xl; x + 4 <= xe;) {
          var id = cd.getUint16(x, true), sz = cd.getUint16(x + 2, true), q = x + 4;
          if (id === 1) {
            if (usize === 0xffffffff) { usize = Number(cd.getBigUint64(q, true)); q += 8; }
            if (csize === 0xffffffff) { csize = Number(cd.getBigUint64(q, true)); q += 8; }
            if (lho === 0xffffffff) lho = Number(cd.getBigUint64(q, true));
            break;
          }
          x += 4 + sz;
        }
      }
      out.push({ name: name, method: method, csize: csize, lho: lho, enc: !!(flags & 1), dir: /\/$/.test(name) });
      p += 46 + nl + xl + cl;
    }
    return out;
  }
  async function zipStream(file, e) {
    if (e.enc || (e.method !== 0 && e.method !== 8) || (e.method === 8 && typeof DecompressionStream === "undefined")) return null;
    var h = new DataView(await file.slice(e.lho, e.lho + 30).arrayBuffer());
    if (h.getUint32(0, true) !== 0x04034b50) return null;
    var start = e.lho + 30 + h.getUint16(26, true) + h.getUint16(28, true), s = file.slice(start, start + e.csize).stream();
    return e.method === 8 ? s.pipeThrough(new DecompressionStream("deflate-raw")) : s;
  }

  var TABLE_EXT = { csv: 1, tsv: 1, txt: 1 };
  var SUPPORT_EXT = { md: 1, txt: 1, pdf: 1, json: 1, yml: 1, yaml: 1, png: 1, jpg: 1, jpeg: 1, gif: 1, svg: 1, m: 1, py: 1, r: 1, rmd: 1, ipynb: 1, html: 1, htm: 1, doc: 1, docx: 1, rtf: 1, cff: 1, bib: 1, license: 1, sha256: 1, log: 1 };
  function supportName(n) { var b = n.split("/").pop().toLowerCase(); return /^(readme|license|licence|changelog|citation)/.test(b) || /(^|_)(columns|codebook|data_?dictionary|dictionary)\.(csv|tsv)$/.test(b); }
  function junk(n) { var b = n.split("/").pop(); return /^__MACOSX\//.test(n) || /^\._/.test(b) || b === ".DS_Store" || b === "Thumbs.db"; }
  /* -> {tables: [...], data: number of data files that are not in the format, entries} */
  async function checkFile(f) {
    var ext = extOf(f.name), hook = function (fm) { if (f.check && f.check.state === "checking") { f.check.label = fm.label; if (FILES.indexOf(f) >= 0) renderFiles(); } };
    if (ext === "zip") {
      var list = await zipList(f.file), tables = [], data = 0, scanned = 0;
      for (var i = 0; i < list.length; i++) {
        var e = list[i], ee = extOf(e.name);
        if (e.dir || junk(e.name)) continue;
        if (TABLE_EXT[ee] && scanned < 200) {
          var s = await zipStream(f.file, e);
          if (!s) { if (!SUPPORT_EXT[ee]) data++; continue; }
          scanned++;
          var t = await scanStream(s, hook); t.entry = e.name;
          tables.push(t);
          if (t.match !== "full" && !supportName(e.name) && ee !== "txt") data++;
        } else if (!SUPPORT_EXT[ee] && !supportName(e.name)) data++;
      }
      return { tables: tables, data: data, zip: true };
    }
    if (TABLE_EXT[ext]) {
      var r = await scanStream(f.file.stream(), hook);
      return { tables: [r], data: r.match === "full" || supportName(f.name) || ext === "txt" ? 0 : 1 };
    }
    return { tables: [], data: SUPPORT_EXT[ext] || supportName(f.name) ? 0 : 1 };
  }
  function rowsText(t) { return (t.rows_complete ? fmt(t.rows) : "at least " + fmt(t.rows)) + " rows" + (t.participants ? ", " + count(t.participants, "participant") : ""); }
  function noteFor(f) {
    var c = f.check;
    if (!c) return "";
    if (c.state === "checking") return c.label ? "Follows the OpenMotor " + esc(c.label) + " format. Counting rows and checking values..." : "Reading the column names...";
    if (c.state !== "done") return "";
    var full = c.tables.filter(function (t) { return t.match === "full"; }), near = c.tables.filter(function (t) { return t.match === "near"; }), h = "";
    var probs = [], warns = [], personal = [];
    full.forEach(function (t) {
      var where = c.zip ? '<code>' + esc(t.entry) + '</code>' : "This file";
      h += "<p><b>" + where + " follows the OpenMotor " + esc(t.format_label) + " format.</b> " + esc(rowsText(t)) + "." + (t.problems.length || t.warnings.length ? "" : " No problems found.") + "</p>";
      t.problems.forEach(function (p) { probs.push((c.zip && full.length > 1 ? t.entry + ": " : "") + p); });
      t.warnings.forEach(function (p) { warns.push((c.zip && full.length > 1 ? t.entry + ": " : "") + p); });
    });
    if (!full.length) near.slice(0, 1).forEach(function (t) {
      var where = c.zip ? '<code>' + esc(t.entry) + '</code>' : "This file";
      h += t.format === "core" ? "<p>" + where + " has the OpenMotor core columns but not all columns of one task. It will be standardized after upload.</p>"
        : "<p>" + where + " is close to the OpenMotor " + esc(t.format_label) + " format but lacks " + count(t.missing.length, "of its columns", "of its columns") + " (" + esc(names(t.missing, 5)) + "). It will be standardized after upload.</p>";
    });
    c.tables.forEach(function (t) { if (t.match !== "full") t.personal.forEach(function (p) { personal.push((c.zip ? t.entry + ": " : "") + 'column "' + p + '"'); }); });
    if (personal.length) probs.push("These may hold personal information: " + names(personal, 4) + ". Remove them if they do.");
    if (probs.length) h += '<p class="fl-h">To fix before publishing:</p><ul>' + probs.slice(0, 8).map(function (p) { return "<li>" + esc(p) + "</li>"; }).join("") + "</ul>";
    if (warns.length) h += '<p class="fl-h">Also check:</p><ul>' + warns.slice(0, 6).map(function (p) { return "<li>" + esc(p) + "</li>"; }).join("") + "</ul>";
    return h;
  }
  function computeKind() {
    var any = false, raw = false;
    FILES.forEach(function (f) {
      var c = f.check;
      if (!c || c.state !== "done") { raw = true; return; }
      if (c.tables.some(function (t) { return t.match === "full"; })) any = true;
      if (c.data) raw = true;
    });
    return any && !raw ? "standardized" : "raw";
  }
  function clientReport(kind) {
    return { version: 1, made_by: "openmotor-site", kind: kind, checked: new Date().toISOString(), sampled_rows_per_file: SAMPLE,
      files: FILES.map(function (f) {
        var c = f.check || {};
        return { name: f.name, size: f.size, checked: c.state === "done", other_data_files: c.data || 0,
          tables: (c.tables || []).slice(0, 100).map(function (t) { var o = {}; Object.keys(t).forEach(function (k) { o[k] = t[k]; }); return o; }) };
      }) };
  }

  /* ---------- the form ---------- */
  var form = $("#ct-form"), subEl = $("#ct-sub"), mineEl = $("#ct-mine"), reviewEl = $("#ct-review"), crumbs = $("#ct-crumbs");
  if (!form || !subEl) return;
  var drop = $("#om-drop"), picker = $("#om-files"), statusEl = $("#om-status"), submitBtn = $("#om-submit");
  var FILES = [], busy = false;
  function say(t) { statusEl.textContent = t; }

  function renderFiles() {
    $("#om-file-list").innerHTML = FILES.map(function (f, i) {
      var note = noteFor(f);
      return '<li><span class="fl-name">' + esc(f.name) + '</span><span class="fl-size">' + mb(f.size) + "</span>" +
        '<button type="button" class="fl-x" data-rm="' + i + '" aria-label="Remove ' + esc(f.name) + '" title="Remove">' + ICON_X + "</button>" +
        (note ? '<div class="fl-note">' + note + "</div>" : "") + "</li>";
    }).join("");
    var kn = $("#om-kind-note"), done = FILES.length && FILES.every(function (f) { return f.check && f.check.state !== "checking"; });
    kn.hidden = !(done && computeKind() === "standardized");
    kn.textContent = "Your files already follow the OpenMotor format, so they will be checked as they are, without conversion.";
  }
  function addFiles(list) {
    var msgs = [];
    Array.prototype.forEach.call(list, function (file) {
      if (!file.size) { msgs.push(file.name + " is empty and was not added."); return; }
      var i = -1;
      FILES.forEach(function (g, k) { if (g.name === file.name) i = k; });
      if (i >= 0 && FILES[i].size === file.size && FILES[i].file.lastModified === file.lastModified) return;
      var f = { name: file.name, size: file.size, file: file, check: null };
      if (i >= 0) { FILES[i] = f; msgs.push("Replaced " + file.name + " with the file you just added."); } else FILES.push(f);
      startCheck(f);
    });
    renderFiles();
    var total = FILES.reduce(function (t, f) { return t + f.size; }, 0);
    if (FILES.length > MAX_FILES) msgs.push("That is " + FILES.length + " files; the limit is " + MAX_FILES + ". Put them in a ZIP, or give a repository link.");
    else if (total > MAX_BYTES) msgs.push("The files add up to " + mb(total) + "; the limit is 500 MB. For larger datasets, put the files in a repository (OSF, Zenodo, Figshare) and give the link.");
    say(msgs.join(" "));
  }
  function startCheck(f) {
    var ext = extOf(f.name);
    if (!(ext === "zip" || TABLE_EXT[ext]) || typeof TextDecoderStream === "undefined" || !f.file.stream) { f.check = { state: "done", tables: [], data: ext === "zip" || TABLE_EXT[ext] || !(SUPPORT_EXT[ext] || supportName(f.name)) ? 1 : 0 }; return; }
    f.check = { state: "checking" };
    f.promise = checkFile(f).then(function (r) { r.state = "done"; f.check = r; }, function () { f.check = { state: "error", tables: [], data: 1 }; })
      .then(function () { if (FILES.indexOf(f) >= 0) renderFiles(); });
  }
  drop.addEventListener("click", function () { picker.click(); });
  drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); } });
  picker.addEventListener("change", function () { addFiles(picker.files); picker.value = ""; });
  ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) {
    var dt = e.dataTransfer; if (!dt) return;
    var folders = Array.prototype.some.call(dt.items || [], function (it) { var en = it.webkitGetAsEntry && it.webkitGetAsEntry(); return en && en.isDirectory; });
    if (folders) { say("Folders cannot be added here. Put the folder in a ZIP, or choose the files inside it."); return; }
    if (dt.files) addFiles(dt.files);
  });
  $("#om-file-list").addEventListener("click", function (e) {
    var b = e.target.closest("[data-rm]"); if (!b) return;
    var f = FILES[+b.getAttribute("data-rm")]; FILES.splice(+b.getAttribute("data-rm"), 1);
    renderFiles(); say(f ? "Removed " + f.name + "." : ""); drop.focus();
  });

  function val(id) { return $("#" + id).value.trim(); }
  function fillForm(meta) { Object.keys(FIELDS).forEach(function (id) { var v = meta[FIELDS[id]]; $("#" + id).value = v == null ? "" : v; $("#" + id).removeAttribute("aria-invalid"); }); }
  try { var saved = JSON.parse(localStorage.getItem(DRAFT) || "null"); if (saved) Object.keys(FIELDS).forEach(function (id) { if (saved[id] != null) $("#" + id).value = saved[id]; }); } catch (err) {}
  $("#om-save").addEventListener("click", function () {
    var v = {}; Object.keys(FIELDS).forEach(function (id) { v[id] = $("#" + id).value; });
    try { localStorage.setItem(DRAFT, JSON.stringify(v)); say("Draft saved in this browser. Files are not saved; add them again when you come back."); }
    catch (err) { say("This browser would not save the draft; keep this page open instead."); }
  });
  function validate() {
    var bad = null, msg = "";
    REQUIRED.forEach(function (id) {
      var el = $("#" + id), ok = !!el.value.trim() && (id !== "om-email" || /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(el.value.trim()));
      el.setAttribute("aria-invalid", ok ? "false" : "true"); if (!ok && !bad) { bad = el; msg = "Fill in the fields marked with a star; the email needs to be a full address."; }
    });
    var repo = $("#om-repo"), rv = repo.value.trim(), rok = !rv || /^https:\/\/[^\s/]+\.[^\s]+$/i.test(rv);
    repo.setAttribute("aria-invalid", rok ? "false" : "true");
    if (!rok && !bad) { bad = repo; msg = "The repository link needs to be a full address starting with https://."; }
    var yr = $("#om-year"), yv = yr.value.trim(), yok = !yv || (/^\d{4}$/.test(yv) && +yv >= 1950 && +yv <= 2100);
    yr.setAttribute("aria-invalid", yok ? "false" : "true");
    if (!yok && !bad) { bad = yr; msg = "The publication year needs four digits, for example 2025."; }
    if (!bad && !FILES.length && !rv) { bad = drop; msg = "Add your files, or give the repository link where they are."; }
    var total = FILES.reduce(function (t, f) { return t + f.size; }, 0);
    if (!bad && FILES.length > MAX_FILES) { bad = drop; msg = "That is " + FILES.length + " files; the limit is " + MAX_FILES + ". Put them in a ZIP, or give a repository link and remove the files here."; }
    if (!bad && total > MAX_BYTES) { bad = drop; msg = "The files add up to " + mb(total) + "; the limit is 500 MB. Put them in a repository (OSF, Zenodo, Figshare), give the link, and remove the files here."; }
    return bad ? { el: bad, msg: msg } : null;
  }
  function metaFromForm() {
    var m = {};
    Object.keys(FIELDS).forEach(function (id) { var v = val(id); if (v) m[FIELDS[id]] = id === "om-year" ? +v : v; });
    return m;
  }
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy) return;
    var bad = validate();
    if (bad) { say(bad.msg); bad.el.focus(); return; }
    if (!API) { say("Submissions are not open right now. Email " + CONTACT + " with your dataset."); return; }
    submit();
  });
  async function submit() {
    busy = true; submitBtn.disabled = true;
    try {
      var pending = FILES.filter(function (f) { return f.check && f.check.state === "checking"; });
      if (pending.length) { submitBtn.textContent = "Checking files..."; say("Reading your files before the upload."); await Promise.race([Promise.all(pending.map(function (f) { return f.promise; })), sleep(60000)]); }
      submitBtn.textContent = "Sending...";
      say("Creating the submission.");
      var meta = metaFromForm(), kind = computeKind(), files = FILES.slice();
      var body = { meta: meta, files: files.map(function (f) { return { name: f.name, size: f.size }; }), link: files.length ? "" : (meta.repo || ""), kind: kind, website: $("#om-website").value };
      if (files.some(function (f) { return f.check && f.check.tables && f.check.tables.length; })) body.client_check = clientReport(kind);
      var r = await api("/api/submissions", { body: body });
      var urls = {}; (r.uploads || []).forEach(function (u) { urls[u.name] = u.url; });
      remember(r.id, r.key, { name: meta.name, status: r.status || "uploading", created: new Date().toISOString() });
      try { localStorage.removeItem(DRAFT); } catch (err) {}
      SUB = { id: r.id, key: r.key, view: null, meta: meta, kind: kind, fresh: true,
              local: { active: true, files: files.map(function (f, i) {
                var u = urls[f.name] || (r.uploads && r.uploads.length === files.length && r.uploads[i].url);
                return { name: f.name, size: f.size, file: f.file, url: u, loaded: 0, state: "waiting", tries: 0 };
              }) } };
      form.reset(); FILES = []; renderFiles(); say("");
      location.hash = "#contribute/" + encodeURIComponent(r.id);
      startUploads(SUB);
    } catch (err) {
      say(err.status === 429 ? err.message : "The submission was not created. " + err.message);
    } finally {
      busy = false; submitBtn.disabled = false; submitBtn.textContent = "Submit dataset";
    }
  }

  /* ---------- your submissions (this browser) ---------- */
  var WORD = { uploading: "Upload not finished", queued: "Waiting to start", running: "Being processed", needs_answers: "Has questions for you", preview: "Ready for your review",
               publishing: "Publishing", published: "Published", failed: "Stopped with an error", rejected: "Not accepted", withdrawn: "Withdrawn", expired: "Expired", missing: "Not found" };
  var FINAL = { published: 1, failed: 1, rejected: 1, withdrawn: 1, expired: 1, missing: 1 };
  function renderMine(refresh) {
    var list = loadMine();
    mineEl.hidden = !list.length;
    if (!list.length) return;
    mineEl.innerHTML = '<h2 class="ct-h3">Your submissions</h2><ul class="mine-list">' + list.slice(0, 10).map(function (s) {
      return '<li><a href="#contribute/' + esc(encodeURIComponent(s.id)) + '">' + esc(s.name || s.id) + '</a><span class="mine-st">' + esc(WORD[s.status] || "") + '</span><span class="mine-date">' + esc(when(s.created || s.added)) + "</span></li>";
    }).join("") + '</ul><p class="mine-note">Kept in this browser only. On another device, open the private link you saved.</p>';
    if (!refresh || !API) return;
    var todo = list.slice(0, 10).filter(function (s) { return !FINAL[s.status]; });
    Promise.all(todo.map(function (s) {
      return api(sp(s.id), { key: s.key }).then(function (v) { remember(s.id, s.key, { status: v.status, name: v.meta && v.meta.name }); }, function (e) { if (e.status === 404) remember(s.id, s.key, { status: "missing" }); });
    })).then(function () { if (!form.hidden && todo.length) renderMine(false); });
  }

  /* ---------- one submission ---------- */
  var SUB = null, POLL = { t: 0, since: 0, failing: false };
  var HEAD = { uploading: "The upload did not finish", queued: "In the queue", running: "Working on your files", needs_answers: "Questions for you", preview: "Ready for your review",
               publishing: "Publishing", published: "Published", failed: "Processing stopped", rejected: "Not accepted", withdrawn: "Withdrawn", expired: "Expired" };
  var STEPS = ["Upload", "Standardize and check", "Your review", "Published"];
  var STEP_OF = { local: 0, uploading: 0, queued: 1, running: 1, needs_answers: 1, preview: 2, publishing: 3, published: 3 };
  var LIVE = { queued: 1, running: 1, publishing: 1 };
  var OPEN = { local: 1, uploading: 1, queued: 1, running: 1, needs_answers: 1, preview: 1 };

  function mode() { var L = SUB.local; if (L && (L.active || L.failed || L.starting)) return "local"; return SUB.view ? SUB.view.status : (SUB.state || "loading"); }
  function showForm() {
    stopPoll();
    subEl.hidden = true; form.hidden = false; reviewEl.hidden = false;
    crumbs.innerHTML = '<a href="#home">Home</a><span aria-hidden="true">/</span><span>Add your dataset</span>';
    renderMine(true);
  }
  function openSubmission(id, key) {
    form.hidden = true; reviewEl.hidden = true; mineEl.hidden = true; subEl.hidden = false;
    crumbs.innerHTML = '<a href="#home">Home</a><span aria-hidden="true">/</span><a href="#contribute">Add your dataset</a><span aria-hidden="true">/</span><span>Your submission</span>';
    if (SUB && SUB.id === id) {
      if (key) SUB.key = key;
      renderAll(true);
      if (SUB.view && !SUB.local) load(); else schedule();
      return;
    }
    stopPoll();
    SUB = { id: id, key: key || keyOf(id), view: null, local: null };
    renderAll(true);
    if (SUB.key) load();
  }
  function load() {
    var s = SUB;
    api(sp(s.id), { key: s.key }).then(function (v) { if (SUB === s) setView(v); }, function (err) {
      if (SUB !== s) return;
      if (err.status === 404) { s.state = "missing"; remember(s.id, s.key, { status: "missing" }); renderAll(); return; }
      if (s.view && LIVE[s.view.status]) { POLL.failing = true; setChecked(err.message + " Trying again."); schedule(); return; }
      s.state = "error"; s.error = err.message; renderAll();
    });
  }
  function setView(v) {
    var s = SUB, prev = s.view ? s.view.status : null, prevMsg = s.view ? s.view.message : null;
    s.view = v; s.state = null; POLL.failing = false;
    if (v.meta) s.meta = v.meta;
    remember(s.id, s.key, { status: v.status, name: v.meta && v.meta.name, created: v.created });
    if (prev !== v.status || s.fresh) { s.fresh = false; renderAll(); }
    else if (prevMsg !== v.message) renderNow();
    setChecked("Last checked at " + clock() + ".");
    schedule();
  }
  function stopPoll() { clearTimeout(POLL.t); POLL.t = 0; }
  function schedule() {
    stopPoll();
    if (!SUB || !SUB.view || SUB.local && SUB.local.active || !LIVE[SUB.view.status] || subEl.hidden) return;
    if (!POLL.since) POLL.since = Date.now();
    var slow = Date.now() - POLL.since > 600000 || document.hidden || POLL.failing;
    POLL.t = setTimeout(load, slow ? 15000 : 4000);
  }
  document.addEventListener("visibilitychange", function () { if (!document.hidden && SUB && SUB.view && LIVE[SUB.view.status] && !subEl.hidden) load(); });
  window.addEventListener("hashchange", function () { if (!/^#contribute\/./.test(location.hash)) stopPoll(); });
  window.addEventListener("beforeunload", function (e) { if (SUB && SUB.local && SUB.local.active) { e.preventDefault(); e.returnValue = ""; } });

  function renderAll(focus) {
    var m = mode(), name = (SUB.meta && SUB.meta.name) || (SUB.view && SUB.view.meta && SUB.view.meta.name) || "Your submission";
    var created = SUB.view && SUB.view.created;
    /* the status line is a live region that stays in the page; only the parts around it are redrawn */
    if (!$("#sub-now", subEl)) subEl.innerHTML = '<h2 class="ct-h" id="sub-h" tabindex="-1"></h2><p class="sub-meta" id="sub-meta"></p><div id="sub-steps"></div>' +
      '<div class="sub-now" id="sub-now" role="status" aria-live="polite"><p class="sub-state" id="sub-state"></p><p class="sub-msg" id="sub-msg"></p></div>' +
      '<div class="sub-body" id="sub-body"></div><div id="sub-rest"></div>';
    $("#sub-h").textContent = name;
    $("#sub-meta").textContent = (created ? "Submitted " + when(created) + ". " : "") + "Submission " + SUB.id + ".";
    $("#sub-steps").innerHTML = stepsHtml(m);
    $("#sub-body").innerHTML = bodyHtml(m);
    $("#sub-rest").innerHTML = linkHtml(m) + endHtml(m);
    renderNow();
    if (m === "preview") drawPlots();
    /* while queued, say so when no worker is running all the time (/api/health instant: false) */
    if (m === "queued") health().then(function (h) { var w = $("#sub-wait"); if (w && h && h.instant === false) w.hidden = false; });
    if (focus) $("#sub-h").focus({ preventScroll: true });
  }
  function stepsHtml(m) {
    var at = STEP_OF[m];
    if (at == null) return "";
    return '<ol class="sub-steps">' + STEPS.map(function (t, i) {
      return i < at ? '<li class="done">' + t + ' <span class="sr">(done)</span></li>' : i === at ? '<li aria-current="step">' + t + "</li>" : "<li>" + t + "</li>";
    }).join("") + "</ol>";
  }
  function renderNow() {
    var m = mode(), v = SUB.view, head, msg = "";
    if (m === "local") { var L = SUB.local; head = L.starting ? "Starting" : L.failed ? "Some files did not upload" : "Uploading your files"; msg = L.starting ? "All files arrived." : upCount(); }
    else if (m === "loading") head = SUB.key ? "Loading the submission" : "This browser does not have the key for this submission";
    else if (m === "missing") head = "Submission not found";
    else if (m === "error") { head = "Could not load the submission"; msg = SUB.error || ""; }
    else { head = HEAD[m] || m; msg = (v && v.message) || ""; }
    var a = $("#sub-state"), b = $("#sub-msg");
    if (a && a.textContent !== head) a.textContent = head;
    if (b && b.textContent !== msg) b.textContent = msg;
  }
  function setChecked(t) { var c = $("#sub-checked"); if (c) c.textContent = t; }
  var LATER = "You can close this page and come back later with your private link.";
  function bodyHtml(m) {
    var v = SUB.view || {}, newBtn = '<p><button type="button" class="btn btn-outline" data-act="again">Start a new submission</button></p>', mail = '<a href="mailto:' + CONTACT + '">' + CONTACT + "</a>";
    switch (m) {
      case "local": return uploadHtml();
      case "loading": return SUB.key ? "" : "<p>Open the private link you saved when you submitted. It ends with <code>#contribute/" + esc(SUB.id) + "/</code> and a long key.</p>" + newBtn;
      case "missing": return "<p>The link may be incomplete, or the submission was deleted.</p>" + newBtn;
      case "error": return '<p><button type="button" class="btn btn-outline" data-act="reload">Try again</button></p>';
      case "uploading": return resumeHtml(v);
      case "queued": return "<p>Your files arrived. The work starts when a worker picks up the submission.</p>" +
        '<p id="sub-wait" hidden>Right now workers start on a schedule, so it can take up to 15 minutes before the work begins.</p>' +
        "<p>Once it starts, most datasets take 2 to 15 minutes. " + LATER + '</p><p class="sub-checked" id="sub-checked"></p>';
      case "running": return "<p>" + (v.kind === "standardized" ? "Your files are being checked against the OpenMotor format and, when there is one, your paper." :
        "Your files are being read, converted to the OpenMotor format and checked against your paper when there is one.") +
        " This page updates by itself. " + LATER + '</p><p class="sub-checked" id="sub-checked"></p>';
      case "publishing": return "<p>Your approval was received and the dataset is being added to OpenMotor. This page updates by itself.</p><p class=\"sub-checked\" id=\"sub-checked\"></p>";
      case "needs_answers": return questionsHtml(v) + reviseHtml("questions");
      case "preview": return previewHtml(v);
      case "published": return publishedHtml(v);
      case "failed": return "<p>Email " + mail + " and include your private link, so the problem can be traced. You can also start a new submission.</p>" + newBtn;
      case "rejected": return "<p>The reason is above. If you think this is a mistake, email " + mail + " and include your private link.</p>" + newBtn;
      case "expired": return "<p>The submission was not finished in time. Start a new submission to try again.</p>" + newBtn;
      case "withdrawn": return "<p>This submission was withdrawn and its files were deleted.</p>" + newBtn;
    }
    return "";
  }
  function linkHtml(m) {
    if (!SUB.key || !OPEN[m] && m !== "publishing" && m !== "published" && m !== "failed") return "";
    return '<div class="sub-link"><h3 class="ct-h3">Your private link</h3><p>Use it to come back to this submission from any device; this page also remembers it in this browser. Anyone with the link can see and withdraw the submission, so keep it to yourself.</p>' +
      '<div class="sub-linkrow"><input type="text" readonly id="sub-url" aria-label="Private link" value="' + esc(privateLink(SUB.id, SUB.key)) + '"><button type="button" class="btn btn-outline" data-act="copy">Copy link</button></div></div>';
  }
  function endHtml(m) {
    if (!OPEN[m]) return "";
    return '<div class="sub-end"><button type="button" class="btn btn-outline" data-act="wd">Withdraw this submission</button>' +
      '<div class="sub-wd" id="sub-wd" hidden><p>Withdrawing stops the work and deletes your files. It cannot be undone.</p>' +
      '<p class="sub-wd-btns"><button type="button" class="btn btn-primary" data-act="wd-yes">Withdraw and delete the files</button><button type="button" class="btn btn-outline" data-act="wd-no">Keep it</button></p>' +
      '<p class="ct-status" id="sub-wd-status" role="status"></p></div></div>';
  }

  /* ---------- uploads ---------- */
  function upCount() { var L = SUB.local, d = L.files.filter(function (f) { return f.state === "done"; }).length; return d + " of " + count(L.files.length, "file") + " uploaded."; }
  function upState(f) { return f.state === "done" ? "Uploaded" : f.state === "failed" ? "Failed" : f.state === "retrying" ? "Trying again" : f.state === "uploading" ? Math.floor(100 * f.loaded / Math.max(1, f.size)) + "%" : "Waiting"; }
  function uploadHtml() {
    var L = SUB.local;
    if (L.starting) return "<p>Starting the work on your files.</p>";
    var total = L.files.reduce(function (t, f) { return t + f.size; }, 0);
    return (L.failed ? "<p>Check your connection and try again. Files that arrived are kept.</p>" + '<p><button type="button" class="btn btn-primary" data-act="retry">Try again</button></p><p class="ct-status" id="sub-begin" role="status"></p>'
      : "<p>Keep this page open until the upload finishes.</p>") +
      '<p class="up-total" id="up-total">' + mb(L.files.reduce(function (t, f) { return t + f.loaded; }, 0)) + " of " + mb(total) + "</p>" +
      '<ul class="up-list">' + L.files.map(function (f, i) {
        return '<li><span class="up-name">' + esc(f.name) + '</span><span class="up-st" id="up-' + i + '-st">' + upState(f) + "</span>" +
          '<progress id="up-' + i + '" max="' + Math.max(1, f.size) + '" value="' + f.loaded + '" aria-label="Upload of ' + esc(f.name) + '"></progress></li>';
      }).join("") + "</ul>";
  }
  var paintQ = null;
  function paint() {
    if (paintQ) return;
    paintQ = requestAnimationFrame(function () {
      paintQ = null;
      if (!SUB || !SUB.local || subEl.hidden) return;
      var L = SUB.local, total = 0, sent = 0;
      L.files.forEach(function (f, i) {
        total += f.size; sent += f.loaded;
        var p = $("#up-" + i), st = $("#up-" + i + "-st");
        if (p) p.value = f.loaded;
        if (st && st.textContent !== upState(f)) st.textContent = upState(f);
      });
      var t = $("#up-total"); if (t) t.textContent = mb(sent) + " of " + mb(total);
    });
  }
  function put(f) {
    return new Promise(function (res, rej) {
      if (!f.url) { rej(new Error("no upload address")); return; }
      var x = new XMLHttpRequest(); f.xhr = x;
      x.open("PUT", f.url);
      x.upload.onprogress = function (e) { f.loaded = e.loaded; paint(); };
      x.onload = function () { f.xhr = null; if (x.status >= 200 && x.status < 300) res(); else rej(new Error("the storage server answered " + x.status)); };
      x.onerror = function () { f.xhr = null; rej(new Error("network error")); };
      x.onabort = function () { f.xhr = null; rej(new Error("cancelled")); };
      x.send(f.file);
    });
  }
  function upOne(s, f) {
    f.state = "uploading"; f.tries++; f.loaded = 0; paint();
    return put(f).then(function () { f.state = "done"; f.loaded = f.size; paint(); if (SUB === s && mode() === "local") renderNow(); }, function () {
      if (s.local.cancelled) return;
      if (f.tries < 2) { f.state = "retrying"; f.loaded = 0; paint(); return sleep(1500).then(function () { return upOne(s, f); }); }
      f.state = "failed"; f.loaded = 0; paint();
    });
  }
  function startUploads(s) {
    var L = s.local; L.active = true; L.failed = false;
    var queue = L.files.filter(function (f) { return f.state !== "done"; }), running = 0;
    queue.forEach(function (f) { f.tries = 0; f.state = "waiting"; });
    if (SUB === s && !subEl.hidden) renderAll();
    return new Promise(function (resolve) {
      function next() {
        if (L.cancelled) { resolve(); return; }
        while (running < 3 && queue.length) { running++; upOne(s, queue.shift()).then(function () { running--; next(); }); }
        if (!running && !queue.length) resolve();
      }
      next();
    }).then(function () {
      L.active = false;
      if (L.cancelled) return;
      if (L.files.some(function (f) { return f.state === "failed"; })) { L.failed = true; if (SUB === s) renderAll(); return; }
      return begin(s);
    });
  }
  function begin(s) {
    if (s.local) { s.local.starting = true; if (SUB === s) renderAll(); }
    return api(sp(s.id) + "/start", { key: s.key, body: {} }).then(function (v) { return v && v.status ? v : api(sp(s.id), { key: s.key }); }).then(function (v) {
      if (s.local) s.local.starting = false;
      s.files = s.local ? s.local.files : s.files; s.local = null; s.missing = null;
      POLL.since = Date.now(); s.fresh = true;
      if (SUB === s) setView(v);
    }, function (err) {
      var L = s.local;
      if (L) L.starting = false;
      var missing = err.data && err.data.missing;
      if (err.status === 409 && L && Array.isArray(missing) && !L.again && missing.every(function (n) { return L.files.some(function (f) { return f.name === n && f.file; }); })) {
        L.again = true;
        L.files.forEach(function (f) { if (missing.indexOf(f.name) >= 0) f.state = "waiting"; });
        return startUploads(s);
      }
      if (err.status === 409 && Array.isArray(missing)) {
        /* back to the resume view: re-select or re-upload what is missing */
        if (L) s.files = L.files;
        s.local = null; s.missing = missing; s.fresh = true;
        if (SUB === s) load();
        return;
      }
      if (SUB !== s) return;
      if (L) { L.failed = true; renderAll(); }
      var el = $("#sub-begin") || $("#sub-msg"); if (el) el.textContent = err.message;
    });
  }
  /* resuming an upload: fresh links from POST .../uploads, then upload what this tab has (or what the person chose again) */
  function inTab(s) { return ((s.local && s.local.files) || s.files || []).filter(function (f) { return f.file; }); }
  function resume(s, files, keepDone) {
    return api(sp(s.id) + "/uploads", { key: s.key, body: {} }).then(function (r) {
      var u = {}; (r.uploads || []).forEach(function (x) { u[x.name] = x.url; });
      s.missing = null; s.picked = null;
      s.local = { active: true, files: files.map(function (f) {
        var done = keepDone && f.state === "done";
        return { name: f.name, size: f.size, file: f.file, url: u[f.name], loaded: done ? f.size : 0, state: done ? "done" : "waiting", tries: 0 };
      }) };
      return startUploads(s);
    });
  }
  function resumeHtml(v) {
    var s = SUB, have = inTab(s), want = v.files || [], h = "<p>The upload stopped before all files arrived.</p>";
    if (s.missing) h += "<p>Still missing: " + esc(names(s.missing)) + ".</p>";
    if (have.length) h += "<p>This page still has your files, so the upload can continue from here.</p>" +
      '<p><button type="button" class="btn btn-primary" data-act="reup">Upload the files again</button></p>';
    else {
      s.picked = s.picked || {};
      h += "<p>Choose the same files again to finish the upload. They are matched by name and size.</p>" +
        '<ul class="up-list">' + want.map(function (f) {
          return '<li><span class="up-name">' + esc(f.name) + '</span><span class="up-st">' + (s.picked[f.name] ? "Chosen" : mb(f.size) + ", not chosen yet") + "</span></li>";
        }).join("") + "</ul>" +
        '<p class="sub-wd-btns"><button type="button" class="btn btn-outline" data-act="pick">Choose files</button><button type="button" class="btn btn-primary" data-act="reup">Upload and continue</button></p>' +
        '<input type="file" id="sub-pick" multiple hidden>';
    }
    return h + '<p class="ct-status" id="sub-begin" role="status"></p>' +
      '<p class="sub-alt">If every file already arrived, <button type="button" class="linkbtn" data-act="begin">check and continue</button>.</p>';
  }
  subEl.addEventListener("change", function (e) {
    if (e.target.id !== "sub-pick" || !SUB) return;
    var s = SUB, want = {}, notes = [];
    ((s.view && s.view.files) || []).forEach(function (f) { want[f.name] = f.size; });
    s.picked = s.picked || {};
    Array.prototype.forEach.call(e.target.files, function (file) {
      if (!(file.name in want)) notes.push(file.name + " is not part of this submission.");
      else if (want[file.name] !== file.size) notes.push(file.name + " is not the same file: its size differs.");
      else s.picked[file.name] = file;
    });
    renderAll();
    var n = Object.keys(s.picked).length, left = Object.keys(want).length - n;
    $("#sub-begin").textContent = (notes.length ? notes.join(" ") + " " : "") + count(n, "file") + " chosen" + (left ? ", " + left + " still to choose." : ".");
    var go = subEl.querySelector('[data-act="reup"]'); if (go) go.focus();
  });

  /* ---------- questions ---------- */
  function questionsHtml(v) {
    var qs = v.questions || [], ans = v.answers || {};
    if (!qs.length) return "<p>Waiting for the questions to load.</p>";
    return '<form class="q-form" id="sub-qform" novalidate><p>Some things in your files were not clear. Answer these questions and the work continues.</p><div class="q-list">' +
      qs.map(function (q, i) { return qField(q, i, ans[q.id]); }).join("") +
      '</div><div class="q-actions"><button type="submit" class="btn btn-primary">Send answers</button></div><p class="ct-status" id="sub-qstatus" role="status"></p></form>';
  }
  /* an extra "Other" with a text box, unless the choices already cover it or the question is yes/no */
  function hasOwnOther(q) {
    return q.choices.some(function (c) { return /^(other|none\b|not sure|don'?t know|unknown|something else)/i.test(String(c)); }) ||
      q.choices.every(function (c) { return /^(yes|no)$/i.test(String(c)); });
  }
  function qField(q, i, prev) {
    var id = "q" + i, opt = q.optional ? ' <span class="q-opt">(optional)</span>' : "";
    var help = q.help ? '<p class="q-help" id="' + id + '-h">' + esc(q.help) + "</p>" : "", desc = q.help ? ' aria-describedby="' + id + '-h"' : "";
    if (q.kind === "choice" && Array.isArray(q.choices) && q.choices.length) {
      var at = -1; q.choices.forEach(function (c, k) { if (prev != null && String(c) === String(prev)) at = k; });
      var other = !hasOwnOther(q), otherOn = other && prev != null && prev !== "" && at < 0;
      return '<fieldset class="q-f" data-i="' + i + '"' + desc + "><legend>" + esc(q.text) + opt + "</legend>" + help + '<div class="q-choices">' +
        q.choices.map(function (c, k) { return '<label class="q-choice"><input type="radio" name="' + id + '" value="' + k + '"' + (k === at ? " checked" : "") + "><span>" + esc(c) + "</span></label>"; }).join("") +
        (other ? '<label class="q-choice"><input type="radio" name="' + id + '" value="other"' + (otherOn ? " checked" : "") + '><span>Other</span></label>' +
          '<input type="text" class="q-other" id="' + id + '-o" aria-label="Other, please say what" placeholder="Say what" value="' + (otherOn ? esc(prev) : "") + '">' : "") + "</div></fieldset>";
    }
    var field = q.kind === "number" ? '<input type="text" inputmode="decimal" id="' + id + '"' + desc + ' value="' + (prev != null ? esc(prev) : "") + '">'
      : '<textarea id="' + id + '" rows="2"' + desc + ">" + (prev != null ? esc(prev) : "") + "</textarea>";
    return '<div class="q-f ff" data-i="' + i + '"><label for="' + id + '">' + esc(q.text) + opt + "</label>" + help + field + "</div>";
  }
  function collectAnswers(fm, qs) {
    var out = {}, bad = null;
    qs.forEach(function (q, i) {
      var id = "q" + i, v = "", el;
      if (q.kind === "choice" && Array.isArray(q.choices) && q.choices.length) {
        var r = fm.querySelector('input[name="' + id + '"]:checked');
        el = fm.querySelector('input[name="' + id + '"]');
        if (r && r.value === "other") { v = $("#" + id + "-o").value.trim(); if (!v) el = $("#" + id + "-o"); }
        else if (r) v = q.choices[+r.value];
      } else {
        el = $("#" + id); v = el.value.trim();
        if (q.kind === "number" && v !== "") { var n = Number(v.replace(",", ".")); if (!isFinite(n)) { if (!bad) bad = { el: el, msg: "Give a number for: " + q.text }; return; } v = n; }
      }
      if (v === "" || v == null) { if (!q.optional && !bad) bad = { el: el, msg: "Answer this question: " + q.text }; return; }
      out[q.id] = v;
    });
    return { answers: out, bad: bad };
  }
  subEl.addEventListener("submit", function (e) {
    var fm = e.target;
    if (fm.id === "sub-qform") { e.preventDefault(); sendAnswers(fm); }
    else if (fm.id === "pv-form") { e.preventDefault(); approve(fm); }
  });
  function sendAnswers(fm) {
    var s = SUB, qs = (s.view && s.view.questions) || [], c = collectAnswers(fm, qs), st = $("#sub-qstatus"), btn = fm.querySelector('button[type="submit"]');
    fm.querySelectorAll("[aria-invalid]").forEach(function (x) { x.removeAttribute("aria-invalid"); });
    if (c.bad) { st.textContent = c.bad.msg; c.bad.el.setAttribute("aria-invalid", "true"); c.bad.el.focus(); return; }
    btn.disabled = true; st.textContent = "Sending your answers.";
    api(sp(s.id) + "/answers", { key: s.key, body: { answers: c.answers } }).then(function (r) {
      POLL.since = Date.now();
      if (SUB !== s) return;
      if (r && r.status) setView(r); else load();
    }, function (err) {
      btn.disabled = false; st.textContent = err.message;
      var qid = err.data && err.data.question, at = -1;
      qs.forEach(function (q, i) { if (q.id === qid) at = i; });
      if (at >= 0) { var box = fm.querySelector('[data-i="' + at + '"]'), el = box && box.querySelector("input, textarea"); if (el) { el.setAttribute("aria-invalid", "true"); el.focus(); } }
    });
  }

  /* ---------- preview ---------- */
  function num(v) { return typeof v === "number" ? numFmt(v) : v == null || v === "" ? "" : String(v); }
  function doiLink(d) { if (!d) return ""; var x = String(d).replace(/^https?:\/\/(dx\.)?doi\.org\//i, ""); return /^10\.\d+\//.test(x) ? '<a href="https://doi.org/' + esc(x) + '" rel="noopener" target="_blank">' + esc(x) + "</a>" : esc(d); }
  function list(items) { return '<ul class="pv-list">' + items.map(function (t) { return "<li>" + esc(t) + "</li>"; }).join("") + "</ul>"; }
  function arr(a) { return Array.isArray(a) && a.length ? a : null; }
  function previewHtml(v) {
    var p = v.preview || {}, L = p.listing || {}, S = p.summary || {}, h = '<div class="pv">';
    h += "<p>This is what will be published. Check it against your study. Nothing is public until you approve it at the end of this page.</p>";
    var rows = [["Name", esc(L.label)], ["Title", esc(L.title)], ["Authors", esc(L.authors)], ["Year", esc(L.year)], ["Journal", esc(L.venue)], ["DOI", doiLink(L.doi)],
                ["Task", esc(L.task)], ["License", esc(L.license)], ["Format", esc(p.format_label || p.format)], ["Dataset ID", p.dataset_id ? "<code>" + esc(p.dataset_id) + "</code>" : ""]];
    h += '<h3 class="ct-h3">The listing</h3><dl class="pv-dl">' + rows.filter(function (r) { return r[1]; }).map(function (r) { return "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>"; }).join("") + "</dl>";
    var sum = [["Rows", S.rows], ["Participants", S.participants], ["Subjects", S.subjects], ["Experiments", S.experiments], ["Columns", S.columns]].filter(function (r) { return r[1] != null; });
    if (sum.length) h += '<dl class="pv-sum">' + sum.map(function (r) { return "<div><dt>" + r[0] + "</dt><dd>" + esc(typeof r[1] === "number" ? fmt(r[1]) : r[1]) + "</dd></div>"; }).join("") + "</dl>";
    if (arr(p.notes)) h += '<div class="pv-look"><h3 class="ct-h3">Please look at</h3>' + list(p.notes) + "</div>";
    if (arr(p.checks)) {
      var cs = p.checks.slice().sort(function (a, b) { return (a.kind === "internal") - (b.kind === "internal"); });
      h += '<h3 class="ct-h3">Checks</h3><p class="pv-sub">Numbers computed from the standardized data, next to the numbers reported in your paper or expected from the files.</p>' +
        '<div class="table-card"><div class="table-wrap"><table class="data compact pv-checks"><thead><tr><th scope="col">Check</th><th scope="col">From the data</th><th scope="col">Reported</th><th scope="col">Result</th><th scope="col">Source</th></tr></thead><tbody>' +
        cs.map(function (c) {
          var res = c.ok === true ? "Matches" : c.ok === false ? '<b class="pv-bad">Differs</b>' : "Not compared";
          var src = c.kind === "internal" ? "Internal check" + (c.source ? ": " + esc(c.source) : "") : esc(c.source || "Paper");
          return "<tr" + (c.ok === false ? ' class="pv-row-bad"' : "") + '><td class="pv-what">' + esc(c.what) + '</td><td data-label="From the data">' + esc(num(c.computed)) + '</td><td data-label="Reported">' + esc(num(c.reported)) + '</td><td data-label="Result">' + res + '</td><td data-label="Source">' + src + "</td></tr>";
        }).join("") + "</tbody></table></div></div>";
    }
    if (arr(p.plots)) h += '<h3 class="ct-h3">Plots</h3><div class="pv-plots">' + p.plots.map(plotShell).join("") + "</div>";
    if (arr(p.how)) h += '<h3 class="ct-h3">How the data were standardized</h3>' + list(p.how);
    if (arr(p.changes)) h += '<h3 class="ct-h3">What was changed</h3>' + list(p.changes);
    if (arr(p.differences)) h += '<h3 class="ct-h3">Differences from the paper</h3>' + list(p.differences);
    if (p.sample && arr(p.sample.columns) && arr(p.sample.rows)) {
      h += '<h3 class="ct-h3">Sample rows</h3><p class="pv-sub">The first ' + count(p.sample.rows.length, "row") + (S.rows ? " of " + fmt(S.rows) : "") + ". Scroll sideways to see every column.</p>" +
        '<div class="table-card"><div class="table-wrap"><table class="data compact pv-sample"><thead><tr>' + p.sample.columns.map(function (c) { return '<th scope="col">' + esc(c) + "</th>"; }).join("") + "</tr></thead><tbody>" +
        p.sample.rows.slice(0, 30).map(function (r) { return "<tr>" + r.map(function (x) { var t = num(x); return t.length > 40 ? '<td title="' + esc(t) + '">' + esc(t.slice(0, 38)) + "...</td>" : "<td>" + esc(t) + "</td>"; }).join("") + "</tr>"; }).join("") +
        "</tbody></table></div></div>";
    }
    if (arr(p.columns)) {
      h += '<details class="pv-more"><summary>All columns (' + p.columns.length + ')</summary><div class="table-card"><div class="table-wrap"><table class="data compact pv-cols"><thead><tr><th scope="col">Column</th><th scope="col">Type</th><th scope="col">Filled</th><th scope="col">Min</th><th scope="col">Max</th><th scope="col">Example</th><th scope="col">Description</th></tr></thead><tbody>' +
        p.columns.map(function (c) {
          var fill = typeof c.filled === "number" ? Math.round(c.filled * 100) + "%" : "";
          return "<tr><td><code>" + esc(c.name) + "</code></td><td>" + esc(c.type) + '</td><td class="num">' + fill + '</td><td class="num">' + esc(num(c.min)) + '</td><td class="num">' + esc(num(c.max)) + "</td><td>" + esc(String(num(c.example)).slice(0, 40)) + "</td><td>" + esc(c.description) + "</td></tr>";
        }).join("") + "</tbody></table></div></div></details>";
    }
    if (p.readme) h += '<details class="pv-more"><summary>README</summary><pre class="pv-readme">' + esc(p.readme) + "</pre></details>";
    if (arr(p.files)) h += '<h3 class="ct-h3">Files</h3><ul class="pv-files">' + p.files.map(function (f) {
      return '<li><a href="' + esc(API + sp(SUB.id) + "/file?key=" + encodeURIComponent(SUB.key) + "&name=" + encodeURIComponent(f.name)) + '">' + esc(f.name) + "</a>" + (f.bytes != null ? ' <span class="pv-size">' + mb(f.bytes) + "</span>" : "") + "</li>";
    }).join("") + "</ul>";
    var lic = L.license || (SUB.meta && SUB.meta.license) || "the license you chose";
    h += '<form class="pv-confirm" id="pv-form" novalidate><h3 class="ct-h3">Approve</h3><p>Publishing makes the dataset public on OpenMotor. Tick all three to continue.</p>' +
      '<div class="pv-check"><input type="checkbox" id="pv-rights"><label for="pv-rights">I have the right to share these data under ' + esc(lic) + "</label></div>" +
      '<div class="pv-check"><input type="checkbox" id="pv-consent"><label for="pv-consent">Participants agreed to their data being shared, and the files contain no names, contact details, dates of birth or other identifying information</label></div>' +
      '<div class="pv-check"><input type="checkbox" id="pv-checked"><label for="pv-checked">I checked the preview and it matches my study</label></div>' +
      '<div class="ff pv-note"><label for="pv-note">Note (optional)</label><textarea id="pv-note" rows="2" placeholder="Anything you want recorded with your approval"></textarea></div>' +
      '<div class="q-actions"><button type="submit" class="btn btn-primary">Approve and publish</button></div><p class="ct-status" id="pv-status" role="status"></p></form>';
    h += reviseHtml("preview");
    return h + "</div>";
  }
  /* "Something is wrong?": a note for another round of processing (POST .../revise), in the preview and with the questions */
  function reviseHtml(where) {
    return '<details class="pv-wrong"' + (where === "questions" ? ' style="margin-top: 26px"' : "") + "><summary>Something is wrong?</summary>" +
      "<p>" + (where === "preview" ? "Say what should change. Your files are then processed again with your note, and you get a new preview to check."
        : "If these questions show that your files were misread, say what is wrong. Your files are then processed again with your note.") + "</p>" +
      '<div class="ff"><label for="rv-note">What should change</label><textarea id="rv-note" rows="3" placeholder="For example: group 1 is the young adults, not the older adults."></textarea></div>' +
      '<p class="sub-wd-btns"><button type="button" class="btn btn-outline" data-act="revise">Send and process again</button></p><p class="ct-status" id="rv-status" role="status"></p>' +
      '<p class="sub-alt">To stop instead, withdraw the submission below. Questions: <a href="mailto:' + CONTACT + '">' + CONTACT + "</a>.</p></details>";
  }
  function approve(fm) {
    var s = SUB, st = $("#pv-status"), btn = fm.querySelector('button[type="submit"]'), boxes = ["pv-rights", "pv-consent", "pv-checked"].map(function (id) { return $("#" + id); });
    var off = boxes.filter(function (b) { return !b.checked; });
    boxes.forEach(function (b) { b.setAttribute("aria-invalid", b.checked ? "false" : "true"); });
    if (off.length) { st.textContent = "Tick all three statements to publish."; off[0].focus(); return; }
    btn.disabled = true; st.textContent = "Sending your approval.";
    var note = $("#pv-note").value.trim(), body = { confirm: { rights: true, consent: true, checked: true } };
    if (note) body.note = note;
    api(sp(s.id) + "/approve", { key: s.key, body: body }).then(function (r) { POLL.since = Date.now(); if (SUB !== s) return; if (r && r.status) setView(r); else load(); },
      function (err) { btn.disabled = false; st.textContent = err.message; });
  }
  function publishedHtml(v) {
    var p = v.published || {}, h = "<p>Thank you. Your dataset is now part of OpenMotor.</p><ul class=\"pv-list\">";
    if (p.url) h += '<li><a href="' + esc(p.url) + '">Open the dataset page</a></li>';
    if (p.zip_url) h += '<li><a href="' + esc(p.zip_url) + '">Download the dataset ZIP</a></li>';
    h += "</ul>";
    if (p.dataset_id) h += "<p>Dataset ID: <code>" + esc(p.dataset_id) + "</code></p>";
    return h + '<p>To correct or remove it later, email <a href="mailto:' + CONTACT + '">' + CONTACT + '</a> with the dataset ID.</p><p><a class="btn btn-outline" href="#contribute">Add another dataset</a></p>';
  }

  /* ---------- plots: inline SVG line charts with a crosshair, a tooltip and a table of the values ---------- */
  var PAL = ["#e40335", "#2a78d6", "#1baf7a", "#eda100", "#6250d6", "#eb6834"], GREY = "#a1a1a6";
  function numFmt(v, step) {
    if (typeof v !== "number" || !isFinite(v)) return v == null ? "" : String(v);
    if (step != null) {
      if (Math.abs(v) >= 10000 && step >= 1000) return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
      var d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
      return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    }
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    var a = Math.abs(v);
    return v.toLocaleString("en-US", { maximumFractionDigits: a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : 3 });
  }
  function niceTicks(lo, hi, n) {
    if (!(hi > lo)) { var d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))), err = (hi - lo) / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var t = [], a = Math.floor(lo / step) * step;
    for (var x = a; x <= hi + step * 0.5; x += step) t.push(+x.toPrecision(12));
    while (t.length > 2 && t[1] <= lo) t.shift();
    while (t.length > 2 && t[t.length - 2] >= hi) t.pop();
    return { ticks: t, step: step };
  }
  function seriesOf(p) {
    var cats = null, S = (p.series || []).filter(function (s) { return s && Array.isArray(s.x) && Array.isArray(s.y); });
    if (S.some(function (s) { return s.x.some(function (x) { return x != null && x !== "" && !isFinite(+x); }); })) {
      cats = []; S.forEach(function (s) { s.x.forEach(function (x) { if (cats.indexOf(String(x)) < 0) cats.push(String(x)); }); });
    }
    return { cats: cats, series: S.map(function (s, k) {
      var pts = [];
      for (var j = 0; j < Math.min(s.x.length, s.y.length); j++) {
        var x = cats ? cats.indexOf(String(s.x[j])) : +s.x[j], y = s.y[j] == null || s.y[j] === "" ? NaN : +s.y[j];
        if (s.x[j] != null && s.x[j] !== "" && isFinite(x)) pts.push([x, y]);
      }
      return { label: s.label == null || s.label === "" ? "Series " + (k + 1) : String(s.label), pts: pts, color: k < PAL.length ? PAL[k] : GREY };
    }) };
  }
  function plotShell(p, i) {
    var D = seriesOf(p), S = D.series, legend = "";
    if (S.length > 1) legend = '<ul class="pv-legend">' + S.slice(0, PAL.length).map(function (s) {
      return '<li><svg width="18" height="10" aria-hidden="true"><line x1="1" y1="5" x2="17" y2="5" stroke="' + s.color + '" stroke-width="2" stroke-linecap="round"/></svg>' + esc(s.label) + "</li>";
    }).join("") + (S.length > PAL.length ? '<li><svg width="18" height="10" aria-hidden="true"><line x1="1" y1="5" x2="17" y2="5" stroke="' + GREY + '" stroke-width="2"/></svg>' + (S.length - PAL.length) + " more, in grey</li>" : "") + "</ul>";
    var ux = unionX(S), cap = 400;
    var table = '<table class="data compact"><thead><tr><th scope="col">' + esc(p.xlabel || "x") + "</th>" + S.map(function (s) { return '<th scope="col">' + esc(s.label) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      ux.slice(0, cap).map(function (x) { return "<tr><td>" + esc(D.cats ? D.cats[x] : numFmt(x)) + "</td>" + S.map(function (s) { var y = s.map.get(x); return '<td class="num">' + (y == null || !isFinite(y) ? "" : esc(numFmt(y))) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody></table>";
    return '<figure class="pv-plot" data-plot="' + i + '"><figcaption>' + esc(p.title || "Plot " + (i + 1)) + "</figcaption>" + legend + '<div class="pv-svg"></div>' +
      '<details class="pv-table"><summary>Show the numbers</summary>' + (ux.length > cap ? '<p class="pv-sub">The first ' + cap + " of " + fmt(ux.length) + " points.</p>" : "") + '<div class="table-wrap">' + table + "</div></details></figure>";
  }
  function unionX(S) {
    var seen = new Set();
    S.forEach(function (s) { s.map = new Map(); s.pts.forEach(function (pt) { s.map.set(pt[0], pt[1]); seen.add(pt[0]); }); });
    return Array.from(seen).sort(function (a, b) { return a - b; });
  }
  var plotRO = null;
  function drawPlots() {
    var v = SUB && SUB.view, plots = v && v.preview && arr(v.preview.plots);
    if (!plots) return;
    subEl.querySelectorAll(".pv-plot").forEach(function (fig) { drawPlot(fig, plots[+fig.getAttribute("data-plot")]); });
    var box = subEl.querySelector(".pv-plots");
    if (plotRO) plotRO.disconnect();
    if (box && window.ResizeObserver) {
      var tm = 0;
      plotRO = new ResizeObserver(function () { clearTimeout(tm); tm = setTimeout(function () { subEl.querySelectorAll(".pv-plot").forEach(function (fig) { var host = fig.querySelector(".pv-svg"); if (Math.abs(host.clientWidth - (+host.getAttribute("data-w") || 0)) > 4) drawPlot(fig, plots[+fig.getAttribute("data-plot")]); }); }, 120); });
      plotRO.observe(box);
    }
  }
  function drawPlot(fig, p) {
    var host = fig.querySelector(".pv-svg"), W = Math.max(260, Math.floor(host.clientWidth || 480)), H = W < 420 ? 220 : 250;
    host.setAttribute("data-w", W);
    var D = seriesOf(p), S = D.series, ux = unionX(S), ys = [];
    S.forEach(function (s) { s.pts.forEach(function (pt) { if (isFinite(pt[1])) ys.push(pt[1]); }); });
    if (!ux.length || !ys.length) { host.innerHTML = '<p class="pv-sub">No values to plot.</p>'; return; }
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys), Y = niceTicks(ymin, ymax, H < 240 ? 4 : 5), y0 = Y.ticks[0], y1 = Y.ticks[Y.ticks.length - 1];
    var x0 = ux[0], x1 = ux[ux.length - 1];
    var xT = D.cats ? D.cats.map(function (c, k) { return k; }).filter(function (k, _, a) { return a.length <= 8 || k % Math.ceil(a.length / 8) === 0; }) : niceTicks(x0, x1, W < 420 ? 4 : 6).ticks.filter(function (t) { return t >= x0 - 1e-9 && t <= x1 + 1e-9; });
    var xStep = D.cats ? null : niceTicks(x0, x1, W < 420 ? 4 : 6).step;
    var ylab = Y.ticks.map(function (t) { return numFmt(t, Y.step); }), mL = 12 + Math.max.apply(null, ylab.map(function (s) { return s.length; })) * 7 + (p.ylabel ? 18 : 0);
    var mT = 10, mB = 26 + (p.xlabel ? 20 : 0), mR = 14, pH = H - mT - mB;
    function py(y) { return mT + (y1 - y) / (y1 - y0) * pH; }
    var ends = null;
    if (S.length > 1 && S.length <= 4 && W >= 480) {
      ends = S.map(function (s) { for (var j = s.pts.length - 1; j >= 0; j--) if (isFinite(s.pts[j][1])) return { s: s, pt: s.pts[j], y: py(s.pts[j][1]) }; return null; }).filter(Boolean);
      var sorted = ends.map(function (e) { return e.y; }).sort(function (a, b) { return a - b; });
      for (var k = 1; k < sorted.length; k++) if (sorted[k] - sorted[k - 1] < 15) { ends = null; break; }
      if (ends) mR = 16 + Math.min(16, Math.max.apply(null, ends.map(function (e) { return e.s.label.length; }))) * 7;
    }
    var pW = W - mL - mR;
    function px(x) { return x1 === x0 ? mL + pW / 2 : mL + (x - x0) / (x1 - x0) * pW; }
    var g = [];
    Y.ticks.forEach(function (t, k) {
      var yy = py(t).toFixed(1);
      g.push('<line x1="' + mL + '" x2="' + (W - mR) + '" y1="' + yy + '" y2="' + yy + '" stroke="' + (t === 0 && y0 < 0 ? "#c9c9cc" : "#ececee") + '" stroke-width="1"/>');
      g.push('<text x="' + (mL - 8) + '" y="' + yy + '" dy="0.32em" text-anchor="end" class="pv-tick">' + esc(ylab[k]) + "</text>");
    });
    g.push('<line x1="' + mL + '" x2="' + (W - mR) + '" y1="' + (H - mB) + '" y2="' + (H - mB) + '" stroke="#c9c9cc" stroke-width="1"/>');
    xT.forEach(function (t) {
      var xx = px(t), anchor = xx - mL < 12 ? "start" : W - mR - xx < 12 ? "end" : "middle";
      g.push('<line x1="' + xx.toFixed(1) + '" x2="' + xx.toFixed(1) + '" y1="' + (H - mB) + '" y2="' + (H - mB + 4) + '" stroke="#c9c9cc"/>');
      g.push('<text x="' + xx.toFixed(1) + '" y="' + (H - mB + 17) + '" text-anchor="' + anchor + '" class="pv-tick">' + esc(D.cats ? String(D.cats[t]).slice(0, 14) : numFmt(t, xStep)) + "</text>");
    });
    if (p.xlabel) g.push('<text x="' + (mL + pW / 2).toFixed(1) + '" y="' + (H - 5) + '" text-anchor="middle" class="pv-axis">' + esc(p.xlabel) + "</text>");
    if (p.ylabel) g.push('<text transform="rotate(-90)" x="' + (-(mT + pH / 2)).toFixed(1) + '" y="13" text-anchor="middle" class="pv-axis">' + esc(p.ylabel) + "</text>");
    S.slice().reverse().forEach(function (s) {
      var d = "", pen = false;
      s.pts.forEach(function (pt) { if (!isFinite(pt[1])) { pen = false; return; } d += (pen ? "L" : "M") + px(pt[0]).toFixed(1) + " " + py(pt[1]).toFixed(1); pen = true; });
      if (d) g.push('<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>');
      if (s.pts.length <= 40) s.pts.forEach(function (pt) { if (isFinite(pt[1])) g.push('<circle cx="' + px(pt[0]).toFixed(1) + '" cy="' + py(pt[1]).toFixed(1) + '" r="4" fill="' + s.color + '" stroke="#ffffff" stroke-width="2"/>'); });
    });
    if (ends) ends.forEach(function (e) { g.push('<text x="' + (px(e.pt[0]) + 8).toFixed(1) + '" y="' + e.y.toFixed(1) + '" dy="0.32em" class="pv-end">' + esc(e.s.label.slice(0, 16)) + "</text>"); });
    g.push('<line class="pv-cross" x1="0" x2="0" y1="' + mT + '" y2="' + (H - mB) + '" stroke="#8e8e93" stroke-width="1" visibility="hidden"/>');
    S.forEach(function (s, k) { g.push('<circle class="pv-hot" data-k="' + k + '" r="4.5" fill="' + s.color + '" stroke="#ffffff" stroke-width="2" visibility="hidden"/>'); });
    g.push('<rect class="pv-hit" x="' + mL + '" y="' + mT + '" width="' + pW + '" height="' + pH + '" fill="transparent"/>');
    var label = (p.title || "Plot") + ". Line chart" + (S.length > 1 ? " of " + S.length + " series" : "") + (p.xlabel ? ", " + p.xlabel + " from " + (D.cats ? D.cats[0] : numFmt(x0)) + " to " + (D.cats ? D.cats[D.cats.length - 1] : numFmt(x1)) : "") + ". Use the arrow keys to read values; the numbers are also in the table below.";
    host.innerHTML = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H + '" role="img" tabindex="0" aria-label="' + esc(label) + '">' + g.join("") + '</svg><div class="pv-tip" hidden></div>';
    var svg = host.querySelector("svg"), tip = host.querySelector(".pv-tip"), cross = svg.querySelector(".pv-cross"), hot = svg.querySelectorAll(".pv-hot"), cur = -1;
    function show(k) {
      cur = Math.max(0, Math.min(ux.length - 1, k));
      var x = ux[cur], xx = px(x);
      cross.setAttribute("x1", xx); cross.setAttribute("x2", xx); cross.setAttribute("visibility", "visible");
      var rows = "";
      S.forEach(function (s, i) {
        var y = s.map.get(x), on = y != null && isFinite(y);
        hot[i].setAttribute("visibility", on ? "visible" : "hidden");
        if (on) { hot[i].setAttribute("cx", xx); hot[i].setAttribute("cy", py(y)); }
        if (i < 8) rows += '<div class="tip-row"><svg width="14" height="6" aria-hidden="true"><line x1="0" y1="3" x2="14" y2="3" stroke="' + s.color + '" stroke-width="2"/></svg><b>' + (on ? esc(numFmt(y)) : "-") + "</b>" + (S.length > 1 ? "<span>" + esc(s.label) + "</span>" : "") + "</div>";
      });
      tip.innerHTML = '<div class="tip-x">' + esc((p.xlabel ? p.xlabel + " " : "") + (D.cats ? D.cats[x] : numFmt(x))) + "</div>" + rows;
      tip.hidden = false;
      var tw = tip.offsetWidth, left = xx + 12 + tw > W ? xx - 12 - tw : xx + 12;
      tip.style.left = Math.max(0, left) + "px"; tip.style.top = mT + "px";
    }
    function hide() { cross.setAttribute("visibility", "hidden"); hot.forEach(function (h) { h.setAttribute("visibility", "hidden"); }); tip.hidden = true; }
    function nearest(x) { var lo = 0, hi = ux.length - 1; while (hi - lo > 1) { var m = (lo + hi) >> 1; if (ux[m] < x) lo = m; else hi = m; } return Math.abs(ux[lo] - x) <= Math.abs(ux[hi] - x) ? lo : hi; }
    var hit = svg.querySelector(".pv-hit");
    hit.addEventListener("pointermove", function (e) { var r = svg.getBoundingClientRect(); show(nearest(x0 + (e.clientX - r.left - mL) / pW * (x1 - x0))); });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", function () { show(cur < 0 ? ux.length - 1 : cur); });
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? Math.max(1, Math.round(ux.length / 20)) : 1, k = { ArrowRight: cur + step, ArrowLeft: cur - step, Home: 0, End: ux.length - 1 }[e.key];
      if (k != null) { e.preventDefault(); show(k); } else if (e.key === "Escape") hide();
    });
  }

  /* ---------- actions in the submission view ---------- */
  subEl.addEventListener("click", function (e) {
    var b = e.target.closest("[data-act]"); if (!b) return;
    var act = b.getAttribute("data-act"), s = SUB;
    if (act === "copy") {
      var inp = $("#sub-url");
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy link"; }, 2000); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(inp.value).then(done, function () { inp.select(); b.textContent = "Selected, press copy"; });
      else { inp.select(); b.textContent = "Selected, press copy"; }
    } else if (act === "wd") { $("#sub-wd").hidden = false; b.hidden = true; $("#sub-wd").querySelector("button").focus(); }
    else if (act === "wd-no") { $("#sub-wd").hidden = true; var w = subEl.querySelector('[data-act="wd"]'); w.hidden = false; w.focus(); }
    else if (act === "wd-yes") { b.disabled = true; withdraw(s, $("#sub-wd-status")).then(function (ok) { if (!ok) b.disabled = false; }); }
    else if (act === "retry" || act === "reup") {
      var files = act === "retry" ? s.local.files : inTab(s).length ? inTab(s) : Object.keys(s.picked || {}).map(function (n) { return { name: n, size: s.picked[n].size, file: s.picked[n] }; });
      if (!files.length) { $("#sub-begin").textContent = "Choose the files first."; return; }
      b.disabled = true;
      resume(s, files, act === "retry").catch(function (err) { b.disabled = false; var st = $("#sub-begin") || $("#sub-msg"); if (st) st.textContent = err.message; });
    }
    else if (act === "pick") $("#sub-pick").click();
    else if (act === "begin") { b.disabled = true; begin(s).then(function () { b.disabled = false; }); }
    else if (act === "reload") { s.state = null; renderAll(); load(); }
    else if (act === "again") again(s, "");
    else if (act === "revise") {
      var ta = $("#rv-note"), note = ta.value.trim(), rs = $("#rv-status");
      ta.removeAttribute("aria-invalid");
      if (!note) { rs.textContent = "Say what should change."; ta.setAttribute("aria-invalid", "true"); ta.focus(); return; }
      b.disabled = true; rs.textContent = "Sending your note.";
      api(sp(s.id) + "/revise", { key: s.key, body: { note: note } }).then(function (r) {
        POLL.since = Date.now();
        if (SUB !== s) return;
        if (r && r.status) setView(r); else load();
      }, function (err) { b.disabled = false; rs.textContent = err.message; });
    }
  });
  function withdraw(s, st) {
    if (s.local) { s.local.cancelled = true; s.local.active = false; s.local.files.forEach(function (f) { if (f.xhr) f.xhr.abort(); }); }
    if (st) st.textContent = "Withdrawing.";
    return api(sp(s.id) + "/withdraw", { key: s.key, body: {} }).then(function () {
      return api(sp(s.id), { key: s.key }).catch(function () { return null; });
    }).then(function (v) {
      if (s.local) { s.files = s.local.files; s.local = null; }
      v = v || Object.assign({}, s.view || {}, { status: "withdrawn", message: "" });
      if (SUB === s) setView(v);
      return true;
    }, function (err) { if (st) st.textContent = err.message; return false; });
  }
  function again(s, what) {
    var m = Object.assign({}, s.meta || (s.view && s.view.meta) || {});
    if (what) m.notes = (m.notes ? m.notes + "\n\n" : "") + "Correction after the preview of submission " + s.id + ": " + what;
    fillForm(m);
    var kept = s.files || (s.local && s.local.files) || [];
    FILES = kept.filter(function (f) { return f.file; }).map(function (f) { var g = { name: f.name, size: f.size, file: f.file, check: null }; startCheck(g); return g; });
    renderFiles();
    say(FILES.length ? "Your details and files are filled in again. Check them and submit." : "Your details are filled in again. Add your files and submit.");
    location.hash = "#contribute";
    setTimeout(function () { $("#om-name").focus({ preventScroll: true }); }, 50);
  }

  /* ---------- route: #contribute, #contribute/<id>, #contribute/<id>/<key> ---------- */
  function onShow(arg) {
    var parts = String(arg || "").split("/"), id = parts[0], key = parts[1];
    if (!id || !/^[A-Za-z0-9_-]{4,64}$/.test(id)) { showForm(); return; }
    if (key && /^[A-Za-z0-9_-]{8,200}$/.test(key)) {
      remember(id, key, {});
      try { history.replaceState(history.state, "", location.pathname + location.search + "#contribute/" + encodeURIComponent(id)); } catch (e) {}
    } else key = null;
    openSubmission(id, key);
    return "Your submission";
  }

  window.OM_PARTS = window.OM_PARTS || {};
  window.OM_PARTS.contribute = { title: "Add your dataset", onShow: onShow };
})();
