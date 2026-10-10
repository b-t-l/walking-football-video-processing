"use strict";
// Game logger front end: plain JavaScript, no build step. Pages: #/ (games), #/game/new, #/game/<id>, #/teams, #/settings

const view = document.getElementById("view");
let META = null;
let dirty = false;
let currentHash = location.hash || "#/";
let suppressNext = false;
const listState = { q: "", type: "", format: "", status: "", archived: false };

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(method, url, body) {
  const res = await fetch(url, { method, headers: body ? { "Content-Type": "application/json", "X-GL-App": "1" } : { "X-GL-App": "1" }, body: body ? JSON.stringify(body) : undefined });
  let data = null;
  try { data = await res.json(); } catch (e) { /* not json */ }
  if (!res.ok) {
    const err = new Error((data && (data.errors || data.detail)) || res.statusText);
    err.errors = data && data.errors ? data.errors : [data && data.detail ? data.detail : res.statusText];
    throw err;
  }
  return data;
}

let toastTimer = null;
function toast(msg, bad) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = "toast" + (bad ? " bad" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 6000 : 2800);
}

function notice(kind, title, items) {
  const list = items && items.length ? `<ul>${items.map(i => `<li>${esc(i)}</li>`).join("")}</ul>` : "";
  return `<div class="notice ${kind}">${title ? `<b>${esc(title)}</b>` : ""}${list}</div>`;
}

function opts(values, selected, blank) {
  const all = values.includes(selected) || !selected ? values : [...values, selected];
  return (blank ? `<option value="">${esc(blank)}</option>` : "") + all.map(v => `<option ${v === selected ? "selected" : ""}>${esc(v)}</option>`).join("");
}

function statusChips(g) {
  const chips = [];
  if (g.archived) chips.push(`<span class="chip">Archived</span>`);
  if (g.ready) chips.push(`<span class="chip ok">Ready to process</span>`);
  else g.missing.forEach(m => chips.push(`<span class="chip warn">${esc(m)}</span>`));
  if (g.processed) chips.push(`<span class="chip info">Processed</span>`);
  return chips.join("");
}

// ---------------------------------------------------------------------------------------------- router

async function route() {
  const hash = location.hash || "#/";
  setNav(hash);
  try {
    if (!META) META = await api("GET", "/api/meta");
    if (hash === "#/" || hash === "") await gamesPage();
    else if (hash === "#/game/new") await gamePage(null);
    else if (hash.startsWith("#/game/")) await gamePage(parseInt(hash.slice(7), 10));
    else if (hash === "#/teams") await teamsPage();
    else if (hash === "#/settings") await settingsPage();
    else view.innerHTML = `<div class="card">Page not found. <a href="#/">Back to the games</a></div>`;
  } catch (e) {
    view.innerHTML = notice("bad", "Something went wrong", e.errors || [e.message]);
  }
}

// The navbar: links on the list pages; back button, title and the page's actions on a game page.
function setNav(hash) {
  const isGame = hash.startsWith("#/game/");
  document.getElementById("nav-links").hidden = isGame;
  document.getElementById("nav-context").hidden = !isGame;
  document.getElementById("nav-actions").innerHTML = "";
  if (isGame) document.getElementById("nav-title").textContent = hash === "#/game/new" ? "New game" : "Game " + hash.slice(7);
  document.querySelectorAll("#nav-links a").forEach(a => {
    const n = a.dataset.nav;
    a.classList.toggle("active", (n === "games" && hash === "#/") || (n !== "games" && hash.startsWith("#/" + n)));
  });
}

window.addEventListener("hashchange", () => {
  if (suppressNext) { suppressNext = false; return; }
  if (dirty && !confirm("You have unsaved changes on this page. Leave without saving?")) {
    suppressNext = true; location.hash = currentHash; return;
  }
  dirty = false; currentHash = location.hash || "#/"; route();
});
window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
route();

// ---------------------------------------------------------------------------------------------- games list

async function gamesPage() {
  const games = await api("GET", `/api/games?archived=${listState.archived}`);
  if ((location.hash || "#/") !== "#/") return;
  view.innerHTML = `
    <div class="row spread" style="margin-bottom:14px"><h1 style="margin:0">Games</h1>
      <a class="btn primary" href="#/game/new">+ New game</a></div>
    <div class="card">
      <div class="filters">
        <div class="grow"><label>Search</label><input id="f-q" placeholder="Title, team, date, id..." value="${esc(listState.q)}"></div>
        <div><label>Type</label><select id="f-type">${opts(META.game_types, listState.type, "All")}</select></div>
        <div><label>Format</label><select id="f-format">${opts(META.game_formats, listState.format, "All")}</select></div>
        <div><label>Status</label><select id="f-status">
          ${[["", "All"], ["ready", "Ready to process"], ["attention", "Needs attention"], ["processed", "Processed"]].map(([v, l]) => `<option value="${v}" ${listState.status === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
        <div><label>&nbsp;</label><label style="display:flex;gap:6px;align-items:center;color:var(--text)"><input type="checkbox" id="f-arch" ${listState.archived ? "checked" : ""}> Show archived</label></div>
      </div>
      <table><thead><tr><th>ID</th><th>Date</th><th>Game</th><th>Type</th><th>Score</th><th>Status</th></tr></thead><tbody id="rows"></tbody></table>
      <p class="muted small" id="count"></p>
    </div>`;
  const draw = () => {
    const q = listState.q.trim().toLowerCase();
    const shown = games.filter(g => {
      if (listState.type && g.game_type !== listState.type) return false;
      if (listState.format && g.game_format !== listState.format) return false;
      if (listState.status === "ready" && !g.ready) return false;
      if (listState.status === "attention" && g.ready) return false;
      if (listState.status === "processed" && !g.processed) return false;
      if (q && ![g.game_id, g.title, g.team_a, g.team_b, g.date_display, g.description].join(" ").toLowerCase().includes(q)) return false;
      return true;
    });
    document.getElementById("rows").innerHTML = shown.map(g => `
      <tr class="click ${g.archived ? "archived" : ""}" data-id="${g.game_id}">
        <td>${g.game_id}</td><td>${esc(g.date_display) || '<span class="muted">-</span>'}</td>
        <td><b>${esc(g.title)}</b>${g.description ? `<div class="muted small">${esc(g.description)}</div>` : ""}</td>
        <td>${esc([g.game_type, g.game_format].filter(Boolean).join(" / "))}</td>
        <td>${esc(g.score) || '<span class="muted">-</span>'}</td><td>${statusChips(g)}</td></tr>`).join("")
      || `<tr><td colspan="6" class="muted">No games match.</td></tr>`;
    document.getElementById("count").textContent = `${shown.length} of ${games.length} games`;
    document.querySelectorAll("#rows tr[data-id]").forEach(tr => tr.onclick = () => { location.hash = "#/game/" + tr.dataset.id; });
  };
  const bind = (id, key, ev = "input") => document.getElementById(id).addEventListener(ev, e => { listState[key] = e.target.value; draw(); });
  bind("f-q", "q"); bind("f-type", "type", "change"); bind("f-format", "format", "change"); bind("f-status", "status", "change");
  document.getElementById("f-arch").onchange = e => { listState.archived = e.target.checked; gamesPage(); };
  draw();
}

// ---------------------------------------------------------------------------------------------- game page

const fmtT = s => { s = Math.max(0, Math.floor(s || 0)); return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); };
const parseT = v => { v = (v || "").trim(); if (!v) return null; const p = v.split(":"); if (p.length > 3 || p.some(x => !/^\d+$/.test(x))) return null; return p.reduce((a, x) => a * 60 + parseInt(x, 10), 0); };

function secOpen(name, dflt) { try { const v = localStorage.getItem("gl-sec-" + name); return v === null ? dflt : v === "1"; } catch (e) { return dflt; } }
function secSave(name, open) { try { localStorage.setItem("gl-sec-" + name, open ? "1" : "0"); } catch (e) { /* ignore */ } }

function sec(name, title, body, isNew) {
  return `<details class="sec" data-sec="${name}" ${isNew || secOpen(name, false) ? "open" : ""}><summary><span class="sec-title">${title}</span><span class="sec-sum" id="sum-${name}"></span></summary><div class="sec-body">${body}</div></details>`;
}

function exclusionRow(ex = {}) {
  return `<tr><td><input class="ex-start" placeholder="MM:SS" value="${esc(ex.start || "")}"></td>
    <td><input class="ex-end" placeholder="MM:SS" value="${esc(ex.end || "")}"></td>
    <td><input class="ex-reason" placeholder="e.g. goalkeeper change, ball out for a long time" value="${esc(ex.reason || "")}"></td>
    <td><button type="button" class="link ex-del" title="Remove">Remove</button></td></tr>`;
}

let keyHandler = null;
let rememberedPlayer = null;      // keeps the video position across the page reload that follows a save

async function gamePage(id) {
  const teams = await api("GET", "/api/teams");
  const isNew = id === null;
  let g = isNew ? { game_id: "", date_display: "", game_type: "league", game_format: "mens", title: "", description: "", notes: "",
    team_a_id: null, team_b_id: null, score_a: null, score_b: null, source_video: "", output_video: "", edited_video: "", youtube_360_video: "",
    running_speed_km_h: null, game_start: "", half_time_start: "", half_time_end: "", game_end: "", exclusions: [], calibrations: [], missing: [], archived: false } : await api("GET", `/api/games/${id}`);
  if (location.hash !== (isNew ? "#/game/new" : "#/game/" + id)) return;
  const teamOpts = sel => `<option value="">- not set -</option>` + teams.filter(t => t.is_active || t.team_id === sel).map(t => `<option value="${t.team_id}" ${t.team_id === sel ? "selected" : ""}>${esc(t.name)}</option>`).join("");
  const v = x => esc(x ?? "");

  const matchBody = `<div class="grid" style="margin-top:14px">
        <div class="c2"><label>Game id</label>${isNew ? `<input id="game_id" type="number" min="0" placeholder="${META.next_game_id}">` : `<input value="${g.game_id}" disabled>`}</div>
        <div class="c3"><label>Date</label><input id="date" type="date" value="${v(g.date_display)}"></div>
        <div class="c3"><label>Type</label><select id="game_type">${opts(META.game_types, g.game_type, "-")}</select></div>
        <div class="c4"><label>Format</label><select id="game_format">${opts(META.game_formats, g.game_format, "-")}</select></div>
        <div class="c5"><label>Team A</label><select id="team_a_id">${teamOpts(g.team_a_id)}</select></div>
        <div class="c2"><label>Score</label><div class="row" style="flex-wrap:nowrap"><input id="score_a" class="score" type="number" min="0" value="${v(g.score_a)}"><span>-</span><input id="score_b" class="score" type="number" min="0" value="${v(g.score_b)}"></div></div>
        <div class="c5"><label>Team B</label><select id="team_b_id">${teamOpts(g.team_b_id)}</select></div>
        <div class="c12"><label>Title <span class="small">(leave blank to use "Team A vs Team B"; it also names the output files)</span></label><input id="title" value="${v(g.title)}"></div>
        <div class="c6"><label>Description</label><textarea id="description">${v(g.description)}</textarea></div>
        <div class="c6"><label>Notes <span class="small">(for you; the pipeline ignores these)</span></label><textarea id="notes">${v(g.notes)}</textarea></div>
      </div>`;
  const slotOf = { edited_video: "edited", source_video: "analysis" };
  const autoInfo = f => (g.videos && g.videos[slotOf[f]]) || null;
  const gamePrefix = isNew ? "" : String(g.game_id).padStart(4, "0");
  const vfield = (id, title, help, browseTitle) => {
    const a = autoInfo(id), ph = a && a.auto && a.path ? "auto: " + a.shown : "";
    return `<div class="c12"><label>${title} <span class="small">${help}</span></label>
          <div class="row" style="flex-wrap:nowrap"><input id="${id}" value="${v(g[id])}" placeholder="${esc(ph)}"><button type="button" data-browse="${id}" data-title="${esc(browseTitle)}">Browse...</button></div><div id="check-${id}" class="small muted" style="margin-top:4px"></div></div>`;
  };
  const videoBody = `<div id="folder-box" class="folder-box"></div>
      <p class="muted small" style="margin:10px 0 0">The analysis and edited videos are found automatically by name in the game's folder (${gamePrefix || "0022"}_analysis.mp4 and ${gamePrefix || "0022"}_edited.mp4). Only fill them in for a video kept somewhere else. All of the game's videos can be played in the player at the top of the page.</p>
      <div class="grid" style="margin-top:10px">
        ${vfield("edited_video", "Edited video", "(the TV-style version that pans to follow the game)", "Choose the edited video")}
        ${vfield("source_video", "Analysis video", "(the video the pipeline uses for detections; game times are measured on this one)", "Choose the analysis video")}
        <div class="c6"><label>Annotated video link <span class="small">(the YouTube link once uploaded; the stats report uses it for chapter links. Annotated video files are found in the game folder.)</span></label><input id="output_video" value="${v(g.output_video)}"></div>
        <div class="c6"><label>YouTube 360 link</label><input id="youtube_360_video" value="${v(g.youtube_360_video)}"></div>
      </div>`;

  const timingsBody = `<p class="muted small" style="margin:12px 0 0">Minutes:seconds into the analysis video, e.g. 12:34 (1:02:30 also works). With the player above you can set these from the current time.</p>
      <div class="grid" style="margin-top:10px">
        <div class="c3"><label>Game start</label><input id="game_start" placeholder="MM:SS" value="${v(g.game_start)}"></div>
        <div class="c3"><label>Half-time start</label><input id="half_time_start" placeholder="MM:SS" value="${v(g.half_time_start)}"></div>
        <div class="c3"><label>Half-time end</label><input id="half_time_end" placeholder="MM:SS" value="${v(g.half_time_end)}"></div>
        <div class="c3"><label>Game end</label><input id="game_end" placeholder="MM:SS" value="${v(g.game_end)}"></div>
        <div class="c3"><label>Running speed km/h <span class="small">(blank = ${META.running_speed_km_h})</span></label><input id="running_speed_km_h" type="number" step="0.5" value="${v(g.running_speed_km_h)}"></div>
      </div>
      <h2 style="margin-top:18px">Extra exclusions <span class="muted small">parts of the video the pipeline should skip</span></h2>
      <table><thead><tr><th style="width:140px">From</th><th style="width:140px">To</th><th>Reason</th><th></th></tr></thead><tbody id="ex-rows">${(g.exclusions || []).map(exclusionRow).join("")}</tbody></table>
      <button type="button" id="ex-add" style="margin-top:8px">+ Add exclusion</button>`;

  const savedSteps = (() => { try { return JSON.parse(localStorage.getItem("gl-run-steps") || "null"); } catch (e) { return null; } })() || ["detections", "transformations", "annotation"];
  const runBody = isNew ? "" : `
      <div class="run-steps">${RUN_STEPS.map(([k, l, h]) => `<label class="run-step"><input type="checkbox" data-step="${k}" ${savedSteps.includes(k) ? "checked" : ""}> <b>${l}</b> <span class="small muted">${h}</span></label>`).join("")}</div>
      <div class="run-test"><label><input type="checkbox" id="run-test"> Test run: only the first <input id="run-mins" type="number" min="1" value="2" style="width:4.5em"> minutes of the game</label>
        <span class="small muted">Results go to separate test files, so your real results are never touched.</span></div>
      <div class="row" style="margin-top:12px;gap:10px"><button type="button" class="primary" id="run-go">Run</button><button type="button" class="danger" id="run-stop" disabled>Stop</button><span id="run-status" class="small"></span></div>
      <div id="run-msg"></div>
      <div class="run-bar" id="run-bar" hidden><div id="run-bar-fill"></div></div>
      <div class="small muted run-line" id="run-line"></div>
      <pre class="console" id="run-console">Nothing has been run yet.</pre>
      <div class="row spread" style="margin-top:8px"><label style="display:flex;gap:6px;align-items:center;margin:0"><input type="checkbox" id="run-follow" checked> Follow new output</label>
        <span class="row" style="gap:8px"><select id="run-past" style="width:auto;max-width:360px"><option value="">Earlier runs...</option></select><button type="button" id="run-past-view">Show log</button><button type="button" id="run-back" hidden>Back to current</button></span></div>
      <div class="run-reports"><b>Reports for this game</b> <span class="small muted">(PDFs, text files and review-sheet spreadsheets in the game's reports folder; click to open)</span><div id="rep-list" class="small"></div></div>`;

  const stage = isNew ? "" : `
    <section class="stage-band" id="stage">
      <div class="stage-main">
        <video id="pv" controls preload="metadata" playsinline></video>
        <div id="pv-msg" class="pv-msg" hidden></div>
      </div>
      <div class="stage-side">
        <div><label>Video</label><select id="vsel"></select><div id="pv-hint" class="small" style="color:#f0b955;margin-top:4px" hidden></div></div>
        <div class="pv-time"><span id="pv-cur">00:00</span><span class="pv-dur"> / <span id="pv-dur">--:--</span></span></div>
        <div class="pv-skip">
          <button type="button" data-skip="-30">&minus;30s</button><button type="button" data-skip="-5">&minus;5s</button><button type="button" data-skip="-0.0417" title="One frame back">&#9666; frame</button>
          <button type="button" data-skip="0.0417" title="One frame forward">frame &#9656;</button><button type="button" data-skip="5">+5s</button><button type="button" data-skip="30">+30s</button>
        </div>
        <div class="row" style="gap:10px"><label style="margin:0">Speed</label><select id="pv-speed" style="width:auto">${[0.5, 1, 2, 4, 8].map(s => `<option value="${s}" ${s === 1 ? "selected" : ""}>${s}x</option>`).join("")}</select>
          <label id="pv-proxy-wrap" style="margin:0;display:none;gap:5px;align-items:center"><input type="checkbox" id="pv-proxy"> lighter preview copy</label></div>
        <div id="pv-proxy-area" class="small"></div>
        <div id="pv-marks" class="pv-marks">
          <div class="pv-mh">Mark from the player <span>(then Save)</span></div>
          ${[["game_start", "Game start"], ["half_time_start", "Half-time start"], ["half_time_end", "Half-time end"], ["game_end", "Game end"]].map(([f, l]) =>
            `<div class="mk"><span class="mk-l">${l}</span><button type="button" class="mk-go" data-go="${f}" title="Jump to this time">--:--</button><button type="button" class="mk-set" data-set="${f}">Set here</button></div>`).join("")}
          <div class="mk"><span class="mk-l">Exclusion</span><button type="button" data-ex="start">Start here</button><button type="button" data-ex="end">End here</button></div>
        </div>
      </div>
    </section>`;

  view.innerHTML = `
    ${isNew ? "" : `<div style="margin-bottom:12px">${statusChips(g)}</div>`}
    <div id="msgs"></div>
    ${stage}
    <div class="sec-tools"><button type="button" class="link" id="sec-open">Expand all</button><button type="button" class="link" id="sec-close">Collapse all</button></div>
    <form id="form" autocomplete="off">
      ${sec("match", "Match", matchBody, isNew)}
      ${sec("video", "Video files", videoBody, isNew)}
      ${sec("timings", "Timings &amp; exclusions", timingsBody, isNew)}
    </form>
    ${isNew ? `<details class="sec" open><summary><span class="sec-title">Pitch calibration</span><span class="sec-sum">Save the game first, then add its pitch calibration here.</span></summary></details>`
      : `<details class="sec" data-sec="cal" id="cal-card" ${secOpen("cal", false) ? "open" : ""}><summary><span class="sec-title">Pitch calibration</span><span class="sec-sum" id="sum-cal"></span><a class="btn primary sec-act" id="sum-act" href="/calibrator?game=${g.game_id}">Calibrate pitch</a></summary><div class="sec-body" id="cal-body"></div></details>
         <details class="sec" data-sec="run" id="run-card" ${secOpen("run", false) ? "open" : ""}><summary><span class="sec-title">Run</span><span class="sec-sum" id="sum-run">detections, reports and the annotated video</span></summary><div class="sec-body">${runBody}</div></details>
         <details class="sec" data-sec="rec" ${secOpen("rec", false) ? "open" : ""} id="rec-details"><summary><span class="sec-title">What the pipeline is given</span><span class="sec-sum">the exact record for this game</span></summary><div class="sec-body"><pre class="rec" id="rec">Loading...</pre></div></details>`}`;

  const $ = id => document.getElementById(id);
  const form = $("form");
  document.getElementById("nav-title").textContent = isNew ? "New game" : `Game ${g.game_id}: ${g.title}`;
  document.getElementById("nav-actions").innerHTML = `<span class="gl-note" id="dirty-note"></span>` +
    (isNew ? "" : `<button type="button" class="gl-btn" id="dup">Duplicate</button><button type="button" class="gl-btn" id="arch">${g.archived ? "Unarchive" : "Archive"}</button><button type="button" class="gl-btn danger" id="del">Delete</button>`) +
    `<button type="submit" form="form" class="gl-btn primary" id="save">Save</button>`;

  // collapsible sections: remember which are open
  const setAll = open => document.querySelectorAll("details.sec[data-sec]").forEach(d => { d.open = open; });
  $("sec-open").onclick = () => setAll(true); $("sec-close").onclick = () => setAll(false);
  document.querySelectorAll("details.sec[data-sec]").forEach(d => d.addEventListener("toggle", () => { if (!isNew) secSave(d.dataset.sec, d.open); }));

  // one-line summaries shown while a section is collapsed
  const videoFound = { edited_video: null, source_video: null };
  const val = k => ($(k) ? $(k).value.trim() : "");
  const teamName = k => { const el = $(k); return el && el.value ? el.options[el.selectedIndex].text : ""; };
  const refreshSummaries = () => {
    const set = (n, html) => { const el = $("sum-" + n); if (el) el.innerHTML = html; };
    const title = val("title") || [teamName("team_a_id"), teamName("team_b_id")].filter(Boolean).join(" vs ");
    const score = val("score_a") !== "" && val("score_b") !== "" ? `${val("score_a")}-${val("score_b")}` : "";
    set("match", esc([title, score, val("date"), [val("game_type"), val("game_format")].filter(Boolean).join(" / ")].filter(Boolean).join("  ·  ")));
    const vparts = [["edited_video", "Edited"], ["source_video", "Analysis"]].map(([f, l]) => {
      const ai = autoInfo(f);
      if (!val(f) && !(ai && ai.path)) return f === "source_video" ? `<span style="color:var(--warn)">no analysis video</span>` : "";
      const fd = val(f) ? videoFound[f] : (ai ? ai.exists : null);
      return `${l} ` + (fd === false ? `<span style="color:var(--warn)">not found</span>` : fd ? `<span style="color:var(--ok)">found</span>` : "set");
    }).filter(Boolean);
    if (val("output_video")) vparts.push("YouTube link set");
    set("video", vparts.length ? vparts.join("  ·  ") : `<span style="color:var(--warn)">no videos chosen</span>`);
    const n = document.querySelectorAll("#ex-rows tr").length;
    const t = [val("game_start") || val("game_end") ? `${val("game_start") || "?"} to ${val("game_end") || "?"}` : "", val("half_time_start") ? `half-time ${val("half_time_start")}-${val("half_time_end") || "?"}` : "", n ? `${n} exclusion${n > 1 ? "s" : ""}` : ""].filter(Boolean);
    set("timings", t.length ? esc(t.join("  ·  ")) : `<span style="color:var(--warn)">not set</span>`);
  };
  const refreshMarks = () => {
    document.querySelectorAll(".mk-go").forEach(b => { b.textContent = val(b.dataset.go) || "--:--"; });
  };
  const markDirty = () => { dirty = true; $("dirty-note").textContent = "Unsaved changes"; };
  form.addEventListener("input", () => { markDirty(); refreshSummaries(); refreshMarks(); }); form.addEventListener("change", () => { markDirty(); refreshSummaries(); });

  // title placeholder follows the teams
  const names = () => teams.filter(t => [$("team_a_id").value, $("team_b_id").value].includes(String(t.team_id)))
    .sort((a, b) => ($("team_a_id").value === String(a.team_id) ? -1 : 1)).map(t => t.name).join(" vs ");
  const upd = () => { $("title").placeholder = names() || "Team A vs Team B"; };
  $("team_a_id").onchange = () => { upd(); refreshSummaries(); }; $("team_b_id").onchange = () => { upd(); refreshSummaries(); }; upd();

  // video path checks (one per video field), and the game folder box
  const VFIELDS = ["edited_video", "source_video"];
  const vt = {};
  const checkVideo = async f => {
    const p = $(f).value.trim(); const el = $("check-" + f);
    if (!p) {
      const ai = autoInfo(f);
      videoFound[f] = null;
      el.innerHTML = ai && ai.path ? (ai.exists ? `<span style="color:var(--ok)">Found automatically: ${esc(ai.shown)}</span>` : `<span style="color:var(--warn)">Not found: ${esc(ai.shown)}</span>`)
        : `<span>Not set${g.folder && g.folder.exists ? "" : " (a game folder is needed for it to be found automatically)"}</span>`;
      refreshSummaries(); return;
    }
    const r = await api("GET", "/api/check-video?path=" + encodeURIComponent(p));
    if ($(f) === null || $(f).value.trim() !== p) return;
    videoFound[f] = r.exists;
    const ai = autoInfo(f);
    el.innerHTML = r.exists ? `<span style="color:var(--ok)">File found</span>`
      : `<span style="color:var(--warn)">File not found at ${esc(r.resolved)}</span>` + (ai && ai.typed_missing && ai.typed_missing === p ? ` - using the file in the game folder instead: ${esc(ai.shown)}` : "");
    if (!r.exists && ai && ai.typed_missing === p) videoFound[f] = true;
    refreshSummaries();
  };
  VFIELDS.forEach(f => {
    $(f).addEventListener("input", () => {
      clearTimeout(vt[f]); vt[f] = setTimeout(() => checkVideo(f), 400);
      const hint = $("pv-hint");
      if (hint) { const changed = VFIELDS.some(k => $(k).value.trim() !== (g[k] || "")); hint.hidden = !changed; hint.textContent = "Save to preview the newly chosen video."; }
    });
    checkVideo(f);
  });
  document.querySelectorAll("[data-browse]").forEach(b => { b.onclick = () => openPicker($(b.dataset.browse), b.dataset.title, isNew ? {} : { game: g.game_id }); });

  const renderFolderBox = () => {
    const box = $("folder-box"); if (!box) return;
    const root = g.data_root || {}, fo = g.folder || {};
    let html;
    if (isNew) html = `<span class="muted">Save the game first, then its folder can be made.</span>`;
    else if (!root.set) html = `<span style="color:var(--warn)">No data folder is set.</span> <button type="button" class="primary" id="root-choose">Choose data folder...</button> <span class="small muted">where all your games are kept, e.g. /Volumes/LaCie/walking-football (also on the <a href="#/settings">Settings page</a>)</span>`;
    else if (!root.exists) html = `<span style="color:var(--warn)">The data folder is not connected:</span> <code>${esc(root.path)}</code>. Connect the drive and reload this page, or <button type="button" id="root-choose">choose another folder</button>.`;
    else if (fo.exists) html = `<b>Game folder:</b> <code>${esc(fo.shown)}</code> <button type="button" id="folder-open">Open in Finder</button>`;
    else html = `<b>No folder for this game yet.</b> <button type="button" class="primary" id="folder-make">Create game folder</button> <span class="small muted">(makes videos, data and reports folders inside <code>${esc(root.path)}/games</code>)</span>`;
    box.innerHTML = html;
    const open = $("folder-open"), make = $("folder-make"), choose = $("root-choose");
    if (choose) choose.onclick = () => {
      const tmp = document.createElement("input"); tmp.value = (root.path || "/Volumes/LaCie/walking-football");
      tmp.addEventListener("input", async () => {
        try {
          META = await api("PUT", "/api/settings", { data_root: tmp.value });
          const fresh = await api("GET", `/api/games/${g.game_id}`);
          g.folder = fresh.folder; g.videos = fresh.videos; g.data_root = fresh.data_root;
          toast("Data folder saved: " + tmp.value);
          renderFolderBox(); VFIELDS.forEach(checkVideo);
        } catch (e) { toast((e.errors || [e.message]).join(" "), true); }
      });
      openPicker(tmp, "Choose the data folder (the one that holds your games)", { folder: true });
    };
    if (open) open.onclick = async () => { try { await api("POST", `/api/games/${g.game_id}/folder/open`); } catch (e) { toast((e.errors || [e.message]).join(" "), true); } };
    if (make) make.onclick = async () => {
      try {
        const r = await api("POST", `/api/games/${g.game_id}/folder`);
        const fresh = await api("GET", `/api/games/${g.game_id}`);
        g.folder = fresh.folder; g.videos = fresh.videos; g.data_root = fresh.data_root;
        toast("Folder made: " + r.shown + " - put the videos in its videos folder");
        renderFolderBox(); VFIELDS.forEach(checkVideo);
      } catch (e) { toast((e.errors || [e.message]).join(" "), true); }
    };
  };
  renderFolderBox();

  // exclusions
  $("ex-add").onclick = () => { $("ex-rows").insertAdjacentHTML("beforeend", exclusionRow()); markDirty(); refreshSummaries(); };
  $("ex-rows").addEventListener("click", e => { if (e.target.classList.contains("ex-del")) { e.target.closest("tr").remove(); markDirty(); refreshSummaries(); } });

  const collect = () => {
    const d = {};
    ["game_id", "date", "game_type", "game_format", "title", "description", "notes", "team_a_id", "team_b_id", "score_a", "score_b",
      "source_video", "output_video", "edited_video", "youtube_360_video", "running_speed_km_h", "game_start", "half_time_start", "half_time_end", "game_end"]
      .forEach(k => { const el = $(k); if (el) d[k] = el.value; });
    d.exclusions = [...document.querySelectorAll("#ex-rows tr")].map(tr => ({
      start: tr.querySelector(".ex-start").value, end: tr.querySelector(".ex-end").value, reason: tr.querySelector(".ex-reason").value }));
    return d;
  };

  const player = isNew ? null : setupPlayer(id, $, { markDirty, refreshSummaries, refreshMarks, val, fmt: fmtT });

  form.onsubmit = async e => {
    e.preventDefault();
    $("save").disabled = true;
    try {
      const r = await api(isNew ? "POST" : "PUT", isNew ? "/api/games" : `/api/games/${id}`, collect());
      dirty = false;
      if (isNew) { sessionStorageSet("flash", JSON.stringify({ warnings: r.warnings })); location.hash = "#/game/" + r.game.game_id; return; }
      toast("Saved");
      if (player) rememberedPlayer = player.state();
      await gamePage(id);
      if (r.warnings.length) $("msgs").innerHTML = notice("warn", "Saved, but check:", r.warnings);
    } catch (err) {
      $("msgs").innerHTML = notice("bad", "Not saved - please fix:", err.errors); window.scrollTo(0, 0);
    } finally { const b = $("save"); if (b) b.disabled = false; }
  };

  refreshSummaries(); refreshMarks();
  if (isNew) return;

  const flash = sessionStorageGet("flash");
  if (flash) { sessionStorageSet("flash", ""); const f = JSON.parse(flash);
    $("msgs").innerHTML = notice("ok", "Game created.", []) + (f.warnings && f.warnings.length ? notice("warn", "Check:", f.warnings) : ""); }

  $("dup").onclick = async () => { const n = await api("POST", `/api/games/${id}/duplicate`); toast(`Created game ${n.game_id}`); location.hash = "#/game/" + n.game_id; };
  $("arch").onclick = async () => { await api("POST", `/api/games/${id}/archive`, { archived: !g.archived }); toast(g.archived ? "Unarchived" : "Archived"); dirty = false; gamePage(id); };
  $("del").onclick = async () => {
    if (!confirm(`Delete game ${id} "${g.title}" and its calibrations? (A backup of the database is made first.)\n\nTo just hide it, use Archive instead.`)) return;
    await api("DELETE", `/api/games/${id}`); dirty = false; toast("Deleted"); location.hash = "#/";
  };
  const loadRecord = async () => { $("rec").textContent = JSON.stringify(await api("GET", `/api/games/${id}/record`), null, 2).replace(/\\n/g, "\n"); };
  $("rec-details").addEventListener("toggle", () => { if ($("rec-details").open) loadRecord(); });
  if ($("rec-details").open) loadRecord();
  renderCalibrations(g);
  if (!isNew) setupRunPanel(g);
}

// ---------------------------------------------------------------------------------------------- run panel + console

const RUN_STEPS = [
  ["detections", "Detections", "find players, goalkeepers and the ball in every frame (the long one)"],
  ["transformations", "Transformations and statistics", "overhead positions, teams, tracks, speed, distance, possession"],
  ["detection_report", "Detection report", "scores each detection stage and says what to fix first (PDF, plus a review sheet to check by eye)"],
  ["match_report", "Match report", "12-page PDF: possession, territory, team shape, distance, speed, running, key moments"],
  ["opposition_report", "Opposition reports", "2 scouting PDFs (one per team, written for the other team's coach): style, strong and weak points, game plan"],
  ["annotation", "Annotated video", "the video with detections drawn on it"],
];
let runTimer = null;

function setupRunPanel(g) {
  const $ = id => document.getElementById(id);
  clearInterval(runTimer);
  const loadReports = async () => {
    const box = $("rep-list"); if (!box) return;
    try {
      const rows = await api("GET", `/api/games/${g.game_id}/reports`);
      box.innerHTML = rows.length ? rows.map(r => `<div class="rep-row"><a href="/api/games/${g.game_id}/reports/file?name=${encodeURIComponent(r.name)}&test=${r.test}" target="_blank" rel="noopener">${esc(r.name)}</a>${r.test ? ' <span class="chip">test</span>' : ""}
        <span class="muted">${esc(r.when)} &middot; ${fmtSize(r.size)}</span></div>`).join("") : '<span class="muted">None yet.</span>';
    } catch (e) { box.innerHTML = '<span class="muted">Not available (is the data folder connected?).</span>'; }
  };
  loadReports();
  if (window.glRepDone) document.removeEventListener("gl-run-finished", window.glRepDone);
  window.glRepDone = loadReports;
  document.addEventListener("gl-run-finished", window.glRepDone);
  let offset = 0, lines = [], cur = "", overwrite = false, shownRun = null, mode = "current", lastStatus = null, busy = false;

  const feed = text => {
    for (const ch of text) {
      if (ch === "\n") { lines.push(cur); cur = ""; overwrite = false; }
      else if (ch === "\r") overwrite = true;
      else { if (overwrite) { cur = ""; overwrite = false; } cur += ch; }
    }
    if (lines.length > 4000) lines.splice(0, lines.length - 4000);
  };
  const paint = () => {
    const pre = $("run-console"); if (!pre) return;
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.textContent = lines.join("\n") + (cur ? "\n" + cur : "");
    if ($("run-follow").checked && (atBottom || mode === "old")) pre.scrollTop = pre.scrollHeight;
    // progress: the last "NN%" a progress bar printed, and the latest line
    const recent = lines.slice(-40).concat([cur]);
    let pct = null, last = "";
    for (let i = recent.length - 1; i >= 0; i--) {
      const t = recent[i].trim();
      if (!last && t) last = t;
      if (pct === null) { const m = /(\d{1,3})%\|/.exec(t); if (m) pct = Math.min(100, +m[1]); }
      if (last && pct !== null) break;
    }
    $("run-line").textContent = last.length > 200 ? last.slice(0, 200) + "..." : last;
    $("run-bar").hidden = pct === null || mode !== "current";
    if (pct !== null) $("run-bar-fill").style.width = pct + "%";
    return pct;
  };
  const reset = () => { lines = []; cur = ""; overwrite = false; offset = 0; };
  const parseStart = s => { const d = new Date((s || "").replace(" ", "T")); return isNaN(d) ? null : d; };
  const elapsed = r => {
    const a = parseStart(r.started), b = r.ended ? parseStart(r.ended) : new Date();
    if (!a || !b) return ""; const t = Math.max(0, Math.round((b - a) / 1000));
    return `${String(Math.floor(t / 3600)).padStart(2, "0")}:${String(Math.floor(t / 60) % 60).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
  };
  const stepNames = r => (r.steps || []).map(k => (RUN_STEPS.find(x => x[0] === k) || [k, k])[1]).join(", ");

  const showStatus = (r, pct) => {
    const mine = r && r.game_id === g.game_id, running = r && r.status === "running";
    $("run-go").disabled = !!running;
    $("run-stop").disabled = !(running && mine);
    document.querySelectorAll("[data-step], #run-test, #run-mins").forEach(el => { el.disabled = !!running; });
    let txt = "", sum = "detections, reports and the annotated video";
    if (r) {
      const who = mine ? "" : `game ${r.game_id}: `;
      if (running) { txt = `<b style="color:var(--ok)">Running</b> ${who}${esc(stepNames(r))}${r.test ? " (test run)" : ""} - ${elapsed(r)}`; sum = `Running${pct != null ? " " + Math.round(pct) + "%" : ""} - ${elapsed(r)}`; }
      else {
        const cls = r.status === "finished" ? "var(--ok)" : "var(--bad)";
        txt = `<b style="color:${cls}">${esc(r.status[0].toUpperCase() + r.status.slice(1))}</b> ${who}${esc(stepNames(r))}${r.test ? " (test run)" : ""} - took ${elapsed(r)}`;
        sum = mine ? `Last run: ${r.status} ${r.ended || ""}` : sum;
      }
      if (!mine) txt += ` <a href="#/game/${r.game_id}">open game ${r.game_id}</a>`;
    }
    $("run-status").innerHTML = txt; $("sum-run").textContent = sum;
  };

  const pollOnce = async () => {
    if (!$("run-console")) { clearInterval(runTimer); return; }
    if (busy || mode !== "current") return;
    busy = true;
    try {
      const st = (await api("GET", "/api/run")).run;
      if (location.hash !== "#/game/" + g.game_id) return;
      const mine = st && st.game_id === g.game_id;
      if (mine) {
        if (shownRun !== st.id) { reset(); shownRun = st.id; }
        const first = offset === 0 && lines.length === 0;
        const r = await api("GET", `/api/run/log?offset=${offset}${first ? "&tail=true" : ""}`);
        if (r.run && r.run.id === st.id) { offset = r.offset; if (r.text) feed(r.text); }
      }
      const pct = paint();
      showStatus(st, pct);
      if (lastStatus === "running" && st && st.status !== "running") {
        toast(st.status === "finished" ? "Run finished" : "Run " + st.status, st.status !== "finished");
        document.dispatchEvent(new Event("gl-run-finished"));
        loadPast();
      }
      lastStatus = st ? st.status : null;
    } catch (e) { /* the app may be restarting; try again next time */ } finally { busy = false; }
  };

  const loadPast = async () => {
    try {
      const runs = await api("GET", `/api/games/${g.game_id}/runs`);
      $("run-past").innerHTML = `<option value="">Earlier runs...</option>` + runs.map(r =>
        `<option value="${esc(r.id)}">${esc(r.started)}  ${r.test ? "[test] " : ""}${esc((r.steps || []).join(", "))}  - ${esc(r.status)}</option>`).join("");
    } catch (e) { /* ignore */ }
  };

  $("run-go").onclick = async () => {
    $("run-msg").innerHTML = "";
    const steps = [...document.querySelectorAll("[data-step]:checked")].map(el => el.dataset.step);
    try { localStorage.setItem("gl-run-steps", JSON.stringify(steps)); } catch (e) { /* ignore */ }
    const body = { steps };
    if ($("run-test").checked) body.duration_s = Math.max(1, Math.round(parseFloat($("run-mins").value || "2") * 60));
    try {
      const r = await api("POST", `/api/games/${g.game_id}/run`, body);
      mode = "current"; $("run-back").hidden = true; reset(); shownRun = r.run.id; lastStatus = "running";
      $("run-console").textContent = "Starting..."; $("run-card").open = true;
      pollOnce();
    } catch (e) { $("run-msg").innerHTML = notice("bad", "Not started:", e.errors || [e.message]); }
  };
  $("run-stop").onclick = async () => {
    if (!confirm("Stop the run? Whatever the current stage has not finished will be lost; stages that completed are kept.")) return;
    try { await api("POST", "/api/run/stop"); toast("Stopping..."); pollOnce(); } catch (e) { toast((e.errors || [e.message]).join(" "), true); }
  };
  $("run-past-view").onclick = async () => {
    const id = $("run-past").value; if (!id) return;
    try {
      const r = await api("GET", `/api/games/${g.game_id}/runs/${encodeURIComponent(id)}/log?tail=true`);
      mode = "old"; reset(); feed(r.text); paint(); $("run-back").hidden = false;
      $("run-line").textContent = "Showing the end of an earlier run's log.";
    } catch (e) { toast((e.errors || [e.message]).join(" "), true); }
  };
  $("run-back").onclick = () => { mode = "current"; $("run-back").hidden = true; reset(); shownRun = null; $("run-console").textContent = ""; pollOnce(); };

  loadPast(); pollOnce();
  runTimer = setInterval(pollOnce, 1500);
}

// ---------------------------------------------------------------------------------------------- video player

function setupPlayer(id, $, ctx) {
  const pv = $("pv"), msg = $("pv-msg"), vsel = $("vsel");
  let items = [], ffmpeg = false, poll = null, pendingSeek = null;
  const key = () => vsel.value;
  const cur = () => items.find(i => i.key === key()) || null;
  const proxy = () => (cur() && cur().proxy) || {};
  const isAnalysis = () => key() === "analysis";
  const useProxy = () => $("pv-proxy").checked && !!proxy().exists;
  const say = (html, kind) => { msg.hidden = !html; msg.className = "pv-msg" + (kind ? " " + kind : ""); msg.innerHTML = html || ""; };
  const pkey = () => "gl-proxy-" + id + "-" + key();
  const lsGet = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } };
  const note = document.createElement("div"); note.className = "small"; note.style.cssText = "color:#f0b955;margin:2px 0 8px"; note.hidden = true;
  $("pv-marks").before(note);

  const load = (seek) => {
    if (!key()) return;
    pendingSeek = seek || 0;
    say("");
    pv.src = `/api/games/${id}/video?key=${encodeURIComponent(key())}` + (useProxy() ? "&proxy=1" : "");
    pv.playbackRate = parseFloat($("pv-speed").value);
    $("pv-marks").classList.toggle("off", !isAnalysis());
    note.hidden = isAnalysis();
    note.textContent = "Game times are measured on the Analysis video, so the Set here buttons only work while that one is selected.";
  };
  pv.addEventListener("loadedmetadata", () => {
    $("pv-dur").textContent = fmtT(pv.duration);
    if (pendingSeek) { pv.currentTime = Math.min(pendingSeek, pv.duration || pendingSeek); pendingSeek = null; }
  });
  pv.addEventListener("timeupdate", () => { $("pv-cur").textContent = fmtT(pv.currentTime); });
  pv.addEventListener("error", () => {
    if (!pv.getAttribute("src")) return;
    if (!useProxy() && proxy().exists) { $("pv-proxy").checked = true; lsSet(pkey(), "1"); load(pv.currentTime); return; }
    if (useProxy()) { say("This browser could not play the preview copy either. Try another browser (Chrome or Safari), or untick \"lighter preview copy\" to try the original.", "bad"); return; }
    say(esc(`Your browser could not play this video (often because it is HEVC/H.265). `) +
      (ffmpeg ? `<button type="button" id="pv-make">Make a lighter preview copy</button> <span>(one-off, takes a few minutes)</span>` : "Install ffmpeg to be able to make a preview copy that plays."), "bad");
    const b = $("pv-make"); if (b) b.onclick = startProxy;
  });

  const showProxyUi = () => {
    const p = proxy(), c = cur();
    $("pv-proxy-wrap").style.display = p.exists ? "flex" : "none";
    $("pv-proxy").checked = !!p.exists && lsGet(pkey()) === "1";
    const area = $("pv-proxy-area");
    if (p.state === "running") area.innerHTML = `Making the preview copy${p.pct != null ? ` ${Math.round(p.pct)}%` : "..."}`;
    else if (p.state === "error") area.innerHTML = `<span style="color:#ff9d90">Preview copy failed: ${esc(p.error || "")}</span>`;
    else if (!p.exists && ffmpeg && c && c.exists) area.innerHTML = `<button type="button" class="link" id="pv-make2">Make a lighter preview copy</button>`;
    else area.innerHTML = "";
    const b = $("pv-make2"); if (b) b.onclick = startProxy;
  };
  const refreshList = async () => {
    const r = await api("GET", `/api/games/${id}/videos`);
    items = r.items; ffmpeg = r.ffmpeg;
    return r;
  };
  async function startProxy() {
    const k = key();
    try { await api("POST", `/api/games/${id}/preview-copy?key=${encodeURIComponent(k)}`); } catch (e) { say(esc((e.errors || [e.message]).join(" ")), "bad"); return; }
    say("Making the preview copy - you can keep working; it will switch over when ready.");
    waitForProxy(k);
  }
  function waitForProxy(k) {
    clearInterval(poll);
    poll = setInterval(async () => {
      if (!document.getElementById("pv")) { clearInterval(poll); return; }
      await refreshList();
      if (key() !== k) { clearInterval(poll); return; }
      showProxyUi();
      const p = proxy();
      if (p.exists) { clearInterval(poll); lsSet(pkey(), "1"); say(""); showProxyUi(); load(pv.currentTime); }
      else if (p.state === "error") { clearInterval(poll); say("Could not make the preview copy: " + esc(p.error || ""), "bad"); }
    }, 2000);
  }

  vsel.addEventListener("change", () => { clearInterval(poll); showProxyUi(); load(0); if (proxy().state === "running") waitForProxy(key()); });
  $("pv-proxy").addEventListener("change", () => { lsSet(pkey(), $("pv-proxy").checked ? "1" : "0"); load(pv.currentTime); });
  $("pv-speed").addEventListener("change", () => { pv.playbackRate = parseFloat($("pv-speed").value); });
  document.querySelectorAll("[data-skip]").forEach(b => b.onclick = () => {
    const d = parseFloat(b.dataset.skip); if (Math.abs(d) < 1) pv.pause();
    pv.currentTime = Math.max(0, Math.min(pv.duration || 1e9, pv.currentTime + d));
  });
  document.querySelectorAll("[data-go]").forEach(b => b.onclick = () => { const t = parseT(ctx.val(b.dataset.go)); if (t !== null) pv.currentTime = t; });
  document.querySelectorAll("[data-set]").forEach(b => b.onclick = () => {
    if (!isAnalysis()) { toast("Switch to the Analysis video to set game times"); return; }
    const el = $(b.dataset.set); el.value = fmtT(pv.currentTime); el.dispatchEvent(new Event("input", { bubbles: true }));
    toast(`${el.previousElementSibling ? el.previousElementSibling.textContent : "Time"} set to ${el.value} - remember to Save`);
  });
  document.querySelectorAll("[data-ex]").forEach(b => b.onclick = () => {
    if (!isAnalysis()) { toast("Switch to the Analysis video to set game times"); return; }
    const rows = $("ex-rows"), t = fmtT(pv.currentTime);
    if (b.dataset.ex === "start") { rows.insertAdjacentHTML("beforeend", exclusionRow({ start: t })); toast(`Exclusion starts at ${t} - now find where it ends and press End here`); }
    else {
      const last = rows.lastElementChild;
      if (!last) { toast("Press Start here first"); return; }
      last.querySelector(".ex-end").value = t; toast(`Exclusion ends at ${t}`);
    }
    ctx.markDirty(); ctx.refreshSummaries();
  });

  if (keyHandler) document.removeEventListener("keydown", keyHandler);
  keyHandler = e => {
    const p = document.getElementById("pv"); if (!p) return;
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || document.activeElement === p || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); p.paused ? p.play() : p.pause(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); p.currentTime -= e.shiftKey ? 30 : 5; }
    else if (e.key === "ArrowRight") { e.preventDefault(); p.currentTime += e.shiftKey ? 30 : 5; }
    else if (e.key === ",") { p.pause(); p.currentTime -= 0.0417; }
    else if (e.key === ".") { p.pause(); p.currentTime += 0.0417; }
  };
  document.addEventListener("keydown", keyHandler);

  function fillOptions() {
    const keep = vsel.value;
    vsel.innerHTML = items.map(i => `<option value="${esc(i.key)}" ${i.exists ? "" : "disabled"}>${esc(i.label)}${i.exists || /(not set|none yet)$/.test(i.label) ? "" : " (file not found)"}</option>`).join("");
    if (keep && items.some(i => i.key === keep && i.exists)) vsel.value = keep;
  }
  if (window.glRunDone) document.removeEventListener("gl-run-finished", window.glRunDone);
  window.glRunDone = async () => { if (!document.getElementById("pv")) return; try { await refreshList(); fillOptions(); showProxyUi(); } catch (e) { /* ignore */ } };
  document.addEventListener("gl-run-finished", window.glRunDone);

  (async () => {
    await refreshList();
    fillOptions();
    const first = items.find(i => i.key === "analysis" && i.exists) || items.find(i => i.exists);
    if (!first) { say("No video to show yet. Choose the videos under Video files and save."); return; }
    vsel.value = first.key;
    const rp = rememberedPlayer && rememberedPlayer.id === id ? rememberedPlayer : null; rememberedPlayer = null;
    if (rp && items.some(i => i.key === rp.key && i.exists)) vsel.value = rp.key;
    showProxyUi();
    if (proxy().state === "running") waitForProxy(key());
    load(rp && rp.key === vsel.value ? rp.time : 0);
  })();

  return { state: () => ({ id, key: key(), time: pv.currentTime }) };
}

function sessionStorageGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
function sessionStorageSet(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* ignore */ } }

function renderCalibrations(g) {
  const body = document.getElementById("cal-body");
  const active = (g.calibrations || []).find(c => c.is_active);
  const pts = active ? META.point_names.map(n => `<div><b>${esc(n.replace("_", " "))}</b>${active.vertices[n] ? active.vertices[n].join(", ") : "-"}</div>`).join("") : "";
  document.getElementById("sum-cal").innerHTML = active ? esc(`In use: ${active.label || "calibration " + active.id}  ·  ${active.created_at}`) : `<span style="color:var(--warn)">No calibration yet</span>`;
  document.getElementById("sum-act").textContent = active ? "Re-calibrate pitch" : "Calibrate pitch";
  body.innerHTML = `
    ${active ? `<p class="muted small" style="margin:12px 0 8px">In use: ${esc(active.label || "calibration " + active.id)} &middot; ${esc(active.created_at)} &middot; from ${esc(active.source)} &middot; ${active.has_goal_posts ? "goal posts included" : "no goal posts"} &middot; ${active.has_calibrator_json ? "full calibrator data" : "outline only (older style; the pipeline uses the simple pitch transform)"}</p><div class="pts">${pts}</div>`
      : `<p class="notice warn" style="margin-top:12px">No calibration yet. Click Calibrate pitch (or paste calibrator JSON below).</p>`}
    <h2 style="margin-top:18px">Or paste a calibration</h2>
    <label>Paste the JSON exported by the Pitch Calibrator</label>
    <textarea id="cal-json" class="code" placeholder='{ "source_frame": "...", "pitch_vertices": { ... } }'></textarea>
    <div class="row" style="margin-top:8px"><input id="cal-label" placeholder="Label (optional, e.g. 'second try with netting edge')" style="flex:1;min-width:220px">
      <button class="primary" id="cal-add" type="button">Add and use</button></div>
    <div id="cal-msg" style="margin-top:10px"></div>
    ${(g.calibrations || []).length ? `<h2 style="margin-top:18px">History <span class="muted small">newest first; the pipeline uses the one marked "in use"</span></h2>` +
      g.calibrations.map(c => `<div class="cal"><span>${c.is_active ? `<span class="chip ok">In use</span>` : ""}</span>
        <span style="flex:1;min-width:200px"><b>${esc(c.label || "Calibration " + c.id)}</b> <span class="muted small">${esc(c.created_at)} &middot; ${esc(c.source)}${c.has_goal_posts ? " &middot; goal posts" : ""}</span></span>
        ${c.is_active ? "" : `<button class="link" data-use="${c.id}" type="button">Use this one</button>`}
        ${c.has_calibrator_json ? `<a class="btn" href="/api/calibrations/${c.id}/download">Download JSON</a>` : ""}
        <button class="link danger" data-del="${c.id}" type="button">Delete</button></div>`).join("") : ""}`;
  const $ = id => document.getElementById(id);
  $("cal-add").onclick = async () => {
    try {
      const r = await api("POST", `/api/games/${g.game_id}/calibrations`, { json: $("cal-json").value, label: $("cal-label").value, make_active: true });
      toast("Calibration added"); renderCalibrations(r.game);
    } catch (e) { $("cal-msg").innerHTML = notice("bad", "Could not add it:", e.errors); }
  };
  body.querySelectorAll("[data-use]").forEach(b => b.onclick = async () => { renderCalibrations(await api("POST", `/api/games/${g.game_id}/calibrations/${b.dataset.use}/activate`)); toast("Calibration switched"); });
  body.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
    if (!confirm("Delete this calibration?")) return;
    renderCalibrations(await api("DELETE", `/api/games/${g.game_id}/calibrations/${b.dataset.del}`));
  });
}

// ---------------------------------------------------------------------------------------------- teams

const hex = rgb => "#" + rgb.map(n => Math.max(0, Math.min(255, n)).toString(16).padStart(2, "0")).join("");
const unhex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));

async function teamsPage() {
  const teams = await api("GET", "/api/teams");
  if (location.hash !== "#/teams") return;
  view.innerHTML = `<div class="row spread" style="margin-bottom:14px"><h1 style="margin:0">Teams</h1><button class="primary" id="t-new">+ New team</button></div>
    <div class="card"><p class="muted small" style="margin-top:0">Kit colours are what the team classifier matches players against, so a team needs at least one colour before a game with that team is processed.</p>
    <table><thead><tr><th>Team</th><th>Kit colours</th><th>Games</th><th></th></tr></thead><tbody>
    ${teams.map(t => `<tr><td><b>${esc(t.name)}</b> ${t.is_active ? "" : '<span class="chip">Inactive</span>'}${t.notes ? `<div class="muted small">${esc(t.notes)}</div>` : ""}</td>
      <td>${t.kit_colours.length ? t.kit_colours.map(c => `<span class="swatch" style="background:rgb(${c.rgb.join(",")})" title="rgb ${c.rgb.join(", ")}"></span>`).join("") : '<span class="chip warn">No colours yet</span>'}</td>
      <td>${t.games}</td><td style="text-align:right"><button class="link" data-edit="${t.team_id}">Edit</button></td></tr>`).join("")}</tbody></table></div>
    <dialog id="dlg"></dialog>`;
  const dlg = document.getElementById("dlg");
  const colourRow = (c = { rgb: [128, 128, 128], tolerance: [10, 50, 50] }) => `<div class="row kc" style="margin-bottom:6px;flex-wrap:nowrap">
      <input type="color" class="kc-rgb" value="${hex(c.rgb)}"><span class="small muted">tolerance H S V</span>
      ${c.tolerance.map((n, i) => `<input class="kc-tol" type="number" min="0" style="width:72px" value="${n}">`).join("")}
      <button type="button" class="link danger kc-del">Remove</button></div>`;
  const edit = t => {
    t = t || { name: "", notes: "", is_active: true, kit_colours: [{ rgb: [128, 128, 128], tolerance: [10, 50, 50] }] };
    dlg.innerHTML = `<h2>${t.team_id ? "Edit team" : "New team"}</h2><div id="dlg-msg"></div>
      <label>Name <span class="small">(must match what you want in game titles)</span></label><input id="t-name" value="${esc(t.name)}">
      <label style="margin-top:10px">Notes</label><input id="t-notes" value="${esc(t.notes || "")}">
      <label style="margin-top:10px;display:flex;gap:6px;align-items:center;color:var(--text)"><input type="checkbox" id="t-active" ${t.is_active ? "checked" : ""}> Active (offered when logging new games)</label>
      <h2 style="margin-top:16px">Kit colours</h2><div id="kcs">${t.kit_colours.map(colourRow).join("")}</div>
      <button type="button" id="kc-add">+ Add colour</button>
      <div class="row spread" style="margin-top:18px"><div>${t.team_id ? `<button class="danger" id="t-del" type="button">Delete team</button>` : ""}</div>
      <div class="row"><button type="button" id="t-cancel">Cancel</button><button class="primary" id="t-save" type="button">Save</button></div></div>`;
    dlg.showModal();
    const $ = id => dlg.querySelector("#" + id);
    $("kc-add").onclick = () => $("kcs").insertAdjacentHTML("beforeend", colourRow());
    $("kcs").onclick = e => { if (e.target.classList.contains("kc-del")) e.target.closest(".kc").remove(); };
    $("t-cancel").onclick = () => dlg.close();
    if ($("t-del")) $("t-del").onclick = async () => {
      const msg = t.games ? `Delete "${t.name}"?\n\nIt is used in ${t.games} game(s). It will be removed from those games (they keep their scores, videos and everything else, but will show "Teams missing" until you pick a team again). A backup of the database is taken first.`
                          : `Delete "${t.name}"?`;
      if (!confirm(msg)) return;
      try { await api("DELETE", `/api/teams/${t.team_id}${t.games ? "?clear_from_games=true" : ""}`); dlg.close(); toast("Team deleted"); teamsPage(); }
      catch (e) { $("dlg-msg").innerHTML = notice("bad", "Not deleted:", e.errors); }
    };
    $("t-save").onclick = async () => {
      const body = { name: $("t-name").value, notes: $("t-notes").value, is_active: $("t-active").checked,
        kit_colours: [...dlg.querySelectorAll(".kc")].map(r => ({ rgb: unhex(r.querySelector(".kc-rgb").value), tolerance: [...r.querySelectorAll(".kc-tol")].map(i => parseInt(i.value, 10) || 0) })) };
      try { await api(t.team_id ? "PUT" : "POST", t.team_id ? `/api/teams/${t.team_id}` : "/api/teams", body); dlg.close(); toast("Saved"); teamsPage(); }
      catch (e) { $("dlg-msg").innerHTML = notice("bad", "Not saved:", e.errors); }
    };
  };
  document.getElementById("t-new").onclick = () => edit(null);
  view.querySelectorAll("[data-edit]").forEach(b => b.onclick = () => edit(teams.find(t => t.team_id === +b.dataset.edit)));
}

// ---------------------------------------------------------------------------------------------- settings

async function settingsPage() {
  META = await api("GET", "/api/meta");
  if (location.hash !== "#/settings") return;
  view.innerHTML = `<h1>Settings</h1><div id="msgs"></div>
    <div class="card"><div class="grid">
      <div class="c4"><label>Default running speed (km/h)</label><input id="s-speed" type="number" step="0.5" value="${esc(META.running_speed_km_h)}"></div>
      <div class="c8" style="grid-column:span 8"><label>Pitch Calibrator link</label><input id="s-cal" value="${esc(META.calibrator_url)}"></div>
    </div><div style="margin-top:14px"><button class="primary" id="s-save">Save settings</button></div></div>
    <div class="card"><h2>Data folder</h2>
      <p class="small muted" style="margin-top:0">Where your games are kept: one folder per game, with its videos, detections database and reports (normally on the external drive). Change it here if the drive is replaced or mounted under another name.</p>
      <label>Data folder</label>
      <div class="row" style="flex-wrap:nowrap"><input id="s-root" value="${esc((META.data_root || {}).path || "")}" placeholder="/Volumes/LaCie/walking-football" ${META.data_root_from_env ? "disabled" : ""}><button type="button" id="s-root-browse" ${META.data_root_from_env ? "disabled" : ""}>Browse...</button></div>
      <div class="small" style="margin-top:6px">${!(META.data_root || {}).set ? `<span class="muted">Not set - the old shared output folder is used.</span>`
        : META.data_root.exists ? `<span style="color:var(--ok)">Connected.</span> Database backups are also copied to <code>${esc(META.data_root.path)}/_app-data/game-logger-backups</code>.`
        : `<span style="color:var(--warn)">Not connected right now.</span>`}${META.data_root_from_env ? " (Set by the WALKING_FOOTBALL_ROOT environment variable.)" : ""}</div>
      <div style="margin-top:12px"><button class="primary" id="s-root-save">Save data folder</button> <span class="small muted">Leave it empty and save to go back to the old output folder.</span></div></div>
    <div class="card"><h2>Your data</h2>
      <p class="small">Database file (kept on this computer): <code>${esc(META.db_path)}</code><br>Daily backups are kept in: <code>${esc(META.backup_dir)}</code>${(META.data_root || {}).exists ? ", and a copy on the data folder." : ""}</p>
      <a class="btn" href="/api/export.xlsx">Download everything as a spreadsheet</a>
      <p class="muted small">A read-only copy in the old games-logger.xlsx layout, for safekeeping. Changes made in that file are not read back in.</p></div>
    <p class="muted small">Game logger ${esc(META.version)}</p>`;
  document.getElementById("s-root-browse").onclick = () => openPicker(document.getElementById("s-root"), "Choose the data folder", { folder: true });
  document.getElementById("s-root-save").onclick = async () => {
    try {
      META = await api("PUT", "/api/settings", { data_root: document.getElementById("s-root").value });
      toast("Data folder saved"); settingsPage();
    } catch (e) { document.getElementById("msgs").innerHTML = notice("bad", "Not saved:", e.errors); }
  };
  document.getElementById("s-save").onclick = async () => {
    try {
      META = await api("PUT", "/api/settings", { running_speed_km_h: document.getElementById("s-speed").value, calibrator_url: document.getElementById("s-cal").value });
      toast("Settings saved");
    } catch (e) { document.getElementById("msgs").innerHTML = notice("bad", "Not saved:", e.errors); }
  };
}

// ---------------------------------------------------------------------------------------------- file picker

const fmtSize = n => n > 1e9 ? (n / 1e9).toFixed(1) + " GB" : n > 1e6 ? (n / 1e6).toFixed(0) + " MB" : Math.max(1, Math.round(n / 1e3)) + " KB";

async function openPicker(input, title, opts = {}) {
  let dlg = document.getElementById("picker");
  if (!dlg) { dlg = document.createElement("dialog"); dlg.id = "picker"; document.body.appendChild(dlg); }
  let showAll = false;
  const load = async path => {
    try {
      render(await api("GET", `/api/browse?path=${encodeURIComponent(path || "")}&all=${showAll}&current=${encodeURIComponent(input.value)}` + (opts.game != null ? `&game=${opts.game}` : "")));
    } catch (e) { toast((e.errors || [e.message]).join(" "), true); }
  };
  const render = r => {
    if (opts.folder) r.entries = r.entries.filter(e => e.type === "dir");
    dlg.innerHTML = `<h2>${esc(title)}</h2>
      <div class="row" style="margin-bottom:8px">${r.shortcuts.map((s, i) => `<button type="button" class="link" data-sc="${i}">${esc(s.label)}</button>`).join("")}</div>
      <div class="row" style="flex-wrap:nowrap;margin-bottom:8px"><button type="button" id="pk-up" ${r.parent ? "" : "disabled"}>Up</button>
        <input id="pk-path" value="${esc(r.path)}"><button type="button" id="pk-go">Go</button></div>
      ${r.note ? notice("warn", "", [r.note]) : ""}
      <div class="pk-list">${r.entries.map((e, i) => `<div class="pk-row" data-i="${i}">
          <span class="pk-name">${e.type === "dir" ? "&#9656; " : ""}${esc(e.name)}</span><span class="muted small">${e.type === "dir" ? "" : fmtSize(e.size)}</span></div>`).join("")
        || '<div class="muted" style="padding:14px">${opts.folder ? "No folders here." : "No folders or video files here."}</div>'}</div>
      <div class="row spread" style="margin-top:12px"><label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--text)">${opts.folder ? "" : `<input type="checkbox" id="pk-all" ${showAll ? "checked" : ""}> Show all file types`}</label>
        <span>${opts.folder ? `<button type="button" class="primary" id="pk-use">Use this folder</button> ` : ""}<button type="button" id="pk-cancel">Cancel</button></span></div>`;
    const $ = id => dlg.querySelector("#" + id);
    dlg.querySelectorAll("[data-sc]").forEach(b => b.onclick = () => load(r.shortcuts[+b.dataset.sc].path));
    $("pk-up").onclick = () => load(r.parent);
    $("pk-go").onclick = () => load($("pk-path").value);
    $("pk-path").onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); load($("pk-path").value); } };
    if ($("pk-all")) $("pk-all").onchange = e => { showAll = e.target.checked; load(r.path); };
    if ($("pk-use")) $("pk-use").onclick = () => { input.value = r.path; input.dispatchEvent(new Event("input", { bubbles: true })); dlg.close(); };
    $("pk-cancel").onclick = () => dlg.close();
    dlg.querySelectorAll(".pk-row").forEach(row => row.onclick = async () => {
      const e = r.entries[+row.dataset.i];
      if (e.type === "dir") { load(e.path); return; }
      input.value = e.stored;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      try { await api("PUT", "/api/video-folder", { path: r.path }); } catch (err) { /* remembering the folder is optional */ }
      dlg.close();
    });
  };
  dlg.showModal();
  load("");
}
