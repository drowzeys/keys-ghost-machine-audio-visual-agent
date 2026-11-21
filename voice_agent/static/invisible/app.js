// keys Ghost Multiverse Studio, invisible mode: the Ghost perched above a timeline of everything the agents make.
// The first creation on the left, the newest on the right; forks (a retake, a second option, another agent's take
// on the same thing) branch into the lanes below. Search finds any creation by its words or the Ghost's description;
// a click opens it: the picture, clip or sound, the Ghost's summary and description, who made it, from what.

import * as ghost from "/multiverse/ghost.js";
import { COL, CARD_W, CARD_H, layout, rootOf, edges, edgePath, marks, cardLine, agentHue, ago } from "./timeline.js";

const $ = (id) => document.getElementById(id);
const all = new Map();          // id -> creation
let pos = {}, laid = null, view = { tx: 40, ty: 20, s: 1 }, follow = true;
let match = null;               // the ids the search found (null: no search)
let kind = "", agent = "", selected = null, status = null;
const cards = new Map();

// ---------------------------------------------------------------------------------------------------- the Ghost
let replyLine = "";
ghost.mount($("face"), {
  state: (s) => { document.body.dataset.ghost = s; },
  status: () => {},
  mic: (on) => { $("mic").classList.toggle("on", on); $("micLbl").textContent = on ? "Listening" : "Talk"; },
  meta: () => {},
  transcript: (t) => { replyLine = ""; say(`you · ${t}`, true); },
  reply: (t) => { replyLine += (replyLine ? " " : "") + t; say(replyLine); },
  event: (e) => {
    if (e.type === "video" && e.action === "watch") {
      replyLine = "";
      if (e.what === "find") { $("q").value = e.q || ""; setKind(e.kind || ""); }
      if (e.what === "agent") { $("q").value = ""; }
      show(e.ids || [], true);
    }
  },
});
$("mic").onclick = () => ghost.toggleMic();
function say(text, you = false) { const l = $("line"); l.textContent = text; l.classList.toggle("you", you); }

// ---------------------------------------------------------------------------------------------------- data
async function load() {
  const r = await fetch("/api/watch/timeline");
  const data = await r.json();
  for (const c of data.creations) all.set(c.id, c);
  stats(data.status);
  redraw();
  if (location.hash.length > 2 && all.has(location.hash.slice(1))) open(location.hash.slice(1));
  else fitNewest();
  listen();
}
function listen() {
  const src = new EventSource("/api/watch/events");
  src.onmessage = (m) => {
    let e; try { e = JSON.parse(m.data); } catch (_) { return; }
    if (e.status) stats(e.status);
    if (e.type === "put" && e.creation) {
      const fresh = !all.has(e.creation.id);
      all.set(e.creation.id, e.creation);
      if (fresh) { noticed(e.creation); }
      soon();
      if (selected === e.creation.id) detail(selected);
    }
  };
  src.onerror = () => { src.close(); setTimeout(listen, 3000); };
}
let soonT = null;
function soon() { clearTimeout(soonT); soonT = setTimeout(() => { redraw(); if (follow) fitNewest(true); }, 250); }
function noticed(c) {  // the Ghost looks up at anything new
  ghost.setEmotion("curious", 1800);
  if (ghost.current() === "idle") say(`${c.agent} made ${c.kind === "image" ? "a picture" : c.kind === "video" ? "a clip" : c.kind === "audio" ? "a sound" : "a note"}: ${cardLine(c)}`);
}
function stats(s) {
  if (!s) return;
  status = s;
  const agents = Object.keys(s.agents || {});
  const bits = [`watching · ${s.total} creation${s.total === 1 ? "" : "s"} · ${agents.length} agent${agents.length === 1 ? "" : "s"}`];
  if (s.describing) bits.push("describing…");
  else if (s.waiting) bits.push(`${s.waiting} to describe${s.busy ? " (after the renders)" : ""}`);
  $("stats").textContent = bits.join(" · ");
  const sel = $("agent"), have = new Set([...sel.options].map((o) => o.value));
  for (const a of agents) if (!have.has(a)) { const o = document.createElement("option"); o.value = a; o.textContent = a; sel.append(o); }
}

// ---------------------------------------------------------------------------------------------------- drawing
function visible() {
  return [...all.values()].filter((c) => (!kind || c.kind === kind) && (!agent || c.agent === agent));
}
function redraw() {
  const list = visible();
  laid = layout(list, (c) => rootOf(c.id, (id) => all.get(id)));
  pos = laid.pos;
  drawBands(laid.bands);
  stick();
  $("empty").hidden = all.size > 0;
  // edges
  const svg = $("edges");
  svg.setAttribute("width", laid.width + COL); svg.setAttribute("height", laid.height + 60);
  svg.innerHTML = edges(list, pos).map((e) => {
    const lit = match && match.has(e.from) && match.has(e.to), sel = selected && (e.from === selected || e.to === selected);
    return `<path class="edge${e.merge ? " merge" : ""}${lit ? " lit" : ""}${sel ? " sel" : ""}" d="${edgePath(pos[e.from], pos[e.to])}"/>`;
  }).join("");
  // ruler
  $("ruler").innerHTML = marks(list, pos).map((m) => `<span class="mark${m.strong ? " strong" : ""}" style="left:${m.x}px">${m.label}</span>`).join("");
  // cards
  const keep = new Set(list.map((c) => c.id));
  for (const [id, el] of cards) if (!keep.has(id)) { el.remove(); cards.delete(id); }
  for (const c of list) drawCard(c);
  apply();
}
// every project's band: a header with its name and what it is (the Ghost's words for the first creation)
function when(at) { return new Date(at * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }); }
// a project's final film: the narrated cut, else the cut, else the newest Multiverse export; else its latest clip
function finalFilm(key) {
  let best = null, rank = -1;
  for (const c of all.values()) {
    if (c.kind !== "video" || !c.thumb || rootOf(c.id, (id) => all.get(id)) !== key) continue;
    const r = /:final$/.test(c.ref) ? 3 : /:film$/.test(c.ref) ? 2 : /:film:/.test(c.ref) ? 1 : 0;
    if (r > rank || (r === rank && c.at > best.at)) { best = c; rank = r; }
  }
  return best && { c: best, final: rank > 0 };
}
function drawBands(bands) {
  const box = $("bands");
  box.replaceChildren(...bands.map((b) => {
    const root = all.get(b.key) || {}, members = b.members.map((id) => all.get(id)).filter(Boolean);
    const agents = [...new Set(members.map((m) => m.agent))];
    const title = (root.title || cardLine(root) || "a project").trim();
    const about = [root.summary, root.description].map((x) => (x || "").trim()).find((x) => x && x.toLowerCase() !== title.toLowerCase()) || "";
    const first = Math.min(...members.map((m) => m.at)), last = Math.max(...members.map((m) => m.at));
    const film = finalFilm(b.key);
    const el = document.createElement("div");
    el.className = "band"; el.dataset.key = b.key;
    if (match) el.classList.toggle("miss", !b.members.some((id) => match.has(id)));
    el.style.cssText = `left:${b.x - 14}px; top:${b.y}px; width:${b.w + 28}px; height:${b.h + 14}px`;
    const head = document.createElement("button");
    head.type = "button"; head.className = "head";
    head.title = [title, about, root.description && root.description !== about ? root.description : ""].filter(Boolean).join("\n\n");
    head.innerHTML = `<b></b><span class="about"></span><em></em>`;
    if (film) {  // the project's final film (or its latest clip while there is none yet): hover plays it, click opens it
      const f = document.createElement("span");
      f.className = "film" + (film.final ? "" : " sofar");
      f.innerHTML = `<img alt="" loading="lazy" src="${film.c.thumb}"><i>${film.final ? "▶ film" : "so far"}</i>`;
      f.title = `${film.final ? "The final film" : "No film yet: the latest clip"}: ${cardLine(film.c)}`;
      f.onmouseenter = () => { if (!film.c.media || f.querySelector("video")) return; const v = document.createElement("video"); Object.assign(v, { src: film.c.media, muted: true, loop: true, playsInline: true, autoplay: true }); f.prepend(v); };
      f.onmouseleave = () => f.querySelector("video")?.remove();
      f.onclick = (e) => { e.stopPropagation(); open(film.c.id, false); };
      head.prepend(f); head.classList.add("has-film");
    }
    head.querySelector("b").textContent = title;
    head.querySelector(".about").textContent = about;
    head.querySelector("em").textContent = `${agents.join(" + ")} · ${b.count} creation${b.count === 1 ? "" : "s"} · ${when(first)}${last - first >= 60 ? " – " + when(last) : ""}`;
    head.onclick = (e) => { e.stopPropagation(); if (all.has(b.key)) open(b.key, false); };
    head.style.setProperty("--hue", agentHue(root.agent || agents[0]));
    el.append(head);
    return el;
  }));
}
function drawCard(c) {
  let el = cards.get(c.id);
  if (!el) {
    el = document.createElement("button");
    el.type = "button"; el.className = "card";
    el.innerHTML = `<div class="media"></div><div class="who"><i></i><span></span></div><div class="sum"></div>`;
    el.onclick = (e) => { e.stopPropagation(); open(c.id, false); };
    el.onmouseenter = () => hoverPlay(el, true);
    el.onmouseleave = () => hoverPlay(el, false);
    $("cards").append(el); cards.set(c.id, el);
  }
  const p = pos[c.id];
  el.style.transform = `translate(${p.x}px, ${p.y}px)`;
  el.dataset.id = c.id; el.dataset.kind = c.kind;
  el.classList.toggle("new", c.status === "new");
  el.classList.toggle("hit", !!(match && match.has(c.id)));
  el.classList.toggle("miss", !!(match && !match.has(c.id)));
  el.classList.toggle("selected", c.id === selected);
  const key = `${c.thumb}|${c.kind}|${c.prompt?.length}`;
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    const m = el.querySelector(".media");
    if (c.thumb) m.innerHTML = `<img loading="lazy" alt="" src="${c.thumb}">` + (c.kind === "video" ? `<b class="play">▶</b>` : "");
    else if (c.kind === "audio") m.innerHTML = `<svg viewBox="0 0 120 40" class="wave" aria-hidden="true">${wave(c.id)}</svg>`;
    else { m.innerHTML = ""; const q = document.createElement("q"); q.textContent = (c.prompt || c.title || "").slice(0, 140); m.append(q); }
  }
  const who = el.querySelector(".who");
  who.querySelector("i").style.background = `hsl(${agentHue(c.agent)} 80% 62%)`;
  who.querySelector("span").textContent = c.agent;
  el.querySelector(".sum").textContent = cardLine(c);
  el.title = `${cardLine(c)}\n${c.agent} · ${new Date(c.at * 1000).toLocaleString()}`;
}
function wave(seed) {  // a sound's card: a little waveform, the same for the same sound
  let h = 0, out = "";
  for (const ch of seed) h = (h * 33 + ch.charCodeAt(0)) >>> 0;
  for (let i = 0; i < 30; i++) { h = (h * 1103515245 + 12345) >>> 0; const a = 4 + (h % 30); out += `<rect x="${i * 4}" y="${20 - a / 2}" width="2.4" height="${a}" rx="1"/>`; }
  return out;
}
function hoverPlay(el, on) {
  const c = all.get(el.dataset.id); if (!c || c.kind !== "video" || !c.media) return;
  const m = el.querySelector(".media");
  let v = m.querySelector("video");
  if (on && !v) { v = document.createElement("video"); v.src = c.media; v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true; m.append(v); }
  else if (!on && v) v.remove();
}

// ---------------------------------------------------------------------------------------------------- the view
const vp = $("viewport"), world = $("world");
function apply() {
  world.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.s})`;
  world.style.setProperty("--z", view.s.toFixed(3)); world.classList.toggle("far", view.s < 0.6);
  stick();
}
const HEADER_H = 102;   // the header's height (invisible.css)
function stick() {  // a project's header stays in view (top and left edges) while the rest of its band still is
  if (!laid) return;
  const top = 30, { w } = size(), s = view.s;   // top: the bands' offset inside the world
  for (const el of $("bands").children) {
    const b = laid.bands.find((x) => x.key === el.dataset.key), head = el.firstChild;
    if (!b || !head) continue;
    const dy = Math.max(0, Math.min(b.h - CARD_H - HEADER_H, (8 - (view.ty + (top + b.y) * s)) / s));
    const full = b.w + 28, dx = Math.max(0, Math.min(full - 240, (8 - (view.tx + (b.x - 14) * s)) / s));
    head.style.transform = dy || dx ? `translate(${dx}px, ${dy}px)` : "";
    head.style.width = `${Math.max(200, Math.min(full - dx, (w - 16) / s))}px`;
    head.classList.toggle("stuck", dy > 0);
  }
}
function size() { return { w: vp.clientWidth, h: vp.clientHeight }; }
function unfollow() { if (follow) { follow = false; $("follow").classList.remove("on"); } }
function fitNewest(smooth) {  // the right end of the timeline in view: the newest creations
  if (!laid) return;
  const { w, h } = size();
  const s = view.s;
  // across the newest card's lane too: with many forks it can sit far below the first lanes
  let last = null;
  for (const [id, p] of Object.entries(pos)) if (!last || p.x > last.x || (p.x === last.x && (all.get(id)?.at || 0) > (all.get(last.id)?.at || 0))) last = { id, ...p };
  let want = last ? h * 0.45 - (last.y + CARD_H / 2) * s : view.ty;
  const band = last && laid.bands.find((b) => b.members.includes(last.id));
  if (band) {  // the whole project in view when it fits; else its header sticks to the top
    const head = 16 - band.y * s;
    if (head + (last.y + CARD_H) * s <= h - 90) want = head;
  }
  const tx = Math.min(40, w - 60 - laid.width * s), ty = Math.min(40, Math.max(-laid.height * s + h - 80, want));
  if (smooth) animateTo({ s, tx, ty }); else { view = { s, tx, ty: Math.min(ty, 40) }; apply(); }
}
function fit() {
  if (!laid) return;
  const { w, h } = size();
  const s = Math.min(1.1, Math.max(0.12, Math.min((w - 80) / (laid.width + 40), (h - 80) / (laid.height + 60))));
  animateTo({ s, tx: (w - laid.width * s) / 2, ty: 30 }); unfollow();
}
function focusOn(id) {
  const p = pos[id]; if (!p) return;
  const { w, h } = size(), s = Math.max(view.s, 0.8);
  animateTo({ s, tx: w * 0.4 - (p.x + CARD_W / 2) * s, ty: h * 0.42 - (p.y + CARD_H / 2) * s });
}
let anim = null;
function animateTo(to) {
  const from = { ...view }, t0 = performance.now();
  cancelAnimationFrame(anim);
  const step = (now) => {
    const t = Math.min(1, (now - t0) / 450), k = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    view = { s: from.s + (to.s - from.s) * k, tx: from.tx + (to.tx - from.tx) * k, ty: from.ty + (to.ty - from.ty) * k };
    apply(); if (t < 1) anim = requestAnimationFrame(step);
  };
  anim = requestAnimationFrame(step);
}
function zoomAt(k, cx, cy) {
  const { w, h } = size(); cx = cx ?? w / 2; cy = cy ?? h / 2;
  const s = Math.min(2.4, Math.max(0.12, view.s * k));
  view.tx = cx - (cx - view.tx) * (s / view.s); view.ty = cy - (cy - view.ty) * (s / view.s); view.s = s; apply();
}
let drag = null;
vp.addEventListener("pointerdown", (e) => { if (e.target.closest(".card")) return; drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }; vp.setPointerCapture(e.pointerId); vp.classList.add("dragging"); });
vp.addEventListener("pointermove", (e) => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) unfollow(); view.tx = drag.tx + dx; view.ty = drag.ty + dy; apply(); });
vp.addEventListener("pointerup", () => { drag = null; vp.classList.remove("dragging"); });
vp.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) { zoomAt(Math.exp(-e.deltaY * 0.0022), e.clientX, e.clientY - vp.getBoundingClientRect().top); return; }
  unfollow();
  if (e.deltaX === 0 && !e.shiftKey) view.tx -= e.deltaY;  // a mouse wheel scrolls through time
  else { view.tx -= e.deltaX; view.ty -= e.deltaY; }
  apply();
}, { passive: false });
new ResizeObserver(() => apply()).observe(vp);
$("zoomIn").onclick = () => zoomAt(1.2);
$("zoomOut").onclick = () => zoomAt(1 / 1.2);
$("fit").onclick = fit;
$("follow").onclick = () => { follow = true; $("follow").classList.add("on"); fitNewest(true); };

// ---------------------------------------------------------------------------------------------------- search
let qT = null;
$("q").addEventListener("input", () => { clearTimeout(qT); qT = setTimeout(search, 180); });
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { const first = $("results").querySelector("button"); if (first) first.click(); }
  if (e.key === "Escape") { $("q").value = ""; search(); }
});
document.addEventListener("click", (e) => { if (!e.target.closest("#searchBox")) $("results").hidden = true; });
$("q").addEventListener("focus", () => { if ($("results").children.length) $("results").hidden = false; });
async function search() {
  const q = $("q").value.trim();
  if (!q) { match = null; $("results").hidden = true; $("results").innerHTML = ""; redraw(); return; }
  const url = `/api/watch/creations?q=${encodeURIComponent(q)}&kind=${kind}&agent=${encodeURIComponent(agent)}&limit=60`;
  const data = await (await fetch(url)).json();
  show(data.creations.map((c) => c.id), false);
}
function show(ids, fromVoice) {
  match = new Set(ids);
  const box = $("results"); box.innerHTML = "";
  if (!ids.length) { box.innerHTML = `<p class="none">Nothing found.</p>`; }
  for (const id of ids.slice(0, 10)) {
    const c = all.get(id); if (!c) continue;
    const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "option");
    b.innerHTML = `${c.thumb ? `<img alt="" src="${c.thumb}">` : `<span class="k">${c.kind}</span>`}<span class="t"></span><span class="a"></span>`;
    b.querySelector(".t").textContent = cardLine(c);
    b.querySelector(".a").textContent = `${c.agent} · ${ago(c.at)}`;
    b.onclick = () => { $("results").hidden = true; open(id); };
    box.append(b);
  }
  if (ids.length > 10) { const p = document.createElement("p"); p.className = "none"; p.textContent = `…and ${ids.length - 10} more, lit on the timeline`; box.append(p); }
  box.hidden = false;
  redraw();
  if (ids.length) { unfollow(); focusOn(ids[0]); if (fromVoice) open(ids[0]); }
}
function setKind(k) {
  kind = k;
  document.querySelectorAll(".kind").forEach((b) => b.classList.toggle("on", b.dataset.kind === k));
}
document.querySelectorAll(".kind").forEach((b) => b.onclick = () => { setKind(b.dataset.kind); redraw(); if ($("q").value.trim()) search(); });
$("agent").onchange = () => { agent = $("agent").value; redraw(); if ($("q").value.trim()) search(); };

// ---------------------------------------------------------------------------------------------------- detail
async function open(id, move = true) {
  selected = id; history.replaceState(null, "", `#${id}`);
  redraw();
  if (move) { unfollow(); focusOn(id); }
  await detail(id);
}
async function detail(id) {
  const r = await fetch(`/api/watch/creations/${id}`);
  if (!r.ok) return;
  const c = await r.json();
  const d = $("detail"); d.hidden = false; d.innerHTML = "";
  const close = el("button", "close", "×"); close.onclick = () => { selected = null; d.hidden = true; history.replaceState(null, "", location.pathname); redraw(); };
  d.append(close);
  const media = el("div", "visual");
  if (c.kind === "image" && c.media) { const i = el("img"); i.src = c.media; i.alt = c.summary || c.title; media.append(i); }
  else if (c.kind === "video" && c.media) { const v = el("video"); v.src = c.media; v.controls = true; v.autoplay = true; v.muted = true; v.loop = true; v.playsInline = true; if (c.thumb) v.poster = c.thumb; media.append(v); }
  else if (c.kind === "audio" && c.media) { const a = el("audio"); a.src = c.media; a.controls = true; media.append(a); }
  else media.append(el("blockquote", "", c.prompt || c.title || ""));
  d.append(media);
  d.append(el("h3", "", c.summary || c.title || c.kind));
  const by = el("p", "by");
  const dot = el("i"); dot.style.background = `hsl(${agentHue(c.agent)} 80% 62%)`;
  by.append(dot, ` ${c.agent} · ${c.kind} · ${new Date(c.at * 1000).toLocaleString()} (${ago(c.at)})`);
  d.append(by);
  if (c.status === "new") d.append(el("p", "pending", "The Ghost hasn't looked at this one yet; it describes them when no render is running."));
  if (c.description) d.append(el("p", "desc", c.description));
  if (c.tags && c.tags.length) { const t = el("p", "tags"); for (const g of c.tags) { const b = el("button", "", g); b.onclick = () => { $("q").value = g; search(); }; t.append(b); } d.append(t); }
  if (c.title && c.title !== c.summary) { d.append(el("div", "k", "title")); d.append(el("p", "", c.title)); }
  if (c.prompt && c.kind !== "text") { d.append(el("div", "k", "prompt")); d.append(el("p", "prompt", c.prompt)); }
  const chips = (label, list) => {
    if (!list || !list.length) return;
    d.append(el("div", "k", label));
    const row = el("p", "chips");
    for (const x of list) { const b = el("button", "", cardLine(x)); b.title = `${x.agent} · ${x.kind}`; b.onclick = () => open(x.id); row.append(b); }
    d.append(row);
  };
  chips("made from", c.parent_list);
  chips(c.child_list && c.child_list.length > 1 ? `forks (${c.child_list.length})` : "led to", c.child_list);
  const acts = el("p", "acts");
  if (c.meta && c.meta.open) { const a = el("a", "", "Open in Multiverse ↗"); a.href = c.meta.open; acts.append(a); }
  if (c.media) { const a = el("a", "", "Open the file ↗"); a.href = c.media; a.target = "_blank"; acts.append(a); }
  d.append(acts);
  const meta = el("p", "meta", `${c.source} · ${c.ref}${c.meta && c.meta.described_by ? " · described by " + c.meta.described_by : ""}`);
  d.append(meta);
}
function el(tag, cls = "", text = "") { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; return e; }
document.addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, select")) return;
  if (e.key === "/") { e.preventDefault(); $("q").focus(); }
  else if (e.key === "Escape") { $("detail").hidden = true; selected = null; redraw(); }
  else if (e.key === "0") fit();
  else if (e.key === "n" || e.key === "N") $("follow").click();
});

// ---------------------------------------------------------------------------------------------------- credits
$("credits").addEventListener("toggle", async () => {
  if (!$("credits").open || $("credits").dataset.loaded) return;
  try {
    const list = await (await fetch("/api/credits")).json(), box = $("clines"); box.textContent = "";
    for (const c of list) { const d = document.createElement("div"), b = document.createElement("b"); b.textContent = c.name; d.append(b, ` by ${c.by}: ${c.what} (${c.licence})`); box.append(d); }
    $("credits").dataset.loaded = "1";
  } catch (_) {}
});

load();
