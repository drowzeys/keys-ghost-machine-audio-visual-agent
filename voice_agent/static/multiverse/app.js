// keys Ghost Multiverse Studio: the page. The Ghost first (its face, its voice); then the film as a tree, past to the
// left and futures to the right, the Ghost standing on its node with the voice docked to it, the walked path lit and
// the roads not taken dimmed. Plain modules, no build step.

import * as ghost from "./ghost.js";
import { CARD_W, CARD_H, COL_W, layout, nodeClasses, edgeClass, edgePath, mediaFor, fileUrl, metaLine, stateOf,
         centreOn, lerpPoint, childrenOf } from "./tree.js";

const $ = (id) => document.getElementById(id);
const api = (path) => "/api/multiverse" + path;
let pid = null, tree = null, pos = {}, source = null;
let view = { tx: 80, ty: 120, s: 1 }, follow = true, mixFirst = null, selected = null;
let ghostAt = null, ghostFrom = null, ghostTo = null, ghostT0 = 0, ghostId = null;
const GHOST_MS = 1100;
const cards = new Map();

// ---------------------------------------------------------------------------------------------------- the voice
let ghostLine = null;
function lineOut(cls, text) {
  const p = document.createElement("p"); p.className = cls; p.textContent = text; $("lines").append(p);
  while ($("lines").children.length > 30) $("lines").firstChild.remove();
  $("lines").scrollTop = 1e9; return p;
}
ghost.mount($("face"), {
  state: (s, label) => { $("chip").textContent = label; $("dock").dataset.state = s; },
  status: (t) => { $("statusText").textContent = t; },
  mic: (on) => { $("mic").classList.toggle("on", on); $("micLbl").textContent = on ? "Listening" : "Talk"; },
  meta: (t) => lineOut("meta", t),
  transcript: (t) => { lineOut("you", t); ghostLine = null; },
  reply: (t) => { if (!ghostLine) ghostLine = lineOut("ghost", ""); ghostLine.textContent += (ghostLine.textContent ? " " : "") + t; },
  event: (e) => {
    if (e.type === "video" && e.action === "look") { ghostLine = null; lookUpdate(e, e.what === "ask"); }
    if (e.type === "video" && e.action === "multiverse") {
      ghostLine = null;
      if (e.project && e.project !== pid) open(e.project);
      else if (e.what === "export") showFilm();
    }
  },
});
$("mic").onclick = () => ghost.toggleMic();
$("say").onkeydown = (e) => { if (e.key === "Enter" && $("say").value.trim()) { ghostLine = null; ghost.sayText($("say").value); $("say").value = ""; } };

function toast(text, ms = 2600) {
  $("toast").textContent = text; $("toast").classList.add("show");
  clearTimeout(toast.t); toast.t = setTimeout(() => $("toast").classList.remove("show"), ms);
}

async function post(path, body) {
  const r = await fetch(api(path), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// ---------------------------------------------------------------------------------------------------- landing
async function landing() {
  document.body.classList.add("landing");
  pid = null; tree = null; if (source) { source.close(); source = null; }
  history.replaceState(null, "", location.pathname);
  $("filmTitle").textContent = "";
  try {
    const { projects } = await (await fetch(api("/projects"))).json();
    const box = $("recent"); box.textContent = "";
    for (const p of projects.slice(0, 6)) {
      const b = document.createElement("button"); b.type = "button"; b.title = p.idea;
      if (p.poster) { const img = document.createElement("img"); img.loading = "lazy"; img.alt = ""; img.src = api(`/projects/${p.id}/files/${p.poster}`); b.append(img); }
      const s = document.createElement("span"); s.textContent = `${p.title} · ${p.nodes} clips`; b.append(s);
      b.onclick = () => open(p.id); box.append(b);
    }
  } catch (_) {}
}
setInterval(() => { if (!pid) fetch(api("/projects")).catch(() => {}); }, 15000);  // the voice knows the page is open
$("ideaForm").onsubmit = async (e) => {
  e.preventDefault();
  const idea = $("idea").value.trim(); if (!idea) return;
  try { const t = await post("/projects", { idea }); $("idea").value = ""; open(t.id, t); }
  catch (err) { toast(err.message); }
};
$("newFilm").onclick = () => landing();

// ---------------------------------------------------------------------------------------------------- a film
async function open(id, first) {
  pid = id; ghostId = null; ghostAt = null; follow = true; selected = null; mixFirst = null;
  document.body.classList.remove("mixing"); $("modebar").hidden = true; $("othersPop").hidden = true;
  for (const c of cards.values()) c.el.remove(); cards.clear();
  document.body.classList.remove("landing");
  history.replaceState(null, "", "#" + id);
  $("follow").classList.add("on"); $("detail").hidden = true; $("film").hidden = true;
  if (first) update(first);
  else try { update(await (await fetch(api("/projects/" + id))).json()); } catch (_) { toast("That film is gone."); landing(); return; }
  listen();
}

function listen() {
  if (source) source.close();
  source = new EventSource(api(`/projects/${pid}/events`));
  source.onmessage = (m) => { try { update(JSON.parse(m.data)); } catch (_) {} };
  source.onerror = () => { setTimeout(() => { if (pid && source && source.readyState === 2) listen(); }, 3000); };
}

function update(next) {
  if (!next || !next.nodes || (pid && next.id !== pid)) return;
  tree = next;
  $("filmTitle").textContent = tree.title;
  const laid = layout(tree);
  pos = laid.pos;
  drawEdges(laid);
  drawNodes();
  sampleEdges();
  drawWriting();
  drawPath(); drawHint(); if (mixFirst && mixFirst !== "?" && stateOf(tree, mixFirst) !== "option") endMix();
  if (tree.ghost !== ghostId) moveGhost(tree.ghost);
  $("film").hidden = !tree.film;
  if (tree.film) $("film").href = filmUrl();
  if (selected && tree.nodes[selected]) detail(selected);
}

function drawEdges(laid) {
  const svg = $("edges");
  svg.setAttribute("width", laid.width + COL_W); svg.setAttribute("height", laid.height + 200);
  const parts = [];
  // dimmed first, then the options, then the lit path on top
  const order = { "edge dimmed": 0, "edge option": 1, "edge active": 2 };
  const edges = [];
  for (const n of Object.values(tree.nodes)) if (n.parent && pos[n.parent] && pos[n.id]) edges.push([edgeClass(tree, n.parent, n.id), n.parent, n.id]);
  edges.sort((a, b) => order[a[0]] - order[b[0]]);
  for (const [cls, a, b] of edges) parts.push(`<path class="${cls}" d="${edgePath(pos[a], pos[b])}"/>`);
  svg.innerHTML = parts.join("");
}

function drawWriting() {
  document.querySelectorAll(".writing").forEach((w) => w.remove());
  const id = tree.writing; if (!id || !pos[id]) return;
  const p = pos[id], n = tree.nodes[id], kids = childrenOf(tree, id);
  const w = document.createElement("div"); w.className = "writing";
  if (n.quality !== "full") { w.textContent = "The Ghost is writing the opening"; w.style.left = p.x + "px"; w.style.top = p.y + CARD_H + 18 + "px"; }
  else {
    w.textContent = kids.length ? "The Ghost is writing other paths" : "The Ghost is writing the paths";
    const below = kids.length ? Math.max(...kids.map((k) => (pos[k.id] || p).y)) + CARD_H + 18 : p.y + 40;
    w.style.left = p.x + COL_W + "px"; w.style.top = below + "px";
  }
  $("nodes").append(w);
}

// ----- a card -----
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      if (v.dataset.play === "1") v.play().catch(() => {});
    } else if (!v.paused) v.pause();
  }
}, { root: $("viewport"), rootMargin: "200px" });

function cardFor(id) {
  let c = cards.get(id);
  if (c) return c;
  const el = document.createElement("div");
  el.innerHTML = '<div class="media"></div><span class="letter"></span><span class="q"></span><div class="label"></div><div class="meta"></div><button class="backhere" type="button" hidden>↩ Back here</button>';
  el.querySelector(".backhere").onclick = (e) => { e.stopPropagation(); goto(id); };
  el.querySelector(".letter").style.display = "none";
  el.onclick = (e) => { e.stopPropagation(); clicked(id); };
  el.onmouseenter = () => { const v = el.querySelector("video"); if (v && v.dataset.play !== "1") { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); } };
  el.onmouseleave = () => { const v = el.querySelector("video"); if (v && v.dataset.play !== "1") v.pause(); };
  $("nodes").append(el);
  c = { el, media: "" };
  cards.set(id, c);
  return c;
}

function drawNodes() {
  for (const [id, c] of cards) if (!tree.nodes[id]) { c.el.remove(); cards.delete(id); }
  for (const id of Object.keys(tree.nodes)) {
    const n = tree.nodes[id], p = pos[id]; if (!p) continue;
    const c = cardFor(id), el = c.el;
    el.className = nodeClasses(tree, id, tree.working).join(" ") + (selected === id ? " selected" : "") + (mixFirst === id ? " picking" : "");
    el.style.left = p.x + "px"; el.style.top = p.y + "px";
    el.dataset.id = id; el.dataset.state = stateOf(tree, id);
    const letter = el.querySelector(".letter");
    letter.style.display = n.letter ? "" : "none"; letter.textContent = n.letter || "";
    const q = el.querySelector(".q");
    q.textContent = n.quality === "full" ? "full" : n.quality === "draft" ? (n.files.draft ? "draft" : "keyframe") : "";
    q.className = "q " + (n.quality || ""); q.style.display = n.quality ? "" : "none";
    el.querySelector(".label").textContent = n.label || "…";
    el.querySelector(".backhere").hidden = !(n.taken && id !== tree.ghost);
    el.querySelector(".meta").textContent = n.status === "failed" ? "failed: " + (n.error || "") : metaLine(tree, id, tree.estimates);
    el.title = [n.label, n.beat, n.mood && "mood: " + n.mood, n.camera && "camera: " + n.camera, n.error].filter(Boolean).join("\n");
    // the media: rebuilt only when what it shows changes
    const m = mediaFor(tree, id), key = `${m.kind}:${m.file || ""}:${m.poster || ""}`;
    const box = el.querySelector(".media");
    if (c.media !== key) {
      c.media = key;
      const old = box.querySelector("video"); if (old) { io.unobserve(old); old.removeAttribute("src"); old.load(); }
      box.textContent = "";
      if (m.kind === "video") {
        const v = document.createElement("video");
        v.muted = true; v.loop = true; v.playsInline = true; v.preload = "none";
        v.dataset.src = fileUrl(pid, id, m.file);
        if (m.poster) v.poster = fileUrl(pid, id, m.poster);
        box.append(v); io.observe(v);
      } else if (m.kind === "image") {
        const img = document.createElement("img"); img.loading = "lazy"; img.alt = ""; img.src = fileUrl(pid, id, m.file); box.append(img);
      } else { const d = document.createElement("div"); d.className = "none"; box.append(d); }
      const bar = document.createElement("div"); bar.className = "bar"; box.append(bar);
    }
    const v = box.querySelector("video");
    if (v) {  // live previews ahead of the Ghost; posters (paused) everywhere else
      const play = m.play ? "1" : "0";
      if (v.dataset.play !== play) { v.dataset.play = play; if (play === "1") { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); } else v.pause(); }
    }
    const pr = n.progress, bar = box.querySelector(".bar");
    if (bar) bar.style.width = pr && pr.of ? Math.round(100 * (pr.at || 0) / pr.of) + "%" : "0";
  }
}

// ----- clicks: an option ahead is taken; a walked node is a place to go back to; anything else shows its detail -----
async function clicked(id) {
  const n = tree.nodes[id], st = stateOf(tree, id);
  if (mixFirst) return mixPick(id);
  if (st === "option") return choose(id);
  detail(id);
}

async function choose(id) {
  const n = tree.nodes[id];
  try { const r = await post(`/projects/${pid}/choose`, { node: id }); toast(`Path ${n.letter}: ${n.label}`); update(r.tree); }
  catch (err) { toast(err.message); }
}

async function goto(id) {
  try { const r = await post(`/projects/${pid}/goto`, { node: id }); toast("The Ghost walks back. Branch from here."); update(r.tree); }
  catch (err) { toast(err.message); }
}

function detail(id) {
  selected = id;
  const n = tree.nodes[id], st = stateOf(tree, id), box = $("detail");
  box.hidden = false; box.textContent = "";
  const close = document.createElement("button"); close.className = "close"; close.textContent = "×"; close.setAttribute("aria-label", "Close");
  close.onclick = () => { selected = null; box.hidden = true; drawNodes(); };
  const h = document.createElement("h3"); h.textContent = (n.letter ? n.letter + " · " : "") + (n.label || "");
  box.append(close, h);
  const field = (k, v) => { if (!v) return; const a = document.createElement("div"); a.className = "k"; a.textContent = k; const b = document.createElement("p"); b.textContent = v; box.append(a, b); };
  field("where", { ghost: "the Ghost is here", active: "on the lit path", option: "a path ahead", dimmed: n.taken ? "walked before, off the lit path" : "a road not taken" }[st]);
  field("beat", n.beat); field("mood", n.mood); field("camera", n.camera); field("prompt", n.prompt);
  field("cost", metaLine(tree, id, tree.estimates));
  const seam = (n.qc || {}).seam_full || (n.qc || {}).seam_draft;
  if (seam && seam.corr != null) field("the cut", `${seam.ok ? "holds" : "jumps"} (layout ${seam.corr}, brightness gap ${seam.luma_gap})`);
  if (n.error) field("error", n.error);
  const acts = document.createElement("div"); acts.className = "acts";
  const act = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn; acts.append(b); };
  if (!n.taken && n.parent) act(st === "option" ? "Take this path" : "Take this road instead", () => choose(id));
  if (n.taken && id !== tree.ghost) act("Go back here", () => goto(id));
  const queued = (tree.queue || []).some((q) => q.node === id) || (tree.working && tree.working.node === id);
  if (n.status === "planned" && n.quality == null && !queued)
    act("Draft it", async () => { try { update((await post(`/projects/${pid}/retry`, { node: id })).tree); } catch (e) { toast(e.message); } });
  if (n.status === "failed") act("Try again", async () => { try { update((await post(`/projects/${pid}/retry`, { node: id })).tree); } catch (e) { toast(e.message); } });
  const f = n.files || {};
  if (f.clip || f.draft) { const a = document.createElement("a"); a.href = fileUrl(pid, id, f.clip || f.draft); a.target = "_blank"; a.textContent = "Open clip"; acts.append(a); }
  box.append(acts);
  drawNodes();
}

// ----- toolbar -----
$("others").onclick = async () => { try { await post(`/projects/${pid}/options`, { fresh: true }); toast("Other directions from here…"); } catch (e) { toast(e.message); } };
$("mixBtn").onclick = () => (mixFirst ? endMix() : startMix());
function aheadIds() { return (tree && tree.ahead) || Object.keys((tree && tree.nodes) || {}).filter((i) => stateOf(tree, i) === "option"); }
function startMix() {
  closeOthers();
  if (aheadIds().length < 2) { toast("Mixing needs two paths ahead of the Ghost"); return; }
  mixFirst = "?"; $("mixBtn").classList.add("on"); document.body.classList.add("mixing"); modebar(); drawHint(); drawNodes();
}
function endMix() { mixFirst = null; $("mixBtn").classList.remove("on"); document.body.classList.remove("mixing"); modebar(); if (tree) { drawHint(); drawNodes(); } }
function modebar() {
  const bar = $("modebar");
  if (!mixFirst) { bar.hidden = true; bar.textContent = ""; return; }
  bar.hidden = false; bar.textContent = "";
  const t = document.createElement("span");
  if (mixFirst === "?") t.innerHTML = "<b>⊕ Mix two</b> · click the <b>first</b> path ahead (or press its letter)";
  else { const n = tree.nodes[mixFirst]; t.innerHTML = `<b>⊕ Mix two</b> · ${n.letter} picked — now click the <b>second</b> path to mix with it`; }
  const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancel"; cancel.onclick = endMix;
  bar.append(t, cancel);
}
async function mixPick(id) {
  if (stateOf(tree, id) !== "option") { toast("Mix two of the paths ahead of the Ghost (dashed outline)"); return; }
  if (mixFirst === "?") { mixFirst = id; modebar(); drawNodes(); return; }
  const a = mixFirst;
  if (a === id) { mixFirst = "?"; modebar(); drawNodes(); return; }
  endMix();
  try { const r = await post(`/projects/${pid}/mix`, { a, b: id }); toast(`Mixing ${tree.nodes[a].letter} with ${tree.nodes[id].letter}: a new path is being written`, 4000); update(r.tree); }
  catch (e) { toast(e.message); }
}
// new options: a small panel says what it does and takes an optional steer
$("others").onclick = () => { if (!$("othersPop").hidden) return closeOthers(); endMix(); $("othersPop").hidden = false; $("steer").focus(); };
function closeOthers() { $("othersPop").hidden = true; }
$("othersCancel").onclick = closeOthers;
$("steer").addEventListener("keydown", (e) => { if (e.key === "Enter") $("othersGo").click(); if (e.key === "Escape") closeOthers(); });
$("othersGo").onclick = async () => {
  const note = $("steer").value.trim(); closeOthers();
  try { await post(`/projects/${pid}/options`, { fresh: true, note }); toast(note ? `New paths, steered: “${note}”` : "Writing 3 new paths from here…", 4000); $("steer").value = ""; }
  catch (e) { toast(e.message); }
};
// back a step: the Ghost walks to the node before it
function backStep() {
  const g = tree && tree.nodes[tree.ghost];
  if (!g || !g.parent) { toast("The Ghost is at the opening: nothing before it"); return; }
  goto(g.parent);
}
$("back").onclick = backStep;
// your path, as steps you can click to go back
function drawPath() {
  const bar = $("pathbar"); bar.textContent = "";
  if (!tree) return;
  const path = tree.active || [];
  const lbl = document.createElement("span"); lbl.className = "lbl"; lbl.textContent = "your path ›"; bar.append(lbl);
  const upto = path.indexOf(tree.ghost);
  path.slice(0, upto < 0 ? path.length : upto + 1).forEach((id, i) => {
    if (i) { const s = document.createElement("span"); s.className = "sep"; s.textContent = "›"; bar.append(s); }
    const n = tree.nodes[id], b = document.createElement("button"); b.type = "button";
    b.textContent = `${i + 1} · ${n.label || "…"}`;
    if (id === tree.ghost) { b.className = "here"; b.title = "The Ghost is here"; }
    else { b.title = "Go back to this step (the path after it stays, dimmed)"; b.onclick = () => goto(id); }
    bar.append(b);
  });
  bar.scrollLeft = bar.scrollWidth;
}
function drawHint() {
  const h = $("hint"), ahead = aheadIds(), g = tree && tree.nodes[tree.ghost];
  $("back").disabled = !g || !g.parent;
  $("mixBtn").disabled = ahead.length < 2 && !mixFirst;
  if (mixFirst) { h.textContent = ""; return; }
  if (tree && tree.writing) h.textContent = "The Director is writing the paths ahead…";
  else if (ahead.length) h.textContent = `Pick a path: click ${ahead.map((i) => tree.nodes[i].letter).filter(Boolean).join(" / ")} (or press the letter, or say “take B”)`;
  else h.textContent = "";
}
$("export").onclick = async () => {
  toast("Joining the lit path…", 6000);
  try { const r = await post(`/projects/${pid}/export`); update(r.tree); showFilm(); } catch (e) { toast(e.message); }
};
// The film's look (gma/look.py): Natural, HDR, Neon or Sonar, by a chip here, by voice or typed to the Ghost. The download is
// the film's copy in the look once that is made (a minute or more after an export): the real HDR copy on a screen that
// shows HDR, else the SDR copy.
const HDR_SCREEN = !!(window.matchMedia && matchMedia("(dynamic-range: high)").matches) && !!document.createElement("video").canPlayType('video/mp4; codecs="hvc1.2.4.L120.B0"');
let look = { look: "natural", stamp: 0, looks: [] };
function filmUrl() {
  return api(`/projects/${pid}/film.mp4?at=${Math.round(tree.film.at)}`) + (look.look === "natural" ? "" : `&look=${look.look}-${look.stamp}` + (HDR_SCREEN ? "&hdr=1" : ""));
}
function lookUpdate(l, ask) {
  look = l;
  const box = $("looks"); box.hidden = false;
  if (box.children.length !== l.looks.length) {
    box.textContent = "";
    for (const one of l.looks) {
      const b = document.createElement("button"); b.type = "button"; b.dataset.look = one.id; b.textContent = one.label; b.title = "Look: " + one.about;
      b.onclick = () => fetch("/api/look", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ look: one.id }) }).then((r) => r.json()).then((x) => { if (x.looks) lookUpdate(x); }).catch(() => {});
      box.append(b);
    }
  }
  for (const b of box.children) b.setAttribute("aria-pressed", String(b.dataset.look === l.look));
  if (ask) { box.classList.remove("ask"); void box.offsetWidth; box.classList.add("ask"); }
  if (l.error) toast(l.error, 5000);
  if (tree && tree.film) { $("film").href = filmUrl(); if (l.look !== "natural") fetch(filmUrl(), { headers: { Range: "bytes=0-0" } }).catch(() => {}); }  // asks for the copy
}
function lookPoll() { fetch("/api/look").then((r) => r.ok ? r.json() : null).then((l) => { if (l && l.looks && (l.look !== look.look || l.stamp !== look.stamp || !look.looks.length)) lookUpdate(l); }).catch(() => {}); }
lookPoll(); setInterval(lookPoll, 5000);
function showFilm() { $("film").hidden = false; toast("The film of this path is ready"); }
$("follow").onclick = () => { follow = !follow; $("follow").classList.toggle("on", follow); };
$("zoomIn").onclick = () => zoomAt(1.2);
$("zoomOut").onclick = () => zoomAt(1 / 1.2);
$("fit").onclick = fit;

// ---------------------------------------------------------------------------------------------------- the view
const vp = $("viewport"), world = $("world");
function vpSize() { return { w: vp.clientWidth, h: vp.clientHeight }; }
function apply() { world.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.s})`; placeGhost(); }
function unfollow() { if (follow) { follow = false; $("follow").classList.remove("on"); } }
function zoomAt(k, cx, cy) {
  const { w, h } = vpSize(); cx = cx ?? w / 2; cy = cy ?? h / 2;
  const s = Math.min(2.2, Math.max(0.25, view.s * k));
  view.tx = cx - (cx - view.tx) * (s / view.s); view.ty = cy - (cy - view.ty) * (s / view.s); view.s = s; apply();
}
function fit() {
  if (!tree) return;
  const laid = layout(tree), { w, h } = vpSize();
  const s = Math.min(1.2, Math.max(0.25, Math.min((w - 120) / (laid.width + 40), (h - 220) / (laid.height + 40))));
  view = { s, tx: (w - laid.width * s) / 2, ty: 120 + (h - 220 - laid.height * s) / 2 }; unfollow(); apply();
}
let drag = null;
vp.addEventListener("pointerdown", (e) => { if (e.target.closest(".node")) return; drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false }; vp.setPointerCapture(e.pointerId); vp.classList.add("dragging"); });
vp.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; unfollow(); }
  view.tx = drag.tx + dx; view.ty = drag.ty + dy; apply();
});
vp.addEventListener("pointerup", () => { drag = null; vp.classList.remove("dragging"); });
vp.addEventListener("click", (e) => { if (!e.target.closest(".node") && !drag) { selected = null; $("detail").hidden = true; if (tree) drawNodes(); } });
vp.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) { zoomAt(Math.exp(-e.deltaY * 0.0022), e.clientX, e.clientY); return; }
  unfollow();
  if (e.deltaX === 0 && !e.shiftKey) view.tx -= e.deltaY;  // a mouse wheel scrolls through time
  else { view.tx -= e.deltaX; view.ty -= e.deltaY; }
  apply();
}, { passive: false });
document.addEventListener("keydown", (e) => {
  if ((e.target.closest && e.target.closest("input")) || document.body.classList.contains("landing")) return;
  if (e.key === "Escape") { endMix(); closeOthers(); return; }
  if (e.key === "Backspace") { e.preventDefault(); backStep(); return; }
  const step = 120;
  if (e.key === "ArrowLeft") { view.tx += step; unfollow(); apply(); }
  else if (e.key === "ArrowRight") { view.tx -= step; unfollow(); apply(); }
  else if (e.key === "ArrowUp") { view.ty += step; unfollow(); apply(); }
  else if (e.key === "ArrowDown") { view.ty -= step; unfollow(); apply(); }
  else if (e.key === "+" || e.key === "=") zoomAt(1.2);
  else if (e.key === "-") zoomAt(1 / 1.2);
  else if (e.key === "0") fit();
  else if (e.key === "f" || e.key === "F") { follow = true; $("follow").classList.add("on"); }
  else if (/^[a-hA-H]$/.test(e.key) && tree) {
    const o = aheadIds().map((i) => tree.nodes[i]).find((x) => x.letter === e.key.toUpperCase());
    if (o) (mixFirst ? mixPick(o.id) : choose(o.id));
  }
});
new ResizeObserver(() => apply()).observe(vp);

// ----- the Ghost on the tree: it walks (eased) to its node, and the view follows it unless the person scrolled away -----
function ghostPoint(id) { const p = pos[id]; return p ? { x: p.x + CARD_W / 2, y: p.y } : null; }
function moveGhost(id) {
  const to = ghostPoint(id); if (!to) return;
  ghostFrom = ghostAt || to; ghostTo = to; ghostT0 = performance.now(); ghostId = id;
  follow = true; $("follow").classList.add("on");  // a step of the Ghost brings the view back to it
}
// The thinking box (the voice) floats beside the Ghost and moves to the spot that covers the least: playing clips first,
// then the lit path and the lines to the options, then any card, line or panel. It eases to its new spot and only moves
// when the new one is clearly better, so it does not jitter while the view pans.
let edgeSamples = [], bubble = null, bubbleAt = null, bubbleCheck = 0;
const BUBBLE_GAP = 14;
function sampleEdges() {  // world points along every drawn edge (the same cubic as edgePath), weighted by what it is
  edgeSamples = [];
  if (!tree) return;
  const W = { "edge active": 3, "edge option": 2, "edge dimmed": 0.6 };
  for (const n of Object.values(tree.nodes)) {
    const a = pos[n.parent], b = pos[n.id]; if (!n.parent || !a || !b) continue;
    const w = W[edgeClass(tree, n.parent, n.id)] || 1;
    const x1 = a.x + CARD_W, y1 = a.y + CARD_H / 2 - 14, x2 = b.x, y2 = b.y + CARD_H / 2 - 14, m = (x1 + x2) / 2;
    for (let i = 0; i <= 14; i++) {
      const t = i / 14, u = 1 - t;
      edgeSamples.push({ x: u * u * u * x1 + 3 * u * u * t * m + 3 * u * t * t * m + t * t * t * x2,
                         y: u * u * u * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y2, w });
    }
  }
}
function overlap(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x), h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? w * h : 0;
}
function obstacles(face) {
  const out = [], s = view.s;
  for (const [id, p] of Object.entries(pos)) {
    const c = cards.get(id), st = stateOf(tree, id);
    const w = st === "ghost" || st === "option" ? 4 : st === "active" ? 2 : 1;  // a playing clip is the worst to cover
    out.push({ x: view.tx + p.x * s - 6, y: view.ty + p.y * s - 6, w: CARD_W * s + 12, h: (c ? c.el.offsetHeight : CARD_H) * s + 12, k: w, card: true });
  }
  if (face) out.push({ ...face, k: 5 });
  const vr = vp.getBoundingClientRect();
  for (const id of ["top", "pathbar", "modebar", "toolbar", "hint", "detail", "othersPop", "toast"]) {
    const el = $(id); if (!el || el.hidden) continue;
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    out.push({ x: r.left - vr.left - 8, y: r.top - vr.top - 8, w: r.width + 16, h: r.height + 16, k: 3 });
  }
  return out;
}
function rectScore(r, obs, near, distW) {
  let cost = 0;
  for (const o of obs) cost += overlap(r, o) * o.k;
  const s = view.s;
  for (const e of edgeSamples) {
    const x = view.tx + e.x * s, y = view.ty + e.y * s;
    if (x > r.x - 6 && x < r.x + r.w + 6 && y > r.y - 6 && y < r.y + r.h + 6) cost += 900 * e.w;
  }
  const { w: vw, h: vh } = vpSize(), inside = overlap(r, { x: 8, y: 8, w: vw - 16, h: vh - 16 });
  cost += (r.w * r.h - inside) * 20;  // off screen is the worst
  const dx = r.x + r.w / 2 - (near.x + near.w / 2), dy = r.y + r.h / 2 - (near.y + near.h / 2);
  return cost + Math.hypot(dx, dy) * distW;  // and stay close
}
// the best spot for a w x h box around `near` (a rect); it keeps its spot (`prev`, an offset from near's centre)
// unless a new one is clearly better, so it does not jitter while the view pans
function bestSpot(near, w, h, obs, prev, gap, distW, extra = [], stick = 1.25, more = null) {
  const cx = near.x + near.w / 2, cy = near.y + near.h / 2;
  const rx = w / 2 + near.w / 2 + gap, ry = h / 2 + near.h / 2 + gap;
  const cands = extra.map((e) => ({ x: cx + e.dx, y: cy + e.dy, w, h }));
  for (let a = 0; a < 360; a += 22.5) for (const k of [1, 1.3, 1.7, 2.3]) {
    const t = a * Math.PI / 180;
    cands.push({ x: cx + Math.cos(t) * rx * k - w / 2, y: cy + Math.sin(t) * ry * k - h / 2, w, h });
  }
  let best = null, bestCost = Infinity;
  const score = (c) => rectScore(c, obs, near, distW) + (more ? more(c) : 0);
  for (const c of cands) { const v = score(c); if (v < bestCost) { best = c; bestCost = v; } }
  if (prev) {
    const keep = { x: cx + prev.dx, y: cy + prev.dy, w, h };
    if (score(keep) <= bestCost * stick + 4000) best = keep;
  }
  return { dx: best.x - cx, dy: best.y - cy };
}
function chooseBubble(face) {
  const dock = $("dock"), obs = obstacles(face);
  for (const t of tetherPts) obs.push({ x: t.x - 5, y: t.y - 5, w: 10, h: 10, k: 60 });  // keep the tether clear
  bubble = bestSpot(face, dock.offsetWidth, dock.offsetHeight, obs, bubble, BUBBLE_GAP, 40, [],
                    document.activeElement === $("say") ? 1.6 : 1.25);
}
// The Ghost floats near its clip, wherever covers the least (like its box), and a glowing yellow tether points from it
// to its video box on the lit path, so you always see which step it stands on.
let float = null, floatAt = null, tetherPts = [];
function ghostCard() {  // the Ghost's clip on screen (between two clips while it walks)
  const s = view.s, c = cards.get(ghostId), h = (c ? c.el.offsetHeight : CARD_H) * s;
  return { x: view.tx + ghostAt.x * s - CARD_W * s / 2, y: view.ty + ghostAt.y * s, w: CARD_W * s, h };
}
function tetherAnchor(card, fc) {  // the nearest point on the rim of the Ghost's video box (the clip it stands on)
  const pts = [], n = 8;
  for (let i = 0; i <= n; i++) {
    const x = card.x + card.w * i / n, y = card.y + card.h * i / n;
    pts.push({ x, y: card.y }, { x, y: card.y + card.h }, { x: card.x, y }, { x: card.x + card.w, y });
  }
  let best = pts[0], d = Infinity;
  for (const pt of pts) { const v = Math.hypot(pt.x - fc.x, pt.y - fc.y); if (v < d) { d = v; best = pt; } }
  return best;
}
function placeGhost() {
  const face = $("face"), dock = $("dock"), trail = $("trail");
  if (document.body.classList.contains("landing") || !ghostAt) {
    face.style.transform = dock.style.transform = ""; $("ghost").classList.remove("docked"); trail.innerHTML = "";
    bubble = bubbleAt = float = floatAt = null; return;
  }
  $("ghost").classList.add("docked");
  const fs = Math.max(70, Math.min(150, 120 * view.s)), now = performance.now(), card = ghostCard();
  face.style.width = face.style.height = fs + "px";
  const check = !float || !bubble || now - bubbleCheck > 120;
  if (check) {
    const obs = obstacles(null).concat([{ ...card, k: 8, card: true }]);  // never on its own clip
    const cardObs = obs.filter((o) => o.card);  // the clips, its own most of all
    const crossing = (c) => {  // the tether should not cut across a clip
      const fc = { x: c.x + c.w / 2, y: c.y + c.h / 2 }, at = tetherAnchor(card, fc);
      let cost = 0;
      for (let i = 2; i < 12; i++) {
        const t = i / 12, x = fc.x + (at.x - fc.x) * t, y = fc.y + (at.y - fc.y) * t;
        for (const o of cardObs) if (x > o.x + 8 && x < o.x + o.w - 8 && y > o.y + 8 && y < o.y + o.h - 8) cost += 2500 * o.k;
      }
      return cost;
    };
    float = bestSpot(card, fs, fs, obs, float, 46, 55, [{ dx: -fs / 2, dy: -card.h / 2 - fs - 40 }], 1.25, crossing);  // far enough off for its tether to show
  }
  const fw = { x: card.x + card.w / 2 + float.dx, y: card.y + card.h / 2 + float.dy };
  if (!floatAt) floatAt = { ...fw };
  floatAt.x += (fw.x - floatAt.x) * 0.14; floatAt.y += (fw.y - floatAt.y) * 0.14;
  const fx = Math.round(floatAt.x), fy = Math.round(floatAt.y + Math.sin(now / 1200) * 5);  // it floats
  face.style.transform = `translate(${fx}px, ${fy}px)`;
  const fr = { x: fx, y: fy, w: fs, h: fs };
  if (check) { chooseBubble(fr); bubbleCheck = now; }
  const want = { x: fr.x + fr.w / 2 + bubble.dx, y: fr.y + fr.h / 2 + bubble.dy };
  if (!bubbleAt) bubbleAt = { ...want };
  bubbleAt.x += (want.x - bubbleAt.x) * 0.16; bubbleAt.y += (want.y - bubbleAt.y) * 0.16;  // glide there
  const bob = Math.sin(now / 900) * 3, bx = Math.round(bubbleAt.x), by = Math.round(bubbleAt.y + bob);
  dock.style.transform = `translate(${bx}px, ${by}px)`;
  drawTrail(fr, { x: bx, y: by, w: dock.offsetWidth, h: dock.offsetHeight }, card);
}
// the yellow neon tether from the Ghost to its video box, and thought-bubble puffs from the Ghost to its box
function drawTrail(f, b, card) {
  const fx = f.x + f.w / 2, fy = f.y + f.h / 2, r0 = f.w * 0.36, parts = [];
  const at = tetherAnchor(card, { x: fx, y: fy }), td = Math.hypot(at.x - fx, at.y - fy);
  tetherPts = [];
  if (td > r0 + 6) {
    const sx = fx + (at.x - fx) * r0 / td, sy = fy + (at.y - fy) * r0 / td;
    const mx = (sx + at.x) / 2, my = (sy + at.y) / 2 + Math.min(40, td * 0.18);  // a slight sag
    const d = `M${sx.toFixed(1)},${sy.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${at.x.toFixed(1)},${at.y.toFixed(1)}`;
    parts.push(`<path class="tether glow" d="${d}"/><path class="tether core" d="${d}"/>`,
               `<circle class="tether-dot" cx="${at.x.toFixed(1)}" cy="${at.y.toFixed(1)}" r="5"/>`);
    for (let i = 1; i < 8; i++) { const t = i / 8, u = 1 - t; tetherPts.push({ x: u * u * sx + 2 * u * t * mx + t * t * at.x, y: u * u * sy + 2 * u * t * my + t * t * at.y }); }
  }
  const px = Math.max(b.x, Math.min(fx, b.x + b.w)), py = Math.max(b.y, Math.min(fy, b.y + b.h));  // nearest point of the box
  const d = Math.hypot(px - fx, py - fy), p0 = f.w * 0.42;
  if (d >= p0 + 4) for (const [t, r, i] of [[0.25, 3, 0], [0.5, 5, 1], [0.76, 7.5, 2]]) {
    const k = (p0 + (d - p0) * t) / d, x = fx + (px - fx) * k, y = fy + (py - fy) * k;
    parts.push(`<circle class="puff p${i}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}"/>`);
  }
  $("trail").innerHTML = parts.join("");
}
function tick(now) {
  requestAnimationFrame(tick);
  if (!tree || !ghostTo) return;
  const t = Math.min(1, (now - ghostT0) / GHOST_MS);
  ghostAt = lerpPoint(ghostFrom, ghostTo, t);
  if (pos[ghostId]) ghostTo = ghostPoint(ghostId);  // the layout may move under it
  if (follow) {
    const want = centreOn(ghostAt, vpSize(), view.s), k = 0.12;
    view.tx += (want.tx - view.tx) * k; view.ty += (want.ty - view.ty) * k;
  }
  apply();
}
requestAnimationFrame(tick);

// ---------------------------------------------------------------------------------------------------- credits, start
$("credits").addEventListener("toggle", async () => {
  if (!$("credits").open || $("credits").dataset.loaded) return;
  try {
    const list = await (await fetch("/api/credits")).json(), box = $("clines"); box.textContent = "";
    for (const c of list) { const d = document.createElement("div"), b = document.createElement("b"); b.textContent = c.name; d.append(b, ` by ${c.by}: ${c.what} (${c.licence})`); box.append(d); }
    $("credits").dataset.loaded = "1";
  } catch (_) {}
});

if (location.hash.length > 2) open(location.hash.slice(1)); else landing();
