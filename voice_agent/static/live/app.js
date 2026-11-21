// The live stage's page: plays the pictures as they arrive, sends typed and spoken directions, speaks the cast's lines.
"use strict";
const $ = (id) => document.getElementById(id);
const FPS = 16, LOADED = Date.now() / 1000;
const canvas = $("picture"), draw = canvas.getContext("2d");
let state = { phase: "off", log: [], takes: [] };
let pictures = [];          // decoded pictures waiting for their turn on the screen
let playing = false, nextAt = 0, stalls = 0, spokenTo = LOADED, saidTimer = 0;
let shown = { still: null, idle: null, talk: null, rush: null }, speaking = 0, speakingWho = "", picked = null;   // picked: a take chosen from the list
const ACTION = location.pathname.startsWith("/live-action");   // Live Action: the rehearsal and the take; /live: the stream
document.body.classList.add(ACTION ? "action" : "stream");
document.title = (ACTION ? "Live Action" : "Live stage") + " · keys Ghost Multiverse Studio";
{ const on = $("navAction"); on.classList.add("on"); on.setAttribute("aria-current", "page"); }
let toldTo = LOADED;
let gap = 1000 / FPS;       // ms between two pictures on the screen: longer when the engine is slower than playback

const EXAMPLE = "Medium shot, slow handheld camera. A woman waits under a street lamp at a tram stop at night in light rain, " +
  "wet cobblestones and shop windows glowing behind her, cinematic, realistic.\n\nMara: a woman in her thirties with short black hair and a red raincoat";

// ------------------------------------------------------------------------------------------------ the pictures
function connect() {
  const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live");
  socket.binaryType = "arraybuffer";
  socket.onmessage = async (message) => {
    if (typeof message.data === "string") return onEvent(JSON.parse(message.data));
    const view = new DataView(message.data), size = view.getUint32(0);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(message.data, 4, size)));
    let offset = 4 + size;
    const made = [];
    for (const length of header.sizes || []) {
      made.push(createImageBitmap(new Blob([new Uint8Array(message.data, offset, length)], { type: "image/jpeg" })));
      offset += length;
    }
    for (const picture of await Promise.all(made)) pictures.push(picture);
    // an engine slower than playback is shown evenly in slow motion, not in bursts: four pictures over the time they took
    if (header.ms && made.length > 1) gap += 0.2 * (Math.max(1000 / FPS, header.ms / made.length) - gap);
    if (pictures.length > 24) pictures.splice(0, pictures.length - 8).forEach((p) => p.close());  // fell behind: catch up
  };
  socket.onclose = () => setTimeout(connect, 1500);
}

function tick(now) {
  requestAnimationFrame(tick);
  if (!playing) {
    if (pictures.length < (gap > 80 ? 1 : 6)) return;      // a little in hand before the first picture
    playing = true; nextAt = now;
  }
  if (now < nextAt) return;
  const picture = pictures.shift();
  if (!picture) { playing = false; stalls += 1; return; }   // nothing ready: hold the last picture
  if (canvas.width !== picture.width || canvas.height !== picture.height) { canvas.width = picture.width; canvas.height = picture.height; }
  draw.drawImage(picture, 0, 0);
  picture.close();
  nextAt += gap;
  if (now - nextAt > 500) nextAt = now;
}

// ------------------------------------------------------------------------------------------------ the state
function onEvent(event) {
  if (event.type === "stats") {
    const speed = 250 / event.ms;
    $("meter").textContent = `${speed.toFixed(2)}× · ${Math.round(event.ms)} ms` + (speed < 0.9 ? " · shown in slow motion, the take plays at full speed" : "");
    $("meter").classList.toggle("slow", speed < 1);
    $("clock").textContent = clock(event.frames / 4);
    return;
  }
  if (event.type !== "state") return;
  const before = state.phase;
  state = event;
  if (before !== "live" && state.phase === "live") { pictures.forEach((p) => p.close()); pictures = []; playing = false; gap = 1000 / FPS; }
  render();
  const wrong = (state.log || []).filter((e) => e.kind === "error").pop();   // what went wrong is said on the picture, not only in the list
  if (wrong && wrong.clock > toldTo) { toldTo = wrong.clock; caption(`<b>That did not work.</b> ${safe(wrong.text)}`, 9000); }
  speakNew();
}

const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

function render() {
  const phase = state.phase, live = phase === "live";
  const filming = phase === "filming", h3 = state.render === "h3", set = state.set || {};
  const running = phase === "running";
  const table = state.table, atTable = !!table && !live && !filming && !running && phase !== "taking";
  dress(set, filming, atTable);
  through();
  document.body.classList.toggle("h3", h3);
  const blocking = state.blocking || {}, block = $("block");
  $("blocking").hidden = !(atTable && (blocking.file || blocking.doing || blocking.error));
  if (blocking.file && block.dataset.file !== blocking.file) { block.dataset.file = blocking.file; block.src = `/api/live/set/${blocking.file}`; block.play().catch(() => {}); }
  $("blockNote").textContent = blocking.error ? `No blocking: ${blocking.error}` : blocking.doing ? "Blocking the scene…"
    : blocking.file ? `Blocking, ${blocking.seconds} s` : "";
  const veil = $("veil"), words = {
    off: "Your scene appears here.", starting: state.detail || "Loading the picture engine",
    ready: state.frames ? "Cut. Print the take, or start another scene." : "The stage is ready.",
    taking: "Printing the take", failed: state.detail || "The stage stopped." };
  veil.hidden = live || !$("set").hidden;
  $("veilText").textContent = atTable ? (state.casting ? "Reading your scene: who is in it and what happens." : "The reference shot is on its way.") : filming ? "Filming" : (words[phase] || "");
  $("board").hidden = !table;
  if (table) {
    $("opening").textContent = table.scene;
    const beats = $("beats"); beats.textContent = "";
    for (const beat of table.beats) {
      const item = document.createElement("li"), when = document.createElement("span");
      when.textContent = clock(beat.s); item.append(when, beat.note); beats.append(item);
    }
    $("cue").textContent = table.beats.length ? "" : "Nothing happens yet. Say what happens: “she waits, then walks to the camera”.";
  }
  roster(set, atTable);
  track();
  turn();
  $("status").textContent = phase === "failed" ? (state.detail || "stopped") : live ? "live" : phase === "starting" ? "loading the picture engine…" : "";
  $("status").classList.toggle("bad", phase === "failed");
  $("note").disabled = $("send").disabled = !(live || atTable || filming || running);
  canSpeak = live || atTable || filming || running;
  $("note").placeholder = filming ? "Or type a direction for the next part" : running ? "Or type what you say to the cast: they answer now" : "Or type it: “she waits, then walks to the camera”, “Mara, how do you feel tonight?”";
  $("stop").disabled = !(live || filming || running);
  $("take").hidden = h3;
  $("frame").classList.toggle("wide", h3 && !live);
  $("action").disabled = !atTable;
  const ran = atTable && state.run && (state.run.state === "cut" || state.run.state === "ended");
  $("action").textContent = ran ? "Retake" : atTable && state.frames ? "Action again" : "Action";
  $("action").classList.toggle("primary", !ran);
  $("print").hidden = !ran;
  $("print").disabled = !!state.rendering;
  $("print").title = state.rendering ? "The take before this one is still being filmed" : "Film it as it was just played, up to the cut (or say “render it”)";
  $("take").disabled = !(phase === "ready" && state.frames > 4);
  $("start").disabled = $("plan").disabled = phase === "taking" || filming || !!state.rendering || (phase === "starting" && !atTable);
  $("start").textContent = ACTION ? (live ? "Restart stream" : "Live stream") : live ? "Restart" : "Start";
  if (!live) $("meter").textContent = "";
  if (state.scene && !$("scene").value) $("scene").value = state.scene;
  $("memory").textContent = phase !== "off" || table ? "" : ACTION ? "Rehearse makes a reference shot of the scene (about a minute), a blocking preview, and a close-up of each character, who answers you. Action films the plan; cut stops it. " + `Live stream plays the scene at once instead, as a rougher moving picture you direct while it runs, with no cast to talk to and no take: its engine loads first (${state.load || "about a minute"}).`
    : (h3 ? "Rehearse makes a reference shot of the scene (about a minute) and then the character, who answers you. Action films the plan with the Studio’s own clip engine. " : "") +
    `Start plays a live stream instead: its engine loads first (${state.load || "about a minute"}, ${state.free_gb} GB of memory free${state.host ? " on its machine" : ""} now).`;

  const log = $("log"); log.textContent = "";
  for (const entry of state.log || []) {
    if (entry.kind === "scene" || entry.kind === "table") continue;
    const item = document.createElement("li"); item.className = entry.kind + (entry.planned ? " planned" : "");
    if (entry.kind === "stop") entry.text = entry.text === "cut" ? "Cut." : /run-through ended/.test(entry.text) ? entry.text : `Cut (${entry.text}).`;
    if (entry.kind === "action") entry.text = `Action: ${entry.text}.`;
    const when = document.createElement("span"); when.className = "when";
    when.textContent = entry.part ? `part ${entry.part}` : entry.queued ? "next" : entry.at_s == null ? "table" : clock(entry.at_s);
    item.append(when);
    if (entry.kind === "line") { const who = document.createElement("b"); who.textContent = (entry.who || "") + ": "; item.append(who); }
    item.append(document.createTextNode(entry.kind === "note" && entry.heard ? `“${entry.text}”` : entry.text));
    if (entry.scene) { const made = document.createElement("span"); made.className = "made"; made.textContent = entry.scene; item.append(made); }
    if (entry.brain_ms != null) { const took = document.createElement("span"); took.className = "took"; took.textContent = `rewritten in ${(entry.brain_ms / 1000).toFixed(1)} s`; item.append(took); }
    log.append(item);
  }
  log.scrollTop = log.scrollHeight;

  const takes = $("takes"); takes.textContent = "";
  $("takesbox").hidden = !(state.takes || []).length;
  for (const take of [...(state.takes || [])].reverse()) {
    const item = document.createElement("li");
    if (take.state === "done") {
      const image = document.createElement("img"); image.src = `/api/live/takes/${take.name}.jpg`; image.alt = "";
      const play = document.createElement("button"); play.type = "button"; play.className = "play"; play.title = "Play this take on the big screen";
      play.append(image, document.createTextNode(`${take.number ? `Take ${take.number}` : "Earlier take"}, ${take.seconds} s${take.cut ? ", cut" : ""}`));
      play.onclick = () => { picked = take.file; shown.closed = null; render(); scrollTo({ top: 0, behavior: "smooth" }); };
      const link = document.createElement("a"); link.href = `/api/live/takes/${take.file}`; link.download = take.file; link.textContent = "save";
      item.append(play, link);
    } else {
      item.textContent = take.state === "failed" ? `No take: ${take.detail || ""}` : take.parts ? `Take ${take.number || ""}: filming part ${Math.min((take.done || 0) + 1, take.frames)} of ${take.frames}`
        : `Filming… ${Math.round(100 * (take.done || 0) / (take.frames || 1))}%`;
      if (take.state === "rendering" && take.behind && !running && !live) {   // filmed behind the rehearsal: stopped from here
        const halt = document.createElement("button"); halt.type = "button"; halt.className = "halt";
        halt.textContent = state.wrapping ? "drop it" : "stop"; halt.title = state.wrapping ? "Drop the part being filmed too" : "End the take after the part being filmed";
        halt.onclick = () => post("/api/live/stop"); item.append(" ", halt);
      }
    }
    takes.append(item);
  }
}

// the set: the reference shot, the character's two loops over it, and the take's parts as they are filmed
function dress(set, filming, atTable) {
  const at = (name) => name ? `/api/live/set/${name}` : null;
  const takes = state.takes || [], take = takes[takes.length - 1];
  if (filming) picked = null;
  if (state.phase === "running") { picked = null; if (take && take.state === "done") shown.closed = take.file; }   // the run-through has the screen, and keeps it after its cut
  const part = picked || (take && take.parts && take.parts.length ? (take.state === "done" ? take.file : take.parts[take.parts.length - 1]) : null);
  if (doneSeen === undefined) {   // a page that opens does not start on an old take: it is under Playback
    if ("scene" in state) {   // the Studio's first word, not the empty page before it
      doneSeen = take && take.state === "done" ? take.file : null;
      if (doneSeen && !filming) shown.closed = doneSeen;
    }
  } else if (take && take.behind && take.state === "done" && doneSeen !== take.file) {   // a take filmed behind the rehearsal is ready
    doneSeen = take.file;
    if (state.phase === "running") { shown.closed = take.file; caption(`<b>Take ${take.number || ""} is ready.</b> It is under Takes.`, 6000); }
  }
  const rush = part && state.phase !== "running" && (picked || filming || take.state === "done") && !(atTable && shown.closed === part) ? `/api/live/takes/${part}` : null;
  $("set").hidden = !(state.render === "h3" && (set.still || rush) && state.phase !== "live");
  const want = { still: at(set.still), idle: at(set.loops && set.loops.idle), talk: at(set.loops && set.loops.talk), rush };
  for (const key of ["still", "idle", "talk", "rush"]) {
    if (want[key] === shown[key]) continue;
    shown[key] = want[key];
    const one = $(key);
    one.classList.toggle("on", !!want[key]);
    if (!want[key]) { one.removeAttribute("src"); if (one.load && key !== "still") one.load(); continue; }
    one.src = want[key];
    if (key !== "still") one.play().catch(() => { if (key === "rush") one.controls = true; });   // a browser that will not start it: its own play button
  }
  // a finished take stays on the big screen (its controls on, to watch it again) until it is closed or the character speaks
  const screen = $("rush"), over = !!want.rush && !filming;
  screen.controls = over || (screen.controls && !!want.rush);
  $("rushClose").hidden = !over;
  $("rushClose").onclick = closeRush;
  $("frame").classList.toggle("watching", over);   // the take has the screen: the plan and the blocking box step aside
}
// ------------------------------------------------------------------------------------------------ the run-through
// after "action" the plan is played in real time and nothing is filmed: the blocking's stick figures on the big screen,
// every line spoken at its second. "Cut" stops it where it is; then it is played again, or filmed up to there.
let doneSeen;   // the last finished take this page knows of (undefined until the first word from the Studio)
let going = null, runShut = 0;   // going: {n, zero (the page's clock at second 0), cues, timer}; runShut: a run closed by hand
const runAt = () => {
  const film = $("runfilm");
  if (!going) return 0;
  return going.rolling && !film.paused && film.readyState > 1 ? film.currentTime : (performance.now() - going.zero) / 1000;
};
function through() {
  const run = state.run, film = $("runfilm"), running = state.phase === "running";
  const after = !!run && (run.state === "cut" || run.state === "ended") && state.phase !== "filming" && runShut !== run.n && !shown.rush;
  const on = running || after;
  $("frame").classList.toggle("running", on);
  $("runbar").hidden = !(on && run && run.state !== "preparing");
  $("runClose").hidden = !after;
  film.classList.toggle("on", on && !!run.film);
  if (on) $("set").hidden = false;
  if (!run || (!on && going)) { stopRun(); if (!run) return; }
  if (!on) return;
  if (run.film && film.dataset.file !== run.film) { film.dataset.file = run.film; film.src = `/api/live/set/${run.film}`; film.load(); }
  marks(run);
  if (run.state === "playing" && (!going || going.n !== run.n)) {
    stopRun();
    going = { n: run.n, zero: performance.now() - run.t * 1000, rolling: false, counted: null,
      cues: run.cues.map((cue) => ({ ...cue, made: null, done: cue.s < run.t - 0.5 })) };
    film.pause(); try { film.currentTime = Math.max(0, run.t); } catch (_) { /* not loaded yet */ }
    going.timer = setInterval(() => step(state.run), 60);
  }
  if (after) {   // it stands where it was cut: the positions at that second
    stopRun();
    film.pause(); try { if (Math.abs(film.currentTime - run.cut_s) > 0.05) film.currentTime = Math.min(run.cut_s, Math.max(0, (film.duration || run.seconds) - 0.05)); } catch (_) { /* not loaded */ }
    showRun(run, run.cut_s);
  }
}
function marks(run) {   // where the beats are on the line
  const line = $("runLine");
  if (line.dataset.n === String(run.n) + run.cues.length) return;
  line.dataset.n = String(run.n) + run.cues.length;
  line.querySelectorAll("s").forEach((one) => one.remove());
  for (const cue of run.cues) { const mark = document.createElement("s"); mark.style.left = `${Math.min(100, 100 * cue.s / (run.seconds || 1))}%`; line.append(mark); }
}
function showRun(run, t) {
  const at = Math.max(0, Math.min(t, run.seconds));
  $("runClock").textContent = `${clock(at)} / ${clock(run.seconds)}`;
  $("runLine").firstElementChild.style.width = `${100 * at / (run.seconds || 1)}%`;
  const now = run.cues.filter((cue) => cue.s <= t + 0.05).pop();
  $("runCue").textContent = now ? (now.line ? `${now.who}: “${now.line}”` : now.note) : run.cues.length ? `first at ${clock(run.cues[0].s)}: ${run.cues[0].note}` : "";
}
function step(run) {
  if (!going || !run || run.state !== "playing") return;
  const film = $("runfilm"), t = runAt();
  if (t < 0) {   // the count before it
    const left = Math.ceil(-t);
    if (going.counted !== left) { going.counted = left; caption(`<b>${left}</b>`, 1100); }
  } else if (!going.rolling) {
    going.rolling = true; $("said").classList.remove("on");
    caption("<b>Action!</b>", 900);
    if (run.film) {
      try { film.currentTime = Math.max(0, t); } catch (_) { /* plays from where it is */ }
      film.play().catch((why) => { going && (going.stepped = String(why && why.name || why)); });
    }
  } else if (run.film && film.paused) {
    // a browser that will not play it by itself (Safari, without a click just before): the picture is moved to the
    // clock's second by hand, so the figures move all the same
    if (!going.movedAt || performance.now() - going.movedAt > 80) {
      going.movedAt = performance.now();
      if (!film.seeking) { try { film.currentTime = Math.min(t, Math.max(0, (film.duration || run.seconds) - 0.05)); } catch (_) { /* not loaded yet */ } }
      if (going.tried !== Math.floor(t / 2)) { going.tried = Math.floor(t / 2); film.play().catch(() => {}); }   // and it is asked again now and then
    }
  }
  if (going.rolling && run.film && !going.told && t > 2) {   // what this browser did with it, for the Studio's log
    going.told = true;
    report(`run-through film ${film.paused ? "stepped by hand" : "playing"} (ready ${film.readyState}, at ${film.currentTime.toFixed(1)} of ${t.toFixed(1)} s${going.stepped ? ", play refused: " + going.stepped : ""}${film.error ? ", error " + film.error.code : ""})`);
  }
  for (const cue of going.cues) {
    if (cue.done) continue;
    if (cue.line && !cue.made && t >= cue.s - 4) cue.made = lineOf({ who: cue.who, text: cue.line, voice: cue.voice }).catch(() => null);   // made ahead, so it starts on its second
    if (t < cue.s) continue;
    cue.done = true;
    if (!cue.line) { caption(safe(cue.note), 3200); continue; }
    const n = going.n;
    cue.made.then((play) => { if (play && going && going.n === n) { caption(`<b>${safe(cue.who || "")}</b> ${safe(cue.line)}`, 6000); play(); } });
  }
  showRun(run, t);
}
function stopRun() {
  if (!going) return;
  clearInterval(going.timer); going = null;
  if (hushLine) hushLine();   // a line being spoken stops with the cut
}
$("runClose").onclick = () => { if (state.run) runShut = state.run.n; render(); };

function closeRush() {
  const rush = $("rush");
  if (!shown.rush || state.phase === "filming") return;
  const last = (state.takes || [])[(state.takes || []).length - 1];   // back to the rehearsal: the last take does not come back on its own
  shown.closed = last && last.state === "done" ? last.file : shown.rush.split("/").pop();
  picked = null; shown.rush = null;
  rush.pause(); rush.classList.remove("on"); rush.controls = false; rush.removeAttribute("src"); rush.load();
  $("rushClose").hidden = true; $("frame").classList.remove("watching");
}

// ------------------------------------------------------------------------------------------------ the cast
// one card for each character: their close-up, which answers with its own lips; a click says who you are talking to
const cards = {};
function roster(set, atTable) {
  const faces = (set && set.faces) || [], box = $("cast");
  box.hidden = !(faces.length && state.table);
  if (box.hidden) return;
  const lips = (set.lips && set.lips.faces) || {};
  for (const name of Object.keys(cards)) if (!faces.some((f) => f.name === name)) { cards[name].card.remove(); delete cards[name]; }
  for (const face of faces) {
    let one = cards[face.name];
    if (!one) {
      const card = document.createElement("button"); card.type = "button"; card.className = "face";
      const image = document.createElement("img"); image.alt = "";
      const film = document.createElement("video"); film.playsInline = true; film.className = "linefilm";
      const who = document.createElement("span"); who.className = "who"; who.textContent = face.name;
      const note = document.createElement("span"); note.className = "note";
      card.append(image, film, who, note);
      card.onclick = () => post("/api/live/face", { name: face.name });
      box.append(card);
      one = cards[face.name] = { card, image, film, note, file: null };
    }
    if (face.file && one.file !== face.file) { one.file = face.file; one.image.src = `/api/live/set/${face.file}`; }
    const chosen = state.facing === face.name && faces.length > 1;
    one.card.classList.toggle("chosen", chosen);
    one.card.classList.toggle("empty", !face.file);
    one.card.setAttribute("aria-pressed", String(chosen));
    one.card.title = faces.length > 1 ? `Talk to ${face.name}` : face.name;
    one.card.dataset.lips = lips[face.name] || "";
    if (!one.card.classList.contains("speaking"))
      one.note.textContent = !face.file ? "close-up on its way" : lips[face.name] === "ready" ? (chosen ? "listening to you" : "") : lips[face.name] === "preparing" ? "lips getting ready" : "answers in sound";
  }
}

// ------------------------------------------------------------------------------------------------ what is being made
// the steps on a line, each as wide as it is long, with the time each has left: counted here between the Studio's words
let progressAt = 0;
function track() {
  progressAt = performance.now();
  const progress = state.progress, list = $("steps");
  $("track").hidden = !(progress && progress.steps && progress.steps.length);
  if ($("track").hidden) return;
  list.textContent = "";
  for (const step of progress.steps) {
    const item = document.createElement("li"); item.dataset.key = step.key; item.dataset.state = step.state;
    item.style.flexGrow = String(Math.max(20, Math.min(step.expect_s, 120)));
    const name = document.createElement("span"); name.className = "name"; name.textContent = step.label;
    const time = document.createElement("span"); time.className = "time";
    const bar = document.createElement("span"); bar.className = "bar"; bar.append(document.createElement("i"));
    item.append(name, time, bar); list.append(item);
  }
  count();
}
function left(step) {
  if (step.state === "done" || step.state === "failed") return 0;
  const elapsed = step.state === "running" ? step.elapsed_s + (performance.now() - progressAt) / 1000 : 0;
  return Math.max(0, step.expect_s - elapsed);
}
function count() {
  const progress = state.progress;
  if (!progress || $("track").hidden) return 0;
  let total = 0, busy = false;
  for (const step of progress.steps) {
    const item = $("steps").querySelector(`[data-key="${step.key}"]`); if (!item) continue;
    const remaining = left(step), running = step.state === "running";
    if (step.key !== "blocking") total += remaining;      // the blocking is drawn beside the pictures, not after them
    busy = busy || running || step.state === "waiting";
    item.querySelector(".time").textContent = step.state === "done" ? "done" : step.state === "failed" ? "failed" : running ? (remaining > 1 ? clock(remaining) : "any moment") : `about ${clock(step.expect_s)}`;
    item.querySelector("i").style.width = step.state === "done" ? "100%" : running ? `${Math.min(97, 100 * (1 - remaining / step.expect_s))}%` : "0";
  }
  const take = progress.of === "take";
  $("track").dataset.done = String(!busy);
  $("trackWhat").textContent = take ? (busy ? "Filming the take" : "The take is filmed") : busy ? "Getting the set ready" : "The set is ready";
  $("trackLeft").textContent = !busy ? "" : total > 1 ? `about ${clock(total)} left` : "any moment now";
  return busy ? total : 0;
}

// ------------------------------------------------------------------------------------------------ whose turn it is
// one band under the picture says it in a word and a colour: speak now, wait, rolling. The microphone, the picture's
// frame and the cast follow the same colour, so nobody has to guess whether they are being heard.
let thinking = 0;
let view = null, stageWas = null;   // view: the lamp chosen for the screen and the side panel; it follows the scene until a lamp is clicked
function go(name) {
  const takes = state.takes || [], table = state.table;
  if (name === "direct") { closeRush(); }
  else if (name === "rehearse") { if (!table) { $("plan").click(); return; } closeRush(); if (state.run) runShut = state.run.n; }
  else if (name === "action") { $("action").click(); return; }
  else if (name === "rolling") { if (!$("print").hidden && !$("print").disabled && !state.rendering) $("print").click(); }
  else if (name === "cut") { if (!$("stop").disabled) { $("stop").click(); return; } closeRush(); runShut = 0; }
  else if (name === "playback") { const last = takes.filter((one) => one.state === "done").pop(); if (!last) return; picked = last.file; shown.closed = null; }
  view = name; render(); view = name; turn();
  if (name === "direct") $("scene").focus();
}
for (const lamp of document.querySelectorAll("#mast li")) lamp.firstElementChild.onclick = () => go(lamp.dataset.stage);
function turn() {
  const phase = state.phase, table = state.table, filming = phase === "filming", set = state.set || {};
  const takes = state.takes || [], take = takes[takes.length - 1];
  const watching = $("frame").classList.contains("watching");
  const who = state.facing || "the cast", many = ((set.faces || []).length > 1);
  let kind = "start", word = "Describe your scene", hint = "Write it in the Scene box, then press Rehearse.";
  const run = state.run, ran = table && run && (run.state === "cut" || run.state === "ended") && phase !== "running";
  if (phase === "live") { kind = "you"; word = "Direct the scene"; hint = "Say or type what happens. Say “cut” to stop."; }
  else if (phase === "running" && run && run.state === "preparing") { kind = "wait"; word = "Stand by"; hint = "Working out the positions and the lines. The run-through starts in a moment."; }
  else if (phase === "running") { kind = "rolling"; word = "Run-through"; hint = (state.cast || []).length ? `This is the scene in real time. Talk to ${who} and they answer now. Say “cut” or press Cut.` : "This is the scene in real time: the positions and the lines. Nothing is filmed yet. Say “cut” or press Cut."; }
  else if (ran && !watching && $("frame").classList.contains("running")) {
    const parts = Math.max(1, Math.round(run.cut_s / 5.17)), each = (set.quality === "sharp" && (set.sizes || {}).sharp) ? 285 : 110;
    kind = "you"; word = run.state === "cut" ? `Cut at ${clock(run.cut_s)}` : "The run-through ended";
    hint = state.rendering ? `Press Retake to play it again, or give a note. This one can be filmed when take ${take && take.number ? take.number : ""} is done.`
      : `Press Retake to play it again, or Render take to film these ${parts * 5} seconds (about ${clock(parts * each)}); you can go on rehearsing while it films. Or give a note first.`;
  }
  else if (filming) {
    const total = count(), part = take ? Math.min((take.done || 0) + 1, take.frames) : 1;
    kind = "rolling"; word = state.wrapping ? "Cut" : "Rolling";
    hint = state.wrapping ? `The part being filmed is finished first${total > 1 ? `: your take is ready in about ${clock(total)}` : ""}. Press Cut again to drop it instead.` : `Part ${part} of ${take ? take.frames : 1}${total > 1 ? `, about ${clock(total)} left` : ""}. Say “cut” or press Cut to stop. Nothing else is heard now.`;
  } else if (table && watching) { kind = "watch"; word = picked ? "Playing a take" : `Take ${take && take.number ? take.number : ""} is ready`.replace("  ", " "); hint = "Watch it on the screen. Press Back to rehearsal to go on, or Action to film it again."; }
  else if (table && state.brain === "loading") { kind = "wait"; word = "Wait"; hint = "Your assistant is waking up, up to a minute and a half after a sharp take. You are heard after that."; }
  else if (table && speaking > 0) { kind = "wait"; word = "Listen"; hint = `${speakingWho || "The cast"} is answering. The microphone waits until the line ends.`; }
  else if (table && thinking > 0) { kind = "wait"; word = "Wait"; hint = "Working out what you said."; }
  else if (table) {
    kind = "you";
    const heard = $("mic").classList.contains("hears");
    word = open ? (heard ? "Listening to you" : "Speak now") : "Your turn";
    const ready = state.rendering ? 0 : count(), behind = state.rendering ? count() : 0;
    hint = (open ? `Talk to ${who}, say what happens in the scene, or say “action” to film it.` : `Open the microphone and talk to ${who}, or type below. Press Action to film.`)
      + (many ? " Click a face to choose who you talk to." : "") + (ready > 1 && !(set.faces || []).some((f) => f.file) ? ` The cast can be seen in about ${clock(ready)}; they already answer in sound.` : "")
      + (state.casting ? " Reading your scene: the cast and the plan appear in a moment." : "")
      + (state.rendering ? ` Take ${take && take.number ? take.number : ""} is being filmed behind this${behind > 1 ? `, about ${clock(behind)} left` : ""}; it plays here when it is made.` : "");
    if (state.rendering) word = "Rolling";
  }
  if (phase === "failed" || (table && set.error && !set.still)) { kind = "stop"; word = "That did not work"; hint = state.detail || set.error || "The stage stopped."; }
  document.body.dataset.turn = kind; $("turn").dataset.turn = kind;
  // the blocking's film is fetched as soon as it is drawn, so "action" has it and a click on Action can start it
  const drawnFilm = state.blocking && state.blocking.file, runFilm = $("runfilm");
  if (drawnFilm && phase !== "running" && runFilm.dataset.file !== drawnFilm) { runFilm.dataset.file = drawnFilm; runFilm.src = `/api/live/set/${drawnFilm}`; runFilm.load(); }
  // the sign on the monitor: wait while the scene is being set up, ready when it is
  let sign = "", signIs = "";
  if (table && phase !== "running" && phase !== "failed" && (filming || state.rendering)) {
    const t = count(); signIs = "rolling"; sign = `<b>Rolling</b> filming take ${take && take.number ? take.number : ""}${t > 1 ? `, about ${clock(t)} left` : ""}`;
  } else if (table && phase !== "running" && !filming && phase !== "failed") {
    const progress = state.progress, setting = progress && progress.of !== "take" && !$("track").hidden && $("track").dataset.done === "false";
    if (state.casting) { signIs = "wait"; sign = "<b>Wait</b> reading your scene"; }
    else if (setting) { const t = count(); signIs = "wait"; sign = `<b>Wait</b> setting the scene up${t > 1 ? `, about ${clock(t)} left` : ""}`; }
    else if (set.still && !set.error) { signIs = "ready"; sign = "<b>Ready</b> the scene is set up"; }
  }
  if (phase === "starting" && /picture engine/.test(state.detail || "")) { signIs = "wait"; sign = `<b>Wait</b> the live stream is loading (${safe(state.load || "about a minute")})`; }
  $("sign").hidden = !sign; $("sign").dataset.is = signIs;
  if ($("sign").innerHTML !== sign) $("sign").innerHTML = sign;
  // the mast on the left: the lit lamp is what the screen and the side panel show. It follows the scene by itself, and
  // a click on a lamp goes there (Rolling also glows while a take films behind the rehearsal)
  let stage = "direct";
  if (phase === "running" || phase === "live") stage = "action";   // the live stream plays as it is directed: that is action too
  else if (filming) stage = state.wrapping ? "cut" : "rolling";
  else if (table && watching) stage = "playback";
  else if (ran && $("frame").classList.contains("running")) stage = "cut";
  else if (table && state.rendering) stage = "rolling";   // a take is filmed behind the rehearsal: Rehearse on the mast goes on meanwhile
  else if (table && !state.casting) stage = "rehearse";
  if (stage !== stageWas) { stageWas = stage; view = stage; }
  document.body.dataset.view = view;
  if (view === "direct" && stage !== "direct") { word = "Set the scene"; hint = "Change the scene in the Scene box, then press Rehearse to set it up again. Or click Rehearse on the left to go back to the cast."; }
  const can = {
    direct: true,
    rehearse: table ? phase !== "running" && !filming : !!$("scene").value.trim() && !$("plan").disabled,
    action: !$("action").disabled,
    rolling: filming || !!state.rendering || (!$("print").hidden && !$("print").disabled),
    cut: !$("stop").disabled || !!ran,
    playback: !filming && phase !== "running" && takes.some((one) => one.state === "done"),
  };
  for (const lamp of document.querySelectorAll("#mast li")) {
    const name = lamp.dataset.stage, on = name === view, button = lamp.firstElementChild;
    lamp.classList.toggle("on", on); lamp.classList.toggle("view", name === view);
    lamp.classList.toggle("behind", !on && name === "rolling" && !!state.rendering);
    button.disabled = !can[name] && !on;
    button.setAttribute("aria-pressed", String(name === view));
    if (on) lamp.setAttribute("aria-current", "step"); else lamp.removeAttribute("aria-current");
  }
  if ($("turnWord").textContent !== word) $("turnWord").textContent = word;
  if ($("turnHint").textContent !== hint) { $("turnHint").textContent = hint; $("turnHint").title = hint; }
  $("micWord").textContent = open ? "Close microphone" : "Open microphone";
  const sizes = set.sizes || {};
  $("quality").hidden = !(sizes.sharp && table);
  for (const one of $("quality").children) {
    const on = one.dataset.quality === (set.quality || "draft");
    one.classList.toggle("on", on); one.setAttribute("aria-pressed", String(on)); one.disabled = filming || !!state.rendering;
  }
  $("mic").disabled = !(table || phase === "live" || filming);
  if (phase === "running" && run && run.state === "playing" && going) $("turnHint").textContent = hint;
  $("tag").textContent = filming ? "Rolling" : phase === "running" ? "Run-through" : watching ? "Take" : $("frame").classList.contains("running") ? "Run-through, cut" : table ? "Rehearsal" : "";
}
setInterval(() => { if (state.progress || state.phase === "filming" || state.rendering) { count(); turn(); } }, 500);

// the microphone's level as bars beside the word: it moves when you are heard and lies flat when you are not
const levels = new Array(28).fill(0);
function level(value) {
  levels.push(Math.min(1, value)); levels.shift();
  const wave = $("wave"), pen = wave.getContext("2d"), w = wave.width, h = wave.height, step = w / levels.length;
  pen.clearRect(0, 0, w, h);
  pen.fillStyle = getComputedStyle($("turn")).getPropertyValue("--sig") || "#58f5a8";
  levels.forEach((v, i) => { const tall = Math.max(2, v * h); pen.fillRect(i * step + 1, (h - tall) / 2, step - 2, tall); });
}

function caption(html, hold = 4000) {
  const box = $("said"); box.innerHTML = html; box.classList.add("on");
  clearTimeout(saidTimer); saidTimer = setTimeout(() => box.classList.remove("on"), hold);
}
const safe = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// what this browser did with a line, told to the Studio's log (a line that stays silent is found there, not guessed)
let told = 0;
function report(what) {
  if (told++ > 40) return;
  try { navigator.sendBeacon("/api/live/client", JSON.stringify({ what: String(what).slice(0, 300) })); } catch (_) { /* no beacon here */ }
}
window.addEventListener("error", (event) => report(`error: ${event.message} at ${event.lineno}`));
// The cast's voices go through one audio output of the page's own (Web Audio), and their films play silent beside it:
// a browser that lets a page start a film only without sound (Safari, every film anew) allows this output for good
// after one click, and the first click is the microphone's or Rehearse's.
let sound = null, waiting = null;   // waiting: a line that plays now and is not heard yet
try { sound = new (window.AudioContext || window.webkitAudioContext)(); } catch (_) { /* made at the first click instead */ }
function unlock() {
  try {
    sound = sound || new (window.AudioContext || window.webkitAudioContext)();
    if (sound.state !== "running") sound.resume().then(() => { if (waiting) waiting(); }).catch(() => {});
    else if (waiting) waiting();
  } catch (_) { /* no audio output here */ }
}
for (const kind of ["pointerdown", "keydown", "touchend"]) document.addEventListener(kind, unlock, true);
const decoded = (bytes) => new Promise((done) => { try { sound.decodeAudioData(bytes, done, () => done(null)); } catch (_) { done(null); } });
// the sound of a line, started ``at`` seconds in (where its film is now); gives what stops it
function voiced(buffer, at, ended) {
  const source = sound.createBufferSource(); source.buffer = buffer; source.connect(sound.destination);
  if (ended) source.onended = ended;
  source.start(0, Math.max(0, Math.min(at, buffer.duration - 0.02)));
  return () => { try { source.onended = null; source.stop(); } catch (_) { /* over already */ } };
}
// a line's film started, silent when its sound comes from the page's own output
async function begin(voice, how, silent) {
  try { voice.muted = !!silent; await voice.play(); report(`line played (${how}${silent ? ", sound beside it" : ""})`); }
  catch (wrong) {
    report(`play refused (${how}): ${wrong.name} ${wrong.message}`);
    if (!(voice instanceof HTMLVideoElement) || silent) throw wrong;
    voice.muted = true; await voice.play();
    caption("Click anywhere to hear the answer.", 15000);
  }
}
document.addEventListener("pointerdown", () => {
  for (const film of document.querySelectorAll("video.linefilm")) if (film.muted && !film.dataset.beside) { film.muted = false; report("sound allowed by a tap"); $("said").classList.remove("on"); }
});

// a film that is still being made, played while it grows (Media Source Extensions; ManagedMediaSource on an iPhone)
const FILM = 'video/mp4; codecs="avc1.64001f,mp4a.40.2"';
const GROWING = [window.ManagedMediaSource, window.MediaSource].find((one) => one && one.isTypeSupported && one.isTypeSupported(FILM)) || null;
let growOk = !!GROWING;
async function growing(path, video, silent) {
  const source = new GROWING(), address = URL.createObjectURL(source);
  video.disableRemotePlayback = true;   // an iPhone asks for this beside ManagedMediaSource
  video.src = address;
  await new Promise((open, fail) => { source.addEventListener("sourceopen", open, { once: true }); setTimeout(fail, 5000); });
  const buffer = source.addSourceBuffer(FILM), reader = (await fetch(path)).body.getReader();
  const add = (piece) => new Promise((added, fail) => {
    buffer.addEventListener("updateend", added, { once: true }); buffer.addEventListener("error", fail, { once: true }); buffer.appendBuffer(piece); });
  let started = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    await add(value);
    if (!started && buffer.buffered.length && buffer.buffered.end(0) > 0.3) { started = true; await begin(video, "growing", silent); }
  }
  if (source.readyState === "open") source.endOfStream();
  if (!started) await begin(video, "growing, short", silent);
  video.addEventListener("ended", () => URL.revokeObjectURL(address), { once: true });
}

// the cast's lines: each new one is spoken once, in that character's voice
async function speakNew() {
  for (const entry of state.log || []) {
    if (entry.kind !== "line" || entry.clock <= spokenTo) continue;
    spokenTo = entry.clock;
    if (entry.film) continue;   // spoken in the picture itself
    caption(`<b>${safe(entry.who || "")}</b> ${safe(entry.text)}`, 6000);
    try { const play = await lineOf(entry); if (play) await play(); }
    catch (wrong) { report(`line failed: ${wrong && wrong.name} ${wrong && wrong.message}`); /* the line stays on the screen, unspoken */ }
  }
}
// a line made ready (its voice, and its film where the character's lips follow): gives what plays it now
let hushLine = null;
async function lineOf(entry) {
      const idle = $("idle"), card = cards[entry.who], lipsOf = (state.set && state.set.lips) || {};
      const synced = card ? (lipsOf.faces || {})[entry.who] === "ready" && !!card.file : lipsOf.phase === "ready" && idle.classList.contains("on");
      const reply = await fetch("/api/live/say", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: entry.text, voice: entry.voice, who: entry.who, lips: !!synced, stream: !!synced && growOk, start: synced && !card ? Math.round(idle.currentTime * 24) : 0 }) });
      if (!reply.ok) { report(`the line was refused: ${reply.status}`); return null; }
      const kind = reply.headers.get("Content-Type") || "";
      let film = kind.startsWith("video/"), url = null, next = Number(reply.headers.get("X-Lips-Next")) || 0, grows = null, buffer = null;
      if (kind.startsWith("application/json")) {   // the film is still being made: it is played while it grows
        const told = await reply.json();
        if (!told.film) return null;
        film = true; next = told.next || 0; grows = told.film;
        if (sound && told.sound) buffer = await decoded(await (await fetch(told.sound)).arrayBuffer());
      } else {
        const bytes = await reply.arrayBuffer();
        url = URL.createObjectURL(new Blob([bytes], { type: kind }));
        if (sound) buffer = await decoded(bytes.slice(0));   // the film's own sound track, or the WAV
      }
  return () => {
      // the character says it: their own lips in step with the words, silent, and the words through the page's sound
      closeRush();   // they answer: the take that was showing gives the screen back
      const voice = film ? (card ? card.film : $("lips")) : buffer ? null : new Audio(url);
      if (film && url) voice.src = url;
      if (film) { if (buffer) voice.dataset.beside = "1"; else delete voice.dataset.beside; }
      return new Promise((done) => {
        let ended = false, hush = null, began = 0, counted = false;
        const over = () => {
          if (ended) return; ended = true;
          if (hushLine === over) hushLine = null;
          if (hush) hush(); if (waiting === hear) waiting = null;
          if (voice && voice.pause) { try { voice.pause(); } catch (_) { /* gone */ } }
          if (counted) speaking = Math.max(0, speaking - 1);
          $("set").classList.toggle("talking", speaking > 0 && !card);
          if (card) { card.card.classList.remove("speaking"); if (film) voice.classList.remove("on"); }
          else if (film) {   // back to the listening loop where the line left her
            idle.currentTime = next / 24; idle.play().catch(() => {});
            voice.classList.remove("on"); $("set").classList.remove("lipping");
          }
          if (url) URL.revokeObjectURL(url);
          turn();
          done();
        };
        hushLine = over;
        const onplay = () => {
          if (!counted) { counted = true; speaking += 1; }
          speakingWho = entry.who || "";
          if (card) { card.card.classList.add("speaking"); card.note.textContent = "speaking"; if (film) voice.classList.add("on"); }
          else { $("set").classList.add("talking"); if (film) { voice.classList.add("on"); $("set").classList.add("lipping"); } }
          turn();
        };
        // the words, from where the film is now (or from the start with no film); asked again by a click when the output was still shut
        const hear = () => {
          if (ended || hush || !buffer) return;
          if (sound.state !== "running") {   // shut until the page was clicked once (any button counts); asked for anyway, which an open microphone allows
            waiting = hear; sound.resume().then(() => { if (waiting === hear) hear(); }).catch(() => {});
            setTimeout(() => { if (waiting === hear && !ended) { caption("Click anywhere once to hear the cast.", 15000); report(`sound waits for a click (${sound.state})`); } }, 400);
            return;
          }
          waiting = null; $("said").classList.remove("on");
          const at = film ? voice.currentTime : (performance.now() - began) / 1000;
          hush = voiced(buffer, at, film ? null : over);
          report(`sound played at ${at.toFixed(2)} s`);
        };
        if (voice) { voice.onplay = () => { onplay(); began = performance.now(); hear(); }; voice.onended = voice.onerror = over; }
        setTimeout(() => { if (!ended) { report("the line never ended"); over(); } }, 45000);   // a film that stalls must not hold the lines after it
        if (!voice) { onplay(); began = performance.now(); hear(); if (!hush) setTimeout(over, buffer.duration * 1000 + 200); }   // sound only
        else if (grows) growing(grows, voice, !!buffer).catch((wrong) => { report(`growing film failed: ${wrong && wrong.name} ${wrong && wrong.message}`); growOk = false; over(); });   // this browser cannot: whole films from now on
        else begin(voice, film ? "whole film" : "sound only", film && !!buffer).catch(over);
      });
  };
}

// ------------------------------------------------------------------------------------------------ directing
async function post(path, body, raw) {
  const reply = await fetch(path, { method: "POST", headers: raw ? { "Content-Type": "application/octet-stream" } : { "Content-Type": "application/json" },
    body: raw || JSON.stringify(body || {}) });
  let said = {};
  try { said = await reply.json(); } catch (_) { /* no body */ }
  if (!reply.ok && said.error) caption(safe(said.error), 5000);
  return said;
}

$("start").onclick = async () => {
  const scene = $("scene").value.trim();
  if (!scene) { $("scene").focus(); return; }
  $("start").disabled = true;
  await post("/api/live/start", { scene, seed: Number($("seed").value) || 0 });
};
$("example").onclick = () => { $("scene").value = EXAMPLE; };
$("stop").onclick = () => post("/api/live/note", { text: "cut" });
$("action").onclick = () => {
  const film = $("runfilm");   // inside the click, so the browser lets the run-through's film play by itself afterwards
  if (film.getAttribute("src")) film.play().then(() => { if (!going || !going.rolling) film.pause(); }).catch(() => {});
  return post("/api/live/action");
};
$("print").onclick = () => post("/api/live/film");
$("plan").onclick = async () => {
  const scene = $("scene").value.trim();
  if (!scene) { $("scene").focus(); return; }
  await post("/api/live/plan", { scene, seed: Number($("seed").value) || 0 });
};
$("take").onclick = () => post("/api/live/take");
$("noteForm").onsubmit = async (event) => {
  event.preventDefault();
  const text = $("note").value.trim();
  if (!text) return;
  $("note").value = "";
  thinking += 1; turn();
  let said = {};
  try { said = await post("/api/live/note", { text }); } finally { thinking = Math.max(0, thinking - 1); turn(); }
  if (said.ack && !said.line) caption(safe(said.ack), 2500);
};
const orders = { action: "Action!", cut: "Cut!", film: "Rendering the take." };

// hold the button, or the space bar, to speak a direction
let recorder = null, canSpeak = false;
async function listen() {
  if (recorder || !canSpeak) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const source = context.createMediaStreamSource(stream), tap = context.createScriptProcessor(4096, 1, 1), chunks = [];
    tap.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(tap); tap.connect(context.destination);
    recorder = { stream, context, tap, source, chunks };
    $("mic").classList.add("on");
    caption('<span class="heard">listening…</span>', 30000);
  } catch (_) { caption("The microphone is not available (it needs permission and a secure page).", 5000); }
}
async function heard() {
  if (!recorder) return;
  const { stream, context, tap, source, chunks } = recorder; recorder = null;
  $("mic").classList.remove("on");
  source.disconnect(); tap.disconnect(); stream.getTracks().forEach((t) => t.stop());
  const rate = context.sampleRate; context.close();
  const all = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0; for (const c of chunks) { all.set(c, at); at += c.length; }
  const out = new Float32Array(Math.floor(all.length * 16000 / rate));   // to 16 kHz, straight lines between samples
  for (let i = 0; i < out.length; i++) { const x = i * rate / 16000, a = Math.floor(x), b = Math.min(a + 1, all.length - 1); out[i] = all[a] + (all[b] - all[a]) * (x - a); }
  if (out.length < 4000) { $("said").classList.remove("on"); return; }
  caption('<span class="heard">…</span>', 8000);
  thinking += 1; turn();
  let said = {};
  try { said = await post("/api/live/hear", null, out.buffer); } finally { thinking = Math.max(0, thinking - 1); turn(); }
  if (said.command) caption(`<b>${orders[said.command]}</b>`, 2500);
  else if (said.heard) caption(`<span class="heard">“${safe(said.heard)}”</span>${said.ack && !said.line ? " · " + safe(said.ack) : ""}`, 3500);
}
// the open microphone: click the button and just talk; the server finds the sentences (Silero VAD) and takes each as a direction
let open = null;
const soundOn = () => { const rush = $("rush"); return !!(rush.classList.contains("on") && !rush.paused && !rush.muted && !rush.ended); };
async function openMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const source = context.createMediaStreamSource(stream), tap = context.createScriptProcessor(4096, 1, 1);
    const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live/mic");
    socket.binaryType = "arraybuffer";
    const rate = context.sampleRate;
    let sound = false, muted = false;
    tap.onaudioprocess = (e) => {
      if (socket.readyState !== 1) return;
      const quiet = (speaking > 0 && state.phase !== "running") || !!recorder;   // in a run-through "cut" is heard over the cast                 // the character is talking, or the space bar has the microphone
      if (quiet !== muted) { muted = quiet; if (quiet) { socket.send(JSON.stringify({ drop: true })); $("mic").classList.remove("hears"); } }
      if (quiet) { level(0); return; }
      { const all = e.inputBuffer.getChannelData(0); let sum = 0; for (let i = 0; i < all.length; i += 8) sum += all[i] * all[i]; level(Math.sqrt(sum / (all.length / 8)) * 6); }
      if (soundOn() !== sound) { sound = soundOn(); socket.send(JSON.stringify({ sound })); }
      const all = e.inputBuffer.getChannelData(0), out = new Int16Array(Math.floor(all.length * 16000 / rate));
      for (let i = 0; i < out.length; i++) { const x = i * rate / 16000, a = Math.floor(x), b = Math.min(a + 1, all.length - 1); out[i] = Math.max(-1, Math.min(1, all[a] + (all[b] - all[a]) * (x - a))) * 32767; }
      socket.send(out.buffer);
    };
    socket.onmessage = (message) => {
      const said = JSON.parse(message.data);
      if (said.type === "speech") { $("mic").classList.toggle("hears", said.on); thinking = said.on ? 0 : 1; if (!said.on) setTimeout(() => { thinking = 0; turn(); }, 12000); turn(); return; }
      thinking = 0; $("mic").classList.remove("hears"); turn();
      if (false) { /* the band says who has the turn */ }
      else if (said.type === "ignored") caption(`<span class="heard">“${safe(said.heard || "")}”</span> · not taken as a direction`, 3000);
      else if (said.command) caption(`<b>${orders[said.command]}</b>`, 2500);
      else if (said.error) caption(`<span class="heard">“${safe(said.heard)}”</span> · ${safe(said.error)}`, 4000);
      else if (said.heard) caption(`<span class="heard">“${safe(said.heard)}”</span>${said.ack && !said.line ? " · " + safe(said.ack) : ""}`, 3500);
    };
    socket.onclose = () => { if (open && open.socket === socket) closeMic(); };
    source.connect(tap); tap.connect(context.destination);
    open = { stream, context, tap, source, socket };
    $("mic").classList.add("open"); $("mic").setAttribute("aria-pressed", "true"); document.body.classList.add("micopen"); turn();
    try { localStorage.setItem("live.mic", "open"); } catch (_) { /* a private window */ }
  } catch (_) { caption("The microphone is not available (it needs permission and a secure page).", 5000); }
}
function closeMic() {
  if (!open) return;
  const { stream, context, tap, source, socket } = open; open = null;
  source.disconnect(); tap.disconnect(); stream.getTracks().forEach((t) => t.stop()); context.close(); try { socket.close(); } catch (_) { /* closed */ }
  $("mic").classList.remove("open", "hears"); $("mic").setAttribute("aria-pressed", "false"); document.body.classList.remove("micopen"); levels.fill(0); level(0); thinking = 0; turn();
}
$("mic").onclick = () => { if (open) { closeMic(); try { localStorage.setItem("live.mic", ""); } catch (_) { /* a private window */ } } else openMic(); };
for (const one of $("quality").children) one.onclick = () => post("/api/live/quality", { quality: one.dataset.quality });
$("blocking").onclick = () => $("blocking").classList.toggle("big");
const typing = () => ["INPUT", "TEXTAREA"].includes(document.activeElement && document.activeElement.tagName);
addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat && !typing()) { e.preventDefault(); listen(); } });
addEventListener("keyup", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); heard(); } });

$("credits").addEventListener("toggle", async () => {
  if (!$("credits").open || $("credits").dataset.loaded) return;
  try {
    const list = await (await fetch("/api/credits")).json(), box = $("clines"); box.textContent = "";
    for (const c of list) { const name = document.createElement("b"); name.textContent = c.name; box.append(name, ` by ${c.by}: ${c.what} (${c.licence})\n`); }
    $("credits").dataset.loaded = "1";
  } catch (_) { /* the short line stays */ }
});

connect();
requestAnimationFrame(tick);
render();
