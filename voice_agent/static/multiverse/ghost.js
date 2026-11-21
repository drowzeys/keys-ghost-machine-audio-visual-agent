// keys Ghost Machine Studio: the Ghost. Boo's face (Three.js), its moods and the live voice (microphone in, the
// reply's audio out, over the app's /ws), taken from the classic page (voice_agent/static/index.html) with its
// camera, hand tracking and video panel left out: the same protocol, the same face, the same voice handling.
//
//   import * as ghost from "./ghost.js";
//   ghost.mount(hostElement, { state, status, mic, meta, transcript, reply, event });
//   ghost.toggleMic(); ghost.sayText("..."); ghost.reset();
//
// The hooks: state(s, label) the face's state; status(text) a short status line; mic(on); meta(text) a note for the
// transcript; transcript(text) the person's words; reply(text, source) the Ghost's words, piece by piece; event(e)
// every other JSON event from the server (e.g. {"type": "video", "action": "multiverse", ...}).

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const mix = (a, b, t) => a + (b - a) * t;
const store = { get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} } };
let ws, ctx, outNode, micNode, micAnalyser, stream, playing = [], playAt = 0, outRate = 44100;
let booHost = null;
const noop = () => {};
let hooks = { state: noop, status: noop, mic: noop, meta: noop, transcript: noop, reply: noop, event: noop };
function send(obj) { if (ws && ws.readyState === 1) ws.send(typeof obj === "string" || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj)); }
function fallbackFace() { if (booHost) booHost.classList.add("noface"); }

// ---------- Boo's moods: each is a set of targets the body eases toward ----------
const EMO = {
  neutral:   { hue: 268, eyeH: 1.00, happyL: 0, happyR: 0, round: 0, droop: 0.00, curve: 0.35, mouthW: 1.00, mopen: 0.00, asym: 0.00, tilt: 0.00, spark: 1.0, bounce: 0 },
  happy:     { hue: 276, eyeH: 1.00, happyL: 1, happyR: 1, round: 0, droop: 0.00, curve: 0.95, mouthW: 1.15, mopen: 0.05, asym: 0.00, tilt: 0.05, spark: 1.4, bounce: 0.4 },
  excited:   { hue: 282, eyeH: 1.15, happyL: 0.7, happyR: 0.7, round: 0, droop: 0.00, curve: 1.00, mouthW: 1.25, mopen: 0.22, asym: 0.00, tilt: 0.00, spark: 2.2, bounce: 1 },
  curious:   { hue: 262, eyeH: 1.08, happyL: 0, happyR: 0, round: 0.2, droop: 0.00, curve: 0.15, mouthW: 0.80, mopen: 0.00, asym: 0.28, tilt: 0.16, spark: 1.2, bounce: 0 },
  thinking:  { hue: 258, eyeH: 0.80, happyL: 0, happyR: 0, round: 0, droop: 0.00, curve: -0.05, mouthW: 0.70, mopen: 0.00, asym: 0.12, tilt: -0.08, spark: 1.6, bounce: 0 },
  surprised: { hue: 272, eyeH: 1.25, happyL: 0, happyR: 0, round: 1, droop: 0.00, curve: 0.00, mouthW: 0.75, mopen: 0.45, asym: 0.00, tilt: 0.00, spark: 1.8, bounce: 0.3 },
  concerned: { hue: 262, eyeH: 0.92, happyL: 0, happyR: 0, round: 0, droop: 0.45, curve: -0.45, mouthW: 0.85, mopen: 0.00, asym: 0.00, tilt: 0.08, spark: 0.8, bounce: 0 },
  sad:       { hue: 256, eyeH: 0.78, happyL: 0, happyR: 0, round: 0, droop: 0.85, curve: -0.85, mouthW: 0.80, mopen: 0.00, asym: 0.00, tilt: 0.12, spark: 0.5, bounce: 0 },
  playful:   { hue: 284, eyeH: 1.00, happyL: 0, happyR: 1, round: 0, droop: 0.00, curve: 0.85, mouthW: 1.10, mopen: 0.06, asym: 0.00, tilt: -0.14, spark: 1.6, bounce: 0.5 },
};
let emotion = "neutral", target = EMO.neutral, mood = { ...EMO.neutral }, emoTimer = null;
function setEmotion(e, holdMs) {
  if (!EMO[e]) return;
  clearTimeout(emoTimer); emotion = e; target = EMO[e];
  if (holdMs) emoTimer = setTimeout(() => setEmotion("neutral"), holdMs);
}

let state = "idle", stateAt = 0, endPending = false;
const LABEL = { idle: "standby", listening: "listening", thinking: "thinking", speaking: "speaking", interrupted: "interrupted" };
function setState(s) { state = s; stateAt = performance.now(); hooks.state(s, LABEL[s] || s); if (s === "interrupted") blinkAt = performance.now(); }

let gaze = { x: 0, y: 0 }, gazeTarget = { x: 0, y: 0 }, nextWander = 0, blinkAt = performance.now() + 2500, micLevel = 0, speakLevel = 0;
let tracker = null, trackerLoading = null, lastDetect = 0, lastFaceAt = -1e9, facePresent = false, faceGaze = null;
const activeTasks = new Map();
let orbUntil = 0;
/*AUDIO*/
// "Futuristic": Boo's voice through a light chorus, a short airy echo and a treble lift, all in the browser (no cost
// on the server; off, the voice plays untouched)
let fxIn = null, fxOn = store.get("boo.fx") === "1";
function buildFx() {
  const input = ctx.createGain(), mix = ctx.createGain();
  input.connect(mix);  // dry
  for (const [ms, rate] of [[22, 0.25], [37, 0.33]]) {  // chorus: two slowly wobbling delays
    const d = ctx.createDelay(0.1), g = ctx.createGain(), lfo = ctx.createOscillator(), depth = ctx.createGain();
    d.delayTime.value = ms / 1000; g.gain.value = 0.32; lfo.frequency.value = rate; depth.gain.value = 0.0018;
    lfo.connect(depth).connect(d.delayTime); lfo.start(); input.connect(d).connect(g).connect(mix);
  }
  for (const [ms, level] of [[55, 0.14], [110, 0.08]]) {  // short airy echo
    const d = ctx.createDelay(0.2), g = ctx.createGain(); d.delayTime.value = ms / 1000; g.gain.value = level;
    input.connect(d).connect(g).connect(mix);
  }
  const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 110;
  const air = ctx.createBiquadFilter(); air.type = "highshelf"; air.frequency.value = 6500; air.gain.value = 4;
  const lim = ctx.createDynamicsCompressor(); lim.threshold.value = -3; lim.ratio.value = 20; lim.attack.value = 0.003; lim.release.value = 0.1;
  const trim = ctx.createGain(); trim.gain.value = 0.8;
  mix.connect(hp).connect(air).connect(trim).connect(lim).connect(outNode);
  return input;
}
function ensureAudio() {
  if (!ctx) ctx = new AudioContext();
  if (!outNode) { outNode = ctx.createAnalyser(); outNode.fftSize = 1024; outNode.connect(ctx.destination); }
  if (!fxIn) fxIn = buildFx();
  return ctx;
}
function stopAudio() { for (const s of playing) { try { s.stop(); } catch (_) {} } playing = []; playAt = 0; }
function play(buf) {
  ensureAudio();
  const pcm = new Int16Array(buf), f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
  const b = ctx.createBuffer(1, f.length, outRate); b.copyToChannel(f, 0);
  const s = ctx.createBufferSource(); s.buffer = b; s.connect(fxOn ? fxIn : outNode);
  playAt = Math.max(playAt, ctx.currentTime + 0.03); s.start(playAt); playAt += b.duration;
  playing.push(s); s.onended = () => { playing = playing.filter((x) => x !== s); };
}
const tbuf = new Float32Array(1024);
function rms(an) { if (!an) return 0; an.getFloatTimeDomainData(tbuf); let s = 0; for (let i = 0; i < tbuf.length; i++) s += tbuf[i] * tbuf[i]; return Math.sqrt(s / tbuf.length); }

/*MIC*/
// ---------- microphone -> 16 kHz int16 frames, in an AudioWorklet (echo cancellation keeps the agent's own voice out) ----------
const worklet = `class Down extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.acc = 0; this.out = []; }
  process(inputs) { const x = inputs[0][0]; if (!x) return true;
    for (let i = 0; i < x.length; i++) { this.acc += 1; if (this.acc >= this.ratio) { this.acc -= this.ratio; this.out.push(Math.max(-1, Math.min(1, x[i])) * 32767); } }
    if (this.out.length >= 512) { this.port.postMessage(Int16Array.from(this.out).buffer, []); this.out = []; }
    return true; } }
registerProcessor("down", Down);`;
let workletLoaded = false;

function micUi(on) { hooks.mic(on); }
// The microphone, kept alive: iPhone Safari ends or mutes a running mic track when the camera starts or stops, and
// interrupts the AudioContext; while the person wants the mic on, a dead track is reopened and the context resumed.
let micWanted = false, micOpening = null;
function closeMic() {
  if (micNode) micNode.disconnect();
  if (stream) stream.getTracks().forEach((t) => { t.onended = t.onmute = null; t.stop(); });
  micNode = null; micAnalyser = null; stream = null;
}
async function openMic() {
  if (micOpening) return micOpening;
  micOpening = (async () => {
    closeMic(); ensureAudio(); await ctx.resume().catch(() => {});
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    if (!workletLoaded) { await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([worklet], { type: "text/javascript" }))); workletLoaded = true; }
    const src = ctx.createMediaStreamSource(stream);
    micNode = new AudioWorkletNode(ctx, "down");
    micNode.port.onmessage = (m) => send(m.data);
    src.connect(micNode);
    micAnalyser = ctx.createAnalyser(); micAnalyser.fftSize = 1024; src.connect(micAnalyser);
    for (const t of stream.getAudioTracks()) t.onended = t.onmute = () => healMic();
  })().finally(() => { micOpening = null; });
  return micOpening;
}
let healTimer = null;
function healMic() {  // reopen a mic that died under us, a moment later (Safari fires ended/mute in bursts)
  if (!micWanted) return;
  clearTimeout(healTimer);
  healTimer = setTimeout(async () => {
    const t = stream && stream.getAudioTracks()[0];
    if (t && t.readyState === "live" && !t.muted && ctx && ctx.state === "running") return;
    try { await openMic(); hooks.status("listening"); } catch (err) { hooks.meta("microphone lost: " + err.message + " — tap Start talking"); micWanted = false; micUi(false); }
  }, 350);
}
function healAudio() {  // an interrupted AudioContext (camera, calls, Siri) is resumed; if Safari wants a tap, the next tap does it
  if (!ctx || ctx.state === "running") return;
  ctx.resume().catch(() => {});
  if (ctx.state !== "running") document.addEventListener("pointerdown", () => ctx.resume().catch(() => {}), { once: true });
}
/*BOO*/
const THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js";
let boo = null, coreSpin = 0.35;  // the chest hexagon's spin speed, rad/s

// the body's radius at height y: a dome on top, a skirt that flares toward the hem
function R(y) { return y > 0.25 ? Math.sqrt(Math.max(0, 1 - (y - 0.25) ** 2)) : 1 + 0.1 * ((0.25 - y) / 1.25) ** 2; }

const HEM = `
  float hem = 1.0 - smoothstep(-1.0, -0.2, pos.y);
  float ang = atan(pos.x, pos.z);
  pos.y += hem * sin(ang * 7.0 + uTime * 2.4) * 0.075;
  pos.xz *= 1.0 + hem * 0.07 * sin(ang * 3.0 - uTime * 1.6);`;

const FACE_FS = `
uniform float uTime, uOpenL, uOpenR, uHappyL, uHappyR, uRound, uDroop, uMouthOpen, uCurve, uMouthW, uEyeH, uAsym, uFlicker;
uniform vec2 uLook; uniform vec3 uEye, uMouth; varying vec2 vUv;
float box(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
float arc(vec2 p, vec2 sc, float ra, float rb) { p.x = abs(p.x); return ((sc.y * p.x > sc.x * p.y) ? length(p - sc * ra) : abs(length(p) - ra)) - rb; }
float eye(vec2 p, float side, float open, float happy, float size) {
  float a = -side * uDroop * 0.35, c = cos(a), s = sin(a);
  vec2 q = mat2(c, -s, s, c) * p;
  float w = 0.085 * size, h = mix(0.012, 0.135 * uEyeH * size, open);
  float d = box(q, vec2(w, h), min(w, h));
  d = mix(d, length(q) - 0.115 * uEyeH * size, uRound * open);
  d = mix(d, arc(q - vec2(0.0, -0.06), vec2(sin(1.05), cos(1.05)), 0.1 * size, 0.024), happy);
  float lid = p.y - h * (1.0 - 0.7 * uDroop) + side * p.x * uDroop * 0.6;
  return mix(d, max(d, lid), step(0.01, uDroop) * (1.0 - happy));
}
float mouth(vec2 m) {
  float ra = 0.1 * uMouthW, ap = mix(0.25, 1.15, clamp(abs(uCurve), 0.0, 1.0));
  vec2 q = uCurve >= 0.0 ? vec2(m.x, -(m.y - ra * 0.8)) : vec2(m.x, m.y + ra * 0.8);
  float d = mix(box(m, vec2(0.07 * uMouthW, 0.012), 0.012), arc(q, vec2(sin(ap), cos(ap)), ra, 0.016), smoothstep(0.05, 0.3, abs(uCurve)));
  float o = clamp(uMouthOpen, 0.0, 1.0);
  vec2 ax = vec2(0.05 * uMouthW + 0.03 * o, 0.014 + 0.08 * o);
  return mix(d, (length(m / ax) - 1.0) * min(ax.x, ax.y), smoothstep(0.03, 0.18, o));
}
void main() {
  vec2 p = (vUv - 0.5) * vec2(1.3, 1.0), look = uLook * vec2(0.05, 0.04);
  float de = min(eye(p - vec2(-0.3, 0.12) - look, -1.0, uOpenL, uHappyL, 1.0 + uAsym),
                 eye(p - vec2(0.3, 0.12) - look, 1.0, uOpenR, uHappyR, 1.0 - uAsym * 0.6));
  float dm = mouth(p - vec2(0.0, -0.17) - look * 0.5);
  float ge = smoothstep(0.005, -0.004, de) + exp(-max(de, 0.0) * 26.0) * 0.5;
  float gm = smoothstep(0.005, -0.004, dm) + exp(-max(dm, 0.0) * 30.0) * 0.4;
  gl_FragColor = vec4((uEye * ge * 1.25 + uMouth * gm * 1.1) * uFlicker, clamp(max(ge, gm), 0.0, 1.0));
}`;

function glowTexture(THREE) {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.25, "rgba(255,255,255,.55)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const additive = (THREE, extra) => ({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, ...extra });

async function initBoo() {
  let THREE, renderer;
  try { THREE = await import(THREE_URL); } catch (err) { fallbackFace(); hooks.meta("3D unavailable: " + err.message); return; }
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" }); }
  catch (err) { fallbackFace(); hooks.meta("WebGL unavailable"); return; }
  const host = booHost;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15;
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 60);
  const look = new THREE.Vector3(0, 0.22, 0);

  const time = { value: 0 };
  const glow = glowTexture(THREE);
  const body = new THREE.Group(); scene.add(body);

  // body: a lathe of the ghost's silhouette, the hem waving in the vertex shader
  const prof = [];
  for (let i = 0; i <= 44; i++) { const y = 1.25 - (i / 44) * 2.25; prof.push(new THREE.Vector2(Math.max(0.0001, R(y)), y)); }
  const shape = new THREE.LatheGeometry(prof, 72);
  // a faint ghost: mostly see-through, brighter only at its edges (fresnel), fading to nothing toward the hem; unlit,
  // so there are no highlights or reflections
  const skin = new THREE.ShaderMaterial({
    uniforms: { uTime: time, uTop: { value: new THREE.Color(0x5b2a9e) }, uBottom: { value: new THREE.Color(0x24104a) },
                uRim: { value: new THREE.Color(0x8e6bff) }, uAlpha: { value: 0.16 }, uEnergy: { value: 1.0 } },
    vertexShader: `uniform float uTime; varying vec3 vN, vV; varying float vY; void main() { vec3 pos = position; ${HEM}
      vY = position.y; vec4 mv = modelViewMatrix * vec4(pos, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform float uTime, uAlpha, uEnergy; uniform vec3 uTop, uBottom, uRim; varying vec3 vN, vV; varying float vY; void main() {
      float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
      vec3 col = mix(uBottom, uTop, smoothstep(-1.0, 1.2, vY)) + uRim * f * 0.8 * uEnergy;
      float fade = smoothstep(-1.1, -0.1, vY), wisp = 0.88 + 0.12 * sin(vY * 7.0 - uTime * 1.4);
      gl_FragColor = vec4(col, (uAlpha + f * 0.5 * uEnergy) * fade * wisp); }`,
    transparent: true, depthWrite: false });
  body.add(new THREE.Mesh(shape, skin));

  // fresnel rim: a slightly larger shell that only glows at the edges
  const rimMat = new THREE.ShaderMaterial(additive(THREE, {
    uniforms: { uTime: time, uColor: { value: new THREE.Color(0x7a5cff) }, uPower: { value: 1.0 } },
    vertexShader: `uniform float uTime; varying float vRim, vFade; void main() { vec3 pos = position; ${HEM}
      vFade = smoothstep(-1.1, -0.1, position.y);
      vec4 mv = modelViewMatrix * vec4(pos, 1.0); vec3 n = normalize(normalMatrix * normal); vRim = pow(1.0 - abs(dot(n, normalize(-mv.xyz))), 3.0);
      gl_Position = projectionMatrix * mv; }`,
    fragmentShader: "uniform vec3 uColor; uniform float uPower; varying float vRim, vFade; void main() { float b = vRim * uPower * vFade * 0.55; gl_FragColor = vec4(uColor * b, b); }" }));
  const rim = new THREE.Mesh(shape, rimMat); rim.scale.setScalar(1.035); body.add(rim);

  // face: eyes and mouth drawn as glowing shapes on a patch bent onto the dome
  const facePlane = new THREE.PlaneGeometry(1.3, 1.0, 26, 20), fp = facePlane.attributes.position;
  for (let i = 0; i < fp.count; i++) { const x = fp.getX(i), y = fp.getY(i) + 0.27; fp.setXYZ(i, x, y, Math.sqrt(Math.max(0.0001, R(y) ** 2 - x * x)) + 0.012); }
  const faceU = { uTime: time, uOpenL: { value: 1 }, uOpenR: { value: 1 }, uHappyL: { value: 0 }, uHappyR: { value: 0 }, uRound: { value: 0 }, uDroop: { value: 0 },
    uMouthOpen: { value: 0 }, uCurve: { value: 0.35 }, uMouthW: { value: 1 }, uEyeH: { value: 1 }, uAsym: { value: 0 }, uFlicker: { value: 1 },
    uLook: { value: new THREE.Vector2() }, uEye: { value: new THREE.Color(0xbff8ff) }, uMouth: { value: new THREE.Color(0xd8ccff) } };
  body.add(new THREE.Mesh(facePlane, new THREE.ShaderMaterial(additive(THREE, { uniforms: faceU, fragmentShader: FACE_FS,
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }" }))));

  // circuit traces on the back and sides, with pulses running along them
  const tp = [], ta = [], nodes = [];
  const surf = (phi, y) => { const r = R(y) * 1.012; return [r * Math.sin(phi), y, r * Math.cos(phi)]; };
  for (let i = 0; i < 22; i++) {
    const side = i % 3 === 0, sign = Math.random() < 0.5 ? -1 : 1;
    let phi = side ? sign * (0.62 + Math.random() * 0.4) : 0.95 + Math.random() * (Math.PI * 2 - 1.9);
    let y = side ? -0.12 + Math.random() * 0.62 : -0.12 + Math.random() * 1.1, along = Math.random();
    const segs = 3 + Math.floor(Math.random() * 4);
    nodes.push(...surf(phi, y));
    for (let s = 0; s < segs; s++) {
      const dPhi = s % 2 === 0 ? (Math.random() < 0.5 ? -1 : 1) * (0.15 + Math.random() * 0.3) : 0;
      const dY = s % 2 === 1 ? (Math.random() < 0.5 ? -1 : 1) * (0.08 + Math.random() * 0.2) : 0;
      const nPhi = side ? sign * clamp(Math.abs(phi + dPhi), 0.58, 1.3) : clamp(phi + dPhi, 0.9, Math.PI * 2 - 0.9), nY = clamp(y + dY, -0.15, side ? 0.55 : 1.08);
      for (let k = 0; k < 5; k++) {
        tp.push(...surf(mix(phi, nPhi, k / 5), mix(y, nY, k / 5)), ...surf(mix(phi, nPhi, (k + 1) / 5), mix(y, nY, (k + 1) / 5)));
        ta.push(along + (k / 5) * 0.12, along + ((k + 1) / 5) * 0.12);
      }
      along += 0.12; phi = nPhi; y = nY;
    }
    nodes.push(...surf(phi, y));
  }
  for (let k = 0; k < 96; k++) {  // the waist seam, a full ring between the mouth and the core
    tp.push(...surf((k / 96) * Math.PI * 2, -0.13), ...surf(((k + 1) / 96) * Math.PI * 2, -0.13)); ta.push(k / 24, (k + 1) / 24);
  }
  const traceGeo = new THREE.BufferGeometry();
  traceGeo.setAttribute("position", new THREE.Float32BufferAttribute(tp, 3)); traceGeo.setAttribute("along", new THREE.Float32BufferAttribute(ta, 1));
  const traceMat = new THREE.ShaderMaterial(additive(THREE, { uniforms: { uTime: time, uColor: { value: new THREE.Color(0x4ff0ff) }, uSpeed: { value: 0.35 } },
    vertexShader: "attribute float along; varying float vA; void main() { vA = along; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "uniform float uTime, uSpeed; uniform vec3 uColor; varying float vA; void main() { float p = smoothstep(0.86, 1.0, fract(vA * 1.4 - uTime * uSpeed)); float b = 0.1 + p * 0.85; gl_FragColor = vec4(uColor * b, b); }" }));
  body.add(new THREE.LineSegments(traceGeo, traceMat));
  const nodeGeo = new THREE.BufferGeometry(); nodeGeo.setAttribute("position", new THREE.Float32BufferAttribute(nodes, 3));
  const nodeMat = new THREE.PointsMaterial(additive(THREE, { size: 0.05, map: glow, color: 0x9ff8ff, opacity: 0.55 }));
  body.add(new THREE.Points(nodeGeo, nodeMat));

  // chest core: an electric-blue hexagonal ring that spins faster with the voice (kept quiet beside the eyes and mouth)
  const core = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.095, 6), new THREE.MeshBasicMaterial(additive(THREE, { color: 0x2f7dff, opacity: 0.85, side: THREE.DoubleSide })));
  core.position.set(0, -0.32, R(-0.32) + 0.015); body.add(core);
  const coreGlow = new THREE.Sprite(new THREE.SpriteMaterial(additive(THREE, { map: glow, color: 0x1f5fff, opacity: 0.3 })));
  coreGlow.scale.setScalar(0.45); coreGlow.position.copy(core.position); body.add(coreGlow);

  // little arms that wave when Boo is happy
  const armGeo = new THREE.SphereGeometry(0.2, 24, 16), arms = [];
  for (const s of [-1, 1]) {
    const pivot = new THREE.Group(); pivot.position.set(s * 0.9, -0.05, 0.05);
    const arm = new THREE.Mesh(armGeo, skin); arm.scale.set(0.85, 1.45, 0.85); arm.position.set(s * 0.12, -0.12, 0); pivot.add(arm);
    pivot.rotation.z = s * 0.55; body.add(pivot); arms.push(pivot);
  }

  // antenna with a blinking tip, and a halo ring above the head
  const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.02, 0.34, 8), new THREE.MeshBasicMaterial({ color: 0x6a4fb8, transparent: true, opacity: 0.7 }));
  antenna.position.set(0.36, 1.3, 0); antenna.rotation.z = -0.55; body.add(antenna);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 12), new THREE.MeshBasicMaterial({ color: 0x9ff8ff }));
  tip.position.set(0.455, 1.445, 0); body.add(tip);
  const tipGlow = new THREE.Sprite(new THREE.SpriteMaterial(additive(THREE, { map: glow, color: 0x6fefff })));
  tipGlow.scale.setScalar(0.4); tipGlow.position.copy(tip.position); body.add(tipGlow);
  const halo = new THREE.Mesh(new THREE.TorusGeometry(0.4, 0.013, 8, 96), new THREE.MeshBasicMaterial(additive(THREE, { color: 0xb98cff, opacity: 0.85 })));
  halo.position.y = 1.7; halo.rotation.x = Math.PI / 2 - 0.25; body.add(halo);

  // the thinking ring: an arc of data that spins around Boo while it thinks
  const thinkRing = new THREE.Mesh(new THREE.TorusGeometry(1.45, 0.008, 6, 160, Math.PI * 1.4), new THREE.MeshBasicMaterial(additive(THREE, { color: 0x4ff0ff, opacity: 0 })));
  thinkRing.rotation.x = Math.PI / 2 - 0.35; scene.add(thinkRing);

  // data motes orbiting
  const N = 130, motes = new Float32Array(N * 3), mp = [];
  for (let i = 0; i < N; i++) mp.push({ r: 1.45 + Math.random() * 0.9, y: -1.2 + Math.random() * 2.8, a: Math.random() * Math.PI * 2, s: 0.12 + Math.random() * 0.32, w: Math.random() * 6 });
  const moteGeo = new THREE.BufferGeometry(); moteGeo.setAttribute("position", new THREE.BufferAttribute(motes, 3));
  const moteMat = new THREE.PointsMaterial(additive(THREE, { size: 0.055, map: glow, color: 0xd6b8ff, opacity: 0.85 }));
  scene.add(new THREE.Points(moteGeo, moteMat));

  // the agent orb: summoned while a back agent works
  const orb = new THREE.Group();
  orb.add(new THREE.Mesh(new THREE.SphereGeometry(0.075, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffe08a })));
  const orbGlow = new THREE.Sprite(new THREE.SpriteMaterial(additive(THREE, { map: glow, color: 0xffc94f }))); orbGlow.scale.setScalar(0.6); orb.add(orbGlow);
  orb.scale.setScalar(0.001); scene.add(orb);

  // aura behind Boo and a holographic floor it hovers over
  const aura = new THREE.Sprite(new THREE.SpriteMaterial(additive(THREE, { map: glow, color: 0x4b22a8, opacity: 0.4 })));
  aura.scale.set(5.2, 5.2, 1); aura.position.set(0, 0.2, -1.4); scene.add(aura);
  const floorMat = new THREE.ShaderMaterial(additive(THREE, { uniforms: { uTime: time, uColor: { value: new THREE.Color(0x9b5cff) }, uLift: { value: 0 } },
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: `uniform float uTime, uLift; uniform vec3 uColor; varying vec2 vUv; void main() {
      float r = length(vUv - 0.5) * 2.0; float rings = pow(abs(sin((r * 7.0 - uTime * 0.5) * 3.14159)), 30.0);
      float fade = 1.0 - smoothstep(0.2, 1.0, r); float pool = (1.0 - smoothstep(0.0, 0.55 + uLift * 0.1, r)) * (0.55 - uLift * 0.25);
      float b = (rings * 0.45 + pool) * fade; gl_FragColor = vec4(uColor * b, b); }` }));
  const floor = new THREE.Mesh(new THREE.CircleGeometry(2.3, 64), floorMat); floor.rotation.x = -Math.PI / 2; floor.position.y = -1.62; scene.add(floor);

  // frame Boo whatever the stage's shape: the camera backs off until the whole ghost fits
  function fit() {
    const w = host.clientWidth || 1, h = host.clientHeight || 1, aspect = w / h, tan = Math.tan((camera.fov * Math.PI) / 360);
    renderer.setSize(w, h, false); camera.aspect = aspect;
    camera.position.set(0, 0.4, Math.max(4.0 / 2 / tan, 2.7 / 2 / (tan * aspect)));
    camera.lookAt(look); camera.updateProjectionMatrix();
  }
  new ResizeObserver(fit).observe(host); fit();

  const C = new THREE.Color(), hsl = (h, s, l) => C.setHSL((((h % 360) + 360) % 360) / 360, s, l);
  coreSpin = 0.35;
  boo = { renderer, scene, camera, time, body, skin, rimMat, faceU, traceMat, nodeMat, core, coreGlow, arms, tipGlow, halo, thinkRing,
          mp, motes, moteGeo, moteMat, orb, aura, floorMat, hsl };
}

// ---------- the loop: mood easing, levels, gaze, blinks, then Boo ----------
let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  if (document.hidden) { last = now; return; }
  const dt = Math.min(0.1, (now - last) / 1000), t = now / 1000; last = now;
  const k = 1 - Math.exp(-dt * 6);
  for (const key in target) mood[key] += (target[key] - mood[key]) * k;
  const sp = clamp(rms(outNode) * 5.5, 0, 1.2), mic = clamp(rms(micAnalyser) * 6, 0, 1);
  speakLevel += (sp - speakLevel) * (sp > speakLevel ? 0.55 : 0.18);
  micLevel += (mic - micLevel) * (mic > micLevel ? 0.5 : 0.12);
  if (state === "speaking" && endPending && ctx && ctx.currentTime >= playAt) { endPending = false; setState("idle"); hooks.status(micNode ? "listening" : "ready"); }
  // gaze: thinking looks up and aside, a tracked face is followed, otherwise Boo glances around
  if (state === "thinking") gazeTarget = { x: -0.6, y: 0.7 };
  else if (faceGaze && now - lastFaceAt < 2500) gazeTarget = { x: faceGaze.x + Math.sin(t * 0.7) * 0.03, y: -faceGaze.y };
  else if (now > nextWander) { const s = state === "idle" ? 0.5 : 0.22; gazeTarget = { x: (Math.random() * 2 - 1) * s, y: (Math.random() * 2 - 1) * s * 0.5 }; nextWander = now + 1200 + Math.random() * 2800; }
  const gk = 1 - Math.exp(-dt * 8); gaze.x += (gazeTarget.x - gaze.x) * gk; gaze.y += (gazeTarget.y - gaze.y) * gk;
  // blinks: every 2-6 s, quick and doubled on interruption
  let blink = 0, bt = now - blinkAt;
  if (bt >= 0 && bt < 150) blink = Math.sin(Math.PI * bt / 150);
  else if (bt >= 150) blinkAt = state === "interrupted" && bt < 400 ? now + 60 : now + 2200 + Math.random() * 4200;
  if (!boo) return;

  const b = boo, u = b.faceU;
  b.time.value = t;
  const wide = state === "listening" ? 1.06 + micLevel * 0.1 : state === "interrupted" ? 1.12 : 1;
  u.uOpenL.value = (1 - blink) * wide; u.uOpenR.value = (1 - blink) * wide;
  u.uHappyL.value = mood.happyL; u.uHappyR.value = mood.happyR; u.uRound.value = mood.round; u.uDroop.value = mood.droop;
  u.uEyeH.value = mood.eyeH; u.uAsym.value = mood.asym; u.uCurve.value = mood.curve; u.uMouthW.value = mood.mouthW;
  u.uMouthOpen.value = clamp(mood.mopen + (state === "speaking" ? speakLevel * 0.9 : 0), 0, 1.1);
  u.uLook.value.set(gaze.x, gaze.y);
  const flick = state === "interrupted" && now - stateAt < 350 ? 0.55 + Math.random() * 0.45 : 1;
  u.uFlicker.value = flick;

  // colour: the mood's hue across body, rim, traces, aura and floor
  const hue = mood.hue;
  const su = b.skin.uniforms;
  su.uTop.value.copy(b.hsl(hue, 0.58, 0.36)); su.uBottom.value.copy(b.hsl(hue - 4, 0.65, 0.16)); su.uRim.value.copy(b.hsl(hue - 8, 0.85, 0.62));
  b.rimMat.uniforms.uColor.value.copy(b.hsl(hue - 10, 0.9, 0.62));
  b.traceMat.uniforms.uColor.value.copy(b.hsl(192, 0.9, 0.6)); b.nodeMat.color.copy(b.hsl(192, 0.9, 0.7));
  b.halo.material.color.copy(b.hsl(hue - 6, 0.7, 0.62)); b.aura.material.color.copy(b.hsl(hue - 4, 0.75, 0.3));
  u.uEye.value.copy(b.hsl(190 + (hue - 275) * 0.4, 1, 0.84));
  b.floorMat.uniforms.uColor.value.copy(b.hsl(hue - 6, 0.75, 0.5));
  b.moteMat.color.copy(b.hsl(hue - 10, 0.6, 0.72));

  // energy: listening follows the microphone, speaking the voice, thinking pulses
  const level = clamp(state === "speaking" ? speakLevel : state === "listening" ? micLevel : state === "thinking" ? 0.35 + 0.2 * Math.sin(t * 5) : 0.08 + 0.05 * Math.sin(t * 1.3), 0, 1);
  b.rimMat.uniforms.uPower.value = (0.85 + level * 1.1) * flick;
  su.uEnergy.value = (0.8 + level * 0.9) * flick;
  b.aura.material.opacity = 0.22 + level * 0.25;
  b.traceMat.uniforms.uSpeed.value = 0.25 + mood.spark * 0.18 + (state === "thinking" ? 0.6 : 0);
  coreSpin += ((state === "speaking" ? 2.2 + speakLevel * 9 : 0.35) - coreSpin) * Math.min(1, dt * 4);  // eased spin speed
  b.coreGlow.scale.setScalar(0.4); b.core.rotation.z -= dt * coreSpin;
  b.tipGlow.material.opacity = Math.sin(t * 4) > 0.6 ? 1 : 0.35;

  // motion: hover, sway, a turn toward the person, a bounce when excited, arms that wave
  const bounce = mood.bounce * Math.abs(Math.sin(t * (state === "speaking" ? 7 : 3.5))) * 0.09;
  b.body.position.y = Math.sin(t * 1.5) * 0.07 + bounce;
  const toward = faceGaze && now - lastFaceAt < 2500 ? faceGaze.x : 0;
  b.body.rotation.y += ((toward * 0.35 + Math.sin(t * 0.6) * 0.12 + gaze.x * 0.08) - b.body.rotation.y) * gk;
  b.body.rotation.x += ((-gaze.y * 0.06) - b.body.rotation.x) * gk;
  b.body.rotation.z = mood.tilt + Math.sin(t * 0.9) * 0.03;
  const wave = mood.bounce > 0.2 ? Math.sin(t * 6) * 0.35 : Math.sin(t * 1.4) * 0.08;
  b.arms[0].rotation.z = -0.55 - wave; b.arms[1].rotation.z = 0.55 + wave * (state === "speaking" ? 0.6 : 0.3);
  b.halo.rotation.z += dt * (0.5 + (state === "thinking" ? 3 : 0));
  b.floorMat.uniforms.uLift.value = b.body.position.y * 6;

  // the thinking ring fades in and spins
  b.thinkRing.material.opacity += ((state === "thinking" ? 0.75 : 0) - b.thinkRing.material.opacity) * k;
  b.thinkRing.rotation.z += dt * 2.6;

  // motes orbit, faster with energy and mood
  const speed = (0.6 + mood.spark * 0.5 + level * 1.4) * (state === "thinking" ? 2.2 : 1);
  for (let i = 0; i < b.mp.length; i++) {
    const m = b.mp[i]; m.a += dt * m.s * speed;
    b.motes[i * 3] = Math.cos(m.a) * m.r; b.motes[i * 3 + 1] = m.y + Math.sin(t * 0.8 + m.w) * 0.1; b.motes[i * 3 + 2] = Math.sin(m.a) * m.r;
  }
  b.moteGeo.attributes.position.needsUpdate = true;

  // the agent orb, while a back agent works
  const orbOn = activeTasks.size > 0 || now < orbUntil ? 1 : 0;
  b.orb.scale.setScalar(Math.max(0.001, b.orb.scale.x + (orbOn - b.orb.scale.x) * (1 - Math.exp(-dt * 4))));
  b.orb.position.set(Math.cos(t * 1.7) * 1.55, 0.55 + Math.sin(t * 2.3) * 0.25, Math.sin(t * 1.7) * 1.55);

  b.renderer.render(b.scene, b.camera);
}

// ---------- the socket: the same protocol as the classic page ----------
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "arraybuffer";
  ws.onmessage = (m) => {
    if (typeof m.data !== "string") { play(m.data); return; }
    const e = JSON.parse(m.data);
    if (e.type === "ready") { hooks.status("ready"); setState("idle"); send({ type: "name", name: "Ghost" }); }
    else if (e.type === "voices" && e.list && e.list.length) {  // a remembered voice of this device, else the profile's
      const want = store.get(e.engine === "lite" ? "boo.voice.lite" : "boo.voice2");
      if (want && want !== e.current && e.list.some((v) => v.id === want)) send({ type: "voice", name: want, quiet: true });
    }
    else if (e.type === "speech_start") { hooks.status("listening…"); setState("listening"); }
    else if (e.type === "speech_end") { hooks.status("thinking…"); setState("thinking"); }
    else if (e.type === "transcript") { if (e.text) hooks.transcript(e.text); }
    else if (e.type === "reply") hooks.reply(e.text, e.source || "");
    else if (e.type === "audio_start") { outRate = e.sample_rate; endPending = false; setState("speaking"); hooks.status("speaking — talk to interrupt"); }
    else if (e.type === "audio_end") endPending = true;
    else if (e.type === "interrupt") {
      stopAudio(); endPending = false; hooks.status("interrupted"); setState("interrupted");
      setTimeout(() => { if (state === "interrupted") setState(e.why === "barge-in" ? "listening" : "idle"); }, 450);
    }
    else if (e.type === "emotion") setEmotion(e.emotion);
    else if (e.type === "error") { hooks.meta("error: " + e.error); setEmotion("concerned", 2500); if (state === "thinking") setState("idle"); }
    else hooks.event(e);
  };
  ws.onclose = () => { hooks.status("reconnecting…"); hooks.state("offline", "offline"); setTimeout(connect, 1500); };
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) { healAudio(); healMic(); } });

export async function toggleMic() {
  ensureAudio(); await ctx.resume();
  if (micWanted) { micWanted = false; closeMic(); micUi(false); hooks.status("ready"); return false; }
  micWanted = true;
  try { await openMic(); }
  catch (err) { micWanted = false; hooks.meta("microphone unavailable: " + err.message + " (open this page on localhost or https)"); return false; }
  micUi(true); hooks.status("listening");
  return true;
}

export async function sayText(text) {
  if (!text || !text.trim()) return;
  ensureAudio(); await ctx.resume().catch(() => {});
  stopAudio(); send({ type: "text", text: text.trim() }); setState("thinking");
}

export function reset() { stopAudio(); send({ type: "reset" }); setEmotion("neutral"); setState("idle"); }
export { setEmotion };
export function current() { return state; }

let started = false;
export function mount(host, given) {
  booHost = host;
  hooks = { ...hooks, ...(given || {}) };
  if (started) return;
  started = true;
  connect();
  initBoo();
  requestAnimationFrame(frame);
}
