// keys Ghost Multiverse Studio: the tree view's pure part (no DOM), so node can test it (tests/js/tree.test.mjs).
// A film walks left to right through time: x is a node's depth, y its row; a parent sits centred on its children.

export const COL_W = 330;   // one step of the film, left to right
export const ROW_H = 236;   // one branch, top to bottom (room for the Ghost above its card)
export const CARD_W = 248;  // a clip's card (16:9 media and two lines under it)
export const CARD_H = 176;

// children in the order they are drawn, top to bottom: older rounds first, then by letter
export function childrenOf(tree, id) {
  const n = tree.nodes[id];
  if (!n) return [];
  return n.children.map((c) => tree.nodes[c]).filter(Boolean)
    .sort((a, b) => (a.round - b.round) || String(a.letter).localeCompare(String(b.letter)));
}

// { id: { x, y } } in world pixels (card centre), and the world's size
export function layout(tree) {
  const pos = {};
  let row = 0;
  const place = (id) => {
    const kids = childrenOf(tree, id);
    let y;
    if (!kids.length) { y = row * ROW_H; row += 1; }
    else {
      const ys = kids.map((k) => place(k.id));
      y = (ys[0] + ys[ys.length - 1]) / 2;
    }
    pos[id] = { x: tree.nodes[id].depth * COL_W, y };
    return y;
  };
  if (tree.root && tree.nodes[tree.root]) place(tree.root);
  const xs = Object.values(pos).map((p) => p.x), ys = Object.values(pos).map((p) => p.y);
  return { pos, width: (xs.length ? Math.max(...xs) : 0) + CARD_W, height: (ys.length ? Math.max(...ys) : 0) + CARD_H };
}

// what a node is drawn as: the server says it (tree.states); recomputed here when a page has an older payload
export function stateOf(tree, id) {
  if (tree.states && tree.states[id]) return tree.states[id];
  const active = activePath(tree);
  if (id === tree.ghost) return "ghost";
  if (active.includes(id)) return "active";
  return aheadOf(tree).includes(id) ? "option" : "dimmed";
}

export function activePath(tree) {
  if (tree.active) return tree.active;
  const out = [];
  for (let at = tree.nodes[tree.ghost]; at; at = at.parent ? tree.nodes[at.parent] : null) out.push(at.id);
  return out.reverse();
}

export function aheadOf(tree) {
  if (tree.ahead) return tree.ahead;
  const kids = childrenOf(tree, tree.ghost);
  const newest = Math.max(0, ...kids.map((k) => k.round));
  return kids.filter((k) => k.round === newest).map((k) => k.id);
}

// the classes on a node's card: its state, plus what it is ("taken", "working", "failed", "mixed")
export function nodeClasses(tree, id, working) {
  const n = tree.nodes[id], st = stateOf(tree, id);
  const out = ["node", st];
  if (n.taken) out.push("taken");
  if (n.status === "failed") out.push("failed");
  if (working && working.node === id) out.push("working");
  if (["keyframe", "drafting", "rendering", "waiting"].includes(n.status)) out.push("busy");
  if (n.mixed_from && n.mixed_from.length) out.push("mixed");
  return out;
}

// an edge parent -> child: lit along the active timeline, open toward the options ahead, dimmed elsewhere
export function edgeClass(tree, parentId, childId) {
  const path = activePath(tree);
  const i = path.indexOf(parentId);
  if (i >= 0 && path[i + 1] === childId) return "edge active";
  return stateOf(tree, childId) === "option" ? "edge option" : "edge dimmed";
}

// a smooth branch from the right of the parent's card to the left of the child's
export function edgePath(a, b) {
  const x1 = a.x + CARD_W, y1 = a.y + CARD_H / 2 - 14, x2 = b.x, y2 = b.y + CARD_H / 2 - 14, mid = (x1 + x2) / 2;
  return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
}

// what a card shows: a playing clip (the Ghost's node and the options ahead), a paused poster (everything dimmed or
// behind), the keyframe while the draft renders, or nothing yet
export function mediaFor(tree, id) {
  const n = tree.nodes[id], st = stateOf(tree, id), f = n.files || {};
  const clip = f.clip || f.draft || null;
  const still = f.poster || f.keyframe || null;
  const live = st === "option" || st === "ghost";
  if (clip && live) return { kind: "video", file: clip, poster: still, play: true };
  if (clip) return { kind: "video", file: clip, poster: still, play: false };
  if (still) return { kind: "image", file: still };
  return { kind: "none" };
}

export function fileUrl(pid, nid, name) {
  return `/api/multiverse/projects/${encodeURIComponent(pid)}/files/${encodeURIComponent(nid)}/${encodeURIComponent(name)}`;
}

const fmt = (s) => (s == null ? "" : s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

// the line under a card's label: where it is, what it cost, what the cut check said
export function metaLine(tree, id, estimates) {
  const n = tree.nodes[id], c = n.cost || {}, p = n.progress || {}, est = estimates || {};
  const bits = [];
  const st = n.status;
  const waiting = !tree.queue || tree.queue.some((q) => q.node === id) || (tree.working && tree.working.node === id);
  if (st === "planned" && !waiting) bits.push("not drafted");
  else if (st === "planned") bits.push("queued" + (est.draft ? ` · ~${fmt(est.draft + (est.keyframe || 0))}` : ""));
  else if (st === "waiting") bits.push("waiting for room");
  else if (st === "keyframe") bits.push("keyframe…");
  else if (st === "drafting" || st === "rendering") {
    const step = p.at != null && p.of ? ` ${p.at}/${p.of}` : "";
    bits.push((st === "drafting" ? "draft" : "full") + step + (p.eta_s != null ? ` · ${fmt(p.eta_s)} left` : ""));
  } else if (st === "failed") bits.push("failed");
  if (n.quality === "full" && c.full_spec) bits.push(`full ${c.full_spec} · took ${fmt(c.full_s)}`);
  else if (n.quality === "draft" && c.draft_spec) bits.push(`draft ${c.draft_spec} · took ${fmt(c.draft_s)}`);
  else if (n.quality === "draft") bits.push(`keyframe · took ${fmt(c.keyframe_s)}`);
  const seam = (n.qc || {}).seam_full || (n.qc || {}).seam_draft;
  if (seam && seam.ok != null) bits.push(seam.ok ? "cut ok" : "cut jumps");
  return bits.join(" · ");
}

// the view's transform that puts a world point at a screen point
export function centreOn(point, view, scale, bias = { x: 0.42, y: 0.45 }) {
  return { tx: view.w * bias.x - point.x * scale, ty: view.h * bias.y - point.y * scale };
}

// eased movement of the Ghost from one node to the next (t: 0..1)
export function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
export function lerpPoint(a, b, t) { const k = ease(Math.max(0, Math.min(1, t))); return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k }; }
