// keys Ghost Multiverse Studio, invisible mode: the timeline's pure part (no DOM), so node can test it
// (tests/js/timeline.test.mjs). Left to right is time: every creation gets its own column, in the order it was made.
// Top to bottom are lanes: a creation continues its parent's lane when it is that parent's first child and the lane is
// still at the parent; any other child (a fork, a retake, a second option) takes the nearest free lane below. A lane
// is free once its last creation is more than one column back, so the graph stays compact.

export const COL = 150;    // one creation, left to right
export const LANE = 168;   // one lane, top to bottom
export const CARD_W = 128;
export const CARD_H = 132;

export function byTime(creations) {
  return [...creations].sort((a, b) => (a.at - b.at) || String(a.id).localeCompare(String(b.id)));
}

// lanes for creations in time order: a child continues its parent's lane when it is that parent's first child and the
// lane is still at the parent; any other child takes the nearest free lane below
function lanesFor(order, colOf, pos) {
  const tip = [], continued = new Set(), lane = {};
  const free = (l, col) => tip[l] === undefined || tip[l] < col - 1;
  for (const c of order) {
    const col = colOf(c), parents = (c.parents || []).filter((p) => lane[p] !== undefined);
    let l = -1;
    if (parents.length) {
      const first = parents[0], pl = lane[first];
      if (!continued.has(first) && tip[pl] === colOf({ id: first })) l = pl;
      else for (let k = pl + 1; l < 0; k++) if (free(k, col)) l = k;
      continued.add(first);
    } else {
      for (let k = 0; l < 0; k++) if (free(k, col)) l = k;
    }
    tip[l] = col; lane[c.id] = l;
  }
  return { lane, lanes: tip.length };
}

export const HEAD = 108;  // a project's header above its lanes
export const GAP = 26;    // between two projects' bands

// the project a creation belongs to: the first creation up its chain of first parents
export function rootOf(id, get) {
  let c = get(id);
  const seen = new Set();
  while (c && (c.parents || []).length && !seen.has(c.id)) {
    seen.add(c.id);
    const up = get(c.parents[0]);
    if (!up) break;
    c = up;
  }
  return c ? c.id : id;
}

// { pos: { id: { col, lane, x, y } }, cols, lanes, width, height, bands }
// With a group (creation -> project key), every project gets its own band: a header, then its lanes, packed so that
// projects made at different times share the same height and projects made at the same time stack.
export function layout(creations, group) {
  const order = byTime(creations), pos = {}, colOf = new Map(order.map((c, i) => [c.id, i]));
  const col = (c) => colOf.get(c.id);
  const cols = order.length, width = Math.max(0, cols - 1) * COL + CARD_W;
  if (!group) {
    const { lane, lanes } = lanesFor(order, col);
    for (const c of order) pos[c.id] = { col: col(c), lane: lane[c.id], x: col(c) * COL, y: lane[c.id] * LANE };
    return { pos, cols, lanes, width, height: Math.max(0, lanes - 1) * LANE + CARD_H, bands: [] };
  }
  const groups = new Map();
  for (const c of order) { const k = group(c); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(c); }
  const bands = [];
  let bottom = 0, total = 0;
  for (const [key, members] of groups) {
    const { lane, lanes } = lanesFor(members, col);
    const c0 = col(members[0]), last = col(members[members.length - 1]);
    const c1 = Math.max(last, c0 + 1);   // room for the header's words over a project of one
    const h = HEAD + Math.max(0, lanes - 1) * LANE + CARD_H;
    const near = bands.filter((b) => b.c0 <= c1 + 1 && c0 <= b.c1 + 1);
    const tops = [0, ...near.map((b) => b.y + b.h + GAP)].sort((a, b) => a - b);
    const top = tops.find((t) => near.every((b) => t + h + GAP <= b.y || b.y + b.h + GAP <= t));
    for (const c of members) pos[c.id] = { col: col(c), lane: lane[c.id], x: col(c) * COL, y: top + HEAD + lane[c.id] * LANE };
    bands.push({ key, c0, c1, x: c0 * COL, y: top, w: (c1 - c0) * COL + CARD_W, h, count: members.length, members: members.map((m) => m.id) });
    bottom = Math.max(bottom, top + h); total = Math.max(total, lanes);
  }
  return { pos, cols, lanes: total, width, height: bottom, bands };
}

// the edges: every parent -> child, the first parent drawn as the line it continues, others as merges
export function edges(creations, pos) {
  const out = [];
  for (const c of creations) {
    (c.parents || []).forEach((p, i) => { if (pos[p] && pos[c.id]) out.push({ from: p, to: c.id, merge: i > 0 }); });
  }
  return out;
}

export function edgePath(a, b) {
  const x1 = a.x + CARD_W, y1 = a.y + CARD_H / 2, x2 = b.x, y2 = b.y + CARD_H / 2, mid = (x1 + x2) / 2;
  if (y1 === y2) return `M${x1},${y1} L${x2},${y2}`;
  return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
}

// the ruler over the timeline: a mark at the first creation of each day, and wherever an hour or more passed between two
export function marks(creations, pos) {
  const out = [];
  let prev = null;
  for (const c of byTime(creations)) {
    const d = new Date(c.at * 1000);
    const day = d.toDateString();
    if (!prev || day !== prev.day) out.push({ x: pos[c.id].x, label: d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }), strong: true });
    else if (c.at - prev.at >= 3600) out.push({ x: pos[c.id].x, label: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }), strong: false });
    prev = { day, at: c.at };
  }
  return out;
}

// a creation's line on its card: the Ghost's summary, else its title, else its kind
export function cardLine(c) {
  return (c.summary || c.title || c.kind || "").trim();
}

// the colour of an agent's badge: a stable hue from its name
export function agentHue(agent) {
  let h = 0;
  for (const ch of String(agent || "")) h = (h * 31 + ch.codePointAt(0)) % 360;
  return h;
}

export function ago(at, now = Date.now() / 1000) {
  const s = Math.max(0, now - at);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
