// The multiverse tree view's pure part (voice_agent/static/multiverse/tree.js): layout, the Ghost's highlighted path,
// the dimmed roads not taken, the cards' media and lines. Run: node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { COL_W, ROW_H, CARD_W, layout, stateOf, activePath, aheadOf, nodeClasses, edgeClass, mediaFor, metaLine,
         centreOn, lerpPoint, fileUrl } from "../../voice_agent/static/multiverse/tree.js";

const node = (id, parent, depth, more = {}) => ({ id, parent, depth, children: [], round: parent ? 1 : 0, letter: null,
  label: id, beat: "", prompt: "", status: "done", quality: "full", taken: false, files: {}, cost: {}, qc: {}, mixed_from: [], ...more });

// root -> A, B, C (B taken) -> B1, B2 (round 1); the Ghost on B
function sample(ghost = "b") {
  const nodes = {
    r: node("r", null, 0, { taken: true, files: { clip: "clip.mp4", poster: "poster.jpg" } }),
    a: node("a", "r", 1, { letter: "A", quality: "draft", status: "draft", files: { draft: "draft.mp4", keyframe: "keyframe.jpg" } }),
    b: node("b", "r", 1, { letter: "B", taken: true, files: { clip: "clip.mp4", draft: "draft.mp4", poster: "poster.jpg" } }),
    c: node("c", "r", 1, { letter: "C", quality: null, status: "keyframe" }),
    b1: node("b1", "b", 2, { letter: "A", quality: "draft", status: "draft", files: { draft: "draft.mp4", poster: "poster.jpg" } }),
    b2: node("b2", "b", 2, { letter: "B", quality: null, status: "drafting", progress: { at: 3, of: 8, eta_s: 20 } }),
  };
  nodes.r.children = ["a", "b", "c"];
  nodes.b.children = ["b1", "b2"];
  return { id: "p1", root: "r", ghost, nodes };
}

test("layout: time runs left to right, a parent sits centred on its children, no two cards share a place", () => {
  const { pos } = layout(sample());
  assert.equal(pos.r.x, 0); assert.equal(pos.b.x, COL_W); assert.equal(pos.b1.x, 2 * COL_W);
  assert.equal(pos.a.y, 0);
  assert.equal(pos.b.y, (pos.b1.y + pos.b2.y) / 2);
  assert.equal(pos.r.y, (pos.a.y + pos.c.y) / 2);
  const seen = new Set(Object.values(pos).map((p) => `${p.x},${p.y}`));
  assert.equal(seen.size, 6);
  for (const [i, p] of Object.entries(pos)) for (const [j, q] of Object.entries(pos))
    if (i < j && p.x === q.x) assert.ok(Math.abs(p.y - q.y) >= ROW_H, `${i} and ${j} overlap`);
});

test("the active path is lit to the Ghost, its options are open, every other road is dimmed", () => {
  const t = sample();
  assert.deepEqual(activePath(t), ["r", "b"]);
  assert.deepEqual(aheadOf(t), ["b1", "b2"]);
  assert.equal(stateOf(t, "b"), "ghost");
  assert.equal(stateOf(t, "r"), "active");
  assert.equal(stateOf(t, "b1"), "option");
  assert.equal(stateOf(t, "a"), "dimmed");
  assert.equal(stateOf(t, "c"), "dimmed");
  assert.equal(edgeClass(t, "r", "b"), "edge active");
  assert.equal(edgeClass(t, "r", "a"), "edge dimmed");
  assert.equal(edgeClass(t, "b", "b1"), "edge option");
  assert.ok(nodeClasses(t, "a").includes("dimmed"));
  assert.ok(nodeClasses(t, "b").includes("ghost") && nodeClasses(t, "b").includes("taken"));
  assert.ok(nodeClasses(t, "b2", { node: "b2" }).includes("working") && nodeClasses(t, "c").includes("busy"));
});

test("when the Ghost walks back, the highlight follows it and B's branch dims", () => {
  const t = sample("r");
  assert.deepEqual(activePath(t), ["r"]);
  assert.equal(stateOf(t, "r"), "ghost");
  assert.equal(stateOf(t, "a"), "option");
  assert.equal(stateOf(t, "b"), "option");
  assert.equal(stateOf(t, "b1"), "dimmed");
  assert.equal(edgeClass(t, "b", "b1"), "edge dimmed");
});

test("the server's states win over the page's own reckoning", () => {
  const t = { ...sample(), states: { r: "active", a: "dimmed", b: "ghost", c: "dimmed", b1: "option", b2: "option" } };
  t.states.a = "option";  // e.g. a mix the page has not laid out yet
  assert.equal(stateOf(t, "a"), "option");
});

test("an older round of options is dimmed; only the newest round is ahead", () => {
  const t = sample();
  t.nodes.b3 = node("b3", "b", 2, { letter: "A", round: 2 });
  t.nodes.b.children.push("b3");
  assert.deepEqual(aheadOf(t), ["b3"]);
  assert.equal(stateOf(t, "b1"), "dimmed");
  assert.equal(stateOf(t, "b3"), "option");
});

test("live previews ahead of the Ghost, paused posters elsewhere, keyframes while a draft renders", () => {
  const t = sample();
  assert.deepEqual(mediaFor(t, "b1"), { kind: "video", file: "draft.mp4", poster: "poster.jpg", play: true });
  assert.deepEqual(mediaFor(t, "b"), { kind: "video", file: "clip.mp4", poster: "poster.jpg", play: true });
  assert.equal(mediaFor(t, "a").play, false);
  assert.equal(mediaFor(t, "r").play, false);
  assert.deepEqual(mediaFor(t, "c"), { kind: "none" });
  t.nodes.c.files = { keyframe: "keyframe.jpg" };
  assert.deepEqual(mediaFor(t, "c"), { kind: "image", file: "keyframe.jpg" });
});

test("the card's line says the cost, the progress and the cut", () => {
  const t = sample();
  t.nodes.b1.cost = { draft_spec: "640x352 · 3 s", draft_s: 41 };
  t.nodes.b1.qc = { seam_draft: { ok: true, corr: 0.93 } };
  assert.equal(metaLine(t, "b1"), "draft 640x352 · 3 s · took 41 s · cut ok");
  assert.equal(metaLine(t, "b2"), "draft 3/8 · 20 s left");
  t.nodes.c.status = "planned";
  assert.equal(metaLine(t, "c", { draft: 30, keyframe: 10 }), "queued · ~40 s");
  t.queue = [];  // dropped when a sibling was taken: honest, not "queued" forever
  assert.equal(metaLine(t, "c"), "not drafted");
  t.queue = [{ node: "c", kind: "draft" }];
  assert.equal(metaLine(t, "c"), "queued");
});

test("the view centres on the Ghost and the Ghost walks eased between nodes", () => {
  const c = centreOn({ x: 100, y: 50 }, { w: 1000, h: 800 }, 2);
  assert.equal(c.tx, 1000 * 0.42 - 200); assert.equal(c.ty, 800 * 0.45 - 100);
  assert.deepEqual(lerpPoint({ x: 0, y: 0 }, { x: 10, y: 20 }, 0), { x: 0, y: 0 });
  assert.deepEqual(lerpPoint({ x: 0, y: 0 }, { x: 10, y: 20 }, 1), { x: 10, y: 20 });
  assert.deepEqual(lerpPoint({ x: 0, y: 0 }, { x: 10, y: 20 }, 0.5), { x: 5, y: 10 });
  assert.equal(fileUrl("p1", "n 1", "clip.mp4"), "/api/multiverse/projects/p1/files/n%201/clip.mp4");
  assert.ok(CARD_W < COL_W);
});
