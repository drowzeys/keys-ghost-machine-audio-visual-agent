// invisible mode's timeline layout (voice_agent/static/invisible/timeline.js): time left to right, forks in lanes below
import test from "node:test";
import assert from "node:assert/strict";
import { layout, rootOf, edges, marks, cardLine, agentHue, COL, LANE, HEAD } from "../../voice_agent/static/invisible/timeline.js";

const c = (id, at, parents = []) => ({ id, at, parents, kind: "video", agent: "a" });

test("each creation gets its own column, in the order it was made", () => {
  const { pos, cols } = layout([c("b", 2), c("a", 1), c("c", 3)]);
  assert.equal(cols, 3);
  assert.deepEqual([pos.a.col, pos.b.col, pos.c.col], [0, 1, 2]);
  assert.equal(pos.b.x, COL);
});

test("a chain stays on one lane, a fork drops to the lane below", () => {
  const { pos } = layout([c("root", 1), c("x", 2, ["root"]), c("y", 3, ["root"]), c("x2", 4, ["x"])]);
  assert.equal(pos.root.lane, 0);
  assert.equal(pos.x.lane, 0);   // the first child continues the line
  assert.equal(pos.y.lane, 1);   // the fork
  assert.equal(pos.x2.lane, 0);  // x's own child carries on
  assert.equal(pos.y.y, LANE);
});

test("a lane is reused once it has been free for a column", () => {
  const { pos, lanes } = layout([c("a", 1), c("b", 2, ["a"]), c("f", 3, ["a"]), c("g", 4), c("h", 5), c("k", 6)]);
  assert.equal(pos.f.lane, 1);
  assert.ok(lanes <= 3, `compact: ${lanes} lanes`);
});

test("a creation with two parents draws a merge edge", () => {
  const list = [c("film", 1), c("voice", 2, ["film"]), c("final", 3, ["film", "voice"])];
  const { pos } = layout(list);
  const e = edges(list, pos);
  assert.deepEqual(e.map((x) => [x.from, x.to, x.merge]), [["film", "voice", false], ["film", "final", false], ["voice", "final", true]]);
});

test("the ruler marks each day and every gap of an hour or more", () => {
  const day = Date.UTC(2026, 10, 8, 12) / 1000;
  const list = [c("a", day), c("b", day + 60), c("c", day + 7200), c("d", day + 86400)];
  const m = marks(list, layout(list).pos);
  assert.equal(m.length, 3);
  assert.deepEqual(m.map((x) => x.strong), [true, false, true]);
});

test("a card says the Ghost's summary first, and agents keep their colour", () => {
  assert.equal(cardLine({ summary: "A fox", title: "t", kind: "image" }), "A fox");
  assert.equal(cardLine({ title: "t", kind: "image" }), "t");
  assert.equal(agentHue("hermes"), agentHue("hermes"));
  assert.notEqual(agentHue("hermes"), agentHue("autopilot"));
});

test("each project gets its own band with a header, and projects made at the same time stack", () => {
  const list = [c("p", 1), c("p1", 2, ["p"]), c("q", 3), c("q1", 4, ["q"]), c("p2", 5, ["p1"]), c("p3", 6, ["p"])];
  const get = (id) => list.find((x) => x.id === id);
  assert.equal(rootOf("p2", get), "p");
  assert.equal(rootOf("q1", get), "q");
  const { pos, bands } = layout(list, (x) => rootOf(x.id, get));
  assert.deepEqual(bands.map((b) => [b.key, b.count]), [["p", 4], ["q", 2]]);
  const [bp, bq] = bands;
  assert.equal(bp.y, 0);
  assert.equal(pos.p.y, HEAD);                    // under its header
  assert.ok(bq.y >= bp.y + bp.h);                 // overlapping in time: below p's band
  assert.equal(pos.q.y, bq.y + HEAD);
  assert.equal(pos.p3.y, HEAD + LANE);            // p's fork stays inside p's band
  for (const id of ["p", "p1", "p2", "p3"]) assert.ok(pos[id].y >= bp.y && pos[id].y < bp.y + bp.h);
});

test("projects made at different times share the same height", () => {
  // a: columns 0-1, x: 2 (its header reaches 3), y: 3 (reaches 4). x is next to a so it stacks below; y is two
  // columns clear of a, so it goes back up to the top row
  const list = [c("a", 1), c("a1", 2, ["a"]), c("x", 3), c("y", 4)];
  const get = (id) => list.find((v) => v.id === id);
  const { bands, height } = layout(list, (v) => rootOf(v.id, get));
  const top = Object.fromEntries(bands.map((b) => [b.key, b.y]));
  assert.equal(top.a, 0);
  assert.ok(top.x > 0);
  assert.equal(top.y, 0);
  assert.equal(height, Math.max(...bands.map((b) => b.y + b.h)));
});
