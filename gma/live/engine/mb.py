"""A blocking preview with MotionBricks (SOMA): a floor plan and timed instructions in, skeleton motion and a cheap
stick-figure clip out. The model runs from its ONNX file with ONNX Runtime; inputs and outputs follow
motionbricks/model/soma/kernels.py (encode/decode), written here in numpy.

    mb.py plan.json out_dir [provider=cpu|cuda]

Runs inside the blocking container of gma/live/blocking.py (no network; the model and its sample poses mounted at
/mb/lfs, NVIDIA's code at /mb/src, ONNX Runtime at /deps). Writes out_dir/preview.rgb (640x480 RGB, 30 fps),
motion.npz and report.json.
"""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/deps")  # run with -I: the side directory is added here
import numpy as np

LFS, HZ = Path("/mb/lfs"), 30.0
NAMES = json.load(open("/mb/src/motionbricks/assets/onnx/soma/soma_n1.json"))["joint_names"]
UP = np.array([0.0, 1.0, 0.0])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def axis(name, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[name]


def load_bvh(path, step=4):
    """World positions [F, 30, 3] (m) and rotations [F, 30, 3, 3] of the model's joints, at 30 Hz, and their parents."""
    lines = Path(path).read_text().split("\n")
    names, parents, offsets, chans, stack, i = [], [], [], [], [], 0
    while not lines[i].startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1); stack.append(len(names) - 1)
        elif t and t[0] == "End":
            names.append(None); parents.append(stack[-1]); stack.append(len(names) - 1); chans.append([])
        elif t and t[0] == "OFFSET":
            offsets.append([float(v) for v in t[1:4]])
        elif t and t[0] == "CHANNELS":
            chans.append(t[2:])
        elif t and t[0] == "}":
            stack.pop()
        i += 1
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 3:] if l.strip()])[::step]
    F, J = len(data), len(names)
    pos, rot, col = np.zeros((F, J, 3)), np.zeros((F, J, 3, 3)), 0
    for j in range(J):
        local_p = np.tile(np.array(offsets[j]), (F, 1)); local_r = np.tile(np.eye(3), (F, 1, 1))
        for c in chans[j]:
            v = data[:, col]; col += 1
            if c.endswith("position"):
                if c == chans[j][0]: local_p = np.zeros((F, 3))  # position channels replace the offset
                local_p[:, "XYZ".index(c[0])] = v
            else:
                local_r = local_r @ np.stack([axis(c[0], d) for d in v])
        if parents[j] < 0:
            pos[:, j], rot[:, j] = local_p, local_r
        else:
            p = parents[j]
            pos[:, j] = pos[:, p] + np.einsum("fab,fb->fa", rot[:, p], local_p); rot[:, j] = rot[:, p] @ local_r
    pick = [names.index(n) for n in NAMES]
    def model_parent(j):
        p = parents[j]
        while p >= 0 and names[p] not in NAMES: p = parents[p]
        return NAMES.index(names[p]) if p >= 0 else -1
    return pos[:, pick] / 100.0, rot[:, pick], [model_parent(j) for j in pick]


RL, LL = NAMES.index("RightLeg"), NAMES.index("LeftLeg")


def heading(pos):
    """The heading of one pose [30, 3]: the angle of its forward direction, atan2(x, z)."""
    a = pos[RL] - pos[LL]
    return math.atan2(a[2], -a[0])


def window(pos, rot, start, travel=False):
    """Four poses from a clip, moved to the origin and turned to face +X."""
    p, r = pos[start:start + 4].copy(), rot[start:start + 4].copy()
    if travel:
        d = p[3, 0] - p[0, 0]; h = math.atan2(d[0], d[2])
    else:
        h = heading(p[0])
    turn = ry(math.pi / 2 - h); o = p[0, 0] * np.array([1, 0, 1])
    return (p - o) @ turn.T, turn @ r


def segment(pos, rot, start, count):
    """``count`` poses from a clip, moved to the origin and turned so the first faces +X: an action played as it was captured."""
    p, r = pos[start:start + count].copy(), rot[start:start + count].copy()
    turn = ry(math.pi / 2 - heading(p[0])); o = p[0, 0] * np.array([1, 0, 1])
    return (p - o) @ turn.T, turn @ r


# the actions a cue can ask for, beside walking and turning: the captured clip each is played from (MotionBricks' sample
# motions), the frames of it that are used, and whether it is played once or goes round until the person's next cue
ACTIONS = {"fight": ("combat_quickstep_ff_idle_180_R_002__A365.bvh", 20, 75, False),
           "jump": ("jump_over_obstacle_0_5m_R_001__A415.bvh", 40, 150, True),
           "crouch": ("crouch_idle_001__A529.bvh", 30, 120, False),
           "dance": ("dance_vouge_vogue_sequence_180_R_002__A316.bvh", 30, 220, False),
           "sit": ("sit_on_chair_loop.bvh", 20, 150, False),
           "look": ("neutral_looking_around_R_001__A542.bvh", 30, 200, False)}
# quick things laid over whatever the body is doing, and how long each lasts unless the cue says
GESTURES = {"punch": 0.45, "kick": 0.6, "blast": 1.6, "wave": 1.4, "fly": 2.5}
J = {n: i for i, n in enumerate(NAMES)}


def gesture(motion, cue, n):
    """One gesture written over a character's positions [F, 30, 3] from its second: a punch or a kick straight ahead,
    both hands thrown forward (a blast), a wave, or the whole body off the floor."""
    kind, secs = cue["do"], float(cue.get("secs") or GESTURES[cue["do"]])
    a, b = int(cue["t"] * HZ), min(len(motion), int((cue["t"] + secs) * HZ))
    side = ("Right", "Left")[n % 2]
    for f in range(max(0, a), b):
        u = (f - a) / max(1, b - a - 1); p = motion[f]
        w = math.sin(math.pi * u) if kind in ("punch", "kick", "fly") else min(1.0, u * 5, (1 - u) * 5)
        h = heading(p); d = np.array([math.sin(h), 0.0, math.cos(h)])

        def reach(names, spots):
            for name, spot in zip(names, spots):
                p[J[name]] = p[J[name]] * (1 - w) + spot * w

        if kind == "punch":
            s = p[J[side + "Arm"]].copy()
            reach([side + "ForeArm", side + "Hand", side + "HandThumbEnd", side + "HandMiddleEnd"], [s + d * 0.28, s + d * 0.58, s + d * 0.62, s + d * 0.68])
        elif kind == "kick":
            s = p[J[side + "Leg"]].copy()
            reach([side + "Shin", side + "Foot", side + "ToeBase"], [s + d * 0.42 + UP * 0.05, s + d * 0.85 + UP * 0.12, s + d * 0.98 + UP * 0.2])
        elif kind == "blast":
            c = p[J["Chest"]].copy()
            for each, out in (("Right", -0.07), ("Left", 0.07)):
                across = np.cross(UP, d) * out
                reach([each + "ForeArm", each + "Hand", each + "HandThumbEnd", each + "HandMiddleEnd"],
                      [c + d * 0.3 + across * 2.4, c + d * 0.58 + across, c + d * 0.62 + across, c + d * 0.68 + across])
        elif kind == "wave":
            s = p[J["RightArm"]].copy(); sway = np.cross(UP, d) * 0.14 * math.sin(u * secs * 2 * math.pi * 2)
            reach(["RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd"],
                  [s + UP * 0.12 - np.cross(UP, d) * 0.22, s + UP * 0.42 + sway, s + UP * 0.46 + sway, s + UP * 0.52 + sway])
        elif kind == "fly":
            p += UP * 1.3 * w
    return {"t": cue["t"], "do": kind, "frames": [max(0, a), b]}


def place(win, where, facing):
    """A window at ground position ``where`` [x, z], facing ``facing`` (radians, atan2(x, z))."""
    turn = ry(facing - math.pi / 2)
    return win[0] @ turn.T + np.array([where[0], 0, where[1]]), turn @ win[1]


def encode(pos, rot, targets=4):
    """The model's inputs from eight world poses (four of context, four of target)."""
    o = pos[3, 0] * np.array([1, 0, 1]); a = heading(pos[3]) - math.pi / 2; R = ry(a)
    p, r = (pos - o) @ R, R.T @ rot  # R^-1 (p - o)
    g, l, lp = np.zeros((1, 8, 5), np.float32), np.zeros((1, 8, 4), np.float32), np.zeros((1, 8, 267), np.float32)
    for f in range(8):
        n = f if f == 3 else min(f + 1, 7)
        h, hn = heading(p[f]), heading(p[n])
        g[0, f] = [*p[f, 0], math.cos(h), math.sin(h)]
        l[0, f] = [math.atan2(math.sin(hn - h), math.cos(hn - h)) * HZ, (p[n, 0, 0] - p[f, 0, 0]) * HZ, (p[n, 0, 2] - p[f, 0, 2]) * HZ, p[f, 0, 1]]
        lp[0, f, :87] = (p[f, 1:] - p[f, 0] * np.array([1, 0, 1])).reshape(-1)
        lp[0, f, 87:] = np.concatenate([r[f, :, :, 0], r[f, :, :, 1]], axis=-1).reshape(-1)
    has = np.array([[1] * (4 + targets) + [0] * (4 - targets)], np.int32)
    has_l = has.copy(); has_l[0, 3] = 0; has_l[0, 4 + targets - 1] = 0
    return {"global_root_values": g, "has_global_root_values": has, "local_root_values": l, "has_local_root_values": has_l,
            "local_poses": lp, "has_local_poses": has}, (o, R)


def decode(out, count, align, relative):
    """World positions and rotations of the valid frames (the three boundary frames at each end dropped)."""
    o, R = align
    m = out[0, 3:4 * count - 3].astype(np.float64)
    root = m[:, :3]
    kids = m[:, 5:92].reshape(-1, 29, 3) + (root[:, None] * np.array([1, 0, 1]) if relative else 0)
    p = np.concatenate([root[:, None], kids], axis=1)
    six = m[:, 92:272].reshape(-1, 30, 6)
    x = six[..., :3] / np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    z = np.cross(x, six[..., 3:]); z /= np.linalg.norm(z, axis=-1, keepdims=True)
    r = np.stack([x, np.cross(z, x), z], axis=-1)
    return p @ R.T + o, R @ r, m[:, -4:]


class Model:
    def __init__(self, provider="cpu"):
        import onnxruntime as ort
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 8
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        t = time.time()
        self.session = ort.InferenceSession(str(LFS / "soma_fp32.onnx"), opts, providers=providers)
        self.load_s, self.calls, self.ms, self.relative = time.time() - t, 0, [], None
        self.providers = self.session.get_providers()

    def generate(self, ctx, tgt, tokens=range(8, 15)):
        """Motion from four context poses to four target poses: (positions, rotations) of the frames after the context."""
        feeds, align = encode(np.concatenate([ctx[0], tgt[0]]), np.concatenate([ctx[1], tgt[1]]))
        feeds["allowed_pred_num_tokens"] = np.array([[1 if 6 + i in tokens else 0 for i in range(11)]], np.int32)
        t = time.perf_counter()
        motion, count = self.session.run(["pred_global_motions", "pred_num_tokens"], feeds)
        self.ms.append((time.perf_counter() - t) * 1e3); self.calls += 1
        if self.relative is None:  # are the joint positions measured from the root's ground point, or absolute? the spine tells
            d = np.linalg.norm(motion[0, 3, 5:8] - motion[0, 3, :3] * np.array([0, 1, 0]))
            self.relative = bool(d < 0.3)
        return decode(motion, int(count[0]), align, self.relative)


def run_character(model, clips, who, seconds):
    """One character's motion for ``seconds``: world positions [F, 30, 3], and the log of what was generated."""
    idle, walk, speed = clips["idle"], clips["walk"], clips["walk_speed"]
    at, face = np.array(who["start"], float), math.radians(who.get("facing", 0))
    p0, r0 = place(idle, at, face)
    hist_p, hist_r = [x for x in p0], [x for x in r0]
    every = sorted(who.get("cues", []), key=lambda c: c["t"])
    cues = [c for c in every if c["do"] not in GESTURES]   # what the body does; the gestures are laid over it afterwards
    goal, log = {"do": "stand", "at": at, "facing": face}, []
    while len(hist_p) - 4 < seconds * HZ:
        now = (len(hist_p) - 4) / HZ
        while cues and cues[0]["t"] <= now:
            c = cues.pop(0)
            if c["do"] in ("walk", "run"):
                goal = {"do": "walk", "to": np.array(c["to"], float), "then": c.get("facing"), "run": c["do"] == "run" and "run" in clips}
            elif c["do"] == "turn": goal = {"do": "stand", "at": goal.get("at", None), "facing": math.radians(c["facing"])}
            elif c["do"] in clips.get("actions", {}): goal = {"do": "act", "clip": c["do"], "i": None}
        here = hist_p[-1][0][[0, 2]]
        if goal["do"] == "act":
            cp, cr, once = clips["actions"][goal["clip"]]
            if goal["i"] is None:  # into it: the model takes the body from where it is to the action's first poses
                goal["from"] = (here.copy(), heading(hist_p[-1]))
                tgt, early, kind, goal["i"] = place((cp[:4], cr[:4]), *goal["from"]), 0, "into " + goal["clip"], 4
            else:  # the action itself, as it was captured, from where the body stood
                count = min(len(cp) - goal["i"], 30)
                if cues:
                    count = max(1, min(count, int(round((cues[0]["t"] - now) * HZ))))
                p, r = place((cp[goal["i"]:goal["i"] + count], cr[goal["i"]:goal["i"] + count]), *goal["from"])
                hist_p += [x for x in p]; hist_r += [x for x in r]; goal["i"] += count
                log.append({"t": round(now, 2), "kind": goal["clip"], "frames": count})
                if goal["i"] >= len(cp):  # the end of the clip: stand where it ended, or go round again from there
                    spot = hist_p[-1][0][[0, 2]]
                    goal = {"do": "stand", "at": spot, "facing": heading(hist_p[-1])} if once else {"do": "act", "clip": goal["clip"], "i": None}
                continue
        elif goal["do"] == "walk":
            gait, pace = (clips["run"], clips["run_speed"]) if goal.get("run") else (walk, speed)
            d = goal["to"] - here; dist = float(np.linalg.norm(d)); toward = math.atan2(d[0], d[1])
            if dist > pace * 1.2:
                tgt, early, kind = place(gait, here + d / dist * pace * 1.0, toward), 4, "run" if goal.get("run") else "walk"
            else:  # the last stretch: arrive standing
                end = math.radians(goal["then"]) if goal["then"] is not None else toward
                tgt, early, kind = place(idle, goal["to"], end), 0, "arrive"
                goal = {"do": "stand", "at": goal["to"], "facing": end}
        else:
            spot = goal["at"] if goal.get("at") is not None else here
            goal["at"] = spot
            tgt, early, kind = place(idle, spot, goal["facing"]), 0, "stand"
        p, r, _ = model.generate((np.stack(hist_p[-4:]), np.stack(hist_r[-4:])), tgt)
        keep = len(p) - early
        if cues:  # a cue cuts the motion short at its time
            keep = max(4, min(keep, int(round((cues[0]["t"] - now) * HZ)) + 1))
        hist_p += [x for x in p[1:keep]]; hist_r += [x for x in r[1:keep]]
        log.append({"t": round(now, 2), "kind": kind, "frames": keep - 1})
    motion = np.stack(hist_p[4:4 + int(seconds * HZ)])
    done = [gesture(motion, c, n) for n, c in enumerate(c for c in every if c["do"] in GESTURES and c["t"] < seconds)]
    return motion, log, done


def render(plan, motions, parents, out, gestures=None):
    """The preview: a fixed camera's view of stick figures on a ground grid, with the floor plan from above in a corner."""
    from PIL import Image, ImageDraw
    W, H = 640, 480
    cam = np.array(plan.get("camera", {}).get("at", [0, 1.5, 7.0]), float); look = np.array(plan.get("camera", {}).get("look", [0, 1.0, 0]), float)
    fwd = look - cam; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, UP); right /= np.linalg.norm(right); upv = np.cross(right, fwd)
    focal = W * 0.9
    def project(p):
        d = p - cam; z = d @ fwd
        return None if z < 0.2 else (W / 2 + focal * (d @ right) / z, H / 2 - focal * (d @ upv) / z)
    colours = [(255, 90, 90), (90, 170, 255), (120, 230, 140), (240, 200, 80)]
    F = min(len(m) for m in motions.values())
    S, pad, span = 150, 8, float(plan.get("floor_m", 10))
    def top(x, z): return (W - S - pad + (x / span + 0.5) * S, pad + (z / span + 0.5) * S)
    frames = np.zeros((F, H, W, 3), np.uint8)
    for f in range(F):
        im = Image.new("RGB", (W, H), (18, 20, 28)); d = ImageDraw.Draw(im)
        for g in np.arange(-span / 2, span / 2 + 0.01, 1.0):
            for a, b in ((np.array([g, 0, -span / 2]), np.array([g, 0, span / 2])), (np.array([-span / 2, 0, g]), np.array([span / 2, 0, g]))):
                pa, pb = project(a), project(b)
                if pa and pb: d.line([pa, pb], fill=(50, 56, 72), width=1)
        for mark in plan.get("marks", []):
            m = project(np.array([mark["at"][0], 0, mark["at"][1]]))
            if m: d.ellipse([m[0] - 4, m[1] - 2, m[0] + 4, m[1] + 2], outline=(200, 200, 120)); d.text((m[0] + 6, m[1] - 6), mark["name"], fill=(200, 200, 120))
        d.rectangle([W - S - pad, pad, W - pad, pad + S], fill=(10, 12, 18), outline=(90, 96, 110))
        for mark in plan.get("marks", []):
            t = top(*mark["at"]); d.rectangle([t[0] - 2, t[1] - 2, t[0] + 2, t[1] + 2], outline=(200, 200, 120))
        c = top(cam[0], cam[2]); d.polygon([(c[0], c[1] - 5), (c[0] - 4, c[1] + 3), (c[0] + 4, c[1] + 3)], fill=(230, 230, 230))
        for i, (name, m) in enumerate(motions.items()):
            col, p = colours[i % 4], m[f]
            for j, pj in enumerate(parents):
                if pj >= 0:
                    a, b = project(p[j]), project(p[pj])
                    if a and b: d.line([a, b], fill=col, width=3)
            head = project(p[NAMES.index("Head")])
            if head: d.ellipse([head[0] - 7, head[1] - 9, head[0] + 7, head[1] + 5], outline=col, width=2); d.text((head[0] - 12, head[1] - 24), name, fill=col)
            for g in (gestures or {}).get(name, []):
                if not g["frames"][0] <= f < g["frames"][1]: continue
                if head: d.text((head[0] - 12, head[1] - 36), g["do"], fill=(255, 255, 255))
                if g["do"] == "blast":  # what leaves the hands: a ball that grows and travels the way they face
                    u = (f - g["frames"][0]) / max(1, g["frames"][1] - g["frames"][0]); hd = heading(p)
                    hands = (p[J["RightHand"]] + p[J["LeftHand"]]) / 2
                    for k in range(1, 7):
                        b = project(hands + np.array([math.sin(hd), 0, math.cos(hd)]) * (0.25 + u * 4.0 * k / 6))
                        size = 3 + 9 * u * k / 6
                        if b and u > 0.15: d.ellipse([b[0] - size, b[1] - size, b[0] + size, b[1] + size], outline=(255, 240, 150), width=2)
            trail = [top(q[0, 0], q[0, 2]) for q in m[max(0, f - 150):f + 1:5]]
            if len(trail) > 1: d.line(trail, fill=tuple(v // 2 for v in col), width=1)
            t = top(p[0, 0], p[0, 2]); h = heading(p)
            d.ellipse([t[0] - 3, t[1] - 3, t[0] + 3, t[1] + 3], fill=col); d.line([t, (t[0] + 9 * math.sin(h), t[1] + 9 * math.cos(h))], fill=col, width=2)
        now = [b for b in plan.get("beats", []) if b["s"] <= f / HZ]
        if now:  # the beat being played, as the director said it
            d.text((8, 8), f"{int(now[-1]['s'])} s  {now[-1]['note'][:64]}", fill=(236, 236, 240))
        d.text((8, H - 18), f"{f / HZ:5.1f} s   blocking preview (MotionBricks SOMA)", fill=(170, 176, 190))
        frames[f] = np.asarray(im)
    frames.tofile(out / "preview.rgb")
    return F


def main():
    plan, out = json.load(open(sys.argv[1])), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    provider = next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("provider=")), "cpu")
    t0 = time.time()
    ip, ir, parents = load_bvh(LFS / "idle_loop_001__A268.bvh"); wp_, wr, _ = load_bvh(LFS / "Neutral_walk_forward_002__A057.bvh")
    mid = len(wp_) // 2
    clips = {"idle": window(ip, ir, 60), "walk": window(wp_, wr, mid, travel=True),
             "walk_speed": float(np.linalg.norm((wp_[mid + 30, 0] - wp_[mid, 0])[[0, 2]]))}
    clips["actions"] = {}
    for name, (file, first, count, once) in ACTIONS.items():  # the actions whose clips are here; a cue for one that is not stands
        if (LFS / file).exists():
            ap, ar, _ = load_bvh(LFS / file)
            sp, sr = segment(ap, ar, min(first, max(0, len(ap) - 8)), count)
            if name != "dance":  # made on the spot: a clip's own run-up or side steps would carry the body off its mark
                sp = sp - (sp[:, :1] - sp[:1, :1]) * np.array([1, 0, 1])
            clips["actions"][name] = (sp, sr, once)
    if (LFS / "run_loop_180_R_003__A326.bvh").exists():
        rp, rr, _ = load_bvh(LFS / "run_loop_180_R_003__A326.bvh"); half = max(0, min(len(rp) // 2, len(rp) - 31))
        clips["run"], clips["run_speed"] = window(rp, rr, half, travel=True), float(np.linalg.norm((rp[half + 30, 0] - rp[half, 0])[[0, 2]]))
        if clips["run_speed"] < clips["walk_speed"] * 1.2: del clips["run"]  # a loop that runs on the spot is no gait to travel with
    t_clips = time.time() - t0
    model = Model(provider)
    t1 = time.time(); motions, logs, gestures = {}, {}, {}
    for who in plan["cast"]:
        motions[who["name"]], logs[who["name"]], gestures[who["name"]] = run_character(model, clips, who, float(plan["seconds"]))
    t_motion = time.time() - t1
    np.savez_compressed(out / "motion.npz", **motions, parents=np.array(parents))
    t2 = time.time(); F = render(plan, motions, parents, out, gestures); t_render = time.time() - t2
    report = {"providers": model.providers, "model_load_s": round(model.load_s, 2), "clips_s": round(t_clips, 2), "walk_speed_mps": round(clips["walk_speed"], 2),
              "positions_root_relative": model.relative, "generations": model.calls, "ms_per_generation_median": round(float(np.median(model.ms)), 1),
              "ms_first": round(model.ms[0], 1), "motion_s": round(t_motion, 2), "render_s": round(t_render, 2), "frames": F, "seconds": F / HZ,
              "bones_cm": {"hips-spine1": round(float(np.linalg.norm(next(iter(motions.values()))[10, 1] - next(iter(motions.values()))[10, 0])) * 100, 1)},
              "ends": {n: [round(float(m[-1, 0, 0]), 2), round(float(m[-1, 0, 2]), 2), round(math.degrees(heading(m[-1])))] for n, m in motions.items()},
              "actions": sorted(clips["actions"]), "run_speed_mps": round(clips.get("run_speed", 0.0), 2), "gestures": gestures, "logs": logs}
    (out / "report.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report))


if __name__ == "__main__":
    main()
