# The video panel

Boo's page has three columns on a desktop or a tablet held sideways: your camera on the left, Boo in the middle (always
the largest), and the **video panel** on the right. On a phone held upright Boo comes first, then the camera, then the
video panel, one under the other. Both side panels fold away, and each device remembers which ones you folded.

The camera column shows the camera only while you have turned it on, and a "camera off" placeholder otherwise. It is
the same local preview that Boo's eyes use. It is never recorded, never uploaded anywhere new, and never used in a
video. The video service has no access to camera frames, and a spoken video request sends only your words.

The video panel shows:

- **the main view**: the clip being made, live while it renders, then the finished clip playing (it loops until the
  next one arrives);
- **status**: `planning`, `keyframe`, `rendering 45%`, `checking`, `done`, `stopped` or `failed`, with the sampler's
  step (`step 4/8 · ~40 s`) and a progress bar;
- **the current beat and prompt**;
- **a strip of recent clips** (tap one to play it);
- **Start** (with an idea, and either "one clip" or "keep going"), **Stop**, **Download film** (the clips joined in
  order), and a **sound** toggle.

Video starts muted. Sound plays only after you turn it on, and even then it stays quiet while Boo is listening,
thinking or speaking. The panel's videos load only when they are shown (`preload="none"`, lazy posters). The panel opens
its event stream only while it is unfolded and the page is visible.

## By voice

A request for a video is found in code before any model sees the turn (`video.asked`, through `backstage.claims`):
"generate a video of a dragon over a castle", "can you make me a short film about …", "show me a clip of …", "I want an
endless video of …". Talk about a video ("I watched a video yesterday", "I made a film", "a video call") is left alone.
The video agent answers that turn itself, with one short spoken line, and the front brain says nothing, so there is
never a second reply. The idea is the words after the request ("of a dragon over a castle"). With no subject ("make a
video") it asks "What should the video show?", and the next thing said (within 90 s) is the idea; "never mind" drops
it.

A refusal is spoken short and plain ("The video machine is busy right now, so I can't render yet."). The full reason,
such as the Studio's memory estimate, goes to the task card and the panel's status line.

When the profile has a `video` section, the front brain's instructions also get one line (`server.VIDEO_NOTE`): it can
make videos in the video panel, the video agent makes them, and it never says it can't.

| Say | Does |
|---|---|
| "Boo, make a video of a lighthouse in a storm" | one clip |
| "make a video" … "a red kite over green hills" | asks what it should show, then one clip of the answer |
| "make an endless video of waves on a reef", "… a story about …" | continue mode: clip after clip until stopped |
| "stop the video" | stops and cancels the render at work |
| "show me the last clip" | plays it in the panel (and unfolds it) |
| "how's the video going?" | the state and the step, said aloud |

Boo speaks short updates without holding up the conversation. It waits until nobody is talking, and it never queues an
update behind a reply. It speaks for the first clip, every `speak_every` clips after that (3 by default), when the run
ends and when it fails. Updates go to the conversation used last. The front brain never sees these lines, so they cannot
end up in its replies.

## The job service

`voice_agent/video.py` runs one run at a time:

    POST /api/video/start   {"prompt": "...", "mode": "single"|"continue", "clips": n, "seconds": s}
    POST /api/video/stop
    GET  /api/video/status  the run as it is now
    GET  /api/video/events  the same as server-sent events, at most 4 a second (a slow page skips to the newest)
    GET  /api/video/clips   the finished clips
    GET  /api/video/clip/{run}/{n}.mp4 | .jpg
    GET  /api/video/preview.jpg   the picture behind the live view (204 when there is none)
    GET  /api/video/film?run=...  the clips joined (stream copy with ffmpeg; re-encoded if the clips differ)

Every change is written to `<dir>/<run>/run.json`. That file is the checkpoint: each clip's plan, its render job and its
state, and `clipNNN.mp4` beside it once fetched. When the agent starts again, a run that was still going picks up from
its checkpoint. A clip that already has a render job is followed, never sent again. The Studio's own paths stay on the
server: the page sees only `/api/video/...` addresses.

### Renderers

Every renderer implements one interface (`Renderer`): `room`, `keyframe`, `submit`, `poll` → `Progress`, `picture`,
`fetch` and `cancel`.

- **`studio`**: Imagination Studio's render queue, through the routes its own page uses. A clip is a `video` job
  (`POST /api/jobs` with `prompt`, `preset`, `quality`, `seconds` and `seed`). It is followed with `GET /api/jobs/{id}`
  (`progress: {stage, at, of}`, `estimate`, `elapsed`, `outputs`) and fetched from `/files/<path in the Studio
  folder>`. A continue clip sets `first_frame_from` to the clip before, so the Studio opens it on that clip's last
  frame. With `"keyframe": true`, a still is made first as a `scout` job and the clip is rendered from it (`image`). The
  Studio's `/api/estimate` memory check runs before a run starts. Its refusal is shown in full in the panel, and
  the voice says only that the video machine is busy.
- **`hyperflow`**: a stub for the infinity engine below. It refuses with "isn't built yet".

### Where clips are made

Video is never made beside the live voice models unless that machine has room. `render_host` says where the renderer
runs:

- `"remote"`: another machine. An ssh tunnel on 127.0.0.1 counts as remote.
- `"local"`: this machine. A run is refused with a clear spoken message unless this machine has `local_min_free_gb`
  free (64 GB by default; MemAvailable on Linux, free + inactive + speculative on macOS).

If `render_host` is left out, a loopback URL counts as local. The Studio's own memory guard is asked as well.

### Profile

```json
"video": {
  "renderer": "studio",
  "url": "http://127.0.0.1:7870",
  "render_host": "remote",
  "key_file": "~/.config/boo/studio-key",
  "key_env": "STUDIO_ACCESS_KEY_VALUE",
  "preset": "small", "quality": "fh8", "seconds": 5,
  "max_clips": 6, "keyframe": false, "retakes": 1,
  "speak_updates": true, "speak_every": 3,
  "dir": "~/.boo-video", "poll_s": 1.0
}
```

`max_clips` caps a continue run (0 means until stopped). The Studio's access key is read from the environment variable
or the file each time it is needed. A Studio reached on its own machine, or through a tunnel, needs no key.

## Live generation: what the engines expose

The live view is driven by what the render host reports, and the page labels it honestly:

- **`live preview`**: real pictures of the clip as it renders. The renderer names each one in `Progress.preview`. The
  service fetches each new one once, and the page redraws it about four times a second.
- **`progress only`**: no pictures, only the sampler's step count and an estimate. The page shows the clip's opening
  still if one is known (the keyframe, or the last frame of the clip before in continue mode), otherwise noise. The
  picture sharpens and the noise fades as the steps go by. This is a progress display, not the model's picture.

**Today's Imagination Studio gives progress only.** On a DGX Spark, its H3 video jobs run in ComfyUI. `spark_generate.py`
reads ComfyUI's websocket and prints `[tensorfold] step i/n`, which the job API passes on as `progress: {stage:
"picture", at, of}`. No picture is passed on, for two reasons:

1. ComfyUI is started without `--preview-method`, so its default is no previews, and the Studio does not ask for one
   per prompt (`extra_data.preview_method`).
2. ComfyUI sends preview frames only to the websocket client that queued the prompt (`server.client_id`). Another
   client listening on the same ComfyUI never gets them, so tapping ComfyUI from Boo is not possible.

The engine side looks able to preview. ComfyUI's `MiniMaxH3AV` latent format has `latent_rgb_factors` (cheap
latent2rgb) and a `taeh3` tiny decoder. The Studio's step count already arrives as ComfyUI `progress` events from the
sampler, and that is the same path that carries preview frames. This has not been tried with the TensorFold int8
blocks yet.
Two small Studio changes would turn it on, and the panel already reads the result:

1. In `spark_generate.py`, queue with `"extra_data": {"preview_method": "latent2rgb"}`. In the websocket loop that
   already follows the sampler, write each binary preview frame (type 1 or 4) to `<job folder>/preview.jpg`, at most
   about 3 a second.
2. In the job API, add `"preview": "<that path>"` to `public(job)` while the job runs.

The `studio` renderer already reads a job's `preview` field. When the field is there, the page switches to
`live preview` with no change on Boo's side. Video jobs on the Mac run H3 in MLX outside ComfyUI, so a step preview
there needs the engine to write one too.

**Keyframes**: with `"keyframe": true`, the still is made as a Studio `scout` job before the clip and gets the same
step progress. The finished still becomes the clip's opening picture in the live view.

**Hand-off**: when a clip finishes, it takes the main view and plays through once. Then the live view of the next clip
comes back, with the clip just made looping small in the corner (tap it to play it full size). When the run ends, the
newest clip loops. Clips in a continue run open on the frame the last one ended on, so the clips run on one into the
next.

## Infinity: continue mode now, the engine next

Continue mode is the first step of HyperFlow-Infinity's idea, an open-ended, continuously directed film that checks
itself. Each turn of the loop:

1. **plan**: the profile's brain (Bolt) writes the next beat and prompt from the idea, the story so far (one beat a
   clip, the last eight) and the last clip's prompt. It must follow on from the last frame, add something new and never
   restage an earlier beat. The planner first waits up to 10 s for a quiet moment in the conversation, so a voice turn
   never queues behind it on Bolt. If the brain gives no answer, the clip carries on with the last prompt.
2. **render**: the clip opens on the last clip's final frame (`first_frame_from`).
3. **check**: `Checker.check(clip_file, clip, run) → Verdict(ok, notes, retake)`. Without a checker it passes
   everything. With `"check": "d1"` in the `video` section and a `d1` server in the profile, **d1-2B** looks at four
   frames spread over the clip and answers four yes/no questions in one call: the same person throughout, no hard cut,
   the same clothes, the same place (`video.D1Checker`; `d1_check` sets `frames`, `checks`, `threshold` and
   `confirm`). A fault it is at least `threshold` sure of (0.7 by default) is a retake; a weaker one is only a note. d1
   misses some person swaps, so a flagged clip can be shown to a stronger vision model first (`confirm`: an
   OpenAI-compatible URL and model, such as Gemma 4 12B), and is retaken only if it agrees. With no answer from d1 the
   clip is kept. The panel's strip outlines a clip with findings and lists them on hover. If a verdict asks for a
   retake, the clip is rendered again with a new seed (`retakes`, 1 by default), and its `notes` go to the planner for
   the next clip.

   Measured on a DGX Spark against our TensorFold build of d1-2B (causal reading, 3 questions, 4 frames), on 20
   synthetic clips (generated pictures, not real footage): 11 of 12 faulty clips retaken (person swaps 5 of 6, cuts to an
   empty scene 2 of 2, clothing colour shifts 4 of 4), 0 of 8 clean clips retaken, 137 ms median per clip (355 ms at
   most, with other work on the GPU). In set reading (`TF_D1_ATTENTION` unset) the three questions share one pass:
   73 ms on an idle GPU.
4. **keep**: the clip is checkpointed, posted to the panel and joined into the film on demand.

### Next, behind the same interfaces

- **Dense Cosmos checks** (a `CosmosChecker`, beside the d1 checker above). These follow Imagination Studio's AutoPilot (`feature/autorl-generation`:
  `app/checks.py`, `app/continuity.py`) as ideas, not code. Frames would be read in windows of 2 s, three frames each,
  plus the join between clips (the last frame before beside the first frame after). Short yes/no questions would be
  built from a small continuity record: who is in the film, what they wear, the place, the era. Pass or fail would be
  decided in code, and a failed major question would be asked again and only fail by majority. The pace of action clips
  would be measured from frame differences. They would run on the same Cosmos3-Edge server Boo's eyes use, between
  clips and only while nobody is talking. Minor findings become planner notes, and confirmed major ones become retakes.
- **The HyperFlow engine** (the `hyperflow` renderer). This would be one long-lived generation that hands over clip after
  clip, each continuing from the latent tail of the one before instead of a decoded frame. That gives seamless joins
  and real step previews from the start (`previews = True`). It would expose the same `submit`, `poll` and `fetch`
  calls, with `Progress.preview` set on each step.
- **Story memory**: a scene graph that folds older events into chapters (HyperFlow-Infinity's design) would replace
  the flat beat list once runs go past a few dozen clips.
- **Branching**: start a new run that shares the first N clips with an earlier one.

## Credits

The idea of an endless film that watches itself and decides what happens next comes from Echo-Infinity (Yuxuan Bian,
Nan Duan, Qiang Xu et al.) and Video Rebirth's [HyperFlow-Infinity](https://github.com/Video-Rebirth/hyperflow-infinity)
(Apache-2.0). Only the idea is used; no code from either is in this repository. Clips are rendered by
[Imagination Studio](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman) and its engines, chiefly
MiniMax H3 (FastH3 on a DGX Spark) through ComfyUI. ffmpeg (LGPL/GPL, not shipped here) makes the posters and joins the
film.
