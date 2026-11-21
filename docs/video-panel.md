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
- **status**: `planning`, `keyframe`, `loading`, `rendering 45%`, `checking`, `stopping`, `done`, `stopped` or
  `failed`, with the DiT forward (`step 4/8 · ~40 s`) and a progress bar;
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
such as the memory governor's budget, goes to the task card and the panel's status line.

When the profile has a `video` section, the front brain's instructions also get one line (`server.VIDEO_NOTE`): it can
make videos in the video panel, the video agent makes them, and it never says it can't.

| Say | Does |
|---|---|
| "Boo, make a video of a lighthouse in a storm" | one clip |
| "make a video" … "a red kite over green hills" | asks what it should show, then one clip of the answer |
| "make an endless video of waves on a reef", "… a story about …" | continue mode: clip after clip until stopped |
| "stop the video" | stops and cancels the render at work |
| "keep going", "make it longer", "make it twenty more seconds longer" | an AutoPilot film: more pieces; a finished one carries on if asked within 10 minutes |
| "now make it rain", "let it snow", "add a seagull" | an AutoPilot film: a Director note from the next piece |
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
its checkpoint. A clip that already has a render job is followed, never sent again. The renderer's own paths stay on the
server: the page sees only `/api/video/...` addresses.

### The renderer

One renderer, this app's own (`gma/studio.py` `HeadlessRenderer`, the `Renderer` interface: `room`, `keyframe`,
`submit`, `poll` → `Progress`, `picture`, `fetch` and `cancel`):

- **keyframes** are `scout` jobs of the headless Imagination Studio (`edit` jobs beside reference pictures);
- **clips** are made by the FP4 clip worker (`gma/clips.py`, `gma/fp4_worker.py`): FastH3 V2 on FastVideo's FP4
  sparse-attention recipe, image-to-video with stereo audio from the keyframe (or the last frame of the clip before,
  or text alone). There is no other route: a clip the worker cannot make fails with its reason.
- **room** is the memory governor's (`gma/governor.py`): a clip's need is the worker's measured peak, and the schedule
  says what stays resident beside it.

### Profile

```json
"video": {
  "renderer": "studio-headless",
  "seconds": 5, "max_clips": 6, "keyframe": true, "retakes": 1,
  "speak_updates": true, "speak_every": 3,
  "dir": "~/ghost-studio/home/ghost/video", "poll_s": 1.0
}
```

`max_clips` caps a continue run (0 means until stopped). The renderer's settings are the profile's `studio`,
`governor.renderer` and `engines.fp4` sections (README, "The FP4 clip worker").

## Live generation: what the engines expose

The live view is driven by what the render host reports, and the page labels it honestly:

- **`live preview`**: real pictures of the clip as it renders. The renderer names each one in `Progress.preview`. The
  service fetches each new one once, and the page redraws it about four times a second.
- **`progress only`**: no pictures, only the sampler's step count and an estimate. The page shows the clip's opening
  still if one is known (the keyframe, or the last frame of the clip before in continue mode), otherwise noise. The
  picture sharpens and the noise fades as the steps go by. This is a progress display, not the model's picture.

**The FP4 clip worker gives progress only** (its stage and the DiT forward count, 8 a clip): the panel shows the
keyframe with the stage ("stages"). The notes below are about the Studio's own clip jobs, which Ghost Studio no longer
uses. On a DGX Spark, its H3 video jobs run in ComfyUI. `spark_generate.py`
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

Video jobs on the Mac run H3 in MLX outside ComfyUI, so a step preview
there needs the engine to write one too.

**Keyframes**: with `"keyframe": true`, the still is made as a Studio `scout` job before the clip and gets the same
step progress. The finished still becomes the clip's opening picture in the live view.

**Hand-off**: when a clip finishes, it takes the main view and plays through once. Then the live view of the next clip
comes back, with the clip just made looping small in the corner (tap it to play it full size). When the run ends, the
newest clip loops. Clips in a continue run open on the frame the last one ended on, so the clips run on one into the
next.

## Infinity: continue mode

Continue mode is the first step of HyperFlow-Infinity's idea, an open-ended, continuously directed film that checks
itself. Each turn of the loop:

1. **plan**: the profile's brain (Bolt) writes the next beat and prompt from the idea, the story so far (one beat a
   clip, the last eight) and the last clip's prompt. It must follow on from the last frame, add something new and never
   restage an earlier beat. The planner first waits up to 10 s for a quiet moment in the conversation, so a voice turn
   never queues behind it on Bolt. If the brain gives no answer, the clip carries on with the last prompt.
2. **render**: the clip opens on the last clip's final frame (taken with ffmpeg, given to the clip worker).
3. **check**: `Checker.check(clip_file, clip, run) → Verdict(ok, notes, retake)`. The checker in this version passes
   everything. If a verdict asks for a retake, the clip is rendered again with a new seed (`retakes`, 1 by default), and
   its `notes` go to the planner for the next clip.
4. **keep**: the clip is checkpointed, posted to the panel and joined into the film on demand.

### Next, behind the same interfaces

- **Dense Cosmos checks** (a `CosmosChecker`). These follow Imagination Studio's AutoPilot (`feature/autorl-generation`:
  `app/checks.py`, `app/continuity.py`) as ideas, not code. Frames would be read in windows of 2 s, three frames each,
  plus the join between clips (the last frame before beside the first frame after). Short yes/no questions would be
  built from a small continuity record: who is in the film, what they wear, the place, the era. Pass or fail would be
  decided in code, and a failed major question would be asked again and only fail by majority. The pace of action clips
  would be measured from frame differences. They would run on the same Cosmos3-Edge server Boo's eyes use, between
  clips and only while nobody is talking. Minor findings become planner notes, and confirmed major ones become retakes.
- **Story memory**: a scene graph that folds older events into chapters (HyperFlow-Infinity's design) would replace
  the flat beat list once runs go past a few dozen clips.
- **Branching**: start a new run that shares the first N clips with an earlier one.

## Credits

The idea of an endless film that watches itself and decides what happens next comes from Echo-Infinity (Yuxuan Bian,
Nan Duan, Qiang Xu et al.) and Video Rebirth's [HyperFlow-Infinity](https://github.com/Video-Rebirth/hyperflow-infinity)
(Apache-2.0); no code from either is in this repository. Keyframes are drawn by
[Imagination Studio](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman); clips are MiniMax H3 (FastH3
V2 NVFP4) on [FastVideo](https://github.com/hao-ai-lab/FastVideo) (Hao AI Lab, Apache-2.0) and its FP4
sparse-attention recipe. MiniMax H3 is under the MiniMax H3
Community License, which excludes the USA, the EU, the UK and South Korea without MiniMax's authorization. ffmpeg
(LGPL/GPL, not shipped here) makes the posters and joins the film.
