# Ghost Studio: build notes and measurements

The working notes of the Ghost Studio branch, kept as they were written while it was built on one DGX Spark: the FP4
clip worker, the self-check, the memory guard, the headless Studio and the four build phases. Paths such as
`~/ghost-studio/...` are that machine's layout. The overview is in the [README](../README.md).

## The FP4 clip worker

`gma/fp4_worker.py` (the process) and `gma/clips.py` (the app's side). Every clip Ghost Studio makes, and nothing else
makes one: **FastH3 V2 on FastVideo's FP4 sparse-attention recipe** (FastVideo's
`basic_fasth3_spark_v2_nvfp4.yaml`: NVFP4 DiT with the `h3_dit_vsa` profile, MiniMax H3 video sparse attention at 0.8
with 64-token tiles and its FP4 attention kernel, the cuDNN NVFP4 GEMM, eight DiT forwards, the light H3 VAE). It is
**image-to-video with stereo audio**: MiniMax H3's first-frame mode (FL2VA) from the piece's keyframe, from the last
frame of the clip before for a piece that continues, or text alone; same 864x480 (1376x768 for "sharp"), 243 frames
(10 s) at 24 fps, the Director's composed prompt (its sound fields included). The Studio's int8 clip route is gone
from the code and the profile; if the worker cannot make a clip, the take fails with the worker's reason.

- **One resident process**, a managed engine (`engines.fp4`, role `clips`, port 8420) in a venv of its own: the
  governor loads it for a clip and keeps it between takes (`live` and `sharp` keep it), and releases it for a keyframe
  or the film's sound (their schedules do not), or when a voice engine could not load beside it while it idles.
  `/health`, `POST /v1/clips`, `GET /v1/clips/{id}` (stage and DiT forward count: the brain's release follows
  "denoising"). A voice turn SIGSTOPs it for at most 4 s, as it did the Studio's ComfyUI.
- **Memory**: the 26 GB of AdaLN projections are never loaded. The release ships their outputs for the text-to-video
  ladder (`transformer/adaln_tables.pt`); first-frame clips add a third row, so Ghost Studio computed those entries
  once with FastVideo's own `FASTVIDEO_H3_ADALN_DUMP` from the real projections and merged both into
  `~/ghost-studio/models/fastvideo/adaln_tables_t2v_fl2va.pt` (370 MB; clips bit-identical to the full model's). The
  worker then holds ~40 GB between clips and peaks at 55.6 GB (864x480x243) or 66.9 GB (1376x768x243), MemAvailable
  measured; `gma/clips.py NEED_MB` holds those.
- **Load in 24 s** (isolated; 41-50 s in the film, the page cache shared with the keyframe model): safetensors' CPU
  tensors are copied out of their mmap as they are read. On GB10 a tensor still backed by the file's mmap moves to the
  GPU ~300x slower (4.8 GB: 28 s against 1 s); unpatched, FastVideo's load took 93 s.
- **Compiled VAE decode** (the default since 2026-11-16; `--no-compile-vae` decodes eagerly): 32.5 s against 44 s for
  864x480x243 and 57 s against 73 s for 1376x768x243, measured on the DGX Spark; the first clip of a size compiles for ~6 s more.
  It used to fail at a size first seen in a later clip (`assert torch._C._is_key_in_tls`): torch keeps a compiled
  module's CUDA-graph trees in the thread that first imported them, FastVideo runs each clip's forward pass in a thread
  of its own, and a new size has to record a new graph. The worker shares the trees across its threads at load
  (`share_cuda_graphs`; one clip renders at a time).
- **FlashInfer's JIT** builds once with `MAX_JOBS=4` (`python -I gma/fp4_worker.py --model x --jit-only`, 47 s) into
  `~/ghost-studio/cache`, never beside the resident model. The external guard (`~/ghost-studio/guard.sh`, started by
  `run-app.sh`) kills the worker's whole session first under 12 GB MemAvailable.

Setup on a DGX Spark, all under `~/ghost-studio`:

```
# the venv: torch 2.12.0+cu130, FlashInfer 0.7.4rc1, transformers, diffusers ... (93 packages, from uv's cache)
uv venv --python 3.12 fv-venv && VIRTUAL_ENV=fv-venv UV_TORCH_BACKEND=cu130 uv pip install torch==2.12.0+cu130 \
  torchvision==0.27.0+cu130 torchaudio==2.11.0+cu130 triton flashinfer-python transformers tokenizers safetensors \
  diffusers accelerate peft einops numpy pillow imageio imageio-ffmpeg av pyyaml huggingface-hub loguru tqdm requests \
  psutil regex cloudpickle remote-pdb ftfy scipy ninja
# FastVideo a01c50a8, its package only (fastvideo-src/fastvideo, on the venv's path by a .pth), and fastvideo-kernel
# 0.3.5 built for sm_121a (fastvideo_kernel, attn_qat_infer, fp4attn_cuda, fp4quant_cuda in site-packages)
hf download FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer --revision 9d0d0d3a \
  --local-dir models/fastvideo/FastH3-V2-Consumer --exclude vae/minimax_h3_video_vae_int8_convrot.safetensors
```

**LoRA**: Ghost Studio's clips use none (AutoPilot's `loras` is empty, and the Studio route never passed one to its
clip jobs either). FastVideo takes a LoRA for an NVFP4 DiT only when the generator is made (packed FP4 linears run it
as an unmerged side branch); changing or merging one at runtime is refused once the BF16 weights are gone, and a LoRA
that targets the AdaLN projections cannot work beside the AdaLN table. So a LoRA here means a worker started with it
(`lora_path`) and without the table for AdaLN LoRAs; the Studio's LoRAs (ComfyUI files for its int8 blocks, such as
`keys_hdr`) were not converted or tried.

### Spark film (2026-11-07, DGX Spark, FP4 clip worker)

The same synthetic spoken request as before ("Make a video of a lighthouse keeper lighting the lamp as a storm rolls
in.", Kokoro voice), "make it rain" said during piece 1's render, stock questions during every DiT pass; 3 pieces of
10 s at 864x480, Qwen3.8-27B brain and Director, d1-2B judge, FLUX 3 Klein Fast keyframes by the Studio. The film:
`~/ghost-studio/out/fp4-film/ghost-studio-fp4-film.mp4` (29.9 s, narration and score), its pieces and frame sheets
beside it; run `r1107-080724-640b2`; the report `~/ghost-studio/out/fp4-film-smoke.json`.

| per piece | FP4 clip worker | before (Studio int8 jobs) |
|---|---|---|
| Director | 47-52 s | 44-55 s |
| keyframe | 44-46 s (`edit` beside the place; the Studio's ComfyUI started cold each time, `after_render: stop`) | 35-39 s |
| clip (load / denoise / decode) | **214.5 / 228.7 / 216.0 s** (41-50 / 121-131 / 43-44) | 211-217 s (21 / 165 / 31) |
| judge | 8-11 s | 4-10 s |
| **a piece** (Director to the join) | **4.7 / 5.8 / 5.4 min** | 4.6-5.3 min |
| the clip alone, nothing else on the GPU | 139-151 s (load 24 s apart) | 187-217 s |

The worker's render (denoise + decode) is ~20 s faster than the Studio's in the film and ~60 s faster alone, but in
the film each piece pays the worker's load (41-50 s, against ComfyUI's ~20 s), its DiT slows from 107 to 121-131 s
beside the voice brain answering a question every 6 s, and the keyframe's ComfyUI now starts cold. So a piece is not
faster than before; the clip is the FP4 recipe's, as asked.

- **Memory**: the lowest MemAvailable over the film was 34.3-35.1 GB (piece 1's clip beside Qwen3.8-27B, Kokoro,
  Parakeet and Cosmos3; boo-transcribe had stepped aside); the 12 GB guard never fired. Before: 21.5 GB.
- **Voice to voice during the DiT pass**: n 23, p50 1.15 s, p95 1.78 s (the brain kept beside the clip; before 2.0 /
  3.3 s); the "make it rain" steer 0.27 s. 10 of the 23 DiT-pass turns got no spoken reply: Qwen ended them itself
  after 1-7 tokens (`finish=stop` in its log, e.g. "[neutral] I am doing") from the second round of the same stock
  questions on, in one conversation of 37 turns (before: 4 of 41 in three shorter ones). Not the render's doing as far
  as the logs show; worth a look on the voice side.

## The self-check (gma/ap/qc.py)

**Design since 2026-11-07 evening (v2, after the user found the retake-heavy film disjointed):** continuity outranks
the per-shot checks. A piece that continues the one before draws no keyframe: it opens on that piece's kept last frame
(FastH3's first-frame mode), and a retake never redraws the opening picture (only the seed and the action's words
change). Pieces are kept in order; a later piece never sends an earlier one back. Every check is asked of **Gemma 4
12B with vision** (vLLM, log-probabilities: yes/no, and A/B asked in both orders), which also writes a free account of
what the frames show for the next Director; d1-2B stays the judge. A retake (budget 1) is only for a confident, visible
failure: a jump at the seam (layout correlation < 0.5, or Gemma <= 0.2), a second copy of a person, the wrong side of a
door, the wrong direction, a lamp lit before the lighting where a piece opens, a hard cut; d1-2B only for its person
or cut faults. Everything else (contact, end states, d1's doubts, the keyframe checklist, a continuing piece's
inherited first frame) is a note the Director reads with the next piece.

Measured (same request, `~/ghost-studio/out/{base-film,qc3-film}`, side by side in `out/side-by-side.mp4`):

| | QC observing, no retakes (r1107-175548-17aa8) | QC v2 (r1107-182630-bcdc8) |
|---|---|---|
| film time | 27.3 min | 25.6 min |
| retakes | 0 (notes only) | 1 keyframe redraw (piece 1 showed the lamp lit), 0 clip retakes |
| seams 1->2, 2->3 (corr / Gemma) | 0.993 / 0.995, 0.971 / 0.994 | 0.972 / 0.994, 0.972 / 0.996 |
| left in the film | lamp glowing from piece 1 on; a hard cut at 0.8 s in piece 3 | lamp dark until he turns the key |

(The retake-heavy v1 film's seams were 0.705 and 0.846; the original FP4 film's -0.074 and 0.014.) With d1-2B as the
looker the same film took 17.1 min; Gemma's ~20 GB does not fit beside the resident FP4 worker and the brain, so the
idle worker gives way and reloads each piece (55-126 s): that, the cold first load and the checks (~1 min a take) are
the gap to the 16 min of a film without checks. Evicting the brain instead of the worker for the check step would save
most of it.

**Ghost 3.8 27B as the brain and the looker (2026-11-08; a selectable profile, not the default):** one resident TensorFold 1.2.4
`--vision` serve (MLX 4-bit + DFlash2, ~21.7 GB, port 8092) is the voice brain, the Director and the self-check's
looker; in that profile Qwen3.8 and Gemma are out of the schedules. The release serve gives no
log-probabilities and accepts PNG only, and a three-picture request crashed the GPU (CUDA 719, then Xid 119 until a
reboot), so `GhostLooker` sends one picture a request (frames tiled into one numbered sheet), asks yes/no as a 0-10
rating, and asks A/B as describe-then-`ANSWER: A/B` in both orders; its ratings agree with almost any statement, so
the people check is a choice against "exactly one". Same request (`~/ghost-studio/out/ghost-film/`, run
`r1108-081411-28336`, side by side with QC v2 in `ghost-vs-qc2-side-by-side.mp4`):

| | d1-only baseline | QC v2 (Qwen + Gemma) | Ghost 3.8 |
|---|---|---|---|
| film time | 17.1 min | 25.6 min | 17.6 min |
| FP4 worker loads | - | reloads each piece (55-126 s) | 1 (the first, 59 s), 0 reloads |
| retakes | 0 | 1 keyframe redraw, 0 clip | 1 keyframe redraw (piece 1, housing), 0 clip |
| seams 1->2, 2->3 (corr / looker) | - | 0.972 / 0.994, 0.972 / 0.996 | 0.995 / 1.0, 0.997 / 1.0 |
| lowest MemAvailable | - | - | 21.6 GB (guard floor 12) |
| voice turns unanswered | - | 3 of 54 | 1 of 45 (0 during the DiT passes; p50 1.3 s) |

Left in the film: piece 3's contact (the hand closing the housing latch) not seen, a note; the cap comes and goes
in piece 3; the lit lamp reads as fire to the looker ("engulfed in flames"). Looker A/B on the old film's defects
(lamp lit early, stair direction, wrong side of the door, end place) plus contact: Gemma caught 3 and called the
flame on the wick touching (wrong); Ghost caught all 4 through the self-check (3 on the strict single questions).

**Verdict (the user, having watched both films):** "the qwen+gemma seems more comprehensive and better detail vs the
ghost studio". QC v2 (Qwen3.8 brain + Gemma 4 12B looker) stays the default for its film's detail and
comprehensiveness, despite Ghost's faster film and better seam numbers; Ghost 3.8 is kept as a selectable profile
(`engines.ghost` on port 8094 in `profiles/spark.example.json`; `~/ghost-studio/profile-spark3-ghost.json`).

**Strict Ghost (2026-11-08, a test profile: `~/ghost-studio/profile-spark3-ghost-strict.json`).** Two opt-in
changes:
- `eyes_choices.ghost.looker = "strict"` (`StrictGhostLooker`): the frames are described first, then each statement is
  asked yes/no in both orders; a check passes only on agreement and ratings of 9-10.
- `settings.rich_shots`: a Director rule that every shot names a camera move, the light and colour, a music cue, pace
  beats and a close detail at the contact or change.

Same request (run `r1108-092902-8a839`, `~/ghost-studio/out/strict-film/`, 3-way side by side
`qc2-ghost-strict-3way.mp4`):

| | QC v2 (Qwen + Gemma) | Ghost | strict Ghost |
|---|---|---|---|
| film time | 25.6 min | 17.6 min | 23.5 min |
| failed checks (notes to the Director) | 15 | 1 | 14 (2 confident) |
| keyframe redraws / clip retakes | 1 / 0 | 1 / 0 | 0 / 1 (piece 1: walks down, not up) |
| seams 1->2, 2->3 (corr / looker) | 0.972 / 0.994, 0.972 / 0.996 | 0.995 / 1.0, 0.997 / 1.0 | 0.988 / 1.0, 0.835 / 1.0 |
| music cue / camera moves | yes / push-ins, tilt | N/A / static | yes / handheld tilt, push-ins |
| FP4 worker loads, lowest MemAvailable | reload each piece | 1, 21.6 GB | 1, 19.8 GB |

The strict looker reported honestly: piece 1 went down the stair in both takes, and it said so. Pieces 2-3 are close
on the hands: oil poured, the match struck, the wick lit inside the glass. QC v2 stays the default.

**Big-picture Ghost (2026-11-08, run `r1108-132110-1aa7d`, `~/ghost-studio/out/bp-film/`; strict looker, fixed
start and people checks, rewritten retakes, `big_picture` with `fix_budget` 1, memory guard and thermal gate on).**

- **Draft:** made with no hang; the two earlier drafts froze on their third back-to-back clip.
- **Thermal gate:** 14 waits, 93 s in all, at most 10 s each; each clip and load started at 70 °C or under. Peak GPU
  was 89 °C, the hottest zone 96.5 °C. Hardware thermal slowdown lasted 47 s and software slowdown 202 s; no live
  thermal stop was needed.
- **Memory:** lowest MemAvailable was 29.8 GB.
- **Fix plan:** 13 findings were merged into 3 fixes (none dropped), and the film was made again from piece 1.
- **Do no harm:** the remake scored 0.383 against the draft's 0.389, and its weakest seam fell to 0.628 from 0.979, so
  the draft was put back. Film time: 35.2 min.
- **On screen:** close shots of the key, the lever and the wheel. But he lights a small lantern, not the lighthouse
  lamp, and he never climbs; that story was the Director's arc, so it is no check failure.

**Attribution (A/B, the same keyframes, prompts and seeds through the Studio's old int8 job, for the comparison only):**
the wrong direction on the stair and the lamp lit from the first frame (and the flame above the glass) came out on both
paths: they are the prompts and keyframes (the piece 3 keyframe already showed the lamp lit; the stair keyframe faced
the camera from above) and the model, not FP4. The hard cut at 6.3 s of piece 1 came out on FP4 only (int8 kept one
continuous shot on that seed). Clips: `~/ghost-studio/out/ab/`.

**Cross-piece continuity.** A piece that continues the one before (no scene change) opens on that piece's own last
frame: its keyframe is an `edit` of the kept take's last frame at full size ("Continue image 1 exactly: the same place,
every person where they stand, every object in the same state"), never a freshly drawn scene, and the Director is told
what that frame was verified to show. Added after the user saw the original film's pieces 2-3 read as two Arthurs (he
walked out onto the gallery in piece 2, and piece 3 opened with him inside at a lit lamp).

**Prompting** (the Director's rules and its answer): one action a segment; where the camera stands and looks, and the
direction of movement relative to it; the exact point of contact for an interaction; the scene renamed when the place
changes; the keyframe is the shot's START, every changing object in its start state, nothing of the outcome (and its
prompt ends with "At this moment, before anything happens: <start states>"); through a door, which way, and the wrong
way beside it; each person once. The Director also fills `qc`: `start`, `end`, `motion`, `wrong_motion`, `contact`,
`start_place`, `end_place`, `wrong_place`, `people`.

**The checks**, every one a yes/no question to d1-2B (`/v1/systemone`, resident, calibrated) except the cut:

| check | asked of | fails when |
|---|---|---|
| keyframe start | the keyframe, before anything moves | a start state is not a yes (the keyframe is drawn again, the Director rewriting its prompt) |
| start state / end state | the first / the last frame | a state is not a yes |
| direction | first, middle and last frames, in time order | `motion` is not a yes and no likelier than `wrong_motion` |
| contact | the last three frames | `contact` is not a yes |
| hard cut | every frame at 12 fps (numeric) | a frame's 16x9 layout correlates under 0.3 with the one 1/3 s later |
| place | the first frame (`start_place`), the last frame (`end_place` against `wrong_place`) | the place is not a yes, or the wrong side of the door is likelier |
| people | the keyframe, the first, middle and last frames | "two or more of this person, through a window, a doorway or in a reflection" is a yes |
| carried | the keyframe and the first frame | the piece before's verified end (its passed end states and end place) is not a yes |

A failed check is a finding that says what failed and what to change; the take is rendered again with the shot
rewritten by the Director from it (`retakes`: 2), and when the retakes run out the take with the fewest failures is kept,
the reasons beside it. Every answer is in the segment folder (`qc-keyframe<N>.json`, `qc-take<N>.json`, and the
take's `continuity-take<N>.json`); the watch record carries `seen` (the answers) beside the plan.

On the old film's own frames d1-2B said: piece 3's keyframe "the oil lamp is unlit" 0.03 (fails), piece 1 "climbs
upward, away from the camera" 0.28 against "walks down toward the camera" 0.41 (fails), the stair cut at 6.33 s
(correlation -0.11, fails; the judge's own cut answer had passed it at 0.19). The flame outside the glass was not
caught: "the match flame touches the wick" came back 0.63. **The film again with the self-check** (2026-11-07, run `r1107-093957-80b18`, `~/ghost-studio/out/qc-film/`, the
checks' answers beside it): Arthur stays one person and moves one way, outside the tower, through the door, up the
stair away from the camera, into the lantern room, and lights the lamp (dark at piece 3's start, lit at its end); each
piece continues from the last frame of the one before. What fired: keyframe start states (6, 7 and 9 keyframes drawn
for pieces 1-3), carried continuity (piece 2 take 1, piece 3 takes 1-3), start and end place, start and end states,
two hard cuts (8.8 s and 1.5 s) and the judge's cut doubt once. Every piece used both retakes; no take passed every
check, so the take with the fewest failures was kept (piece 1 take 1, piece 2 take 3, piece 3 take 3). The film took
63 min against 16 without the check. Limits seen: d1-2B cannot verify a contact (a hand on a handle, a flame on a
wick: 0.40-0.49 on every take, a pass only once), and many answers sit near 0.5 on the Director's longer statements,
so the check is strict and retakes often; a vision model that describes, or contact as a note instead of a retake,
would cut the retakes.

**No describer writes free text on the DGX Spark**: the Cosmos3-Edge
and Qwen3.8-27B TensorFold builds Ghost Studio runs take text only (they answer 400 to an image), so AutoPilot's eyes
(its keyframe checklist and watcher) had not been seeing anything there; the self-check records d1-2B's answers as
what was seen instead. A vision build of either (or Gemma 4 12B) would add descriptions.

## The memory guard (gma/memguard.py, the governor's admission)

Added 2026-11-08 after a big-picture film hung the DGX Spark: the kernel's GPU driver ran out of memory (NVRM NV_ERR_NO_MEMORY)
during a clip while MemAvailable was still above the external guard's 12 GB floor (it fell too fast to catch).

- **Admission before every load and render** (gma/governor.py). A load's peak (`need_gb`) or a render's (the FP4
  worker's measured peak) is compared with MemAvailable less the floor (`governor.floor_gb`). If it does not fit, this
  app's idle models are released first, biggest first, never one that is answering. Then it waits up to
  `memguard.admit_wait_s` for room another program is giving back. Then it is refused, with the biggest memory users
  named. A load that will not fit is never started.
- **Models per stage.** The schedules say which models stay beside each step. On the Ghost profiles the brain and
  looker (Ghost 3.8, ~22 GB) is released for every clip and loads again for the checks and the Director. The FP4 worker
  (~52 GB) gives way when the brain or the looker needs the room between clips.
- **Live guard** (a thread in the app, `governor.memguard`). It polls every `poll_s` (1 s) and acts under `floor_gb`
  (20) or when `/proc/pressure/memory` "some avg10" is over `psi_limit` (20). It acts in this order:
  1. release the next idle model;
  2. stop the current step (the clip worker is ended; the run fails cleanly with the reason and the biggest users
     named).

  Every action goes to `logs/memguard.jsonl` and `/api/governor`. `~/ghost-studio/guard.sh` stays as the backstop
  under it: 14 GB, or pressure over 40.
- **Foreign processes** (not started by Ghost Studio) are named, never killed, unless `memguard.kill_foreign` lists a
  pattern of their command line (opt-in).
- **Thermal gate and stop** (`governor.thermal`, 2026-11-08). A soak of the FP4 worker alone on the DGX Spark held the GPU at 88-89
  °C on back-to-back clips, and the thermal slowdown grew each clip; two big-picture drafts froze on their third
  back-to-back clip.
  - **The gate:** before every clip, keyframe and big load (`big_load_gb`), wait until the GPU is at or under 70 °C and
    the hottest zone at or under 85 °C (every 2 s, at most 300 s). After the timeout the step starts warm with a
    warning, unless the hottest reading is at or over 88 °C: then it is refused cleanly. The wait shows on the film's
    status line ("cooling the GPU ...") and in `/api/governor`.
  - **The live stop:** the live guard stops the step at a GPU of 90 °C or more, or after 10 s of hardware thermal
    slowdown (clock-event bits 0x8 or 0x40). The clip is made again once, after the gate has let it cool.

## The headless Studio

`gma/studio.py`. The Studio is a child engine of the app (`"studio"` in the profile; the registry starts it with the
app, under the leash, or adopts one already answering on its port):

```
PREFIX=~/.local/opt/tensorfold-studio STUDIO_HOME=~/ghost-studio/home HOST=127.0.0.1 PORT=7890 \
STUDIO_PIDFILE=~/ghost-studio/home/app.pid SPARK_COMFY_PORT=8290 \
SPARK_COMFY_ARGS="--use-ck-attention --input-directory <home>/.comfy/input --output-directory <home>/.comfy/output ..." \
  $PREFIX/venv/bin/python <studio checkout>/app/server.py
```

- **Its own port, home and ComfyUI**: 127.0.0.1:7890 (never a worker's 7870), its own Studio home (projects, renders,
  voice and music takes), its ComfyUI on 8290 (never 8190) with its input, output, temp and user folders under that
  home. The install (PREFIX) and its models (`~/spark-studio-models`) are shared read only; the Studio itself still
  writes `comfy-8290.log`/`.mem` into PREFIX and refreshes its own `tensorfold_h3` node files in the install's
  ComfyUI (the same files a worker writes), as any second Studio on one install does.
- **No updater, no Sparks, no background work**: `STUDIO_PIDFILE` is the Studio's switch for an app another app
  starts (a worker Spark's): its self-updater is not managed, the launcher is bypassed and the background captioner
  does not start. Its home has no Sparks configured, so it hands no job to a worker and is no worker itself.
- **One render host at a time**: the Studio's GPU lock (`~/studio-gpu.lock.d`) is the machine's, as for any Studio,
  but only Drama 3, MiniMax Music 3's song builds and LoRA training take it. A worker Studio on the same Spark (`.3`'s,
  port 7870) does not see Ghost Studio's clip worker: keep it stopped while Ghost Studio renders (it was stopped
  throughout the tests). Services that step aside for renders on `.3`
  (boo-transcribe) see the Studio's ComfyUI process and do so.
- **The routes used**: `/api/state`, `/api/jobs` (`scout` keyframes, `edit` keyframes beside the place's picture and
  the cast's portraits, `soundtrack`),
  `/api/jobs/{id}` and `/cancel`, `/api/voice/takes` (Fish Audio Drama 3), `/api/music/takes` (MiniMax Music 3),
  `/api/music/soundtrack`, `/api/upload`, `/files/...`. Files are named relative to the Studio home; Ghost Studio's
  own folders live inside it (`<home>/ghost/`), so keyframes are written in place, nothing copied.

What moved to the Studio, and what stayed here:

| | before (GMA) | Ghost Studio |
|---|---|---|
| keyframes, keyframes beside references | this app's own ComfyUI (`gma/comfy_server.py`), the Studio's graphs vendored (`gm_core/comfy_graphs.py`), the Mac's scripts (`flux3_edit.py` ...) | the Studio's `scout` and `edit` jobs (`HeadlessRenderer`) |
| clips | this app's own ComfyUI, then (until 2026-11-07) the Studio's `video` jobs (TensorFold int8) | the FP4 clip worker (`gma/clips.py`, `gma/fp4_worker.py`), only |
| a render's memory estimate | vendored `gm_core/memory_guard.py`, then the Studio's `POST /api/estimate` | a clip: the worker's measured peak (`gma/clips.py`); a keyframe or the sound: the profile's stated figure |
| the film's narration and score | none | Drama 3 take + MiniMax Music 3 song + the Studio's soundtrack mix (`gma/ap/sound.py`) |
| the voice loop, Boo, the face, the camera, AutoPilot (story, d1-2B judge, continuity, pace, resume), brain lanes, governor schedules, engines (brain, Kokoro, Parakeet, d1, Cosmos3) | here | here (unchanged) |

Retired: `gma/renderer.py`, `gma/comfy.py`, `gma/comfy_server.py`, `gma/comfy_nodes/`, `gm_core/memory_guard.py`,
`gm_core/workload.py`, `gm_core/comfy_graphs.py`, `gm_core/flux3_generate.py`, `gm_core/flux3_edit.py`,
`gm_core/edit_plan.py` (3,129 lines) and their tests. Kept, with their provenance headers: `gm_core/engine.py` and
`leash.py` (the child engines, the Studio itself among them), `memstat.py`, and `h3_prompt.py`, `lengths.py`,
`negation.py` (pure functions AutoPilot calls many times a piece in its own thread; the Studio's `/api/compose` would
be the same words over HTTP).

**The governor.** A clip's need is the FP4 clip worker's measured peak (gma/clips.py), less what the worker already
holds when the schedule kept it from the take before; a keyframe's and the sound's are the profile's stated figures. The
schedules decide what is evicted beside each (`live` and `sharp` keep the worker; `keyframe` and `audio` release it).
After a keyframe or the sound pass the Studio's ComfyUI is stopped (`after_render: "stop"`: its `/free` alone left
25.5 GB held on the DGX Spark), so the clip worker has the room; the brain, when a clip does not fit beside it, is released before
the worker loads (`brain_during_dit`) and loads again after the clip.

**Studio API gaps** (what Ghost Studio makes up itself, each said in `gma/studio.py`):

- *Freeing the keyframe's memory*: no route unloads the Studio's ComfyUI (the page never needs it: it stays warm on
  purpose); Ghost Studio posts `/free` to that ComfyUI's own port and stops it with the Studio's `spark/comfy.sh stop`.
- *Jobs are not kept across a Studio restart*: AutoPilot then finds the file, or makes the keyframe again.

**The Slate director** (`/api/slate/direct`, the Studio's logline-to-cut-film pipeline) was read and not adopted: it
writes, casts and renders a whole Slate film as a batch (three jobs in the queue at once, its QA gate per shot), with no
per-take vision judge, no voice steering while it runs and no resume into a live voice loop; AutoPilot's arc, d1-2B
judge, continuity bible and place anchors are what the phase 3 and 4 numbers below were bought with. **Bolt** stays
Ghost Studio's own engine (or Qwen3.8-27B, the default): the Studio's Studio-managed Bolt is unloaded before any job
that does not fit beside it and held until the queue is empty, coarser than the release for the DiT pass the voice
loop needs, and Boo's voice turns cannot wait behind the render queue. The same holds for Boo's own voice: a Drama 3
take of the Studio is a queue job, so only the film's narration goes through it.

### Spark smoke (2026-11-07, DGX Spark, Studio main 1ae1be4 headless)

The same synthetic spoken request as phase 4 ("Make a video of a lighthouse keeper lighting the lamp as a storm rolls
in.", Kokoro am_michael by `tools/synth_turns.py`) streamed into the app's microphone by `tools/voice_film_smoke.py`,
"make it rain" said during piece 1's DiT pass, stock questions during every DiT pass. 3 pieces of 10 s, 864x480 fh8,
Qwen3.8-27B (MLX 4-bit + DFlash2, TensorFold 1.1.8) as the brain and the Director (thinking 2048), d1-2B as the judge,
Kokoro and Parakeet bf16 as the voice and the ears, Cosmos3-Edge resident as the voice fallback; keyframes FLUX 3
Klein Fast (Qwen-Image is not installed on the DGX Spark) through the Studio. Outputs in `~/ghost-studio/home/ghost/video/`
(`r1107-001507-e02d2`: `film.mp4`, `film-final.mp4` with narration and score) and the reports in
`~/ghost-studio/smoke/`.

| per piece (a take: Director, keyframe, clip, judge) | Ghost Studio (Studio jobs) | GMA phase 4 |
|---|---|---|
| Director | 44-55 s (Qwen resident) | 30 s (Bolt loaded beside the checks) |
| keyframe | 35-39 s `edit` beside the place (FLUX3-K, multi-reference); 68-71 s for the opening `scout` with the Studio's ComfyUI start | 62 s with its check |
| clip (load / DiT / decode) | 211-217 s (21 / 165 / 31) | 210 s (~20 / ~160 / ~30) |
| judge | 4-10 s (d1-2B, its load in it) | 118 s (Gemma's cold load ~77 s, then 41 s) |
| **a take** | **~4.6-5.3 min** (piece 1: 4.6 min from the Director to the join, twice; piece 2: 5.0 min) | ~7.0 min (7.5 measured) |

The faster take is d1-2B's doing (the judge since 2026-10-29), not the Studio's: the clip itself takes what it took on
this app's own ComfyUI (the same TensorFold int8 blocks, now run by the Studio). The first keyframe was in the panel
71 s after the film started (phase 4: 48 s), the Studio's ComfyUI starting cold in it; the arc took 60 s.

- **The film**: 27.7 s, the keeper the same in every piece, one lighthouse interior (each later keyframe drawn beside
  the place's first), rain on the glass from piece 2, the lamp lit at the end. Piece 3 was made at 8 s: its first try
  was stopped by the Studio's own memory guard while it loaded (33.6 GB available, falling 29.6 GB a second, the
  brain still resident before its DiT release) and AutoPilot's memory-first rule made it again shorter.
- **The sound pass** (`tools/film_sound.py`, then by AutoPilot itself at a film's end): the Director's narration (four
  sentences) spoken by Drama 3 (62 s take), a MiniMax Music 3 score of 29 s (86 s) laid under the film by the Studio's
  soundtrack job (2 s), 156 s in all, inside the `audio` schedule.
- **Memory**: the lowest MemAvailable over the films was 21.5 GB (a keyframe beside references with Qwen, d1, Kokoro,
  Parakeet and Cosmos3 resident); the 15 GB external guard never fired.
- **Voice to voice during the DiT pass** (the end of the person's speech to the first reply audio):

  | | n | p50 | p95 |
  |---|---|---|---|
  | the film, Qwen3.8-27B kept beside the render (the `live` schedule fits without releasing it), render held for the turn | 28 | 2.02 s | 3.33 s |
  | one 10 s clip, the brain out for the render, Cosmos3-Edge answers, render held (phase 4's condition) | 12 | **0.80 s** | **0.92 s** |
  | GMA phase 4, same condition | 13 | 0.71 s | 0.93 s |
  | idle, Qwen3.8-27B | 6 | 0.85 s | 0.94 s |

  Qwen's first token beside a DiT pass is ~1.4 s (its prefill, while the stopped render's queued GPU work drains):
  a 27B brain kept through the pass costs about 1.2 s a turn against the 4B fallback. The hold itself: 12 holds of
  p50 0.75 s in the clip test (`GET /api/governor`: `voice_holds`).

**Problems found and fixed on the way** (each with a test): a governor that kept its rendering phase when a fallback
step raised (Parakeet's move to the CPU raised a CUDA illegal-address error mid-film: the move is out of the Spark
profile's fallbacks now, and a failed move releases the brain for the DiT instead); a clip queued the moment the one
before failed raced that one's session end ("a render with schedule live runs already": the session now ends under
the renderer's lock); a clip refused 0.6 GB short while the keyframe's `/free` was still giving memory back (the room
check waits up to 60 s); the Studio's ComfyUI holding 25.5 GB after a FLUX edit's `/free`, which the Studio counts as
the next clip's own and the governor did not (now it does); and AutoPilot's settings dropping `narration` and `score`.
The film was resumed by a spoken "keep going" after each fix, from its checkpoint (pieces already made never
rendered again).

**On the PR branch** (`job-previews`, the same 10 s clip): previews reached the panel at every pass (`previews:
"engine"`, the clip's middle frame), the clip took 204 s (17 / 156 / 31), unchanged.


Tests: `python -m pytest -q` (no models, no GPU, no Studio: `tests/test_studio.py` stands a fake Studio in).

## Phase 1: the core

One app process (`python -m voice_agent.server --profile profiles/<yours>.json`, port 8196 by default) supervises its
child engines on loopback ports:

- **Engine registry** (`gma/registry.py`, lifecycle in `gm_core/engine.py`, from the Studio's `app/bolt.py`): each engine
  is started under the Studio's leash (`gm_core/leash.py`), health-checked, loaded on demand, idled out after `idle_s`,
  watched by a kill-guard, and released around a render. `"mode": "adopt"` (or a server already answering on a managed
  engine's port) adopts a running server instead of starting a duplicate; an adopted server is never stopped here.
  Roles: brain, voice (Drama 3 or Kokoro), vision (Cosmos3-Edge), ASR (Parakeet, in process). A profile without an
  `"engines"` section adopts the servers its brain, voice and eyes URLs name, as Ghost Machine always did.
- **Pluggable brain** (`gma/brain.py`): any OpenAI-compatible server by the profile alone (Bolt, Gemma 4 12B,
  Qwen3.8-27B ...); a split set-up is the `brain` section plus small multimodal `eyes` (and `ears`, used from phase 4).
  New client kinds register in `gma.brain.KINDS`.
- **Brain lanes** (`gma/lanes.py`): voice turns first; the Director lane (the video planner now, AutoPilot later) runs
  only when the conversation is quiet and is cancelled at speech start and barge-in, then asked again.
- **Memory governor** (`gma/governor.py`): the Studio's measured DGX Spark estimates (`gm_core/memory_guard.py`, with
  AutoPilot's render-expectation delta) and its Mac peaks; MemAvailable on a Spark, vm_stat on a Mac
  (`gm_core/memstat.py`). Phases: talking (each load keeps the floor) and rendering, by schedule: on a Spark `live`
  (480p live + Kokoro, Bolt resident beside a ~58 GB render) and `sharp` (1376x768, 10 s, the brain evicted). The
  renderer is phase 2's (below). Every refusal is spoken as one short line.
- Routes: `GET /api/engines`, `POST /api/engines/{name}/load|unload`, `GET /api/governor`,
  `GET /api/governor/plan?schedule=`, `POST /api/governor/phase`, `GET /api/lanes`, `GET /api/credits` (the page's
  credits panel).
- HyperFlow-Infinity's renderer and the remote Studio renderer were removed on 2026-11-07 (every clip is the FP4 clip worker's).

### The Spark "480p live + Kokoro" fit (measured, open for the Spark smoke)

Measured on one DGX Spark in a Studio run (2026-10-27): a FastH3 v2 fh8 864x480 render held ComfyUI at about 54.5 GB
resident (budgeted 61 +- 6 GB), Bolt 30.4 GB, Cosmos3 3.4 GB; with the Studio app (2 GB) the machine bottomed out at
20 GB MemAvailable. With Kokoro (3.5 GB) and Parakeet (5.4 GB) beside them that is ~11 GB, under the 12 GB floor. So:

- the live schedule's fit is explicit (`GET /api/governor/plan?schedule=live` returns its `budget`), and when what is
  left is under the floor its `fallbacks` apply in order: `asr_to_cpu` (Parakeet moves to the CPU for the render;
  counted as freeing 3.0 GB, the 5.4 GB measured less its fp32 weights that stay in RAM: **not measured yet**), then
  `brain_during_dit` (Bolt released when the renderer says its DiT pass starts, `Governor.dit(True)`, which the local
  renderer calls);
- after a render, Bolt could not reload beside ComfyUI's ~50 GB cache (status -6): `leave()` and any load that does not
  fit first ask this app's own ComfyUI to `POST /free` (`unload_models`), and only its own (`governor.renderer.comfy_url`);
- the Studio's `POST /api/bolt/unload` did not free Bolt in that run: engines here are stopped through the leash and
  their process group by PID, never through another app's API.

Open: measure on the Spark what `asr_to_cpu` really frees, and the live render's floor with every engine resident.

Tests: `python -m pytest -q` (no models, no GPU).

## Phase 2: the local renderer (retired by the headless Studio)

Kept as a record: phase 2's renderer is gone, its numbers stay for comparison.


`"video": {"renderer": "local", "keyframe": true}` makes the app render its own FastH3 v2 (MiniMax H3) clips on the
same machine, with no Imagination Studio app running (`gma/renderer.py`, `LocalRenderer`, on the panel's existing
`Renderer` interface). "Make a video of ..." goes keyframe, then clip, into the panel, one clip a request (AutoPilot's
directing is phase 3).

- **DGX Spark**: the app's own ComfyUI (`gma/comfy_server.py`), a child engine with role `renderer` on its own port
  (8391; never the Studio's 8190), started from a Studio install read only: its own input, output, temp, user and
  preview folders, no file written into the install. TensorFold's int8 FastH3 blocks (the install's `tensorfold_h3`
  node), fh8, 864x480 by default; the graphs are the Studio's (`gm_core/comfy_graphs.py`), followed over ComfyUI's
  progress socket (`gma/comfy.py`).
- **Mac**: the Studio install's FastH3 on TensorFold's native Zig + Metal runtime (`zig_fasth3.py`), the MLX engine
  (`h3_generate.py`) where the native one cannot run (or refuses, status 3).
- **Keyframe**: the picture model is a setting (`governor.renderer.picture`: `auto`, `qwen`, `flux3k-fast`,
  `flux3k-full`); `auto` takes Qwen-Image-2.1-Turbo where installed, else FLUX 3 Image Klein (Fast). On a Spark its
  models are unloaded (`POST /free`) before the video's load: never both resident.
- **Live preview**: on a Spark ComfyUI's latent2rgb preview, asked for per prompt (`extra_data.preview_method`), works
  with the TensorFold blocks and arrives at every pass; the app's `GMAPreviewFrame` node (`gma/comfy_nodes/`, loaded
  only by its own ComfyUI) points it at the clip's middle latent frame (the first is the keyframe, held). On a Mac the
  native engine hands over no pictures between passes: the panel shows the keyframe with the stage and the pass count
  and says "previews by stage".
- **Governor**: one rendering phase from the keyframe to the clip (schedule `live` by default), `Governor.dit(True/False)`
  around the DiT pass, the fh8-480 budget of 61 +- 6 GB (54.5 GB resident). Leaving frees the app's ComfyUI first: POST
  /free, and after TensorFold's blocks ran (their int8 weights outlive /free) the server is stopped too
  (`after_render`: `auto`, `free` or `stop`), before an evicted brain may load again.
- Prompts are the Studio's MiniMax H3 form (`gm_core/h3_prompt.py`: `compose`, the first-frame lead), with its
  `lengths.py` and `negation.py` (`gm_core/`).
- `tools/render_smoke.py`: one clip from a typed request through the video agent, the governor and the renderer,
  without the ears and the voice.

### Smoke renders (2026-10-27), 5 s, 864x480, 124 frames, 8 passes, from a typed "make a video of a red paper kite ..."

| | DGX Spark (GB10) | Mac Studio (M5 Ultra) |
|---|---|---|
| keyframe | FLUX 3 Klein Fast (Qwen Turbo not installed), 32.7 s incl. its load, +6.3 s ComfyUI start | Qwen-Image-2.1-Turbo, 13.9 s |
| load (text encoder, graph loaders / case export) | 17.5 s | 12.9 s |
| DiT pass | 81.4 s (first pass 29.4 s with the int8 load, then 7.3 s a pass) | 55.9 s (6.0 s a pass, Zig + Metal) |
| decode | 16.6 s | 15.3 s |
| request to clip in the panel | 157 s | ~105 s |
| memory | machine peak 66.6 GB used, 55.1 GB MemAvailable at the lowest (Cosmos3 resident, no Bolt) | 73.7 GB available at the lowest (Bolt, Drama 3, Cosmos, Kokoro live beside it) |
| live preview | latent2rgb, 8 of 8 passes (and 16 during the keyframe) | by stage |

## Phase 3: AutoPilot

AutoPilot (Imagination Studio's AutoPilot branch, 6cc6e67) runs inside this app, on its local renderer, with no
Studio server, Slate or job queue (`gma/ap/`). Pick "AutoPilot film" in the video panel (or `POST /api/video/start`
with `"mode": "autopilot"`, optional `"seed"`), and it directs, renders, checks and retakes every piece itself. No
step waits for anyone's approval.

- **Director**: Qwen3.8-27B (MLX 4-bit + DFlash2, TensorFold) by default, the app's brain in both example profiles, on
  its Director lane (it yields to every voice turn), reasoning first within a `thinking_budget` (`think_tokens`, 4096).
  An empty answer is asked again without thinking. AP's rules apply: the scene budget, no repeats, no "no X" lines,
  word limits, and the pace plan. `autopilot.director` picks it from `autopilot.director_choices` (`"qwen"`, `"bolt"`,
  or `"brain"` for the app's brain); a choice on another server than the brain gets a client of its own, so a
  profile can keep Bolt for the voice and direct with Qwen, or the other way round. Why Qwen: in the brain-quality
  bake-off (13 plans, 8 rendered films, blinded judges) its plans were valid 13/13 and on the target length 12/13
  (Bolt 7/13), it finished 8/8 films on the first try (Bolt 5/8: one broken word rule, two KV-growth failures), and
  the blinded plan preference was 10-2. Its resident ~22-24 GB (Bolt ~30 GB) is what the Spark schedules budget.
- **Pieces**: a keyframe per piece by default (FLUX3-K or Qwen), checked by the eyes before anything moves, then the
  clip from it (10 s at 864x480 fh8 on a Spark). The film is joined again after every piece (`film.mp4`, AP's fades).
- **Judge** (`gma/ap/judge.py`, since 2026-10-29): **d1-2B alone** decides keep or retake. Eight frames of the take
  and every question go to its `/v1/systemone` route in one pass (set attention): the same person throughout, a hard
  cut, someone's clothes changing colour (these three retake), the same place, a person morphing, the piece's beat,
  the shot's setting and the hero's clothes against the bible (these are notes). Answers are softened by the
  calibrated temperature (1.337); a person under 0.6, a fault at 0.7 or more, or **doubt** (0.35 to 0.65) on any of
  the three renders the piece again with the shot rewritten from the findings. No second model confirms: on doubt it
  retakes. Every question's probability is in the take's `continuity-takeN.json` and in the run report. The memory
  records the plan for a take (d1 does not describe), the keyframe is not checked and the film-level pass is skipped.
  On a Spark d1 is a managed engine (port 8412, 5.3 GB, loads in 2.6 s, evicted by a clip's render, kept through a
  keyframe); on a Mac a persistent launchd server (`ai.keys.d1`, 127.0.0.1:8412). `autopilot.judge: "eyes"` brings back
  the older path below.
- **Eyes** (`gma/ap/eyes.py`, the older judge, selectable): Gemma 4 12B. Cosmos3-Edge is selectable (`autopilot.eyes`: `"gemma"` or
  `"cosmos"`). The checks are AP's: windows of yes/no questions decided in code, votes on a failed major question, and
  a second, closer look. A confirmed major problem renders the piece again at the next seed with the shot rewritten
  (`retakes`, 2; the best take is kept when they run out). A minor one becomes a Director note. On a Spark Gemma
  runs in vLLM (Docker, images only: the build's audio path crashes) and is **never resident beside a render**: no
  render schedule keeps it, so the governor unloads it when a render starts and it loads again for the checks.
  On a Mac it runs in mlx-vlm, and the `<image|>`/`<audio|>` tokens that build leaks are filtered out.
- **Ears** (`gma/ap/ears.py`): the QA gate's line check, heard with Parakeet (the app's own recognizer). Where
  Parakeet is not installed, it falls back to faster-whisper from another environment (`ears.whisper_python`).
- **Checkpoints**: AP's `checkpoint.json` and `checkpoints/S000N.json`, the bible and the scene graph are written into
  the panel's run folder, with the AP run inside the panel's `run.json` shell (`mode: "autopilot"`, `"autopilot": {...}`).
  A restart, a kill -9 included, resumes from it: finished pieces are never rendered again.
- **Panel**: beside the live preview it shows the piece and beat at work, the story so far, the check counts
  (requests, questions, failed answers), the retakes and redrawn keyframes, and the film's length.
- **Not in V1**: fight sequences (`fight.py` is not ported), Ref2VA reference pictures, the Director on the vision
  server, branching from the page (`autopilot.branch` exists as a call).

Why Gemma for the eyes: the brain evaluation's 20 clips (8 clean, 6 person swaps, 2 cuts to an empty scene, 4
clothing colour changes; 52 yes/no questions, 8 frames each):

| eyes | same person: acc / recall of swaps / FP | abrupt cut: acc / recall / FP | clothing colour: acc / recall / FP | all 52 | s / question |
|---|---|---|---|---|---|
| Gemma 4 12B, Spark vLLM w4a16 | 95% / 88% / 0% | 95% / 88% / 0% | 100% / 100% / 0% | 96% | 0.17 |
| Gemma 4 12B, Mac MLX mxfp4 | 95% / 88% / 0% | 100% / 100% / 0% | 100% / 100% / 0% | 98% | 0.96 |
| Cosmos3-Edge, Mac and Spark (live :8099), one 4x2 frame sheet (it takes 4 images at most) | 60% / 0% / 0% | 60% / 0% / 0% | 83% / 50% / 0% | 65% | 0.29 |
| Cosmos3-Edge, Mac, 4 separate frames | 65% / 50% / 25% | 60% / 0% / 0% | 92% / 75% / 0% | 69% | 0.14 |

Cosmos3-Edge said "same person, no cut" to all 8 swaps and all 8 cuts.

Why d1-2B alone (calibrated 2026-10-29 on a DGX Spark, TensorFold `d1-2b-cuda-wip` 38724e55, 8 frames; retake rule
above). The real takes are the 53 takes of the brain-quality films (FastH3, two Directors), looked at by hand: 7 with a
hard cut or a person who vanishes, 46 clean.

| judge | synthetic: faulty retaken (swaps) / clean retaken | real takes: cut retaken / clean retaken | per take |
|---|---|---|---|
| d1-2B, this rule | 12/12 (6/6) / 0/8 | 4/7 / 1/46 (a fox close-up: no person to be sure of) | 0.3 s (5 fixed questions), 0.9-1.3 s with the take's own (the CUDA build splits the pass at 1024 rows) |
| Gemma 4 12B yes/no (Spark vLLM) | 11/12 (5/6) / 0/8 | 6/7, mostly for other reasons (story, identity, brand) / 18/46 (story 8, era 6, wardrobe 4, setting 1, action 1 among them; the audio gate is the ears' and stays) | ~0.2 s a question, plus a 70-120 s load a piece |

One swap (s12) is caught only by doubt on the cut; with `retake_on_doubt` off d1 retakes 11/12. On a Mac (TensorFold Metal,
`lfm3-vl-metal` 2a353473, launchd `ai.keys.d1`, ~4 GB) every question goes in one pass in 0.13 to 0.25 s; its answers
differ from CUDA's by 0.01 at the median (0.26 at most), and on the same frames the rule retakes 11/12 (6/6 swaps; one
clothing shift at 0.68 is kept) and 0/8 clean, the real takes as on CUDA. Through the app's own frame grab of the
clips (the smoke) it retakes 12/12 and keeps 8/8. The place and morph
questions stay notes because they flagged clean takes; the beat question stays a note because its answers on good
takes spread from 0.05 to 0.95.

### Spark smoke (2026-10-27, DGX Spark, one machine)

Run with `tools/autopilot_smoke.py`. The typed brief was "a lighthouse keeper's evening", 3 pieces of 10 s, 864x480
fh8, Bolt as the Director (thinking budget 4096), and Gemma as the eyes. Kokoro and Parakeet are not installed on that
Spark, so the line check used faster-whisper large-v3-turbo on the CPU. The live Cosmos3 (:8099, 3.4 GB) and an idle
Studio app stayed up. The governor's floor was 20 GB, with an external kill-guard at 15 GB.

| piece | Director (incl. Bolt load) | keyframe + its check (incl. Gemma load) | render (load / DiT / decode) | checks (Gemma load, line, watcher, bible, dense) | retakes |
|---|---|---|---|---|---|
| 1 | 99 s (Bolt load 38 s) | 182 s (keyframe ~42 s) | 208 s | 221 s; 6 requests, 40 questions | 0 |
| 2 | 92 s | 176 s (drawn again after the first kill) | 193 s (15.8 / 137.9 / 32.2) after the second kill | 202 s; 6 requests, 41 questions | 0 |
| 3 | 89 s | 178 s (keyframe 41.7 s) | 202 s (17.2 / 149.8 / 31.6) | 216 s; 12 requests, 44 questions | 0 |

The film came out at 29.9 s (864x480 with sound). The run took 18 min 14 s after the second restart, about 12 min a
piece. Gemma loads in 117 to 120 s, twice a piece, which is the largest overhead (see Open below).

**Live fit, measured with MemAvailable every 0.5 s.** The lowest was 32.4 GB, at the moment the DiT pass began, with
Bolt (27.7 GB) resident beside the loaded renderer. The governor's live fit (61 to 65 GB budget for 243 frames, floor
20 GB) did not fit with Bolt through the DiT pass, so its `brain_during_dit` fallback released Bolt there (60.2 GB
available right after). Bolt then reloaded for the next Director turn. During DiT without Bolt the lowest was 51.9 GB.
An estimate, not measured: with Bolt kept through the DiT pass, about 24 GB would be left, or about 20.7 GB with
Kokoro, right at the 20 GB rule. Gemma was unloaded before every render and loaded again after it. That schedule
is logged in the engine events, and `tests/test_autopilot_service.py` holds it for the Spark profile.

**Resume, measured.** The app was killed with kill -9 twice in piece 2: once during its keyframe render, then during
its clip's DiT pass. After each restart the leash had ended every child (Bolt, ComfyUI) and nothing was left on the
ports. The run continued from `checkpoint.json`:

- The first restart drew piece 2's keyframe again, from the plan already in the checkpoint.
- The second restart went straight to piece 2's clip, from its checked keyframe.
- Piece 1's take was never rendered again: same file and mtime.

Open:

- Gemma's vLLM load, about 2 minutes, twice a piece. Options: vLLM sleep mode, or letting the eyes stay through the
  keyframe phase, which needs far less memory than the clip.
- A keyframe per piece loses the place between pieces. The third piece's tower is not the first's, and the checks
  passed it because the setting question asks about the Director's own setting line.
- The Director was slow to move the story: three pieces of climbing, and the lamp was never lit.
- The Mac smoke was not run.

(Phase 4 below addresses the first three.)

## Phase 4: the voice loop, and phase 3's fixes

### Phase 3's three problems

**Speed.** A 10 s piece took about 12 minutes, about 4 of them two Gemma vLLM loads (117 to 120 s each).

- *Why Gemma loaded so slowly*: the weights' read (52 to 62 s of it) is mmap page faults into unified memory, not the
  disk. vLLM's `--safetensors-load-strategy eager` reads the file at once: **8 s**, and a cold start of **~69-78 s**
  instead of ~125 s. With it the KV cache is given in bytes (`--kv-cache-memory`), because the eager read's buffer
  counts against `--gpu-memory-utilization` at start-up.
- *vLLM sleep mode, measured on the Spark* (`--enable-sleep-mode`, MemAvailable): level 1 frees only the KV cache
  (5.2 GB of 20.2 GB: the weights' "CPU" copy is the same unified memory) and, after its first wake, holds ~20 GB asleep
  and ~32.5 GB awake for good (wake 0.6 s). Level 2 holds ~10.5 GB asleep, wakes in 7.5 s with the eager read, and grew
  ~2.5 GB a cycle. Neither leaves the room a render needs, so the eyes are still stopped for a clip and started cold
  (`profiles/spark.example.json`, the gemma engine's `_note`).
- *One eyes window a piece*: the keyframe has a governor schedule of its own (`keyframe`, ~31 +- 4 GB): the eyes stay
  resident through it, so the keyframe is checked without a load; Bolt stays where it fits (the opening keyframe, drawn
  while the Director plans) and is evicted where it does not (`fallbacks: evict bolt`). A clip never runs inside a
  keyframe's session (`LocalRenderer.begin` enters the clip's schedule afresh).
- *The checks*: the 200 s of phase 3 were mostly the load. The watcher, the continuity checklist and the dense windows
  now go to the eyes at once (`eyes_parallel`, 4; vLLM batches them, `--max-num-seqs 8`), the eyes start loading while
  the ears hear the lines, and the Director's brain loads beside them for the next piece (`director_warm`). The yes/no
  questions were already batched a window. Measured: ~42 s for 26 requests and 55 questions after the load.
- Bolt's thinking budget in the smoke profile is 2048 (the arc gives each piece its plan; 4096 is still the default).

**Place.** The kept keyframe of the piece that opens a place (the opening, or a piece the Director declares a scene
change) becomes that place's reference picture (`run["locations"]`). Later keyframes of the place are drawn beside it
(FLUX 3 Klein's multi-reference graph on the Spark, `gm_core/comfy_graphs.flux3_edit_workflow`; the vendored
`flux3_edit.py` on a Mac), with the cast's portraits after it. The eyes are asked "Is this the same building or place
as in the reference picture of the place ...?" with that PICTURE, on every keyframe and on a clip's first and last
windows. The place changes only with `scene_change: true` (or a spoken "new scene: ...").

**Story pace.** The Director plans the whole arc before the first piece (`gma/ap/story.py`): a beat a piece, the state
each piece must END on, and the payoff of the last. Every piece is told its beat and end state (the last one, the
payoff), and its last frames are asked "By the end of these frames, is this shown: <end state>?". A miss is a note: the
next piece is told to catch up first. On the last piece a miss is a retake. An Infinity film plans 4 pieces at a time,
and a film made longer by voice gets the rest of its arc planned.

### The voice loop

- **Speech to film.** "Make a video of ..." is found in code (as before) and Boo says one short line. Words that follow
  the request at once (a breath in the middle of it) are the rest of the request: the film waits until the person has
  been quiet for 2.5 s. The brain (no reasoning, ~4-6 s) turns the words into AutoPilot's seed: title, world, situation,
  the cast with a wardrobe line each, the style, the pace and the opening still; the length is read in code ("a 30
  second video", "an endless video"). The opening keyframe is drawn from the seed at once, while the Director plans the
  arc and the first piece, and the panel shows it the moment it is made. With no subject it asks "What should the video
  show?". Nothing asks for approval.
- **Steering by voice, mapped in code** (`VideoAgent.film_action`):

  | said | does |
  |---|---|
  | "keep going", "twenty seconds more", "one more piece" | the film is made longer (pieces of its length); a stopped film carries on |
  | "make it rain", "add a dog ...", "have Elias light the lantern" | a steer: a Director note and a bible fact from the next piece, and a yes/no check on it (a retake once if it does not show) |
  | "faster", "more action" | pace: the Director is told, the run's guidance changes |
  | "redo that" | the piece shown is taken out of the film (its memory and bible put back) and made again; one at work is cancelled and made again |
  | "new scene: the harbour at dawn", "cut to ..." | the next piece opens a new scene (and a new place) |
  | "stop", "stop the film" | stops after the piece at work |
  | "stop now", "cancel the render" | stops now (the render is cancelled) |
  | "go sharper" | the next pieces at 1376x768 on the governor's `sharp` schedule (the brain evicted for them) |

  While a film runs, "add ...", "make it ...", "bring in ..." are the film's only when they are about the film: a film
  word, a picture word (weather, light, camera) or a word of the film's own (its idea, cast and beats), or the person
  spoke about the film in the last 20 s; never with an errand's or the conversation's words ("add milk to my shopping
  list", "make it a double espresso", "can you talk faster" stay with Boo; tests/test_phase4_voice.py). A bare "stop"
  right after the person cut Boo off is for Boo.
- **Brain lanes.** Voice turns preempt the Director (its request is cancelled at speech start and barge-in and asked
  again once it is quiet); a Director whose brain is held for a render waits for it instead of failing.
- **Spoken updates**: the first piece, every `speak_every` pieces, a piece sent back for a retake (one short line) and
  the end, each through `when_quiet`; one update waits at a time (a newer one replaces it), so they never queue behind
  a reply.
- **Voice while Bolt is out.** On a Spark Bolt cannot stay beside a 10 s 864x480 DiT pass with Kokoro and Parakeet
  resident: the fit is ~12 GB under the 20 GB floor, so the live schedule's `brain_during_dit` releases it. Voice turns
  then go to `brain_fallback` (Cosmos3-Edge, resident anyway; the last 6 messages of the conversation) instead of a
  "paused" line (`gma.brain.HeldFallback`); a brain released by a render is never loaded for one voice turn.
- Parakeet runs in bf16 (`"asr": {"dtype": "bfloat16"}`, cast before its first use): 21 ms a short turn instead of 40,
  ~1.1 GB less.

### Spark smoke (2026-10-27, DGX Spark)

From a SYNTHETIC spoken request ("Make a video of a lighthouse keeper lighting the lamp as a storm rolls in.", Kokoro's
am_michael voice made by tools/synth_turns.py) streamed into the app's microphone socket by `tools/voice_film_smoke.py`,
with "make it rain" said during piece 1's DiT pass and stock questions asked during every DiT pass. 3 pieces of 10 s,
864x480 fh8, Bolt as the Director (thinking 2048), Gemma 4 12B eyes (vLLM, eager load), Parakeet bf16 and Kokoro on
the GPU, Cosmos3-Edge (the live :8099) as the voice fallback while Bolt is out. Film: 29.9 s with sound; contact sheet
and logs in the head's scratchpad `gma-phase4/spark/`.

- **The request**: it came in as two turns at a breath ("... a lighthouse keeper." / "lighting the lamp as a storm rolls
  in."); merged, the film's idea was the whole sentence (the first run, before the merge, made a film of "a lighthouse
  keeper"). The seed took 3.6 s; the film started 6 s after the last words (the 2.5 s quiet window); **its first
  keyframe was in the panel 48 s later** (FLUX 3 Klein with ComfyUI's start), while the arc (25 s) and piece 1 were
  planned.
- **The arc**: 1. climbs the spiral stairs, ends at the lantern door; 2. opens the door, ends beside the unlit lamp;
  3. strikes a match and lights the lamp, ends on the lamp burning bright (the payoff). Every piece reached its end
  state by the eyes' answer; piece 3 on its third take (takes 1 and 2 were sent back as "the last piece did not reach
  the payoff"). **The lamp is lit at the end of the film.**
- **The place**: piece 1's kept keyframe became the place's picture; all six later keyframes were drawn beside it (FLUX
  3 Klein multi-reference), and the sheet shows one lighthouse interior, its stairs, windows and green door, across the
  three pieces (phase 3: a different tower each piece).
- **The steer**: "make it rain" (0.32 s to the spoken reply) was a Director note and a bible fact from piece 2. Rain
  streaks are on the windows of piece 2 in the sheet, but the eyes answered "no" to "Is this clearly visible: make it
  rain" on all three takes of piece 2 (an interior shot), so piece 2 was made three times and its third take kept with
  the reason. A steer that does not show is now sent back once, then carried as a note.
- **Per take** (DGX Spark, a take = Director, keyframe and its check, clip, checks; phase 3's in brackets):

  | stage | phase 4 | phase 3 |
  |---|---|---|
  | Director | 30 s with Bolt loaded beside the checks; 58 s when it loaded first (38 s) | 89-99 s (Bolt's 38 s load in it) |
  | keyframe + its check | 62 s (no eyes load: they stay through the keyframe) | 176-182 s (a Gemma load in it) |
  | clip (load, DiT, decode) | 210 s (~20 + ~160 + ~30) | 193-208 s |
  | checks | 118 s: the eyes' cold load ~77 s, then 41 s for 26 requests and 55 questions at once | 202-221 s |
  | **a take** | **~7.0 min with Bolt pre-loaded (7.5 min measured on piece 2's first take)** | ~11.5 min |

  A keyframe redrawn after a failed check adds 50 to 150 s. The film took 70 minutes of wall time for 3 pieces made in
  7 takes, with two stops on the way, both resumed by a spoken "keep going": a keyframe refused for memory because Bolt
  had been measured as ~0 GB when it loaded beside Gemma (fixed: a load measured under half its resident size counts
  as its resident size; and a keyframe refused for room now waits and tries again), and Bolt answering HTTP 500 when
  its sequence passed 4096 tokens with the eyes resident (TensorFold's 10 GiB memory reserve against cudaMemGetInfo,
  which counts the page cache as used on a GB10; the smoke profile now reserves 2 GiB).
- **Memory**: the lowest MemAvailable during the film was 29.2 GB (the 20 GB floor, the 15 GB external guard never
  fired).
- **Voice latency during the DiT pass** (the end of the person's speech to the first reply audio, measured at the
  socket; Bolt is out for the pass, Cosmos3-Edge answers):

  | | n | p50 | p95 |
  |---|---|---|---|
  | the film, fallback with the whole history, the render running on | 52 | 1.11 s | 2.38 s |
  | the film's last piece, the render stopped until the first audio (SIGSTOP), fallback with 6 messages | 26 | 0.63 s | 1.18 s |
  | **after the fixes**: one 10 s clip, render stopped until the first audio, fallback reads the question alone | 13 | **0.71 s** | **0.93 s** |
  | idle (no render), for reference | 2 | 0.44-0.49 s | |

  The fallback's first audio grew from ~0.7 s to ~2 s over 50 turns as the whole history was prefilled beside the DiT
  pass, and with the last 6 messages it repeated the reply before; it now reads the question alone.
  Stopping the render: `LocalRenderer.hold_for_voice` stops ComfyUI's process group from the end of the person's
  speech to the reply's first audio (4 s at most, a thread's timer), so CUDA's queue drains in milliseconds and the
  small brain and Kokoro get the GPU. The DiT pass took ~160 s with ~25 turns in it (about 150 s without).

### Mac smoke

Mac Studio M5 Ultra (256 GB), adopting the live Bolt :8091, Drama 3 :8098 (the voice), Cosmos :8099 and Kokoro :8197,
Gemma 4 12B mxfp4 in mlx-vlm as the eyes (this app's engine on :8311), the same synthetic request and steer, 2 pieces.
Contact sheet and logs in `gma-phase4/mac/`.

- The first keyframe (Qwen-Image-2.1-Turbo) was in the panel about 19 s after the request.
- 2 pieces in 8.2 minutes (~4 min a piece, arc and checks included; the eyes stay loaded on a Mac). Each 10 s clip was
  stopped once by the renderer's own 40 GB guard (37.7 and 36.5 GB available with Drama 3, Bolt, Cosmos, Gemma and
  Kokoro resident) and made again at 8 s, as AutoPilot's memory-first rule says: the film is 15.8 s.
- The place: piece 2's keyframe was drawn beside piece 1's (FLUX 3 Klein, the vendored `flux3_edit.py`); the same
  lantern room. The steer: rain on the glass in piece 2 (its check passed). The payoff: the lamp lit at the end of
  piece 2 (its end-state check passed).
- **Voice latency during the DiT pass** (Bolt resident, Drama 3 speaking):

  | | n | p50 | p95 |
  |---|---|---|---|
  | the film | 28 | 1.65 s | 4.96 s |
  | one 10 s clip, render running on | 14 | 1.07 s | 4.09 s |
  | one 10 s clip, render stopped until the first audio | 14 | 0.98 s | 4.53 s |
  | idle, for reference | 2 | 0.46 s | 0.91 s |

  The 0.8 s target is missed. The slow turns wait for Bolt's first token (up to 2.7 s) behind the native engine's
  Metal work; stopping its process does not stop a pass's command buffers already queued, so it does not help (off
  by default on a Mac, `pause_for_voice`).

Open:

- The Spark take is ~7.0 min with everything warm, at the 7 min target, not under it; the eyes' cold start (~77 s a
  piece) is most of what is left. A Gemma build that holds its weights outside the GPU's view, or a smaller eyes model
  beside the render, would take it out.
- Bolt cannot stay beside a 10 s 864x480 DiT pass on a Spark with the voice engines resident (~12 GB under the floor);
  voice turns during the pass are answered by Cosmos3-Edge, a much weaker conversationalist.
- The Mac misses the 0.8 s voice target during a render (p50 ~1.0-1.6 s, p95 ~4-5 s). A fix needs the native engine
  to yield between passes (a pause it checks), or Bolt's work at a higher Metal priority.
- The yes/no eyes are strict about steers seen through glass ("make it rain" in an interior); the arc's end states must
  be one frame's worth (the prompt now says so) or the final payoff is sent back.
- A latency test after the film found two Bolt requests left "in flight" by turns cut while Bolt loaded: the DiT pass
  could not release Bolt, the machine sat at 15.0 GB and the engines' own 15 GB guards stopped ComfyUI and Kokoro.
  Fixed (a cut load releases its count; a turn's brain stream is closed at once; the DiT release waits 10 s at most)
  and measured again above: Bolt released at the pass, the lowest MemAvailable 24.8 GB.

---
