# Credits

What Ghost Studio (Ghost Machine + AutoPilot on a headless Imagination Studio) is built from, and under which terms. It
is a private, local fork of **Ghost Machine** by keys (drowzeys), MIT (see LICENSE), joined with **Imagination Studio**
(driven through its HTTP API, never copied for rendering) and its **AutoPilot** (Apache-2.0). Everything below keeps
its own licence. The page shows the same list (its "credits" panel, `GET /api/credits`).

## Imagination Studio, headless (gma/studio.py), and the code kept from it (gm_core/)

**Imagination Studio** by Keys ([github.com/drowzeys](https://github.com/drowzeys)) and Sam Wasserman / Wasserman
Productions ([wassermanproductions.com](https://wassermanproductions.com)), Apache-2.0
([drowzeys/Imagination-Studio-by-keys-Wasserman](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman)).
Ghost Studio runs it, unmodified, as a headless child engine (`gma/studio.py`): its own port, its own Studio home, its
own ComfyUI port, the install's models read only. Every keyframe, keyframe beside reference pictures, clip, Drama 3
narration take, MiniMax Music 3 score and the score's mix under the film is a job of its queue, through its own routes
(`/api/jobs`, `/api/estimate`, `/api/voice/takes`, `/api/music/takes`, `/api/music/soundtrack`, `/files/`). Its memory
guard's estimates (`/api/estimate`) are the governor's. None of its rendering code is copied here any more.

Per its NOTICE (Section 4(d) of the Apache License 2.0) that NOTICE is kept, unchanged, in
`LICENSES/Imagination-Studio-NOTICE`, with the licence in `LICENSES/Imagination-Studio-LICENSE-Apache-2.0`. The files
still taken from it (what the Studio does not expose as a service) carry a header with their source path and commit,
and say what was changed:

- `gm_core/leash.py`: `app/bolt_leash.py` at 912374e, unchanged (a child engine never outlives the app).
- `gm_core/engine.py`: ported from `app/bolt.py` at 912374e (the managed engine's lifecycle: load on demand, hold and
  release around a render, idle unload, kill-guard, leash), generalised to any child engine (the headless Studio is
  one), with an adopt mode.
- `gm_core/memstat.py`: `parse_vm_stat` and `mac_pressure_critical` from `app/telemetry.py`, and the MemAvailable
  reading and `settle` from `app/voice.py`, at 912374e (the Mac and Spark memory readers).
- `gm_core/h3_prompt.py`: `compose`, `lead` and their helpers from `app/server.py` at 912374e (MiniMax H3's prompt form;
  AutoPilot composes every piece's prompt with it, in its own thread, many times a piece).
- `gm_core/lengths.py` and `gm_core/negation.py`: `app/lengths.py` and `app/negation.py` at 912374e, unchanged.
- `gma/governor.py` keeps the Studio's floor rule (a tenth of the machine, never under 8 GB; `spark/memory_guard.py`)
  for when no estimate has been read, and the Studio's measured Mac peaks (`MAC_PEAK_GB`, `app/bolt.py`).

Retired with the headless Studio (they were copies of what its queue now does): `gm_core/memory_guard.py`,
`gm_core/workload.py`, `gm_core/comfy_graphs.py`, `gm_core/flux3_generate.py`, `gm_core/flux3_edit.py`,
`gm_core/edit_plan.py`, `gma/renderer.py`, `gma/comfy.py`, `gma/comfy_server.py` and the `gma/comfy_nodes/` preview node.

**The look** (`gma/look.py`): the HDR look is Imagination Studio's HDR finish (its `hlg` job, `app/hlg.py`, by Keys and
Sam Wasserman, Apache-2.0), run unmodified through the headless Studio: BT.2020 / HLG in 10-bit HEVC, encoded by
**x265** (GPL-2.0-or-later, the MulticoreWare project) inside **FFmpeg**, with an SDR copy. The Neon look lays Ghost
Studio's own colour table first (`gma/neon.py`, MIT with this repo; FFmpeg's `lut3d` filter applies it), tuned by eye on
2026-11-17 against a reference clip the owner picked (cobalt shadows, hot red and magenta lights); no part of that clip
is in this repo. The Sonar look (`gma/sonar.py`, `gma/sonar_worker.py`, MIT with this repo) is Ghost Studio's own effect
over the HDR look's copies; the depth of every frame that its wave travels over is estimated by **Depth Anything V2**
(Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao; the metric outdoor Small
model, Apache-2.0, `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`, fetched from Hugging Face on first use,
not shipped here), run with **PyTorch** (BSD-3-Clause) and Hugging Face **Transformers** (Apache-2.0); the picture is
drawn with **NumPy** (BSD-3-Clause).

**AutoPilot** (Imagination Studio's autonomous director, by Keys and Sam Wasserman), from the AutoPilot branch at
6cc6e67, in `gma/ap/` (phase 3), each file with a header naming its source and what changed:

- `gma/ap/continuity.py`, `gma/ap/checks.py` and `gma/ap/pace.py`: `app/continuity.py`, `app/checks.py` and
  `app/pace.py`, unchanged (the continuity bible, the dense yes/no checks decided in code, the pace measured from the
  pixels).
- `gma/ap/autopilot.py`: ported from `app/autopilot.py` (the loop: the Director's prompts and rules, keyframes, the
  watcher, the continuity checklist, votes and second looks, retakes, the scene graph, checkpoints and the film's
  joins), decoupled from the Studio's server, Slate and queue; `extract_json` and `mend_json` from `app/slate.py`.
- `gma/ap/eyes.py`: from `app/cosmos.py` (the vision request, the frame-sheet fallback), for any vision server.
- `gma/ap/ears.py`: the line check of **Sam Wasserman's QA gate** (`app/qa.py`), heard with Parakeet.
- `tests/test_autopilot.py`, `tests/test_autopilot_checks.py` and `tests/test_autopilot_continuity.py`: ported from the
  branch's tests.

AutoPilot's scene-graph memory (its delta schema, the merge and id folding, the chapter fold), the Director's rules
against repetition and stagnation and the branch are adapted from **HyperFlow-Infinity** by Video Rebirth
(github.com/Video-Rebirth/hyperflow-infinity), Copyright Video Rebirth, Apache License 2.0
(`LICENSES/HyperFlow-Infinity-LICENSE`). The idea of an endless film that watches itself and decides what happens next
is **Echo-Infinity**'s (Yuxuan Bian, Nan Duan, Qiang Xu et al., CUHK and Joy Future Academy, JD).

## The supervisor's own pieces (gma/)

New in this fork: the engine registry, the memory governor, the brain lanes and the pluggable brain client. They run
on **FastAPI** by Sebastián Ramírez (MIT), **Uvicorn** by Encode (BSD-3-Clause) and **HTTPX** by Encode
(BSD-3-Clause), as the agent always did, with **NumPy** (BSD-3-Clause) for audio and **ONNX Runtime** by Microsoft (MIT)
for Silero VAD. The tests use **pytest** (MIT).

## Brain candidates, AutoPilot's judge and eyes

The brain is chosen by the profile. Besides Nemotron 3.5 Bolt (below), a brain evaluation compared **Gemma 4** 12B by
Google and **Qwen3.8** 27B by the Qwen team, Alibaba Cloud (the licence on its model card), each behind any
OpenAI-compatible server; Bolt stays the brain.

**d1-2B** by Liquid AI (LFM Open License v1.0; commercial use only under US$10M annual revenue, see the licence) is
AutoPilot's clip judge since 2026-10-29 (`gma/ap/judge.py`): eight frames of each take and every question in one
pass of its `/v1/systemone` route, on **TensorFold** (CUDA on a DGX Spark, Metal on a Mac; Ash Hart and the TensorFold
contributors, Apache-2.0, our d1-2B branches). The weights are not shipped.

**Gemma 4** 12B by Google (Apache 2.0) was AutoPilot's eyes and stays selectable as the older judge (`autopilot.judge`
`"eyes"`): it answers the yes/no checks of every keyframe, window and join (`gma/ap/eyes.py`). On a DGX Spark it runs in **vLLM** by the vLLM project (Apache-2.0), loaded between renders;
on a Mac in **mlx-vlm** by Prince Canuma and contributors (MIT). NVIDIA's **Cosmos3-Edge** stays selectable as the
eyes. Neither model is shipped. Where Parakeet is not installed the line check hears with **faster-whisper** by
SYSTRAN (MIT) running OpenAI's Whisper large-v3-turbo (MIT), from another install's environment.

## Brain and Director

- **Qwen3.8** 27B by the Qwen team, Alibaba Cloud (the licence on its model card): the brain and AutoPilot's Director,
  as an MLX 4-bit checkpoint with its **DFlash2** drafter, served by TensorFold (CUDA and Metal).

## Ears, brain and voice

- **Silero VAD** v5 by [Silero](https://github.com/snakers4/silero-vad) (MIT): when someone is speaking.
  `silero_vad.onnx` is shipped here under that licence.
- **Parakeet TDT 0.6B v3** by NVIDIA (CC-BY-4.0): speech to text, through
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (Apache-2.0) on the Mac and NeMo (Apache-2.0) on the DGX
  Spark.
- **Nemotron 3.5 Bolt** by NVIDIA (NVIDIA Open Model License; the Studio's engine notes the OpenMDW License
  Agreement 1.1 for its 24 GB checkpoint), served by TensorFold: the front brain that answers most turns, with mood
  tags for the face. Its CUDA engine recipe is **MiaAI-Lab**'s (MIT, with ChriZzn's speed work) on TensorFold.
- **Fish Audio Drama 3** by [Fish Audio](https://huggingface.co/fishaudio/drama-3) (Fish Audio Community License),
  served by TensorFold: the voice. The 4-bit pruned checkpoint
  [drama-3-NVFP4-Pruned](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) and its CUDA engine and named voices
  (`--voices`) by [plotarmordev](https://github.com/plotarmordev/TensorFold). Hosted services that generate speech for
  third parties need a separate licence from Fish Audio. The film's narration is a Drama 3 take of the headless
  Studio's (its TensorFold CUDA engine on a DGX Spark). Built with Fish Audio.
- **MiniMax-Music3** by MiniMax (MiniMax-Music3 Community License: commercial use allowed, a commercial product shows
  "MiniMax-Music3" in its interface): the film's score, a song of the headless Studio's, laid under the film by its
  soundtrack job (ducked under the dialogue).
- **TensorFold** by [Ash Hart](https://github.com/ashhart/TensorFold) (Apache-2.0): the engine for Bolt, Drama 3 and
  Cosmos3-Edge, on the Mac's GPU (Metal) and on a DGX Spark (CUDA).
  The prebuilt builds in this repository's releases are unofficial builds of TensorFold 1.1.5 (source:
  [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold)); the DGX Spark's Drama 3 engine is
  [plotarmordev's NVFP4 Drama fork](https://github.com/plotarmordev/TensorFold) (branch `drama-nvfp4`) with our
  voice-prefix cache, and the Mac's Drama 3 engine is the keys 1.1.4-keys.1 build (it starts speaking faster on Metal).
  TensorFold 1.1.5 carries upstream Drama 3 work: the pruned NVFP4 checkpoint on Metal by
  [jerryfane](https://github.com/jerryfane) (#1268), the voice reference prefill reuse by plotarmordev with
  BobClawblaw and jerryfane (#1237), and voice descriptions in `/v1/voices` (#1270). The Docker image is built on
  NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`)
  and gives the TensorFold engines NVIDIA cuBLAS 13.1 (the `nvidia-cublas` wheel, under NVIDIA's licence).

## Lite voice (optional)

`voice_agent/kokoro_server.py` puts Kokoro-82M behind the same API as Drama 3, for `--voice lite` or `both`.

- **Kokoro-82M** by [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0): the lite voice and its stock
  voices (bf_emma by default). Built on the **StyleTTS 2** architecture by
  [Yinghao Aaron Li et al.](https://github.com/yl4579/StyleTTS2) (MIT), with an iSTFTNet decoder.
- **kokoro**, the PyTorch package by [hexgrad](https://github.com/hexgrad/kokoro) (Apache-2.0): runs it on the DGX Spark
  (CUDA).
- **mlx-audio** by Prince Canuma ([Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), MIT): runs it on the Mac
  (MLX), with the `mlx-community/Kokoro-82M-bf16` weights.
- **misaki** by [hexgrad](https://github.com/hexgrad/misaki) (Apache-2.0): Kokoro's text-to-phoneme step, with
  **spaCy** and its **en_core_web_sm** model by [Explosion](https://github.com/explosion/spaCy) (MIT), and
  **espeak-ng** ([espeak-ng](https://github.com/espeak-ng/espeak-ng), GPL-3.0, through phonemizer-fork and
  espeakng-loader) as the fallback for words its dictionary lacks. These are installed into the lite voice's own
  environment, not shipped here.

## Eyes and memory

- **Cosmos3-Edge** by [NVIDIA](https://huggingface.co/nvidia/Cosmos3-Edge) (OpenMDW 1.1): its reasoner tower (the
  SigLIP2 image tower, projector and dense text model; the world-generation tower is left out), converted to MLX 8-bit
  and served by TensorFold: camera questions and the memory captions.
- **Eyes of Kai** by [tonyd2wild](https://github.com/tonyd2wild/eyes-of-kai) (MIT): the continuity log, captions of
  what the camera saw in a timestamped log, so the agent can answer about the past as well as the present. It runs
  beside the agent with one small addition, `patches/eyes-of-kai-retention.patch` (set how many days it keeps while
  running). `voice_agent/memory.py` follows its question-about-the-past pattern, its caption prompt and its labelled
  prompt layout.
- **MediaPipe** Face Detector and Hand Landmarker by Google (Apache-2.0): face tracking for the ghost's gaze, and exact
  finger counts and which hand, in the page.
- **Three.js** (MIT): Boo, the 3D ghost.

## Back agents and live facts

- **Hermes Agent** by [Nous Research](https://github.com/NousResearch/hermes-agent) over ACP (Agent Client Protocol),
  following Eyes of Kai's Hermes client.
- **qwen-audio-agent** by the [Qwen Audio team](https://github.com/QwenAudio/qwen-audio-agent) (Apache-2.0): design
  inspiration, the frontstage/backstage split (a fast voice in front, a tool-using agent behind that reports back); no
  code from it is used.
- **RED-SNOW 5.3 Flash** (a GLM-5.3-Flash build), the back agent's model in the reference setup; any OpenAI-compatible
  model works.
- **ddgs** (Dux Distributed Global Search, MIT) by deedy5: the web search behind quick live facts.
- **wttr.in** by Igor Chubin (Apache-2.0), with data from WorldWeatherOnline: the weather lookup.

## Video panel

`voice_agent/video.py` and the page's video panel (docs/video-panel.md).

- **Imagination Studio** by keys and Sam Wasserman
  ([drowzeys/Imagination-Studio-by-keys-Wasserman](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman)):
  the render host, run headless by this app (above), or any Studio at a URL (the `studio` renderer).
- **MiniMax H3** by MiniMax (MiniMax H3 Community License): the video model behind every clip. Personal, local use
  only here. Ghost Studio ships none of its weights. Powered by MiniMax H3. The MiniMax H3 Community License excludes
  the USA, the EU, the UK and South Korea unless MiniMax authorizes the use.
- **ComfyUI** by Comfy Org ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), GPL-3.0), run by the
  headless Studio on a DGX Spark on its own port for keyframes and the film's sound; this app sends it only `/queue`
  and `/free` and stops it after each job (`after_render: stop`). Never the worker's ComfyUI.

- Keyframes: **FLUX 3 Image Klein** (FLUX3-K Fast, with BFL's prompt upsampler; also the keyframes drawn beside the
  place's picture and the cast's portraits) by Black Forest Labs (FLUX Kommunity License) and **Qwen-Image-2.1-Turbo**
  by the Qwen team, Alibaba Cloud (Apache-2.0), whichever the Studio has installed (a setting). No weights here.
- **HyperFlow-Infinity** by Video Rebirth ([Video-Rebirth/hyperflow-infinity](https://github.com/Video-Rebirth/hyperflow-infinity),
  Apache-2.0) and **Echo-Infinity** (Yuxuan Bian, Nan Duan, Qiang Xu et al., CUHK and Joy Future Academy, JD): the
  idea behind continue mode, an endless film that plans its next clip from the story so far and checks itself.
  (The `hyperflow` renderer that drove a HyperFlow-Infinity service was removed with the remote Studio route on
  2026-11-07; AutoPilot keeps the ideas credited above.) The dense checks take their approach (many small yes/no
  questions, decided in code) from Imagination Studio's AutoPilot.
- **websockets** by Aymeric Augustin and contributors (BSD-3-Clause): the client of the app's own socket in the
  voice bench and the smoke tools.
- **FFmpeg** (LGPL-2.1+/GPL, [ffmpeg.org](https://ffmpeg.org)): the posters, the joined film and the narration laid
  over it, used if it is installed. It is not shipped here.

## keys Ghost Multiverse Studio (gma/multiverse/, voice_agent/static/multiverse/)

The page that starts with the Ghost and walks a film as a tree of clips: options ahead, the active timeline lit, roads
not taken dimmed, any node a place to branch from.

- **keys Ghost Multiverse Studio** by keys (drowzeys): the idea and its design, a visual decision tree of a film's
  possible futures that the Ghost walks with you, picked by voice.
- **Loom** by Janus (@repligate) and contributors ([socketteer/loom](https://github.com/socketteer/loom), MIT): the
  idea of a "multiverse" interface, a tree of generated continuations to read, compare and branch from; credited as an
  idea, no code taken.
- **Jorge Luis Borges**, "The Garden of Forking Paths" (1941): the image of a story that takes every road at once;
  credited as an idea.
- **Three.js** (MIT): the Ghost on the tree is the same Boo, drawn by the same code as the classic page.
- **Pillow** (Alex Clark and contributors, MIT-CMU licence): the development stand-in renderer's generated scenes
  (`gma/multiverse/stub.py`); no photographs, recordings or real people.
- **FFmpeg**: the stand-in renderer's clips, each option's poster, first and last frames, and the exported path.
- **Node.js** (OpenJS Foundation, MIT): runs the tree view's tests (`node --test tests/js/`), nothing at run time.

## Invisible mode (gma/watch/, voice_agent/static/invisible/)

The Multiverse Studio's watching mode: the Ghost perched above a timeline of everything the agents make (pictures,
clips, sounds, ideas), first on the left, newest on the right, forks in the lanes below; every creation found by its
words and opened with the Ghost's own summary and description.

- **Invisible mode** by keys (drowzeys): the idea and its design, a Ghost that watches over a studio's agents and keeps
  the catalogue of what they made, from what.
- **SQLite** with its **FTS5** full-text index (D. Richard Hipp and the SQLite developers, public domain): the ledger
  of creations and its search, through Python's own `sqlite3`.
- The Ghost's look at each creation uses the engines credited above: **Cosmos3-Edge** (the eyes, on a picture or three
  frames of a clip), **Parakeet TDT 0.6B v3** (the ears, on a sound) and the Director's brain (**Qwen3.8** on .3) for
  its words; **FFmpeg** for the frames and the thumbnails.
- The watched sources are **AutoPilot** and the video panel (Keys and Sam Wasserman, Imagination Studio), the
  **Multiverse** films, and any **Imagination Studio** whose jobs the profile names.

## Clips: the FP4 clip worker (gma/clips.py, gma/fp4_worker.py)

Every clip Ghost Studio makes comes from one resident process, and from nothing else.

- **FastVideo** by Hao AI Lab ([hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo), Apache-2.0; licence
  text in `LICENSES/FastVideo-LICENSE-Apache-2.0`): the runtime, used unmodified at commit
  `a01c50a8063386895c6d8895a1e2509d11d55284` (its `fastvideo` package and `fastvideo-kernel`, built for sm_121a) in a
  venv of the worker's own; its MiniMax H3 pipeline (first-frame FL2VA and text-to-video with stereo audio), MiniMax
  H3 video sparse attention with FP4 attention, NVFP4 linears and the light H3 VAE decode. The worker wraps its
  stages to read the clip's stage and forward count, and copies safetensors' tensors out of their memory map on load
  (faster on GB10); FastVideo's code is not changed.
- **The FastH3 V2 FP4 recipe**: FastVideo's `examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml` and its
  "FastH3 V2 NVFP4 on one Spark" guide (`docs/getting_started/installation/spark_performance.md`): NVFP4 DiT with the
  `h3_dit_vsa` profile, VSA sparsity 0.8 with 64-token tiles and FP4 attention, eight DiT forwards, the cuDNN NVFP4
  GEMM, eager DiT; by FastVideo (Hao AI Lab), Apache-2.0.
- **FastH3 V2** (`FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`, revision `9d0d0d3a`) by FastVideo (Hao AI
  Lab), its model card's terms: the 8-step distilled MiniMax H3 transformer in NVFP4, the NVFP4 **Qwen3-VL** text
  encoder (Qwen team, Alibaba Cloud, Apache-2.0) trimmed to 50 layers, **LynnReal**'s lightweight video VAE and the H3
  audio VAE, and the precomputed AdaLN tables. Ghost Studio adds the first-frame entries to that table, computed once
  by FastVideo's own `FASTVIDEO_H3_ADALN_DUMP` from the same projections (bit-identical clips); no weights are shipped.
- **FlashInfer** (Apache-2.0): NVFP4 quantize and GEMM; **PyTorch** (BSD-3-Clause), **Triton** (OpenAI, MIT),
  Hugging Face **Transformers**, **Diffusers** and **safetensors** (Apache-2.0), NVIDIA **cuDNN** (NVIDIA SLA).

## Ghost 3.8 (the Ghost profile)

- **Ghost 3.8 27B** (Apache-2.0): Diffusion-Qwen3.8 (from **Qwen3.8** 27B by the Qwen team, Alibaba Cloud) fused with
  **Arete** by Nous Research (Apache-2.0), served autoregressively as MLX 4-bit with its vision tower. In
  `profile-spark3-ghost` it is the voice brain, AutoPilot's Director and the self-check's looker (gma/ap/qc.py
  `GhostLooker`), one resident model. Weights are not shipped here.
- **TensorFold 1.2.4** by Ash Hart and the TensorFold contributors (Apache-2.0; its NOTICE and third-party notices ship
  with the release): the CUDA server, `--vision` (the shared Qwen-VL tower).
- **DFlash2** drafter for Qwen3.8-27B by z-lab (Apache-2.0): speculative decoding beside Ghost 3.8 and Qwen3.8.

## Earlier versions

Earlier versions captioned with Qwen3.5-0.8B (Qwen team, Alibaba Cloud, Apache-2.0) and answered camera questions with
AEON-7's Nemotron 3 Nano Omni build (from NVIDIA's Nemotron 3 Nano Omni, NVIDIA Open Model License). Until
2026-11-07 clips were jobs of the headless Studio (TensorFold's int8 FastH3 blocks in ComfyUI, by Ash Hart and the
TensorFold contributors, Apache-2.0), and the video panel could also drive a remote Studio or a HyperFlow-Infinity
service; those routes were removed when every clip moved to the FP4 clip worker.

## The live stage (gma/live/)

A scene that plays as it is made and is directed while it plays (`/live`). Its picture engine runs in its own container
with no network.

- **FastFrames** by Apple ([apple-aiml-research/ml-fastframes](https://github.com/apple-aiml-research/ml-fastframes);
  code under the Apple Sample Code License, the FastFrames-S weights under the Apple Model License for Research): the
  picture engine. FastFrames-S makes one latent frame at a time in two steps, with a KV refinement network and a pixel
  decoder. Its code is used unmodified from a local clone; `gma/live/engine/worker.py` runs it as an endless stream (a
  ring of the last twelve latent frames, positions that keep counting, the prompt changed while it plays), and
  `gma/live/engine/fastpipe.py`, `fp4.py` and `fattn.py` are Ghost Studio's own faster forward for a DGX Spark.
- **The rehearsal and the take** (`gma/live/film.py`, `gma/live/director.py`): rehearse, "action" and "cut" follow
  **Pika Enact** (the idea only). With `"render": "h3"` the reference shot is the Studio's keyframe (**FLUX 3 Image
  Klein** or **Qwen-Image**, as listed above), the character's two loops and the take are **MiniMax H3** through
  **FastVideo's FastH3** clip worker, the director is heard by **Parakeet** and answered in a **Kokoro** voice, and the
  plan is kept by the app's brain. All are credited in their own sections above.
- **Causal Forcing++** by thu-ml ([thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing), see its
  repository for terms): the frame-wise two-step generator FastFrames-S runs.
- **Wan 2.1** by the Alibaba Wan team (Apache-2.0): the T2V 1.3B video model under FastFrames-S, and its umT5 text
  encoder (Google).
- **FlashInfer** (Apache-2.0): the engine's NVFP4 quantize, GEMM and attention kernels. The attention kernel derives
  from **SageAttention** by thu-ml (Apache-2.0).
- **OpenH264** by Cisco (BSD-2-Clause): the encoder of the takes.
- **MuseTalk 1.5** by [Tencent Music Entertainment Lyra Lab](https://github.com/TMElyralab/MuseTalk) (MIT, code and
  weights): Live Action's talking character (`gma/live/lips.py`, `gma/live/lipsync/server.py`). Her lips follow each
  line she says at the rehearsal. It works in the latent space of **sd-vae-ft-mse** by Stability AI (MIT), led by
  **Whisper tiny** by OpenAI (MIT). The face and its landmarks come from **face-alignment** by Adrian Bulat (BSD-3-Clause;
  S3FD and 2D-FAN, in place of MuseTalk's DWPose), the blend mask from **face-parsing (BiSeNet)** by zllrunning (MIT) on
  torchvision's **ResNet-18** (BSD-3-Clause). None of them is shipped here: the profile's `live.lipsync.home` points
  at a local copy.
- **MotionBricks** by [NVIDIA](https://github.com/NVlabs/motionbricks) (code Apache-2.0; weights under the NVIDIA Open
  Model License; the sample pose clips are for evaluation): Live Action's blocking preview (`gma/live/blocking.py`,
  `gma/live/engine/mb.py`). Its SOMA model, run from its ONNX file with **ONNX Runtime** (Microsoft, MIT) on the CPU,
  makes the cast's walks, runs and turns between their marks; a fight stance, a jump, a crouch, a dance, sitting and
  looking around are played from MotionBricks' own sample motions (NVIDIA Sample Data Evaluation License: for evaluating
  and testing NVIDIA technologies), and the punches, kicks, blasts, waves and flights over them are drawn by this
  project's own code. They are drawn as stick figures on a floor plan before anything is filmed. Nothing of
  MotionBricks is shipped here: the profile's `live.blocking.home` points at a local copy.
- **Silero VAD** by Silero (MIT): Live Action's open microphone (`gma/live/ear.py`) finds the director's sentences with it.
- **Pika Enact** by Pika: the idea of directing a scene while it plays, with notes to the cast and the camera and a
  clean take afterwards. An idea, credited; none of its code or weights.
- **Kokoro-82M** by hexgrad (Apache-2.0): the cast's voices. **Parakeet TDT 0.6B v3** by NVIDIA (CC-BY-4.0): the spoken
  notes. The brain that rewrites the scene is the profile's Director.

