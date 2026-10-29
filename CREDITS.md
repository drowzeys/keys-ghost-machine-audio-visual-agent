# Credits

What this agent is built from, and under which terms. The code in this repository is MIT (see LICENSE); everything
below keeps its own licence.

## Ears, brain and voice

- **Silero VAD** v5 by [Silero](https://github.com/snakers4/silero-vad) (MIT): when someone is speaking.
  `silero_vad.onnx` is shipped here under that licence.
- **Parakeet TDT 0.6B v3** by NVIDIA (CC-BY-4.0): speech to text, through
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (Apache-2.0) on the Mac and NeMo (Apache-2.0) on the DGX
  Spark.
- **Nemotron 3.5 Bolt** by NVIDIA (NVIDIA Open Model License), served by TensorFold: the front brain that answers
  most turns, with mood tags for the face.
- **Fish Audio Drama 3** by [Fish Audio](https://huggingface.co/fishaudio/drama-3) (Fish Audio Community License),
  served by TensorFold: the voice. The 4-bit pruned checkpoint
  [drama-3-NVFP4-Pruned](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) and its CUDA engine and named voices
  (`--voices`) by [plotarmordev](https://github.com/plotarmordev/TensorFold). Hosted services that generate speech for
  third parties need a separate licence from Fish Audio.
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
- **EmbeddingGemma-2** by [Google DeepMind](https://huggingface.co/google/embeddinggemma-2): the optional memory search
  by meaning (`voice_agent/recall.py`), embedding captions and conversation lines. Its model card states Apache-2.0
  (the card's licence link points at the Gemma 4 licence page, labelled Apache 2.0; the repository ships no LICENSE
  file), and says deployments must follow the Gemma Prohibited Use Policy. Its task prefixes (`task: search result |
  query:`, `title: none | text:`) are the card's. On the DGX Spark it runs on TensorFold's CUDA engine.
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

## d1-2B assist (optional)

- **d1-2B** by [Liquid AI](https://huggingface.co/LiquidAI/d1-2B) (LFM Open License v1.0): a decision model built on
  LFM3-VL-2B that answers questions with probabilities in one pass. Boo asks it where a turn the patterns miss should
  go and what a reply to a tool-permission question means (`voice_agent/d1.py`). The licence allows use,
  modification and redistribution (with the licence and notices kept); commercial use is licensed only to a company
  under US$10 million in annual revenue (non-profits are exempt for non-commercial or research use).
- **d1-2B on TensorFold CUDA** (the bf16 LFM3 text decoder, the LFM3-VL vision tower, d1's set attention and decision
  head) is our port, on [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) branch `d1-2b-cuda-wip`, built
  on the open LFM3 work upstream ([ashhart/TensorFold](https://github.com/ashhart/TensorFold) #1513, #1514, #1508).

## Earlier versions

Earlier versions captioned with Qwen3.5-0.8B (Qwen team, Alibaba Cloud, Apache-2.0) and answered camera questions with
AEON-7's Nemotron 3 Nano Omni build (from NVIDIA's Nemotron 3 Nano Omni, NVIDIA Open Model License).
