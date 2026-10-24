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
  following Eyes of Kai's Hermes client and the frontstage/backstage design of
  [qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent).
- **RED-SNOW 5.3 Flash** (a GLM-5.3-Flash build), the back agent's model in the reference setup; any OpenAI-compatible
  model works.
- **ddgs** (Dux Distributed Global Search, MIT) by deedy5: the web search behind quick live facts.
- **wttr.in** by Igor Chubin (Apache-2.0), with data from WorldWeatherOnline: the weather lookup.

## Earlier versions

Earlier versions captioned with Qwen3.5-0.8B (Qwen team, Alibaba Cloud, Apache-2.0) and answered camera questions with
AEON-7's Nemotron 3 Nano Omni build (from NVIDIA's Nemotron 3 Nano Omni, NVIDIA Open Model License).
