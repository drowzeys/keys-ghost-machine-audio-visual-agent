# Boo (keys-ghost-machine-audio-visual-agent) for an NVIDIA DGX Spark (linux/arm64, CUDA 13).
# VOICE_ENGINE=full (Drama 3, default), lite (Kokoro-82M) or both (a Full/Lite switch in the page).
# Model weights are not included: mount them under /models (see docker/README.md).
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG TF_VERSION=1.1.5
ARG KAI_COMMIT=7f2f02b
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Node 22 for the optional Eyes of Kai memory server
RUN curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz | tar xJ -C /opt \
    && ln -s /opt/node-v22.14.0-linux-arm64/bin/node /usr/local/bin/node \
    && ln -s /opt/node-v22.14.0-linux-arm64/bin/npm /usr/local/bin/npm

# Stock TensorFold 1.1.5 for Bolt, and the Boo stack's CUDA builds (Cosmos3-Edge, Drama 3 NVFP4 with voices),
# all from the build context (see docker/README.md)
COPY tensorfold-${TF_VERSION}-linux-aarch64 /opt/tensorfold
COPY keys-boo-tensorfold-${TF_VERSION}-linux-aarch64 /opt/keys-tensorfold
# The CUDA 13.1 cuBLASLt the TensorFold engines are qualified with, for them only (the entrypoint puts it first on their
# library path; PyTorch keeps the base image's 13.6). On 13.6 Drama 3 turns its one-row tower kernel off and speaks
# slower.
RUN pip install --no-deps --target /tmp/cublas nvidia-cublas==13.1.1.3 && mkdir -p /opt/tf-cuda/lib \
    && mv /tmp/cublas/nvidia/cu13/lib/libcublasLt.so.13 /tmp/cublas/nvidia/cu13/lib/libcublas.so.13 /opt/tf-cuda/lib/ \
    && rm -rf /tmp/cublas

# The agent and its Python dependencies (PyTorch comes from the base image)
WORKDIR /opt/boo
COPY voice_agent ./voice_agent
COPY profiles ./profiles
COPY patches ./patches
COPY silero_vad.onnx requirements.txt requirements-spark.txt requirements-kokoro-spark.txt LICENSE CREDITS.md README.md ./
RUN pip install -r requirements.txt "nemo_toolkit[asr]>=3.0"

# The optional lite voice, Kokoro-82M (started when VOICE_ENGINE is lite or both), in its own environment on the
# image's PyTorch: misaki pins spaCy's tokenizer libraries. Its weights download into /models/hf-cache on first start.
RUN python3 -m venv --system-site-packages /opt/kokoro-venv \
    && /opt/kokoro-venv/bin/pip install -r requirements-kokoro-spark.txt

# Eyes of Kai (MIT, tonyd2wild) with the retention patch, started only when KAI=1
RUN git clone -q https://github.com/tonyd2wild/eyes-of-kai /opt/eyes-of-kai && cd /opt/eyes-of-kai \
    && git checkout -q ${KAI_COMMIT} && git apply /opt/boo/patches/eyes-of-kai-retention.patch \
    && npm ci --omit=dev --no-audit --no-fund && rm -rf .git

# The 14 synthetic Drama 3 voices (Spark format) from the Hugging Face repo
COPY voices /voices

COPY docker/entrypoint.sh /usr/local/bin/boo
RUN chmod +x /usr/local/bin/boo
EXPOSE 8095
ENTRYPOINT ["/usr/local/bin/boo"]
