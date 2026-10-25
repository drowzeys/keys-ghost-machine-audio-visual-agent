# Boo (keys-ghost-machine-audio-visual-agent) for an NVIDIA DGX Spark (linux/arm64, CUDA 13).
# Model weights are not included: mount them under /models (see docker/README.md).
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG TF_VERSION=1.1.4
ARG KAI_COMMIT=7f2f02b
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Node 22 for the optional Eyes of Kai memory server
RUN curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz | tar xJ -C /opt \
    && ln -s /opt/node-v22.14.0-linux-arm64/bin/node /usr/local/bin/node \
    && ln -s /opt/node-v22.14.0-linux-arm64/bin/npm /usr/local/bin/npm

# Stock TensorFold 1.1.4 for Bolt, and the Boo stack's CUDA builds (Cosmos3-Edge, Drama 3 NVFP4 with voices),
# all from the build context (see docker/README.md)
COPY tensorfold-${TF_VERSION}-linux-aarch64 /opt/tensorfold
COPY keys-boo-tensorfold-1.1.4-linux-aarch64 /opt/keys-tensorfold

# The agent and its Python dependencies (PyTorch comes from the base image)
WORKDIR /opt/boo
COPY voice_agent ./voice_agent
COPY profiles ./profiles
COPY patches ./patches
COPY silero_vad.onnx requirements.txt requirements-spark.txt LICENSE CREDITS.md README.md ./
RUN pip install -r requirements.txt "nemo_toolkit[asr]>=3.0"

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
