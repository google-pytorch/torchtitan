ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}

# Install tcmalloc (enabled via LD_PRELOAD at the end of this file)
RUN apt-get update && apt-get install -y --no-install-recommends libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

ARG CACHE_BUSTER=1

# Install PyTorch CPU nightly first so downstream packages do not pull CUDA wheels
ARG CACHE_BUSTER
ARG TORCH_VERSION=""
RUN pip install --pre "torch${TORCH_VERSION:+==${TORCH_VERSION}}" --index-url https://download.pytorch.org/whl/nightly/cpu

# Install TPU requirements
COPY torchtitan/experiments/tpu/requirements.txt ./requirements-tpu.txt
RUN pip install -r ./requirements-tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

COPY torch_tpu-*.whl ./
RUN pip install ./torch_tpu-*.whl --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .ci/docker/requirements-flux.txt .
RUN pip install -r requirements-flux.txt

# Install test utilities (CI runs with this image)
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY torchtitan /torchtitan/torchtitan
COPY assets /torchtitan/assets
COPY tests /torchtitan/tests

WORKDIR /torchtitan

# Enable tcmalloc at runtime (set after all RUN steps so build steps are unaffected)
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
