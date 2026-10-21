ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}

# Install tcmalloc (enabled via LD_PRELOAD at the end of this file)
RUN apt-get update && apt-get install -y --no-install-recommends libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

ARG CACHE_BUSTER=1

# Install PyTorch CPU nightly first so downstream packages do not pull CUDA wheels.
# To build against a pinned release instead (e.g. torch 2.15.0 for torch_tpu 0.2.0),
# set TORCH_VERSION and point TORCH_INDEX_URL at the matching CPU index.
ARG CACHE_BUSTER
ARG TORCH_VERSION=2.15.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/test/cpu
RUN pip install --pre "torch${TORCH_VERSION:+==${TORCH_VERSION}}" --index-url "${TORCH_INDEX_URL}"

# Install TPU requirements
COPY torchtitan/experiments/tpu/requirements.txt ./requirements-tpu.txt
RUN pip install -r ./requirements-tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install torch_tpu: the release from PyPI if TORCH_TPU_VERSION is set (e.g. 0.2.0,
# with the matching TORCH_VERSION above), otherwise the torch_tpu-*.whl in the build
# context. requirements.txt is listed so the COPY succeeds when there is no wheel.
ARG TORCH_TPU_VERSION=0.2.0
COPY requirements.txt torch_tpu-*.whl* ./
RUN if [ -n "${TORCH_TPU_VERSION}" ]; then \
      pip install "torch_tpu==${TORCH_TPU_VERSION}" --extra-index-url "${TORCH_INDEX_URL}"; \
    else \
      pip install ./torch_tpu-*.whl --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN pip install -r requirements.txt
COPY .ci/docker/requirements-flux.txt .
RUN pip install -r requirements-flux.txt

# Install test utilities (CI runs with this image)
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Log the installed versions; later installs must not have replaced torch.
RUN python -c "import importlib.metadata as m; print('torch', m.version('torch'), 'torch_tpu', m.version('torch_tpu'))"

COPY torchtitan /torchtitan/torchtitan
COPY assets /torchtitan/assets
COPY tests /torchtitan/tests

WORKDIR /torchtitan

# Enable tcmalloc at runtime (set after all RUN steps so build steps are unaffected)
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
