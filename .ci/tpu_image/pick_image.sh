#!/usr/bin/env bash
# Picks the image the TPU CI suite runs in.
#
# :latest is built nightly from main, so a change that edits the image's inputs
# (pins, Dockerfile) would otherwise be tested against the old dependencies.
# Diffs those inputs against the commit :latest was built from and emits:
#   image=<ref>     when an image can be used as is (:latest, or a previously
#                   rebuilt image for the same inputs), pinned by digest
#   build=true      when the image must be rebuilt from this checkout, along
#                   with deps_tag=<tag to push> and base_ref=<:latest by digest>
#
# Needs: IMAGE_REPO, a full-history checkout, docker logged in to the registry,
# and jq. Outputs go to $GITHUB_OUTPUT (stdout when run locally).
set -euo pipefail

: "${IMAGE_REPO:?}"
out="${GITHUB_OUTPUT:-/dev/stdout}"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"

# Everything the Dockerfile reads from the build context, other than the
# torchtitan/ and tests/ sources that CI runs from the checkout instead.
# requirements.txt and requirements-dev.txt are symlinks into .ci/docker/;
# `git diff` only sees the link itself, so the targets are listed as well.
image_inputs=(
  torchtitan/experiments/tpu/Dockerfile
  torchtitan/experiments/tpu/requirements.txt
  requirements.txt
  .ci/docker/requirements.txt
  requirements-dev.txt
  .ci/docker/requirements-dev.txt
  .ci/docker/requirements-flux.txt
)

manifest_digest() {
  docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' | jq -r .digest
}

# Pin :latest by digest so every shard runs the same image even if the nightly
# promotes a new one mid-run.
base_digest=$(manifest_digest "${IMAGE_REPO}:latest")
base_ref="${IMAGE_REPO}@${base_digest}"
base_rev=$(docker buildx imagetools inspect "${base_ref}" --format '{{json .Image.Config.Labels}}' \
  | jq -r '."org.opencontainers.image.revision" // empty')
echo "base_ref=${base_ref}" >> "${out}"

if [[ -n "${base_rev}" ]] && git cat-file -e "${base_rev}^{commit}" 2>/dev/null; then
  if git diff --quiet "${base_rev}" HEAD -- "${image_inputs[@]}"; then
    echo "Image inputs match ${base_ref} (built from ${base_rev}); using it."
    echo "image=${base_ref}" >> "${out}"
    echo "Test image: \`${base_ref}\` (:latest)" >> "${summary}"
    exit 0
  fi
  echo "Image inputs changed since ${base_rev}, which :latest was built from:"
  git diff --stat "${base_rev}" HEAD -- "${image_inputs[@]}"
else
  echo "::warning::Cannot tell which commit :latest was built from (revision '${base_rev}'); rebuilding the image."
fi

# Tag by inputs and base image: same inputs on the same nightly reuse one
# build; a new nightly rebuilds on top of the newer torch_tpu.
inputs_hash=$(git ls-tree -r HEAD -- "${image_inputs[@]}" | sha256sum | cut -c1-12)
base_short=${base_digest#sha256:}
deps_tag="${IMAGE_REPO}:deps-${inputs_hash}-${base_short:0:12}"
echo "deps_tag=${deps_tag}" >> "${out}"

if deps_digest=$(manifest_digest "${deps_tag}" 2>/dev/null); then
  echo "Reusing ${deps_tag}"
  echo "image=${IMAGE_REPO}@${deps_digest}" >> "${out}"
  echo "Test image: \`${deps_tag}\` (rebuilt for changed dependencies, reused)" >> "${summary}"
else
  echo "build=true" >> "${out}"
fi
