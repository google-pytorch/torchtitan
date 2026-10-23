## TorchTPU Support for TorchTitan

This directory provides a demonstration of running TorchTitan on Google Cloud TPUs using the TorchTPU framework. This document describes how to provision Cloud TPU resources and initiate TorchTitan training jobs.

---

## Before you begin

Before you start, complete the following steps:

1. **Verify your project:** Ensure you have a Google Cloud project with billing enabled.
2. **Request TPU access:** To obtain access to Cloud TPU resources, contact your Google Cloud account team.
3. **Install the gcloud CLI:** Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

## Single-host model training on a TPU VM

For debugging and development, you can train small models directly on TPU VMs. You can create a TPU VM and connect to it using SSH.

### Prerequisites

1. **Project setup:** [Set up a Google Cloud project for Cloud TPUs](https://cloud.google.com/tpu/docs/setup-gcp-account).
2. **Planning:** [Plan your Cloud TPU resources](https://cloud.google.com/tpu/docs/plan-tpus).
3. **Environment variables:** Set the following variables (the following example uses a `v6e` TPU):
4. **Enable the API:** Enable the [Cloud TPU API](https://docs.cloud.google.com/tpu/docs/setup-gcp-account).

### Create a TPU VM

You can create a TPU VM using the **Create TPU** page in the [Google Cloud console](https://console.cloud.google.com/compute/tpus/add) or by using the following gcloud CLI commands.

```bash
export TPU_NAME=your-tpu-name
export PROJECT_ID=your-project
export ZONE=your-zone  # For example, us-central2-b
export ACCELERATOR_TYPE=your-accelerator # For example, v6e-4
export VERSION=matching-runtime # v2-alpha-tpuv6e for v6e

```

To create a TPU VM, run the following command:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$VERSION

```

Consult the **Quotas** page in the Google Cloud console to identify zones where Cloud TPUs are available and where your project has sufficient quota: [https://console.cloud.google.com/apis/api/tpu.googleapis.com/quotas](https://console.cloud.google.com/apis/api/tpu.googleapis.com/quotas)

**Flags:**

* `--zone`: Specifies the zone for the Cloud TPU. For more information on availability, see [Cloud TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) and your project's **Quotas** page.
* `--accelerator-type`: Specifies the TPU version and size (for example, `v6e-4`).
* `--version`: Specifies the Cloud TPU software version (runtime). Ensure you use the correct version for your hardware. For a complete list, see [Cloud TPU software versions](https://docs.cloud.google.com/tpu/docs/runtimes#tpu_software_versions).
* `--reserved`: Use this flag if you are creating the TPU VM from a reservation.
* `--spot`: Use this flag to create a TPU Spot VM.

After you create the TPU VM, connect to it using SSH:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE  \
    --project=$PROJECT_ID

```

### Install TorchTitan and TorchTPU

Choose one of the following options:

* [Option A](#option-a-torchtpu-release-with-pytorch-215-recommended): To run the stable release. Installs TorchTPU 0.2.0 with PyTorch 2.15.0; no build required.
* [Option B](#option-b-build-torchtpu-from-source-for-the-pytorch-nightly-for-development): For development. Builds TorchTPU from source to use the latest PyTorch nightly.

Both options use Python 3.14. Create and activate a virtual environment:

```bash
python3.14 -m venv venv; source venv/bin/activate
```

#### Option A: TorchTPU release with PyTorch 2.15 (recommended)

1. Clone the torchtitan repository:

    ```bash
    git clone https://github.com/google-pytorch/torchtitan
    cd torchtitan
    ```

2. Install PyTorch 2.15.0, TorchTPU 0.2.0, and the TPU dependencies, then the TorchTitan dependencies. The extra index provides the CPU build of PyTorch:

    ```bash
    pip install torch==2.15.0 torch_tpu==0.2.0 -r torchtitan/experiments/tpu/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/test/cpu
    pip install -r requirements.txt -r .ci/docker/requirements-flux.txt
    ```

    The TPU and TorchTitan dependencies are installed in separate commands because `tokamax` and `tyro` declare conflicting `typeguard` versions. pip warns about the conflict but completes the install.

    > [!NOTE]
    > PyTorch 2.15.0 is on the PyTorch test (release candidate) index until its final release. After the release, use `--extra-index-url https://download.pytorch.org/whl/cpu`.

#### Option B: Build TorchTPU from source for the PyTorch nightly (for development)

1. Clone [torch_tpu](https://github.com/google-pytorch/torch_tpu) and build the wheel. For details, see the [TorchTPU build instructions](https://github.com/google-pytorch/torch_tpu?tab=readme-ov-file#build-from-source).

    ```bash
    git clone https://github.com/google-pytorch/torch_tpu.git
    cd torch_tpu
    bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel
    cd ..
    ```

    `--config=wheel_common` is required. It also enables the Remote Build Execution (RBE) cache and executor. If you do not have RBE credentials, add `--config=no_rbe` after `--config=wheel_common`.

    The wheel is written to `torch_tpu/bazel-bin/ci/wheel/`.

2. Clone the torchtitan repository:

    ```bash
    git clone https://github.com/google-pytorch/torchtitan
    cd torchtitan
    ```

3. Install the PyTorch CPU nightly first, then the TorchTitan dependencies and the TorchTPU wheel:

    ```bash
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
    pip install -r torchtitan/experiments/tpu/requirements.txt
    pip install -r requirements.txt -r .ci/docker/requirements-flux.txt
    pip install ../torch_tpu/bazel-bin/ci/wheel/*.whl
    ```

### Run a TorchTitan training job

Only small models can run on a single host. The following example demonstrates Qwen3 1.7B running on a `v6e-4` VM using FSDP. It uses the tokenizer bundled in `tests/assets/tokenizer` and the `c4_test` sample dataset, so it needs no downloads and no Hugging Face token:

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_1_7b \
    --training.steps=10
```

---

## Multi-host model training on GKE

To train larger models, use a multi-host configuration on Google Kubernetes Engine (GKE).

### Provision Cloud TPU resources

To provision Cloud TPUs, use one of the following methods:

* **GKE:** Use GKE to provision and manage Cloud TPUs as a pool of accelerators for containerized machine learning workloads. Use the gcloud CLI to create GKE cluster instances manually for precise customization or to expand existing production GKE environments. For more information, see [TPUs in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus).
* **GKE and Cluster Toolkit:** Cluster Toolkit is an open-source tool that simplifies cluster creation and workload execution on GKE. It is designed for ML practitioners to provision Cloud TPUs and run training jobs without requiring deep Kubernetes expertise. Use Cluster Toolkit blueprints to create GKE clusters, and use its `gcluster` CLI to submit workloads. For more information, see the [Cluster Toolkit overview](https://docs.cloud.google.com/cluster-toolkit/docs/overview) and the [Cluster Toolkit GitHub repository](https://github.com/GoogleCloudPlatform/cluster-toolkit).
* **GKE and TPU Cluster Director:** TPU Cluster Director is available through an All Capacity mode reservation, providing full access to reserved capacity without hold-backs. This mode offers full visibility into TPU hardware topology, utilization, and health status. For more information, see the [All Capacity mode overview](https://cloud.google.com/tpu/docs/all-capacity-overview).

In this document, you use GKE managed by Cluster Toolkit. Cluster Toolkit's `gcluster` command-line tool is designed to simplify provisioning, managing, and running machine learning workloads.

### Install Cluster Toolkit and its dependencies

1. **Install Cluster Toolkit:** Follow the [Cluster Toolkit installation guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment) to install the `gcluster` CLI, and make sure `gcluster` is on your `PATH`. Verify the installation:

```bash
gcluster --version

```

2. **Install Docker:** Follow the official Docker installation instructions. After installation, run the following commands to configure Docker and verify the installation:

```bash
gcloud auth configure-docker
sudo usermod -aG docker $USER
# Restart the terminal and reactivate the virtual environment after running this command.
docker run hello-world # Verify the Docker installation.

```

3. **Set environment variables:**

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=YOUR_REGION
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export COMPUTE_TYPE=YOUR_COMPUTE_TYPE
export TOPOLOGY=YOUR_TOPOLOGY
export AUTHORIZED_CIDR=YOUR_AUTHORIZED_CIDR
export BUCKET_NAME="gs://YOUR_BUCKET_NAME"
export REPOSITORY=YOUR_REPOSITORY

```

Replace the following placeholders:

* `YOUR_PROJECT_ID`: Your Google Cloud project ID.
* `YOUR_REGION`: The region that contains `YOUR_ZONE` (for example, `us-east5`).
* `YOUR_ZONE`: The zone where the cluster's TPU nodes will be created. For TPU availability, see [Cloud TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) and your project's **Quotas** page.
* `YOUR_CLUSTER_NAME`: The name of the new GKE cluster. Cluster Toolkit also uses this name as the deployment name.
* `YOUR_COMPUTE_TYPE`: The TPU machine type (for example, `ct6e-standard-4t` for TPU v6e).
* `YOUR_TOPOLOGY`: The TPU slice topology (for example, `2x4` for a `v6e-8` slice).
* `YOUR_AUTHORIZED_CIDR`: The CIDR block that can access the cluster's control plane. It must contain the IP address of the machine where you run `gcluster`. To allow only your IP address, use your IP address followed by `/32`. To allow all IP addresses, use `0.0.0.0/0`; Identity and Access Management (IAM) restrictions are still enforced.
* `YOUR_BUCKET_NAME`: The name of your Cloud Storage bucket for output.
* `YOUR_REPOSITORY`: The name of your Artifact Registry repository (for example, `test`).

4. **Optional: Create a Cloud Storage bucket:**

```bash
gcloud storage buckets create ${BUCKET_NAME} \
    --project=${PROJECT_ID} \
    --location=US \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access

```

### Create a single-slice cluster

Create the cluster with the Cluster Toolkit [GKE TPU v6e blueprint](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/gke-tpu-v6e.yaml). The blueprint creates a VPC network with an MTU of 8,896 for optimal performance and a firewall rule for traffic between nodes, a GKE cluster, and a TPU node pool for one slice. It also installs Kueue and JobSet, which `gcluster job submit` uses to run workloads. For more information, see [Deploy a GKE TPU v6e cluster](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-v6e).

1. Generate Application Default Credentials so that Cluster Toolkit can provision resources with Terraform:

```bash
gcloud auth application-default login

```

2. From the directory where you extracted the Cluster Toolkit bundle (for example, `~/cluster-toolkit`), create the cluster:

```bash
gcluster deploy examples/gke-tpu-v6e/gke-tpu-v6e.yaml \
    --vars project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME} \
    --vars region=${REGION},zone=${ZONE} \
    --vars machine_type=${COMPUTE_TYPE},tpu_topology=${TOPOLOGY},num_slices=1 \
    --vars authorized_cidr=${AUTHORIZED_CIDR},enable_gcsfuse_csi=true

```

`gcluster` displays a Terraform plan; type `a` and press Enter to apply it. Provisioning can take 10 to 20 minutes or more.

**Notes:**

* The blueprint creates a regional cluster in `${REGION}` and places the TPU node pool in `${ZONE}`.
* `enable_gcsfuse_csi=true` enables the Cloud Storage FUSE CSI driver, which you need to mount a Cloud Storage bucket in your workload.
* The blueprint provisions on-demand TPUs. To use a reservation, Spot VMs, or flex-start instead, see the consumption options in [`gke-tpu-v6e-deployment.yaml`](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/gke-tpu-v6e-deployment.yaml).
* `gcluster deploy` creates a deployment directory named `${CLUSTER_NAME}` in the current directory. It contains the Terraform configuration and state for the cluster. Keep this directory, because you need it to delete the cluster.

### Build the TorchTitan TPU-enabled Docker image

Clone the repository from GitHub

```bash
git clone https://github.com/google-pytorch/torchtitan
cd torchtitan

```

#### Optional: Download the Hugging Face model tokenizer

This step is optional if you are running a dummy model. For more information, see the [TorchTitan README](https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer).

```bash
# Download the Qwen3 1.7B and 8B tokenizers (not gated, no Hugging Face token needed)
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B --assets tokenizer
```

The tokenizers are saved to `assets/hf/<model>` (for example `assets/hf/Qwen3-8B`), which is the
directory to pass as `--hf_assets_path`.

Upload the tokenizer to the GCS bucket

```bash
gcloud storage cp --recursive ./assets/hf ${BUCKET_NAME}/assets/hf
gcloud storage cp --recursive ./tests ${BUCKET_NAME}/tests
```

#### Build the Docker image

Run the following commands from the root of the torchtitan repository. Choose the option that matches [Install TorchTitan and TorchTPU](#install-torchtitan-and-torchtpu).

**Option A: TorchTPU release with PyTorch 2.15 (recommended)**

```bash
docker build -t torchtitan-container:latest -f torchtitan/experiments/tpu/Dockerfile \
  --build-arg TORCH_TPU_VERSION=0.2.0 \
  --build-arg TORCH_VERSION=2.15.0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/test/cpu .
```

> [!NOTE]
> PyTorch 2.15.0 is on the PyTorch test (release candidate) index until its final release. After the release, use `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.

**Option B: TorchTPU built from source with the PyTorch nightly (for development)**

First, build the `torch_tpu` wheel as described in step 1 of [Option B](#option-b-build-torchtpu-from-source-for-the-pytorch-nightly-for-development). The commands below assume the `torch_tpu` repository is cloned next to the torchtitan repository.

```bash
rm -f torch_tpu-*.whl
cp ../torch_tpu/bazel-bin/ci/wheel/torch_tpu-*.whl ./
docker build -t torchtitan-container:latest -f torchtitan/experiments/tpu/Dockerfile .
```

The end of the build log shows the installed `torch` and `torch_tpu` versions, for example `torch 2.15.0+cpu torch_tpu 0.2.0` for Option A.

#### Upload the Docker image to Artifact Registry

```bash
# This example assumes you are using Artifact Registry in us-central1.
# This command only needs to be run once.
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# This command only needs to be run once.
gcloud artifacts repositories create ${REPOSITORY} \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for torchtitan training"

docker tag torchtitan-container:latest us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest
docker push us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest

```

#### Optional: Mount a Cloud Storage bucket with Cloud Storage FUSE

To make data in Cloud Storage accessible to TorchTitan and to export logs and profiles, you can mount a bucket using Cloud Storage FUSE. This allows the bucket to appear as a local directory.

With Cluster Toolkit, you don't attach storage to the cluster in a separate step. Instead, you mount the bucket when you submit the workload, by passing the following flags to `gcluster job submit` (as shown in the next section):

* `--mount "$BUCKET_NAME;/data;rw"`: Mounts the bucket at `/data` in read-write mode. This requires the Cloud Storage FUSE CSI driver, which the cluster creation command enables with `enable_gcsfuse_csi=true`.
* `--service-account workload-identity-k8s-sa`: Runs the workload as the Kubernetes service account that the blueprint creates. This service account uses Workload Identity Federation for GKE to access the Cloud Storage buckets in your project.

For more information, see [Submit Job with Persistent Storage](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/docs/gcluster_job_guide.md#44-example-submit-job-with-persistent-storage).

#### Deploy the training workload

The following sample demonstrates Qwen3 8B training on a `v6e-8` slice (`TOPOLOGY=2x4`, two hosts with four chips each) using FSDP and the `reg_trainer_qwen3_8b` config. It follows the [Qwen3-8B v6e-8 recipe](recipes/trillium/qwen3-8b/v6e-8/README.md):

```bash
export WORKLOAD_NAME=torchtitan-qwen3-8b

# if using GCSFUSE, set the following parameters:
# --dump_folder="/data/assets/hf/Qwen3-8B"
# --dataloader.dataset_path="/data/assets/hf/Qwen3-8B"
# --hf_assets_path="/data/assets/hf/Qwen3-8B"
# if not using GCSFUSE, remove the --service-account and --mount flags.
gcluster job submit \
    --name $WORKLOAD_NAME \
    --cluster $CLUSTER_NAME \
    --location $REGION \
    --project $PROJECT_ID \
    --compute-type $COMPUTE_TYPE \
    --topology $TOPOLOGY \
    --num-slices 1 \
    --image "us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/torchtitan-container:latest" \
    --service-account workload-identity-k8s-sa \
    --mount "$BUCKET_NAME;/data;rw" \
    --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
    --command 'torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_backend=static \
        --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
        --node_rank=$TPU_WORKER_ID \
        -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.qwen3 \
        --config=reg_trainer_qwen3_8b \
        --libtpu-init-flags="xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536" \
        --dataloader.dataset=c4_test \
        --training.steps=10 \
        --hf_assets_path=assets/hf/Qwen3-8B/ \
        --dataloader.dataset_path=tests/assets/c4_test \
        --metrics.enable_tensorboard \
        --profiler.profile_freq=5 \
        --profiler.profiler_warmup=3 \
        --profiler.profiler_active=1 \
        --profiler.enable_profiling'
```

---

### Configuring XLA and libtpu flags

`XLA_FLAGS` and `LIBTPU_INIT_ARGS` are typically set via environment variable by the launcher. We support configuring these as part of the TPUTrainerConfig via the `libtpu_init_flags` and `xla_flags` fields. This codifies these flags as citizens of the TorchTitan training configuration instead of them being documented only at the launcher level.

Their override precedence is as follows:

1. **Config**: Flags are stored as `libtpu_init_flags` and `xla_flags` in the TPUTrainerConfig.
2. **CLI:** `--extra-libtpu-init-flags` and `--extra-xla-flags` is the additive layer: it **adds** flags the config does not set and **overrides** the ones it does, leaving the rest alone.

    (Note: setting `--libtpu-init-flags` and `--xla-flags` in the command line without the `extra` keyword **replaces the entire set** of flags defined in the config. This enables runs with default flags by setting either or both to the empty string.)
3. **Environment**: an exported `LIBTPU_INIT_ARGS` or `XLA_FLAGS` has the last word, **overriding** flags named in the config and **adding** any new flags.

Values are whitespace-separated `flag=value` entries.

#### Usage

Taking [`qwen3_32b`](qwen3/config_registry.py), whose flags include `xla_tpu_dvfs_p_state=7`:

```bash
# 1. Add a flag the config does not set. xla_dump_to is an XLA flag, not a
#    libtpu one: libtpu rejects it as an unknown flag.
--extra-xla-flags=xla_dump_to=/tmp/hlo

# 2. Override a flag the config sets.
--extra-libtpu-init-flags=xla_tpu_dvfs_p_state=5

# 3. Drop single flags (use libtpu/XLA default)
--extra-libtpu-init-flags='!xla_tpu_enable_latency_hiding_layer_scheduler'

# 4. Add or override several flags at once -- quote the whole value.
--extra-libtpu-init-flags='xla_tpu_dvfs_p_state=5 xla_tpu_enable_sparse_core_collective_aggregator=false'

# 5a. Drop the recipe's flags entirely -- set to the empty string.
--libtpu-init-flags=''

# 5b. Replace the *entire* set of flags set by the config
--libtpu-init-flags="xla_tpu_enable_sparse_core_collective_aggregator=true \
xla_tpu_enable_latency_hiding_layer_scheduler=true \
xla_tpu_scoped_vmem_limit_kib=131072"
```

#### Flag bundling and merging flags

Groups of flags that only make sense together can be found in named bundles in [`tpu_flags.py`](tpu_flags.py)

A config can declare its flags by composing bundles and single knobs as so:

```python
libtpu_init_flags=tpu_flags.merge_flags(
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    xla_tpu_scoped_vmem_limit_kib=65472,
)
```

#### Logging

Overridden and removed flags are logged as `INFO`. If the environment sets a flag that overrides one set in the config, a warning is logged:

```
[titan] ... - INFO - Overriding LIBTPU_INIT_ARGS xla_tpu_dvfs_p_state=7 with 5 from the launch command.
[titan] ... - INFO - Removing LIBTPU_INIT_ARGS xla_tpu_enable_latency_hiding_layer_scheduler=true at the request of the launch command.
[titan] ... - WARNING - LIBTPU_INIT_ARGS already sets xla_tpu_scoped_vmem_limit_kib=131072 in the environment; keeping it and ignoring the config value 65472.
```

This also enables the `--debug.print-config` option to print the full configuration including the global XLA and libtpu flags set at launch (however: note that they may still be overridden by a LIBTPU_INIT_ARGS environment var and XLA_FLAGS may also be overridden per compile by torch_tpu via `TORCH_TPU_INTERNAL_XLA_OPTIONS` and the custom_compiler_options context manager)

---

### Profile workloads using TensorBoard

TorchTitan on TorchTPU integrates with a profiler to export profiling data for trained models. Use the `--profiler.enable_profiling` flag to enable profiling and the `--profiler...` flags for configuration.

Profiling data is exported to the directory specified by `--dump_folder` + `profile_trace` for each process. In the example above, this is mapped to an external Cloud Storage bucket via Cloud Storage FUSE and can be loaded directly into TensorBoard.

For more information, see [Profile your TPU VM workload using TensorBoard](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm).

To install TensorBoard locally, run the following commands:

```bash
pip install tensorboard_plugin_profile tensorboard
pip install xprof

```

To view the profile captured for your TorchTitan training job, run:

```bash
tensorboard --logdir=${BUCKET_NAME}/torchtitan-qwen3-8b/rank_0

```

### Clean up

To avoid incurring unnecessary charges, delete your resources when you are finished:

```bash
# Cancel the workload.
gcluster job cancel torchtitan-qwen3-8b \
    --cluster ${CLUSTER_NAME} \
    --location ${REGION} \
    --project ${PROJECT_ID}

# Delete the GKE cluster and the VPC network that Cluster Toolkit created.
# Run this command from the directory where you ran gcluster deploy, and
# confirm when prompted.
gcluster destroy ${CLUSTER_NAME}

```

For more information, see [`gcluster job cancel`](https://docs.cloud.google.com/cluster-toolkit/docs/reference/gcluster-job-cancel) and [`gcluster destroy`](https://docs.cloud.google.com/cluster-toolkit/docs/reference/gcluster-destroy).

## Multi-slice model training on GKE

A multi-slice job spans several TPU slices. Each slice is its own ICI fabric and MegaScale joins
them over DCN, which is much slower than ICI, so the layout that pays off is HSDP: shard inside a
slice, replicate between slices, and only the replicated gradient crosses DCN. TorchTitan arranges
that for you.

Everything below is the single-slice flow from the previous section with `num_slices` raised and
two environment variables added.

### Create a multi-slice cluster

The same blueprint provisions several slices -- pass the slice count instead of `num_slices=1`:

```bash
gcluster deploy examples/gke-tpu-v6e/gke-tpu-v6e.yaml \
    --vars project_id=${PROJECT_ID},deployment_name=${CLUSTER_NAME} \
    --vars region=${REGION},zone=${ZONE} \
    --vars machine_type=${COMPUTE_TYPE},tpu_topology=${TOPOLOGY},num_slices=2 \
    --vars authorized_cidr=${AUTHORIZED_CIDR},enable_gcsfuse_csi=true

```

This creates one node pool per slice. The blueprint installs Kueue and JobSet, which matters here:
a multi-slice workload must be admitted all-or-nothing, or one slice can take the last free chips
and sit holding them while the other never starts.

### Deploy the training workload

One `gcluster job submit`, same shape as the single-slice one, plus `--num-slices` and the two
MegaScale variables that are identical on every pod:

```bash
gcluster job submit \
    --name $WORKLOAD_NAME \
    --cluster $CLUSTER_NAME \
    --location $REGION \
    --project $PROJECT_ID \
    --compute-type $COMPUTE_TYPE \
    --topology $TOPOLOGY \
    --num-slices 2 \
    --image "us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/torchtitan-container:latest" \
    --service-account workload-identity-k8s-sa \
    --mount "$BUCKET_NAME;/data;rw" \
    --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME" \
    --env MEGASCALE_NUM_SLICES=2 \
    --env MEGASCALE_COORDINATOR_ADDRESS="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME:8080" \
    --env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --megascale_grpc_premap_memory_bytes=4294967296 --megascale_use_numa_aware_block_allocator=false" \
    --command '
        VMS=$(echo "$TPU_WORKER_HOSTNAMES" | awk -F, "{print NF}");
        POD_DNS=${TPU_WORKER_HOSTNAMES%%,*};
        POD=${POD_DNS%%.*};
        JOB=${POD%-*};
        SLICE=${JOB##*-};
        torchrun \
        --nnodes=$(( MEGASCALE_NUM_SLICES * VMS )) \
        --nproc_per_node=8 \
        --rdzv_backend=static \
        --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
        --node_rank=$(( SLICE * VMS + TPU_WORKER_ID )) \
        -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.muse_glimmer \
        --config=muse_glimmer_30b_full_logits \
        --parallelism.data_parallel_replicate_degree=$MEGASCALE_NUM_SLICES \
        --parallelism.enable_min_cost_redistribute \
        --training.steps=30'
```

Differences from the single-slice command:

* Multi-slice needs three environment variables. libtpu joins the slices into one device mesh
  through its MegaScale PJRT client, and builds that client from:

  ```
  MEGASCALE_NUM_SLICES           how many slices there are
  MEGASCALE_COORDINATOR_ADDRESS  where slice 0's first worker listens
  MEGASCALE_SLICE_ID             which slice this pod is, set automatically by TorchTitan
  ```

  The command passes the first two with `--env`. The coordinator address is the one that makes
  libtpu create the client at all; without it every slice numbers its devices from zero and the
  mesh cannot be built.

  The third is different, because its value changes from slice to slice -- one `--env` cannot say
  two things. TorchTitan fills it in: it reads the slice index out of `TPU_WORKER_HOSTNAMES` and
  exports `MEGASCALE_SLICE_ID` before the first device is used, logging
  `Exporting MEGASCALE_SLICE_ID=...`. You can still set it yourself, and it is cross-checked
  against that host list, so a wrong value raises instead of building a broken mesh.

* `--parallelism.data_parallel_replicate_degree=$MEGASCALE_NUM_SLICES` is **required**. The
  replicate axis is the one split across DCN, which keeps every FSDP shard group, and so every
  weight all-gather and gradient reduce-scatter, inside a single slice. With the default of 1 the
  only axis left for the split is `dp_shard`, so shard groups would straddle the slices: the job
  trains correctly at a fraction of the throughput, and torch_tpu notes it only at INFO. TorchTitan
  refuses to start in that case rather than correcting it for you, since a job that is merely slow
  is far harder to notice than one that will not launch. A degree the slice count does not divide
  is rejected by torch_tpu when it builds the mesh; any multiple of the slice count works. Writing it
  as the variable rather than a literal cannot drift when you change `--num-slices`.

* `--parallelism.enable_min_cost_redistribute` is worth setting on multi-slice. The default greedy
  DTensor planner all-reduces the **full** gradient on the replicate axis -- the axis that crosses
  DCN -- before the reduce-scatter; the min-cost planner reduce-scatters first, shrinking the DCN
  payload by the shard degree.

* On v7x, shrink the host-memory pool MegaScale reserves for DCN transfers. Multi-slice pins a pool
  of 16 GiB per NUMA node per process by default. JAX runs one process per VM, but torch_tpu runs
  one per TensorCore -- 8 per VM on v7x, which is why the command above uses
  `--nproc_per_node=8` -- so the pool is paid 8 times: 16 GiB x 2 NUMA nodes x 8 processes = 256 GiB
  per VM. That matters when anything else needs host memory, such as activation offloading. The
  two flags already in the `LIBTPU_INIT_ARGS` above shrink the pool to 4 GiB per process, 32 GiB per
  VM, and both are needed:

  ```bash
  --megascale_grpc_premap_memory_bytes=4294967296    # 4 GiB per process
  --megascale_use_numa_aware_block_allocator=false   # one pool per process, not one per NUMA node
  ```
