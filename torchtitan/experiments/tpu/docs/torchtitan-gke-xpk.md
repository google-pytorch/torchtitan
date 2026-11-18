# Multi-Node Training Guide for Llama on A3 Mega XPK-GKE Cluster with TorchTitan

This document describes the complete workflow for provisioning a A3 Mega XPK-GKE
cluster on GCP and launching multi-node training for llama 3.1 70B using the
PyTorch TorchTitan framework.

1. This document is structured into two main parts, cluster setup and training setup.
---

## [Cluster Setup](https://cloud.google.com/cluster-toolkit/docs/setup/configure-environment)

1.  **Setup Environment Variables**

    Export the following environment variables to configure the cluster and training parameters:

    ```bash
    export PROJECT_ID=<project_id>
    export ZONE=<zone>
    export CLUSTER_NAME=<cluster_name>
    export NUM_NODES=<num_nodes>
    export RESERVATION_NAME=<reservation_name>
    export BUCKET_NAME=<bucket_name>
    export WORKLOAD_NAME=<workload_name>
    ```
2.  **Authenticate and Set Project**

    First, authorize the gcloud CLI and configure it to use your target project.

    a. Log in to your Google Account:
    ```bash
    gcloud auth login
    ```

    b. Set the default project for all subsequent commands:

    ```bash
    gcloud config set project $PROJECT_ID
    ```

3.  **Install xpk**

    As of now, XPK must be built from source using the following commands (pip and apt installation methods are not yet available):

    ```bash
    # Clone the XPK repository
    git clone https://github.com/google/xpk.git
    cd xpk

    # Install required dependencies and build XPK with make
    make install && export PATH=$PATH:$PWD/bin
    ```
    a. Due to a bug in xpk which adds an additional placement policy for the reservation and causes errors, we need to comment out the lines shown below in `src/xpk/core/blueprint/blueprint_generator.py` as a workaround:

    ```diff
    --- a/src/xpk/core/blueprint/blueprint_generator.py
    +++ b/src/xpk/core/blueprint/blueprint_generator.py
    @@ -283,9 +283,9 @@ class BlueprintGenerator:
                workload_configmap,
            ],
        )
    -    if set_placement_policy and reservation_placement_policy is None:
    -      a3_megagpu_pool_0.use.append(group_placement_0.id)
    -      primary_group.modules.append(group_placement_0)
    +    # if set_placement_policy and reservation_placement_policy is None:
    +    #   a3_megagpu_pool_0.use.append(group_placement_0.id)
    +    #   primary_group.modules.append(group_placement_0)
    ```

4.  **Create GKE cluster with xpk**

    ```bash
    python3 xpk.py cluster create \
      --cluster $CLUSTER_NAME \
      --zone $ZONE \
      --project $PROJECT_ID \
      --device-type h100-mega-80gb-8 \
      --num-nodes=$NUM_NODES \
      --reservation=$RESERVATION_NAME
    ```

5.  **Create NCCL test workload with xpk**

    a. Authorize docker registry

    ```bash
    gcloud auth configure-docker gcr.io,us.gcr.io,us-docker.pkg.dev,us-east5-docker.pkg.dev
    ```

    b. Create NCCL test workload

    ```bash
    python3 xpk.py workload create \
      --workload=nccl-test \
      --command="./examples/nccl/nccl-a3mega.sh" \
      --base-docker-image=us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev:v1.0.8-1 \
      --cluster=$CLUSTER_NAME \
      --device-type=h100-mega-80gb-8 \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --num-nodes=$NUM_NODES
    ```

    c. You’ll see log link in the terminal, click the link then you’ll navigate to GCP log explorer

### 6. Build and Push Training Image

This step creates the software environment required for TorchTitan. You must build the Docker image locally and push it to the Google Artifact Registry (GAR).

**a. Create Artifact Registry Repository**

If you haven't already, create a repository in your project to store the training images:

```bash
gcloud artifacts repositories create <repo_name> \
    --repository-format=docker \
    --location=us-east5 \
    --description="Docker repository for TorchTitan training"
```
**b. Create the Dockerfile**

In your working directory, create a file named Dockerfile:
```dockerfile
# Use the official CUDA 12.8 devel image as the base
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Set non-interactive for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Nightly and TorchTitan
WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python3 -m pip install --no-cache-dir --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python3 -m pip install --no-cache-dir einops torchao

# Copy entrypoint scripts (ensure these exist in your local dir)
# Note: These scripts are usually part of the base NVIDIA image,
# so COPYing them might not be necessary if not modifying them.
# COPY nvidia_entrypoint.sh /opt/nvidia/
# COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
# RUN chmod +x /opt/nvidia/nvidia_entrypoint.sh
# Ensure the TCPXo plugin is found by NCCL at runtime
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH 

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
```
**c. Build and Push the Image**

Define your image tag using the repository created above, then build and push:
```bash
export IMAGE_TAG="us-east5-docker.pkg.dev/$PROJECT_ID/<repo_name>/<image_name>"

# Build the image
docker build -t $IMAGE_TAG .

# Authenticate with GAR
gcloud auth configure-docker us-east5-docker.pkg.dev

# Push to Google Cloud
docker push $IMAGE_TAG
```

---

## Training Setup

1. **FUSE adapter**

   To use the GCS FUSE with XPK you need to create a [Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets).

    Set the mount point environment variable:

    ```bash
    export MOUNT_POINT="/data"
    ```

    Run the following command to attach the storage. This ensures consistency, as the storage name should follow the format `${WORKLOAD_NAME}-storage`.

    ```bash
    xpk storage attach ${WORKLOAD_NAME}-storage \
    --type=gcsfuse \
    --project=$PROJECT_ID \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --mount-point=$MOUNT_POINT \
    --readonly=false \
    --bucket=$BUCKET_NAME \
    --size=1 \
    --auto-mount=false
    ```
2. **Prepare training workload script**

   Create the storage bucket and upload the Llama tokenizer once so we don't waste GPU time downloading it later.

    ```bash
    ## Clone repo to get the download script
    git clone https://github.com/pytorch/torchtitan
    cd torchtitan
    pip3 install -r requirements.txt

    ## Download tokenizer (requires HF Token)
    python3 scripts/download_hf_assets.py \
      --repo_id meta-llama/Llama-3.1-70B \
      --assets tokenizer \
      --hf_token=<YOUR_HF_TOKEN> \
      --save_dir ./my_assets

    ## Upload to your Bucket
    gcloud storage cp -r ./my_assets/Llama-3.1-70B/* gs://$BUCKET_NAME/assets/
    gcloud storage cp ./my_assets/Llama-3.1-70B/original/tokenizer.model gs://$BUCKET_NAME
    ```
    #### Download the base config:
    ```bash
    # Copy to xpk folder
    cp torchtitan/models/llama3/train_configs/llama3_70b.toml ../llama3_70b_fp8.toml

    # Navigate back to xpk folder
    cd ..
    ```

    #### Upload the configuration to your GCS bucket:
    ```bash
    gcloud storage cp llama3_70b_fp8.toml gs://$BUCKET_NAME/
    ```
    c. Create `training_workload.sh` with execute permission. (Remember to replace `<HF_TOKEN>` with your huggingface token in the script)
    This script will be run as command in the container.
        We override parameters via command-line flags to configure FP8 training with FSDP, and reduce training steps for a quick run:
    - `training.steps=100`: reduce training steps to 100 for quick test run.
    - `lr_scheduler.warmup_steps=20`: reduce warmup steps to 20 accordingly.
    - `parallelism.data_parallel_shard_degree=4`: enable FSDP across 4 nodes.
    - `compile.enable=true`: enable torch compile for better performance.
    - `quantize.linear.float8.enable_fsdp_float8_all_gather=true`: enable FP8 all-gather for FSDP.
    - `quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp=true`: precompute FP8 dynamic scale for FSDP.
    - `checkpoint.enable=true`: enable model checkpointing.
    - `checkpoint.folder="$GCS_MOUNT/checkpoints"`: set checkpoint directory in GCS bucket.
    ```bash
    #!/bin/bash
    set -ex

    # 1. Setup NCCL Environment
    NCCL_LIB_DIR="/usr/local/nvidia/lib64"
    source ${NCCL_LIB_DIR}/nccl-env-profile.sh

    # 2. Master Node Discovery (PYTHON VERSION)
    sleep 5
    echo "NODE_RANK: $NODE_RANK"
    MASTER_ADDR_DNS_NAME="${JOBSET_NAME}-${REPLICATED_JOB_NAME}-0-0.${JOBSET_NAME}"
    export MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('$MASTER_ADDR_DNS_NAME'))")
    export MASTER_PORT=6002
    echo "MASTER_ADDR: $MASTER_ADDR"

    # 3. Enter the Repo
    cd /workspace/torchtitan

    # 4. Link Assets
    echo "Linking pre-downloaded assets..."
    mkdir -p /workspace/torchtitan/assets/hf/Llama-3.1-70B
    ln -sf /data/assets/tokenizer.model /workspace/torchtitan/assets/hf/Llama-3.1-70B/tokenizer.model
    ln -sf /data/assets/tokenizer.json /workspace/torchtitan/assets/hf/Llama-3.1-70B/tokenizer.json
    ln -sf /data/assets/tokenizer_config.json /workspace/torchtitan/assets/hf/Llama-3.1-70B/tokenizer_config.json
    ln -sf /data/assets/special_tokens_map.json /workspace/torchtitan/assets/hf/Llama-3.1-70B/special_tokens_map.json
    # 5. Configure Networking
    echo "Setting NCCL environment variables for TCPX..."
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_FASTRAK_CTRL_DEV=eth0
    export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # 6. Launch Training
    CONFIG_FILE="/data/llama3_70b_fp8.toml"
    GCS_MOUNT="/data"
    export HF_TOKEN=<HF_TOKEN>

    echo "Starting training..."
    torchrun \
      --nnodes $NNODES \
      --nproc_per_node 8 \
      --rdzv_id 101 \
      --rdzv_backend c10d \
      --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
      -m torchtitan.train \
      --job.config_file "$CONFIG_FILE" \
      --job.dump_folder "$GCS_MOUNT/outputs/dump" \
      --profiling.save_traces_folder "$GCS_MOUNT/outputs/traces" \
      --metrics.save_tb_folder "$GCS_MOUNT/outputs/metrics" \
      --model.converters '["quantize.linear.float8"]' \
      --training.steps 100 \
      --lr_scheduler.warmup_steps 20 \
      --parallelism.data_parallel_shard_degree 4 \
      --compile.enable true \
      --quantize.linear.float8.enable_fsdp_float8_all_gather true \
      --quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp true \
      --checkpoint.folder "$GCS_MOUNT/checkpoints" \
      --checkpoint.enable

    ```

3.  **Submit training workload through xpk**

    ```bash
    python3 xpk.py workload create \
      --workload=$WORKLOAD_NAME \
      --command="./training_workload.sh" \
      --base-docker-image=$IMAGE_TAG \
      --cluster=$CLUSTER_NAME \
      --device-type=h100-mega-80gb-8 \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --num-nodes=$NUM_NODES \
      --storage=${WORKLOAD_NAME}-storage
    ```

4.  **Delete training workload (after checked logs & copied training outputs)**

    ```bash
    python3 xpk.py workload delete \
      --workload $WORKLOAD_NAME \
      --cluster $CLUSTER_NAME \
      --zone $ZONE
    ```

5.  **See the training log**

    ```bash
    kubectl get pods
    kubectl logs <pod_name>
    ```

6.  **Serving TensorBoard on localhost**

    a. Get training result on tensorboard

    After training finishes, copy the TensorBoard metrics from your GCS bucket to a local directory:

    ```bash
    # 1. Clear local logs and create a tb_logs folder
    rm -rf tb_logs && mkdir tb_logs

    # 2. Copy metrics from gcs bucket to local
    gcloud storage cp -r gs://$BUCKET_NAME/outputs/metrics tb_logs/
    ```
    b. Serving tensorboard

    ```bash
    # [Optional]
    pip install tensorboard

    tensorboard --logdir tb_logs --port 6006
    ```

    c. Access tensorboard in local environment

    ```bash
    http://localhost:6006/
    ```
---