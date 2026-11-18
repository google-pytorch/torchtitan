# (Slurm) Multi node training with torchtitan

**Multi-Node Training Guide - Llama 3.1 (70B) model on A3 Mega Slurm Cluster with TorchTitan**

This document describes the complete workflow for provisioning a A3 Mega Slurm cluster on GCP
and launching multi-node training for llama 3.1 70B using the PyTorch TorchTitan framework.

This document is structured into two main parts: Cluster Setup and Training Setup.

**Note:**

The relevant information and documentation for this deployment are as follows:

* A3 Mega Slurm cluster deployment: https://cloud.google.com/cluster-toolkit/docs/deploy/deploy-a3-mega-cluster
* TorchTitan’s Repo: https://github.com/pytorch/torchtitan/tree/main

---

## [Cluster Setup](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/deploy-a3-mega-cluster)

The following setup commands should be executed on the user's local machine, unless otherwise specified.
1.  **Authenticate and Set Project**

    First, authorize the gcloud CLI and configure it to use your target project.

    a. Log in to your Google account:

    ```bash
    gcloud auth login
    ```

    b. Set the default project for all subsequent commands:

    ```bash
    gcloud config set project <project_id>
    ```

2.  **Install cluster toolkit**

    ```bash
    git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
    cd cluster-toolkit
    make

    # Validation
    ./gcluster --version
    ```

3.  **Create a cloud storage bucket**

    ```bash
    gcloud storage buckets create gs://BUCKET_NAME \
        --default-storage-class=STANDARD \
        --project=<PROJECT> \
        --location=<LOCATION> \
        --uniform-bucket-level-access 

    gcloud storage buckets update gs://BUCKET_NAME --versioning
    ```

4.  **Modify the a3mega-slurm.blueprint.yaml**
    
    The following modifications to `a3mega-slurm.blueprint.yaml` are needed to
    mount a GCS bucket onto all nodes (including login and compute nodes) in the
    Slurm cluster using GCS Fuse. This allows for seamless access to training
    data, checkpoints, and other large files stored in GCS, treating the GCS
    bucket like a shared network file system. This is highly recommended for
    large-scale training to simplify data management and avoid limitations of
    local node storage.
    
    To apply these changes automatically, save the following script as
    `apply_gcsfuse_patch.sh` in the `cluster-toolkit` directory:
    
    ```bash
    #!/bin/bash
    # Run this from the 'cluster-toolkit' root directory

    # Set your bucket name here or pass it as an argument
    BUCKET_NAME="${1:-<YOUR_BUCKET_NAME>}"
    TARGET_FILE="examples/machine-learning/a3-megagpu-8g/a3mega-slurm-blueprint.yaml"

    if [ ! -f "$TARGET_FILE" ]; then
        echo "Error: Target file not found at $TARGET_FILE"
        exit 1
    fi

    patch -p1 <<EOF
    --- a/$TARGET_FILE
    +++ b/$TARGET_FILE
    @@ -14,3 +14,5 @@
    validators:
    -  - validator: test_deployment_variable_not_used
    + - validator: test_deployment_variable_not_used
        inputs: {}
        skip: true
    + - validator: test_module_not_used
    +   skip: true
    @@ -226,7 +228,8 @@
        - id: data-bucket
    -      source: community/modules/file-system/pre-existing-network-storage
    +      source: modules/file-system/pre-existing-network-storage
        settings:
    +        remote_mount: $BUCKET_NAME
    +        fs_type: gcsfuse
            local_mount: /gcs
            mount_options: defaults,rw,_netdev,implicit_dirs,allow_other,implicit_dirs,file_mode=777,dir_mode=777
    -        random_suffix: true
    @@ -446,2 +449,3 @@
        - gpunets
    +      - data-bucket
        settings:
    @@ -542,2 +546,3 @@
        - sysnet
    +      - data-bucket
        settings:
    EOF

    echo "Successfully patched $TARGET_FILE with bucket: $BUCKET_NAME"
    ```

    Then, make the script executable and run it from `cluster-toolkit`:
    
    ```bash
    chmod +x apply_gcsfuse_patch.sh
    # To patch with placeholder <YOUR_BUCKET_NAME>
    ./apply_gcsfuse_patch.sh
    # Or patch with your bucket name
    ./apply_gcsfuse_patch.sh YOUR_BUCKET_NAME
    ```

    **IMPORTANT**: If you ran `apply_gcsfuse_patch.sh` without providing a bucket name, you must edit `examples/machine-learning/a3-megagpu-8g/a3mega-slurm-blueprint.yaml` and replace `<YOUR_BUCKET_NAME>` with the name of your GCS bucket.

    <details>
    <summary>Click here to see manual edits</summary>

    The patch above makes the following changes:

    ```diff
    --- a3mega-slurm-blueprint.yaml.orig
    +++ a3mega-slurm-blueprint.yaml.new

    @@ -14,6 +14,8 @@
    validators:
    -  validator: test_deployment_variable_not_used
       inputs: {}
       skip: true
    +- validator: test_module_not_used
    +  skip: true

    vars:
       sys_net_range: 172.16.0.0/16

    @@ -226,9 +226,9 @@
        - id: data-bucket
    -     source: community/modules/file-system/pre-existing-network-storage
    +     source: modules/file-system/pre-existing-network-storage
          settings:
    +       remote_mount: <YOUR_BUCKET_NAME>    # REQURIED: Put your actual bucket name here
    +       fs_type: gcsfuse                 # REQUIRED: Explicitly set file system type
            local_mount: /gcs
            mount_options: defaults,rw,_netdev,implicit_dirs,allow_other,implicit_dirs,file_mode=777,dir_mode=777
    -       random_suffix: true              # DELETE THIS: Cannot be used with existing buckets
    
        - id: gpunets
          source: modules/network/multivpc
    @@ -446,6 +446,7 @@
        use:
        - sysnet
        - gpunets
    +   - data-bucket                      # ADD THIS: Connects storage to Compute Nodes
        settings:
            node_count_static: $(vars.a3mega_cluster_size)
            node_count_dynamic_max: 0
    @@ -542,6 +543,7 @@
        source: community/modules/scheduler/schedmd-slurm-gcp-v6-login
        use:
        - sysnet
    +   - data-bucket                      # ADD THIS: Connects storage to Login Node
        settings:
            enable_login_public_ips: $(vars.enable_login_public_ips)
            name_prefix: login
     ```
    </details>
    
5. **Provision a Slurm cluster**

    ```bash
    ./gcluster deploy -d \
      examples/machine-learning/a3-megagpu-8g/a3mega-slurm-deployment.yaml \
      examples/machine-learning/a3-megagpu-8g/a3mega-slurm-blueprint.yaml \
      --backend-config "bucket=BUCKET_NAME" \
      --vars "deployment_name=<DEPLOYMENT_NAME>" \
      --vars "project_id=<PROJECT>" \
      --vars "region=<REGION>" \
      --vars "zone=<ZONE>" \
      --vars "network_name_system=<NETWORK_NAME>" \
      --vars "subnetwork_name_system=<SUBNET_NAME>" \
      --vars "enable_ops_agent=true" \
      --vars "enable_nvidia_dcgm=true" \
      --vars "enable_nvidia_persistenced=true" \
      --vars "disk_size_gb=200" \
      --vars "final_image_family=slurm-a3mega" \
      --vars "slurm_cluster_name=DEPLOYMENT_NAME" \
      --vars "a3mega_reservation_name=<RESERVATION_NAME>" \
      --vars "a3mega_cluster_size=<NUM_NODES>" \
      --auto-approve -w
    ```

    Note: Variable 'slurm_cluster_name' must be a match of regex '^[a-z](?:[a-z0-9]{0,9})$'.

    If the creation somehow failed, please remember to clean up the resources that you have built from last deployment, or it may affect your next deployment.

    ```bash
    ./gcluster destroy --auto-approve <deployment_name>
    ```

    After destroying the slurm cluster, please ensure the following resources have been deleted: VM instance, VPC network, VPC firewall rules, Routes, Filestore instance.

    If the creation got this error: Error waiting for Create Service Networking Connection: ``` Error code 9, message: Cannot modify allocated ranges in CreateConnection.``` ...... run this command to clear VPC peerings:

    ```bash
    gcloud services vpc-peerings delete --project <project_id> --network <network_name> --service=servicenetworking.googleapis.com
    ```

6.  **SSH to Slurm login node**

    **Option 1: IAP Tunneling** Create a firewall rule to allow-ssh-iap for your vm network. 

    Create a firewall rule to allow-ssh-iap for your vm network.
    This is a workaround if you are unable to ssh to your vm instance.

    ```bash
    gcloud compute firewall-rules create "allow-ssh-iap" \
      --network=<network_name> \
      --allow=tcp:22 \
      --source-ranges="35.235.240.0/20" \
      --priority=1000 \
      --direction=INGRESS
    ```

    SSH to Slurm login node with port 6006 forwarding

    ```bash
    gcloud compute ssh <DEPLOYMENT_NAME>-login-001 --tunnel-through-iap --zone <zone> -- -L 6006:localhost:6006
    ```

    **Option 2: Direct SSH**

    ```bash
    ssh -i ~/.ssh/google_compute_engine <username>@nic0.<hostname>-login-001.<zone>.c.<project_id>.internal.gcpnode.com
    ```

7.  **[Login node] Run NCCL test for a3mega Slurm cluster ([**Doc link**](https://cloud.google.com/cluster-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo))**

---

## Training Setup


The following setup commands should be executed on the user's Login Node, unless otherwise specified.
1.  **SSH to Slurm login node**

    You should already have an active SSH session to the Slurm login node to perform the NCCL test. If you haven’t done so, please connect now.

2.  **Clone the torchtitan repo**

    ```bash
    git clone https://github.com/pytorch/torchtitan
    cd torchtitan
    ```

3.  **Install torchtitan from source code (recommended to use cuda12.8)**

    ```bash
    # Optional
    python3 -m venv venv
    source venv/bin/activate
    ```

    ```bash
    pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
    pip3 install -r requirements.txt

    # Optional, if you would like to use nightly builds
    pip3 install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu128
    ```

4.  **[Login Node] Make sure einops package is installed**

    ```bash
    # Check the einops package is installed
    pip show einops
    # If not, install it
    pip install einops
    ```

5.  **[Login Node] Download tokenizer (should sign in & apply for token through [**huggingface**](https://huggingface.co/meta-llama/Llama-3.1-70B))**

    ```bash
    python3 scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-70B --assets tokenizer --hf_token=<hugging_face_token>
    ```

6.  **[Login Node] Copy the `multinode_trainer.slurm` script to `run-slurm.sh` and make the following modifications:**

    `run-slurm.sh`

    ```bash
    #!/bin/bash
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree.
    # --- This script is optimized for AWS with EFA
    # --- adjust NCCL_BUFFSIZE if you encounter memory
    # --- constraint issues or to tune for improved performance.
    # ---
    
    #SBATCH --job-name=torchtitan_multi_node
    
    #SBATCH --ntasks=4
    
    #SBATCH --nodes=<num_nodes>
    
    #SBATCH --gpus-per-task=8

    #SBATCH --cpus-per-task=96

    # Because the partition name is defined as `a3mega` within the `a3mega-slurm-blueprint.yaml` file of the cluster-toolkit, you must use `a3mega` here
    #SBATCH --partition=a3mega

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo Node IP: $head_node_ip
    export LOGLEVEL=INFO
    
    # Ensure that P2P is available
    # export NCCL_P2P_DISABLE=1
    # export NCCL_IB_DISABLE=1

    # debugging flags (optional)
    export NCCL_DEBUG=WARN
    export PYTHONFAULTHANDLER=1
    # optional debug settings
    # export NCCL_DEBUG=INFO
    # NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    export CUDA_LAUNCH_BLOCKING=0

    # NCCL variables
    NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
    export NCCL_BUFFSIZE=2097152
    export NCCL_FASTRAK_CTRL_DEV=enp0s12
    export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
    export NCCL_SOCKET_IFNAME=enp0s12
    export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

    CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_70b.toml"}
    
    dcgmi profile --pause
    # adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
    # to your specific node count, and update target launch file.
    srun torchrun --nnodes <num_nodes> --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
    dcgmi profile --resume
    ```

7.  **[Login Node] Submit multi-node training**

    ```bash
    sbatch run-slurm.sh
    ```

    The results in a `slurm-XX.out` file that contains the training logs. You can use the `squeue` command to check if the training is complete.

8.  **[Login Node] Serving TensorBoard on localhost**

    ```bash
    tensorboard --logdir ./outputs/tb
    ```

    Note: If you didn’t ssh to the login node with port forwarding, you can’t access TensorBoard directly. You must first copy the data from the outputs directory to your local machine to perform the actions mentioned above.

    ```bash
    # [Login node] Create a gz archive
    cd ~/torchtitan
    tar czf outputs.tar.gz --directory=outputs .

    # [Local] Copy the archive to your local machine
    scp -i ~/.ssh/google_compute_engine <username>@nic0.<hostname>-login-001.<zone>.c.<project_id>.internal.gcpnode.com:~/torchtitan/outputs.tar.gz ~/Downloads/outputs.tar.gz

    # [Local] Extract the archive
    cd ~/Downloads
    tar xf outputs.tar.gz
    ```

9.  **[Local] Access tensorboard on Chrome by your env.**

    ```
    http://localhost:6006/
    ```

    You should see a tensorboard webpage.

---
