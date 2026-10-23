# TPU Non-Colocated RL with Ray

This directory contains the Ray-based orchestrator and TPU-specific components to run torchtitan Reinforcement Learning (GRPO) in a non-colocated (decoupled) architecture.

In a non-colocated setup, the Policy Trainer (PyTorch FSDP) and the Rollout Generator (vLLM) run on entirely separate slices of TPU hardware. This avoids out-of-memory errors by ensuring the training weights/gradients and the KV cache do not compete for the same HBM.

The orchestrator (`grpo.py`) runs on the Ray head node, launching remote `RayTPUPolicyTrainer` and `RayVLLMGenerator` actors on their respective TPU slices, and coordinating the state dictionary transfer via Ray's object store.

## Scripts

### 1. `smoke_test_vm_tp1.sh`
- Runs a local, single-slice smoke test utilizing CPU for the Trainer and TPU for the Generator.
- Useful for fast local debugging of the Generator/vLLM loop without needing a full multi-slice Ray cluster.

### 2. `smoke_test_vm_tp4.sh`
- Similar to `tp1`, but configures the Generator to use Tensor Parallelism degree 4.
- Validates the `torchtpu-vllm` multi-process worker initialization and TP overlapping configurations locally.

### 3. `e2e_recipe_gke.sh`
- The main production script for running an end-to-end training job across a full multi-host GKE Ray cluster.
- Supports training multiple model sizes: Qwen3-0.6B (default), Qwen3-1.7B, and Qwen3-4B.
- Configures 8-way Tensor Parallelism for the vLLM Generator and 2D sharding (FSDP x DP) for the Policy Trainer.
- Accepts a model size argument and forwards extra arguments directly to `grpo.py`.


---

### GKE KubeRay Deployment Walkthrough

The following instructions walk through building the custom Docker image, configuring GCS storage connectivity, establishing the Ray cluster, and executing the decoupled training job.

> [!NOTE]
> These steps target a `v6e-8` GKE cluster. You may need to adapt parameters (e.g., node counts or TPU accelerator limits) depending on your actual cluster topology.

#### A. Build and Push the Ray/vLLM Docker Image (Optional)
If you need to rebuild the Docker container containing the correct Torch-TPU, Ray, and vLLM dependencies, utilize the provided Dockerfile:

```bash
# Build the Docker image
docker build -t <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/torchtitan-ray:latest -f torchtitan/experiments/tpu/rl_nc_ray/scripts/Dockerfile.ray.vllm .

# Push to your Google Artifact Registry
docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/torchtitan-ray:latest
```

> [!TIP]
> Ensure your KubeRay cluster configuration YAML file (`torchtitan/experiments/tpu/rl_nc_ray/scripts/ray-v6e8-2.yaml`) points to this built image (replace `YOUR_IMAGE`):
> ```yaml
> image: <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/torchtitan-ray:latest
> ```

#### B. Configure GCS Bucket Access (Workload Identity)
The Ray configuration (`ray-v6e8-2.yaml`) automatically mounts your GCS bucket (`bucketName: YOUR_BUCKET`) to `/data` in all pods using the GKE GCS Fuse CSI driver.

To authorize secure bucket access from your GKE nodes:
1. Ensure the **GCS Fuse CSI driver** is enabled on your GKE cluster.
2. Bind the Kubernetes Service Account (`ray-ksa` in the `default` namespace) to your Google Service Account (GSA) containing Storage Object Admin permissions via Workload Identity:

```bash
gcloud iam service-accounts add-iam-policy-binding <GSA_NAME>@<PROJECT_ID>.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:<PROJECT_ID>.svc.id.goog[default/ray-ksa]"
```

> [!IMPORTANT]
> Set `YOUR_GCP_SERVICE_ACCOUNT` in the Workload Identity annotation on the `ray-ksa` ServiceAccount in `ray-v6e8-2.yaml`:
> ```yaml
> metadata:
>   name: ray-ksa
>   annotations:
>     iam.gke.io/gcp-service-account: YOUR_GCP_SERVICE_ACCOUNT
> ```

#### C. Provision the KubeRay Cluster
Connect to your GKE cluster, enable the Ray Operator addon, and apply the cluster resource manifest:

```bash
# Authenticate with the GKE cluster
export CLUSTER_NAME="my-gke-cluster"
export REGION="my-cluster-region"
export PROJECT="my-project"

gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT --dns-endpoint

# Enable the GKE Ray Operator addon
gcloud container clusters update $CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT \
    --update-addons=RayOperator=ENABLED 

# Launch the decoupled KubeRay cluster resources
kubectl apply -f torchtitan/experiments/tpu/rl_nc_ray/scripts/ray-v6e8-2.yaml
```

#### D. Establish Dashboard & Client Port-Forwarding
In a separate terminal, forward the Ray dashboard/client ports to allow local job submissions to reach the head node service:

```bash
# Verify all cluster pods are successfully running
kubectl get pods

# Port-forward the dashboard (8265) and Ray client ports (10001)
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 8265:8265 10001:10001
```

#### E. Submit the Decoupled Training Job
With the port-forward connection active, launch the training job from your local workspace by specifying the model size (`0.6b`, `1.7b`, or `4b`):

```bash
# Run Qwen3-0.6B (default)
bash torchtitan/experiments/tpu/rl_nc_ray/scripts/e2e_recipe_gke.sh 0.6b

# Run Qwen3-1.7B
bash torchtitan/experiments/tpu/rl_nc_ray/scripts/e2e_recipe_gke.sh 1.7b

# Run Qwen3-4B
bash torchtitan/experiments/tpu/rl_nc_ray/scripts/e2e_recipe_gke.sh 4b
```

#### F. Monitor Job Status & Training Logs
You can easily list the active jobs running on your cluster to fetch your `raysubmit_xxxxxxxx` ID:

```bash
ray list jobs --address="http://127.0.0.1:8265" --filter "status=RUNNING"
```

To tail the real-time logging output or pipe logs to a local file, execute:

```bash
ray job logs <raysubmit_id> --address="http://127.0.0.1:8265" > ray.log
```

---

### 4. `plot_metrics_from_log.py`
- A utility script to visualize RL training metrics (Reward, Loss, Gen Time, Train Time) from the Ray job logs.
- **Usage:**
  Save the logs from a completed or running Ray job, and generate a plot:
  ```bash
  ray job logs <raysubmit_id> --address="http://127.0.0.1:8265" > training.log
  python3 torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log.py --log_file training.log --output rl_metrics.png
  ```
