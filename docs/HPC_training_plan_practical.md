# HPC Training Plan for Starting on Capella

This document is for getting a first training workflow running on **Capella** at TU Dresden.

Content:

- Logging in
- Creating a workspace
- Cloning the repo
- Uploading the dataset
- Uploading an initialization checkpoint, if training starts from an existing model
- Creating a Python environment
- Submitting the job
- Monitoring jobs
- Downloading the ready checkpoint


#### Replace the placeholders below before running commands:

- `roge097b` = your ZIH username
- `<LOCAL_CODE_DIR>` = local path of your code repository
- `<LOCAL_DATA_DIR>` = local path of your dataset(s)
- `<REMOTE_WS_NAME>` = workspace name, e.g. `fastumi`
- `<MAIL>` = your TU Dresden mail address
- `<DATASET_NAME>` = dataset file or folder name
- `<TRAIN_SCRIPT>` = your training entry point or wrapper script



## 1. Logging in

### Connecting via SSH

Connect to the Capella login node (from the campus network or via VPN) with agent forwarding, so GitHub authentication works without storing keys on the cluster (though you can store them there if you prefer).

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/wherever_you_store_ssh_keys
ssh -A roge097b@login1.capella.hpc.tu-dresden.de
```

### Running some checks

Check you are in:

```bash
hostname
whoami
ws_list -l
ssh -T git@github.com
```

Note that login nodes (the ones you connect to via `ssh -A ...`) are only for setting up your environment, moving datasets, submitting jobs, and checking logs. The way it works is that from the login node you submit a job using `sbatch`, which schedules and runs your workload on a GPU node that reads data from your login node's filesystem.

Find your default and available accounts (used for billing HPC hours - though the system is a bit opaque, as there doesn't seem to be a straightforward way to check your remaining hours):

```bash
sacctmgr --parsable2 --noheader show user where name=$USER format=defaultaccount
sacctmgr --noheader --parsable2 show assoc user=$USER format=account
```


## 2. Creating a Workspace

All workspaces have a limited lifespan, after which the data is permanently deleted. There is a clever system of creating a short-lived workspace (~30 days) and syncing it with a long-lived one (>30 days) - but if you remember to download your trained models and keep a reproducible environment setup, you can simply recreate the workspace every 30 days and avoid overcomplicating things.

### Allocate a workspace

```bash
ws_allocate --filesystem cat --reminder 7 --mailaddress <MAIL> <REMOTE_WS_NAME> 30
```

Also define an environment variable for convenience:

```bash
export CAT_WS=/data/cat/ws/$USER-<REMOTE_WS_NAME>
```

Verify:

```bash
ls -lah $CAT_WS
```



## 3. Cloning the Repo

```bash
cd $CAT_WS
git clone git@github.com:your_username/your_repo.git
cd $CAT_WS/your_repo
```

This is a convenient approach, since you only have a console view on the cluster and editing job submission files with `nano` is a pain - especially given that you often don't know how many resources you need upfront and have to run test jobs to tune them, balancing enough compute against wasting HPC hours. With Git, you can simply fix your launch file locally, `git push`, `git pull` on the cluster, and the changes are already there.



## 4. Uploading the Dataset

There are 2 ways: either download a dataset using `wget` (from some online source) or push it using `rsync` from your local machine.

If your training setup starts from an existing checkpoint, that checkpoint is a separate required artifact and must also be present on the cluster before the first `sbatch`.

### wget

```bash
mkdir -p $CAT_WS/your_repo/data/<DATASET_NAME>
wget -O $CAT_WS/your_repo/data/<DATASET_NAME>/<FILE_NAME> "<DATASET_URL>"
```

Example:

```bash
mkdir -p $CAT_WS/fastumi_dp/data/dataset/fold_towel
wget -O $CAT_WS/fastumi_dp/data/dataset/fold_towel/dataset.zarr.zip \
  "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/fold_towel/dataset.zarr.zip"
```

### rsync

```bash
rsync -avh <LOCAL_DATA_DIR>/ roge097b@dataport1.hpc.tu-dresden.de:$CAT_WS/your_repo/data/<DATASET_NAME>/
```

Important:

- create the target directory on the cluster first with `mkdir -p`
- run `rsync` from your local machine, not from inside the cluster shell
- avoid relying on local `$USER` or `$CAT_WS` expansion unless you intentionally exported them in your local shell to the exact remote values

Example:

```bash
rsync -avh ~/datasets/fold_towel/ roge097b@dataport1.hpc.tu-dresden.de:$CAT_WS/fastumi_dp/data/dataset/fold_towel/
```

### Uploading an initialization checkpoint

If your Slurm script uses a path like `INIT_CHECKPOINT=.../latest.ckpt`, copy that file separately.

Example:

```bash
ssh roge097b@login1.capella.hpc.tu-dresden.de 'mkdir -p /data/cat/ws/roge097b-fastumi/fastumi_dp/data/checkpoints/pour_water'
rsync -avh \
  /home/umi/fastumi_dp/data/checkpoints/pour_water/latest.ckpt \
  roge097b@dataport1.hpc.tu-dresden.de:/data/cat/ws/roge097b-fastumi/fastumi_dp/data/checkpoints/pour_water/latest.ckpt
```

Before the first debug run, verify both the dataset and the checkpoint exist on the cluster.



## 5. Creating a Python Environment

Despite HPC prefers venv, conda is imho superior because it allows specifying a Python version, installing non-Python packages like CUDA and OpenCV, and provides pre-built binaries. Note that on HPC it's important to create the conda env on the `cat` workspace (hence `--prefix`).

### Load conda and cmake

```bash
module load Miniconda3/25.5.1-1
module load CMake/3.18.4
conda env create --prefix $CAT_WS/envs/<ENV_NAME> -f conda_environment.yaml
```

### Log out, log back in, run checks

Log in again and activate the env for a quick check:

```bash
logout
ssh -A roge097b@login1.capella.hpc.tu-dresden.de
module load Miniconda3/25.5.1-1
module load CMake/3.18.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CAT_WS/envs/<ENV_NAME>
hash -r
echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python -c 'import sys; print(sys.executable)'
```

If `which python` or `sys.executable` does not point to `$CAT_WS/envs/<ENV_NAME>/bin/python`, the shell is not using the intended interpreter even if the prompt looks activated. In that case, run checks through the prefix explicitly:

```bash
conda run -p $CAT_WS/envs/<ENV_NAME> python -c 'import sys; print(sys.executable)'
```

At this point there is a chance you will run into dependency conflicts, even if the environment installed perfectly on your local machine. This is hard to cover generically - if it happens, reinstall the conflicting packages inside the conda env (just paste your conda yaml and the error text into an LLM and it will help you). For example, in our case it was:

```bash
pip install wandb==0.15.8 --force-reinstall
pip install "setuptools<70" --force-reinstall
```

Then check that things work (CUDA not available is expected - there is no GPU on the login node):

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

Then check the packages that previously yielded errors:

```bash
python -c 'import hydra, hydra.version; print(hydra.__file__); print(hydra.version.__version__)'
python -c 'import torch; import accelerate; import timm; import wandb; print("All good!")'
```



## 6. Submitting the Job

The prerequisite is that before starting training on HPC you have already debugged your script locally and have a `.sh` file that launches training (and you ran it for several epochs to make sure validation/test/checkpoint saving functionalities run correctly).

### Adjusting the launch file

As an example we had this initial training script:

```bash
task_name="pour_water_depth"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
echo ${run_dir}

accelerate launch --mixed_precision 'bf16' ../train.py \
--config-name=train_diffusion_unet_timm_umi_depth_workspace \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path=data/datasets/processed_pour_water/dataset_subsampled.zarr.zip \
training.num_epochs=250 \
dataloader.batch_size=64 \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
task.dataset.use_ratio=1.0 \
task.obs_down_sample_steps=[3,15] \
task.action_down_sample_steps=3 \
task.low_dim_obs_horizon=3 \
task.img_obs_horizon=3 \
training.debug=True
```

which we then wrapped into an sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=fastumi_depth
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=100:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/data/cat/ws/roge097b-fastumi/fastumi_dp/logs/%j.out
#SBATCH --error=/data/cat/ws/roge097b-fastumi/fastumi_dp/logs/%j.err
#SBATCH --account=p_roborescue

CAT_WS=/data/cat/ws/roge097b-fastumi
module load Miniconda3/25.5.1-1
module load CMake/3.18.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/cat/ws/roge097b-fastumi/envs/fastumi
export PYTHONPATH="$CAT_WS/fastumi_dp:${PYTHONPATH:-}"
echo "python_executable=$(command -v python)"
python - <<'PY'
import importlib
import sys

required = ["hydra", "accelerate", "torch", "timm", "wandb"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

print(f"python_sys_executable={sys.executable}")
if missing:
  print("ERROR: missing Python modules in active conda env:", ", ".join(missing), file=sys.stderr)
  sys.exit(1)

print("env_check=ok")
PY
export HYDRA_FULL_ERROR=1
cd "$CAT_WS/fastumi_dp/train_scripts"

task_name="pour_water"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
echo "${run_dir}"

accelerate launch --mixed_precision bf16 --num_processes 4 ../train.py \
--config-name=train_diffusion_unet_timm_umi_depth_workspace \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path="$CAT_WS/fastumi_dp/data/datasets/pour_water/dataset.zarr.zip" \
training.num_epochs=75 \
dataloader.batch_size=32 \
dataloader.num_workers=8 \
val_dataloader.num_workers=8 \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
task.dataset.use_ratio=1.0 \
task.obs_down_sample_steps=[3,15] \
task.action_down_sample_steps=3 \
task.low_dim_obs_horizon=3 \
task.img_obs_horizon=3 \
training.gradient_accumulate_every=2 \
task.dataset.cache_dir="$CAT_WS/fastumi_dp/data/cache" \
training.lr_warmup_steps=1000
```

Key differences from the original launch script:
- included conda setup (with loading the cmake and conda modules, since they are not loaded on GPU yet, only on login node)
- added all `#SBATCH` directives at the top - note these do **not** expand shell variables, so always use full absolute paths there. %j is the job id alias
- `source $(conda info --base)/etc/profile.d/conda.sh` is required before `conda activate` in Slurm scripts - plain `conda activate` does not work without it
- added `export PYTHONPATH=$CAT_WS/fastumi_dp:${PYTHONPATH:-}` so Python can find the local `diffusion_policy` package without failing when `PYTHONPATH` is initially unset
- used `$CAT_WS` variable for all paths inside the script body (but not in `#SBATCH` lines)

Also note that `--num_processes` must match `--gres=gpu:N`.

### Test run (single GPU, 30 minutes)

Before launching full training, always do a short test run to catch errors early:

```bash
#!/bin/bash
#SBATCH --job-name=fastumi_test
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/data/cat/ws/roge097b-fastumi/fastumi_dp/logs/%j.out
#SBATCH --error=/data/cat/ws/roge097b-fastumi/fastumi_dp/logs/%j.err
#SBATCH --account=p_roborescue

CAT_WS=/data/cat/ws/roge097b-fastumi
module load Miniconda3/25.5.1-1
module load CMake/3.18.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/cat/ws/roge097b-fastumi/envs/fastumi
export PYTHONPATH="$CAT_WS/fastumi_dp:${PYTHONPATH:-}"
echo "python_executable=$(command -v python)"
python - <<'PY'
import importlib
import sys

required = ["hydra", "accelerate", "torch", "timm", "wandb"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

print(f"python_sys_executable={sys.executable}")
if missing:
  print("ERROR: missing Python modules in active conda env:", ", ".join(missing), file=sys.stderr)
  sys.exit(1)

print("env_check=ok")
PY
export HYDRA_FULL_ERROR=1
cd "$CAT_WS/fastumi_dp/train_scripts"

task_name="pour_water"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
echo "${run_dir}"

accelerate launch --mixed_precision bf16 --num_processes 1 ../train.py \
--config-name=train_diffusion_unet_timm_umi_depth_workspace \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path="$CAT_WS/fastumi_dp/data/datasets/pour_water/dataset.zarr.zip" \
training.num_epochs=2 \
dataloader.batch_size=32 \
dataloader.num_workers=8 \
val_dataloader.num_workers=8 \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
task.dataset.use_ratio=1.0 \
task.obs_down_sample_steps=[3,15] \
task.action_down_sample_steps=3 \
task.dataset.cache_dir="$CAT_WS/fastumi_dp/data/cache" \
task.low_dim_obs_horizon=3 \
task.img_obs_horizon=3 \
training.debug=True
```

### Avoiding silly mistakes

If the dataset contains images it can be huge. If you try to load it fully into RAM you can get an OOM that makes the node completely unresponsive. For fastumi, use `task.dataset.cache_dir` to cache the dataset on disk so workers read data from disk instead of loading everything into memory. Otherwise refer to LLM to help you with that if needed.

### Submit

```bash
mkdir -p $CAT_WS/your_repo/logs
sbatch submit_train.sh
```

Useful job management commands:

```bash
squeue -u roge097b          # check job status
scancel <jobid>               # cancel a specific job
scancel -u roge097b         # cancel all your jobs
sinfo                         # check available partitions and GPUs
```



## 7. Monitoring Jobs

Ideally use wandb. If your repo uses an older wandb (like fastumi), export the API key before starting training due to key format incompatibility between old and new wandb versions:

```bash
export WANDB_API_KEY=<YOUR_KEY>
```

This lets you monitor training loss from the wandb dashboard on your phone first thing in the morning.

Alternatively, tail the logs directly:

```bash
tail -fn 1000 $CAT_WS/your_repo/logs/<JOBID>.out
tail -fn 1000 $CAT_WS/your_repo/logs/<JOBID>.err
```

Switch between `.out` and `.err` - `.out` has training progress, `.err` has errors and warnings.



## 8. Downloading the Ready Checkpoint

Once training finishes, pull the checkpoint from the cluster to your local machine using `rsync`:

```bash
rsync -avz --progress \
  roge097b@login1.capella.hpc.tu-dresden.de:$CAT_WS/your_repo/data/outputs/<run_dir>/checkpoints \
  <LOCAL_CODE_DIR>/checkpoints/
```

Example:

```bash
rsync -avz --progress \
  roge097b@login1.capella.hpc.tu-dresden.de:/data/cat/ws/roge097b-fastumi/fastumi_dp/data/outputs/2026.03.20/14.58.03/checkpoints \
  ~/fastumi_dp/checkpoints/
```