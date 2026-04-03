# 1. Update Packages

```bash
sudo apt update && sudo apt install -y build-essential
```

# 2. default package for setup

```bash
pip install --upgrade pip setuptools wheel ninja packaging
pip install einops timm h5py scipy scikit-image matplotlib tqdm pytorch_wavelets PyWavelets
```

# 3. NOTION! Depends of you GPU, drivers, versions, CUDA, etc.

For example my is 4070super nvidia

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# NOTION! WAIT! the loading can be freeze, wait untill command end!!!, 30-60 minutes

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation -v
```

# NOTION: WAIT! the loading can be freeze, wait untill command end!!!, 30-60 minutes

```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

# 4. Training notes

The training pipeline is patch-based now, because whole-scene training on CAVE usually gives too few optimization steps.

Key options in `config.yaml`:

```yaml
data:
  train_patch_size: [128, 128]
  patches_per_scene: 128
  normalization: "fixed_255"

model:
  ssm_version: "mamba3"
```

If GPU memory is limited, reduce `train_patch_size` first and only then `batch_size`.
