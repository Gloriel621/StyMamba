# StyMamba

StyMamba is a Mamba‑based multimodal style‑transfer framework that supports both image‑guided and text‑guided stylization.


**Warning:** The software stack is extremely sensitive to exact package versions. 

Even with the instructions below the build may still fail on some machines. 

Mamba‑SSM, Triton, and PyTorch all need to agree on the same CUDA toolkit and GCC version, and small mismatches can break compilation or cause runtime seg‑faults.

## Tested configuration

| Component  | Version                      |
| ---------- | ---------------------------- |
| GPU driver | ≥ 525.85 (CUDA 11.8 runtime) |
| CUDA       | 11.8                         |
| Python     | 3.10                         |
| GCC / G++  | 11                           |
| PyTorch    | 2.2.0                        |
| Triton     | 2.1.0                        |

If you deviate from these versions you may need to patch the build scripts or recompile kernels manually.

## Quick start

```bash
# 1. Create a fresh environment
conda create -n StyMamba python=3.10 -y
conda activate StyMamba

# 2. Install GPU‑enabled PyTorch
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Add the CUDA runtime libraries
conda install nvidia/label/cuda-11.8.0::cuda -y

# 4. Ensure a modern toolchain (needed for Triton)
conda install gcc_linux-64=11 gxx_linux-64=11 -y

# 5. Python dependencies
pip install causal-conv1d==1.5.0.post8 \
           mamba-ssm==2.2.2 \
           scikit-learn==1.6.1 \
           numpy==1.26.4 \
           triton==2.1.0 \
           openai-clip

# 6. Clone the repository
git clone /address/of/this/repo
cd StyMamba

# 7. Build VMamba custom kernels
# errors can happen when installing selective_scan.
cd VMamba2
pip install -r requirements.txt
cd kernels/selective_scan
pip install .
```

All versions should match those in the table above.

## Training

1. **Prepare content images** — any photo dataset works; we recommend the MS‑COCO 2014 training split or your own collection.


2. **Download WikiArt styles** — grab the curated WikiArt archive from **Google Drive** and unpack it:
   [https://drive.google.com/drive/folders/1nJ6ThkwWmP3nfn4q596rhyLFeVmEBxYR?usp=sharing](https://drive.google.com/drive/folders/1nJ6ThkwWmP3nfn4q596rhyLFeVmEBxYR?usp=sharing)


3. **Launch training** — activate the environment and point the script at the two folders:

   ```bash
   conda activate StyMamba
   python3 train.py \
       --content_dir $CONTENT_DIR \
       --style_dir   $STYLE_DIR
   ```

4. **Using other style sets** — The WikiArt folder includes text‑prompt annotations (artist names) required for text‑guided training. If you substitute a different style dataset you must generate equivalent text files and update the preprocessing scripts.

## Inference

We provide two inference files, inference_image.py and inference_text.py

We also provide test images for style and content in /examples.

## Benchmark Images

Benchmark result images by Mamba-ST, SaMam, and our StyMamba are provided at:
[https://drive.google.com/drive/folders/1eJUD3QtDhR9rhAG3fZ5GzY6rwUlc7kIj?usp=sharing](https://drive.google.com/drive/folders/1eJUD3QtDhR9rhAG3fZ5GzY6rwUlc7kIj?usp=sharing)

## Checkpoints

Our model checkpoints are provided at:

[https://drive.google.com/drive/folders/1bMPb0Q0QkTQMd1di_e6msKOujsBVhI5A?usp=sharing](https://drive.google.com/drive/folders/1bMPb0Q0QkTQMd1di_e6msKOujsBVhI5A?usp=sharing)

## Acknowledgements

StyMamba builds upon VMamba, Mamba‑SSM, and Triton selective‑scan kernels.
