#!/bin/bash
# Run pMF reproduction inside HF Job
set -e

VARIANT=${1:-pMF-H/16}
NUM_SAMPLES=${2:-50000}

echo "=== pMF Reproduction: $VARIANT ==="
echo "Samples: $NUM_SAMPLES"

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers transformers scipy numpy clean-fid Pillow huggingface_hub

# Map variant to subfolder
case "$VARIANT" in
    pMF-B/16) SF=pMF-B-16; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_256_fid_stats.npz ;;
    pMF-L/16) SF=pMF-L-16; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_256_fid_stats.npz ;;
    pMF-H/16) SF=pMF-H-16; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_256_fid_stats.npz ;;
    pMF-B/32) SF=pMF-B-32; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_512_fid_stats.npz ;;
    pMF-L/32) SF=pMF-L-32; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_512_fid_stats.npz ;;
    pMF-H/32) SF=pMF-H-32; FID_URL=https://huggingface.co/Lyy0725/pMF/resolve/main/imagenet_512_fid_stats.npz ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

# Download variant subfolder locally
echo "Downloading variant $SF from BiliSakura/pMF-diffusers..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BiliSakura/pMF-diffusers', allow_patterns=f'$SF/**', local_dir='./model')
"
MODEL_DIR="./model/$SF"
echo "Model directory: $MODEL_DIR"
ls -la "$MODEL_DIR"

# Download FID stats
echo "Downloading FID stats..."
curl -sL "$FID_URL" -o /tmp/stats.npz
echo "FID stats: $(wc -c < /tmp/stats.npz) bytes"

# Run reproduction
python3 << 'PYEOF'
import json, os, time
import numpy as np
import torch
from pathlib import Path
from diffusers import DiffusionPipeline
from scipy import linalg

VARIANT = os.environ.get("VARIANT", "pMF-H/16")
NUM = int(os.environ.get("NUM_SAMPLES", "50000"))
MODEL_DIR = os.environ.get("MODEL_DIR")
FID_STATS = "/tmp/stats.npz"

def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def compute_fid(imgs, ref_path, bs=100):
    from cleanfid.features import build_feature_extractor, get_image_features
    ref = np.load(ref_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_feature_extractor("inception_v3", device=device)
    model.eval()
    feats = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i+bs]
        bt = torch.from_numpy(b).permute(0,3,1,2).float()/255.0
        feats.append(get_image_features(model, bt.to(device)).cpu().numpy())
    all_f = np.concatenate(feats, axis=0)
    mu = np.mean(all_f, axis=0)
    sg = np.cov(all_f, rowvar=False)
    return compute_fid_from_stats(mu, sg, mu_ref, sigma_ref)

cfg_params = {
    "pMF-B/16":{"gs":7.5,"tmi":0.1,"tma":0.8,"ns":1.0},
    "pMF-L/16":{"gs":7.0,"tmi":0.2,"tma":0.7,"ns":1.0},
    "pMF-H/16":{"gs":7.0,"tmi":0.2,"tma":0.6,"ns":2.0},
    "pMF-B/32":{"gs":6.5,"tmi":0.1,"tma":0.7,"ns":2.0},
    "pMF-L/32":{"gs":7.5,"tmi":0.2,"tma":0.6,"ns":4.0},
    "pMF-H/32":{"gs":5.5,"tmi":0.1,"tma":0.6,"ns":4.0},
}
paper = {"pMF-B/16":3.12,"pMF-L/16":2.52,"pMF-H/16":2.22,"pMF-B/32":3.70,"pMF-L/32":2.75,"pMF-H/32":2.48}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
c = cfg_params[VARIANT]
os.makedirs("outputs", exist_ok=True)

t0 = time.time()
model_path = Path(MODEL_DIR)
print(f"Loading from {model_path}...", flush=True)
pipe = DiffusionPipeline.from_pretrained(
    str(model_path),
    local_files_only=True,
    custom_pipeline=str(model_path / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(device)
pipe.set_progress_bar_config(disable=True)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

classes = [i%1000 for i in range(NUM)]
gen = torch.Generator(device=device)

t1 = time.time()
all_imgs = []
for i in range(0, NUM, 50):
    gen.manual_seed(i)
    with torch.no_grad():
        imgs = pipe(
            class_labels=classes[i:i+50],
            num_inference_steps=1,
            guidance_scale=c["gs"],
            guidance_interval_min=c["tmi"],
            guidance_interval_max=c["tma"],
            noise_scale=c["ns"],
            generator=gen,
        ).images
    all_imgs.extend([np.array(img) for img in imgs])
    if (i+50)%1000==0 or i+50>=NUM:
        print(f"  {len(all_imgs)}/{NUM} ({time.time()-t1:.0f}s)", flush=True)

imgs_np = np.stack(all_imgs, axis=0)
gt = time.time()-t1
print(f"Gen: {gt:.0f}s ({gt/NUM:.4f}s/img)", flush=True)

npz_path = f"outputs/{VARIANT.replace(chr(47),chr(95))}_images.npz"
np.savez_compressed(npz_path, images=imgs_np)
print(f"Saved {npz_path}", flush=True)

t2 = time.time()
fid = round(compute_fid(imgs_np, FID_STATS), 4)
ft = time.time()-t2
print(f"FID: {ft:.0f}s", flush=True)

r = {
    "variant":VARIANT, "num_samples":NUM, "fid":fid,
    "fid_paper":paper[VARIANT], "fid_diff":round(fid-paper[VARIANT],4),
    "gen_time_s":round(gt,1), "fid_time_s":round(ft,1),
    "hardware":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
print(json.dumps(r,indent=2), flush=True)
rp = f"outputs/{VARIANT.replace(chr(47),chr(95))}_result.json"
with open(rp,"w") as f: json.dump(r,f,indent=2)
PYEOF
