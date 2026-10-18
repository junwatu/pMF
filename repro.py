#!/usr/bin/env python3
"""Standalone pMF FID reproduction script."""

import json, os, sys, time
import numpy as np
import torch
from pathlib import Path
from diffusers import DiffusionPipeline
from scipy import linalg

def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    d = mu1 - mu2
    c, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(c).all():
        c = linalg.sqrtm((sigma1+eps*np.eye(sigma1.shape[0])).dot(sigma2+eps*np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(c):
        c = c.real
    return float(d.dot(d) + np.trace(sigma1 + sigma2 - 2*c))

def compute_fid(imgs, ref_path, bs=100):
    from cleanfid.features import build_feature_extractor, get_image_features
    ref = np.load(ref_path)
    mr, sr = ref["mu"], ref["sigma"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_feature_extractor("inception_v3", device=dev)
    m.eval()
    feats = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i+bs]
        bt = torch.from_numpy(b).permute(0,3,1,2).float()/255.0
        feats.append(get_image_features(m, bt.to(dev)).cpu().numpy())
    a = np.concatenate(feats, axis=0)
    return compute_fid_from_stats(np.mean(a,axis=0), np.cov(a,rowvar=False), mr, sr)

CFG = {
    "pMF-B/16":{"gs":7.5,"tmi":0.1,"tma":0.8,"ns":1.0},
    "pMF-L/16":{"gs":7.0,"tmi":0.2,"tma":0.7,"ns":1.0},
    "pMF-H/16":{"gs":7.0,"tmi":0.2,"tma":0.6,"ns":2.0},
    "pMF-B/32":{"gs":6.5,"tmi":0.1,"tma":0.7,"ns":2.0},
    "pMF-L/32":{"gs":7.5,"tmi":0.2,"tma":0.6,"ns":4.0},
    "pMF-H/32":{"gs":5.5,"tmi":0.1,"tma":0.6,"ns":4.0},
}
PAPER = {"pMF-B/16":3.12,"pMF-L/16":2.52,"pMF-H/16":2.22,
         "pMF-B/32":3.70,"pMF-L/32":2.75,"pMF-H/32":2.48}

variants = sorted(CFG.keys())

def run_variant(variant, num_samples, fid_stats):
    c = CFG[variant]
    sf = variant.replace("/", "-")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== {variant} on {dev} ===", flush=True)

    t0 = time.time()
    model_dir = Path(f"./model/{sf}")
    if not model_dir.exists():
        from huggingface_hub import snapshot_download
        print(f"Downloading {variant}...", flush=True)
        snapshot_download("BiliSakura/pMF-diffusers", allow_patterns=f"{sf}/**", local_dir="./model")
        print(f"Downloaded in {time.time()-t0:.1f}s", flush=True)

    # Patch scheduler config to use the custom PMFScheduler (supports 1-step)
    sched_config = model_dir / "scheduler" / "scheduler_config.json"
    import json as _json
    if sched_config.exists():
        with open(sched_config) as f:
            cfg = _json.load(f)
        if cfg.get("_class_name") != "PMFScheduler":
            cfg["_class_name"] = "PMFScheduler"
            with open(sched_config, "w") as f:
                _json.dump(cfg, f, indent=2)
            print(f"Patched scheduler config to use PMFScheduler", flush=True)

    # Also patch model_index.json so the loader uses the custom scheduler module
    model_index = model_dir / "model_index.json"
    if model_index.exists():
        with open(model_index) as f:
            mi = _json.load(f)
        if mi.get("scheduler", [None, None])[1] != "PMFScheduler":
            mi["scheduler"] = ["scheduling_pmf", "PMFScheduler"]
            with open(model_index, "w") as f:
                _json.dump(mi, f, indent=2)
            print(f"Patched model_index.json to use PMFScheduler", flush=True)

    print(f"Loading from {model_dir}...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True, torch_dtype=torch.float16,
    ).to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    classes = [i%1000 for i in range(num_samples)]
    gen = torch.Generator(device=dev)

    t1 = time.time()
    all_imgs = []
    for i in range(0, num_samples, 50):
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
        if (i+50)%1000==0 or i+50>=num_samples:
            pct = min(i+50, num_samples)
            print(f"  {pct}/{num_samples} ({time.time()-t1:.0f}s)", flush=True)

    imgs_np = np.stack(all_imgs, axis=0)
    gt = time.time()-t1
    print(f"Generation: {gt:.0f}s ({gt/num_samples:.4f}s/img)", flush=True)

    os.makedirs("outputs", exist_ok=True)
    npz_path = f"outputs/{variant.replace(chr(47),chr(95))}_images.npz"
    np.savez_compressed(npz_path, images=imgs_np)

    t2 = time.time()
    fid = round(compute_fid(imgs_np, fid_stats), 4)
    ft = time.time()-t2
    print(f"FID: {fid} (paper: {PAPER[variant]}, diff: {round(fid-PAPER[variant],4)}), time: {ft:.0f}s", flush=True)

    result = {
        "variant": variant, "num_samples": num_samples,
        "fid": fid, "fid_paper": PAPER[variant],
        "fid_diff": round(fid-PAPER[variant], 4),
        "gen_time_s": round(gt, 1), "fid_time_s": round(ft, 1),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    rp = f"outputs/{variant.replace(chr(47),chr(95))}_result.json"
    with open(rp, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=variants+["all"])
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--fid_stats_256", default="imagenet_256_fid_stats.npz")
    parser.add_argument("--fid_stats_512", default="imagenet_512_fid_stats.npz")
    args = parser.parse_args()

    if args.variant == "all":
        variants_to_run = variants
    else:
        variants_to_run = [args.variant]

    results = {}
    for v in variants_to_run:
        stats = args.fid_stats_512 if "/32" in v else args.fid_stats_256
        results[v] = run_variant(v, args.num_samples, stats)

    print("\n=== SUMMARY ===")
    for v, r in results.items():
        print(f"  {v}: FID={r['fid']} (paper: {r['fid_paper']}, diff: {r['fid_diff']:+})")

    with open("outputs/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Upload results
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path="outputs", repo_id="junwatu/repro-pmf-results",
                      repo_type="dataset", exist_ok=True)
    print("Results uploaded to junwatu/repro-pmf-results")

if __name__ == "__main__":
    main()
