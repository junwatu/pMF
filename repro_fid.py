"""Compute FID for a given Pixel MeanFlow checkpoint variant.

Usage:
    python repro_fid.py --variant pMF-H/16 --num_samples 50000

Uses pre-trained weights from BiliSakura/pMF-diffusers and official
FID statistics from Lyy0725/pMF.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers import DiffusionPipeline
from scipy import linalg


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def compute_fid(gen_images, ref_stats_path, batch_size=100):
    import cleanfid
    from cleanfid.features import build_feature_extractor, get_image_features

    ref_stats = np.load(ref_stats_path)
    mu_ref = ref_stats["mu"]
    sigma_ref = ref_stats["sigma"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_feature_extractor("inception_v3", device=device)
    model.eval()

    all_feats = []
    for i in range(0, len(gen_images), batch_size):
        batch = gen_images[i : i + batch_size]
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        batch_t = batch_t.to(device)
        feats = get_image_features(model, batch_t)
        all_feats.append(feats.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    mu_gen = np.mean(all_feats, axis=0)
    sigma_gen = np.cov(all_feats, rowvar=False)
    fid = compute_fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
    return float(fid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=[
        "pMF-B/16", "pMF-L/16", "pMF-H/16",
        "pMF-B/32", "pMF-L/32", "pMF-H/32",
    ])
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--fid_stats", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    variant_to_path = {
        "pMF-B/16": "pMF-B-16", "pMF-L/16": "pMF-L-16", "pMF-H/16": "pMF-H-16",
        "pMF-B/32": "pMF-B-32", "pMF-L/32": "pMF-L-32", "pMF-H/32": "pMF-H-32",
    }

    cfg_params = {
        "pMF-B/16": {"guidance_scale": 7.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.8, "noise_scale": 1.0},
        "pMF-L/16": {"guidance_scale": 7.0, "guidance_interval_min": 0.2, "guidance_interval_max": 0.7, "noise_scale": 1.0},
        "pMF-H/16": {"guidance_scale": 7.0, "guidance_interval_min": 0.2, "guidance_interval_max": 0.6, "noise_scale": 2.0},
        "pMF-B/32": {"guidance_scale": 6.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.7, "noise_scale": 2.0},
        "pMF-L/32": {"guidance_scale": 7.5, "guidance_interval_min": 0.2, "guidance_interval_max": 0.6, "noise_scale": 4.0},
        "pMF-H/32": {"guidance_scale": 5.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.6, "noise_scale": 4.0},
    }

    subfolder = variant_to_path[args.variant]
    params = cfg_params[args.variant]

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    print(f"Loading pipeline for {args.variant}...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        "BiliSakura/pMF-diffusers",
        subfolder=subfolder,
        custom_pipeline=f"{subfolder}/pipeline.py",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    class_labels = [i % 1000 for i in range(args.num_samples)]
    generator = torch.Generator(device=device)

    t1 = time.time()
    print(f"Generating {args.num_samples} images...", flush=True)
    all_images = []
    saved = 0

    for i in range(0, args.num_samples, 50):
        batch_labels = class_labels[i : i + 50]
        generator.manual_seed(i)
        with torch.no_grad():
            images = pipe(
                class_labels=batch_labels,
                num_inference_steps=1,
                guidance_scale=params["guidance_scale"],
                guidance_interval_min=params["guidance_interval_min"],
                guidance_interval_max=params["guidance_interval_max"],
                noise_scale=params["noise_scale"],
                generator=generator,
            ).images
        all_images.extend([np.array(img) for img in images])
        saved += len(images)

        if (i + 50) % 500 == 0 or (i + 50) >= args.num_samples:
            print(f"  Generated {saved}/{args.num_samples} images ({time.time()-t1:.1f}s)", flush=True)

    images_np = np.stack(all_images, axis=0)
    gen_time = time.time() - t1
    print(f"Total generation time: {gen_time:.1f}s ({gen_time/args.num_samples:.4f}s/img)", flush=True)

    npz_path = os.path.join(args.output_dir, f"{args.variant.replace('/', '_')}_images.npz")
    np.savez_compressed(npz_path, images=images_np)
    print(f"Saved images to {npz_path}", flush=True)

    t2 = time.time()
    fid = compute_fid(images_np, args.fid_stats, batch_size=100)
    fid_time = time.time() - t2
    print(f"FID computed in {fid_time:.1f}s", flush=True)

    paper_fids = {
        "pMF-B/16": 3.12, "pMF-L/16": 2.52, "pMF-H/16": 2.22,
        "pMF-B/32": 3.70, "pMF-L/32": 2.75, "pMF-H/32": 2.48,
    }

    result = {
        "variant": args.variant,
        "num_samples": args.num_samples,
        "fid": round(fid, 4),
        "fid_paper": paper_fids[args.variant],
        "fid_diff": round(fid - paper_fids[args.variant], 4),
        "gen_time_s": round(gen_time, 1),
        "fid_time_s": round(fid_time, 1),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cfg_params": params,
    }

    result_path = os.path.join(args.output_dir, f"{args.variant.replace('/', '_')}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Result saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
