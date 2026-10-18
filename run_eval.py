#!/usr/bin/env python3
"""Single-model FID evaluation script for HF Jobs."""
import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                       choices=['pmfDiT_B_16', 'pmfDiT_L_16', 'pmfDiT_H_16',
                                'pmfDiT_B_32', 'pmfDiT_L_32', 'pmfDiT_H_32'])
    parser.add_argument('--workdir', default='./eval_output')
    parser.add_argument('--num-images', type=int, default=50000)
    parser.add_argument('--gen-bsz', type=int, default=32)
    parser.add_argument('--save-samples', action='store_true')
    args = parser.parse_args()

    # Model-specific hyperparameters from the paper/README
    model_params = {
        'pmfDiT_B_16': {'cfg_omega': 7.5, 'interval_min': 0.1, 'interval_max': 0.8, 'ckpt': 'pMF-B-16.pt'},
        'pmfDiT_L_16': {'cfg_omega': 7.0, 'interval_min': 0.2, 'interval_max': 0.7, 'ckpt': 'pMF-L-16.pt'},
        'pmfDiT_H_16': {'cfg_omega': 7.0, 'interval_min': 0.2, 'interval_max': 0.6, 'ckpt': 'pMF-H-16.pt'},
        'pmfDiT_B_32': {'cfg_omega': 6.5, 'interval_min': 0.1, 'interval_max': 0.7, 'ckpt': 'pMF-B-32.pt'},
        'pmfDiT_L_32': {'cfg_omega': 7.5, 'interval_min': 0.2, 'interval_max': 0.6, 'ckpt': 'pMF-L-32.pt'},
        'pmfDiT_H_32': {'cfg_omega': 5.5, 'interval_min': 0.1, 'interval_max': 0.6, 'ckpt': 'pMF-H-32.pt'},
    }

    params = model_params[args.model]
    ckpt_path = os.path.join('checkpoints', params['ckpt'])
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    os.makedirs(args.workdir, exist_ok=True)

    cmd = [
        sys.executable, 'evaluate.py', 'evaluate',
        '--ckpt-path', ckpt_path,
        '--workdir', args.workdir,
        '--model', args.model,
        '--cfg-omega', str(params['cfg_omega']),
        '--interval-min', str(params['interval_min']),
        '--interval-max', str(params['interval_max']),
        '--num-images', str(args.num_images),
        '--gen-bsz', str(args.gen_bsz),
    ]
    
    if args.save_samples:
        cmd.append('--save-samples')

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
