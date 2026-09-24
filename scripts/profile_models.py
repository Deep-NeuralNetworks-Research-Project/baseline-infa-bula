"""CLI script for model efficiency profiling.

Usage:
    python scripts/profile.py model=fc_siam_diff
    python scripts/profile.py model=rgb_ssim
"""

import time
import logging
from typing import Any
import numpy as np

import torch
import hydra
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis

from cdlib.models.build import build_model
from cdlib.utils.seed import set_seed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profile")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model: torch.nn.Module, device: torch.device, resolution: int = 256) -> int:
    """Measure FLOPs for a single forward pass."""
    img1 = torch.randn(1, 3, resolution, resolution).to(device)
    img2 = torch.randn(1, 3, resolution, resolution).to(device)
    
    model.eval()
    flops = FlopCountAnalysis(model, (img1, img2))
    return flops.total()


def measure_memory(
    model: torch.nn.Module, device: torch.device, batch_size: int = 1, is_train: bool = False, resolution: int = 256
) -> float:
    """Measure peak GPU memory allocation in MB."""
    if device.type != "cuda":
        return 0.0

    img1 = torch.randn(batch_size, 3, resolution, resolution).to(device)
    img2 = torch.randn(batch_size, 3, resolution, resolution).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if is_train:
        model.train()
        outputs = model(img1, img2)
        # Dummy loss to force backward pass
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"].mean()
        elif isinstance(outputs, dict) and "logits" in outputs:
            loss = outputs["logits"].mean()
        else:
            loss = outputs.mean() if isinstance(outputs, torch.Tensor) else 0.0
            
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            _ = model(img1, img2)

    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def measure_latency(
    model: torch.nn.Module, device: torch.device, batch_size: int = 1, runs: int = 100, warmup: int = 10, resolution: int = 256
) -> float:
    """Measure median forward-pass latency in ms."""
    img1 = torch.randn(batch_size, 3, resolution, resolution).to(device)
    img2 = torch.randn(batch_size, 3, resolution, resolution).to(device)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(img1, img2)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(img1, img2)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return float(np.median(times) * 1000)


def measure_sce_audit_cost(
    model: torch.nn.Module, device: torch.device, resolution: int = 256
) -> tuple[float, float]:
    """Measure the time to compute a single forward pass vs the SCE symmetric audit."""
    img1 = torch.randn(1, 3, resolution, resolution).to(device)
    img2 = torch.randn(1, 3, resolution, resolution).to(device)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(img1, img2)
            _ = model(img2, img1)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        # Single pass
        start = time.perf_counter()
        _ = model(img1, img2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        single_pass_time = (time.perf_counter() - start) * 1000

        # SCE pass (model(A, B) + model(B, A))
        start = time.perf_counter()
        out1 = model(img1, img2)
        out2 = model(img2, img1)
        # minimal correlation mock logic overhead
        if isinstance(out1, dict) and "confidence" in out1 and out1.get("confidence") is not None:
            diff = (out1["confidence"] - out2["confidence"]).abs().mean()
        elif isinstance(out1, dict) and "logits" in out1:
            diff = (out1["logits"] - out2["logits"]).abs().mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
        sce_pass_time = (time.perf_counter() - start) * 1000

    return single_pass_time, sce_pass_time


def main() -> None:
    import sys
    from hydra import compose, initialize
    
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model = build_model(cfg).to(device)
    except Exception as e:
        logger.error(f"Failed to build model {cfg.model.name}: {e}")
        return

    logger.info(f"Profiling Model: {cfg.model.name}")
    print("-" * 50)
    
    # 1. Parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters (Total)     : {total_params:,}")
    print(f"Parameters (Trainable) : {trainable_params:,}")
    
    # 2. FLOPs
    try:
        flops = measure_flops(model, device)
        print(f"FLOPs (1x256x256)      : {flops / 1e9:.2f} G MACs")
    except Exception as e:
        logger.warning(f"Failed to measure FLOPs: {e}")
        
    # 3. Peak GPU Memory
    if device.type == "cuda":
        mem_train_b1 = measure_memory(model, device, batch_size=1, is_train=True)
        mem_train_b16 = measure_memory(model, device, batch_size=16, is_train=True)
        mem_inf_b1 = measure_memory(model, device, batch_size=1, is_train=False)
        mem_inf_b16 = measure_memory(model, device, batch_size=16, is_train=False)
        print(f"Memory (Train, B=1)    : {mem_train_b1:.1f} MB")
        print(f"Memory (Train, B=16)   : {mem_train_b16:.1f} MB")
        print(f"Memory (Infer, B=1)    : {mem_inf_b1:.1f} MB")
        print(f"Memory (Infer, B=16)   : {mem_inf_b16:.1f} MB")
    else:
        print("Memory profiling skipped (no CUDA).")

    # 4. Latency
    latency_b1 = measure_latency(model, device, batch_size=1)
    latency_b16 = measure_latency(model, device, batch_size=16)
    print(f"Latency (Infer, B=1)   : {latency_b1:.2f} ms")
    print(f"Latency (Infer, B=16)  : {latency_b16:.2f} ms")
    
    # 5. SCE Audit Cost
    single_pass, sce_pass = measure_sce_audit_cost(model, device)
    print(f"Latency (1 Forward Pass): {single_pass:.2f} ms")
    print(f"Latency (SCE Audit)    : {sce_pass:.2f} ms")
    print(f"SCE Audit Overhead     : {(sce_pass / single_pass - 1)*100:.1f}% (Expected ~100%)")
    
    print("-" * 50)
    

if __name__ == "__main__":
    main()
