"""Training engine for the 125M from-scratch SLM."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

import config


CommitFn = Callable[[], None] | None


class BinWindowDataset:
    """Random-access packed uint16 windows spread across multiple .bin shards."""

    def __init__(self, root: str | os.PathLike[str], split: str, seq_len: int,
                 source_prefix: str = ""):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.source_prefix = source_prefix
        self.dir = self.root / "tokens" / split
        # bins are named "<source>-<NNN>.bin", so a prefix filter isolates one domain
        # (needed to measure forgetting per source rather than as a blurred aggregate)
        pattern = f"{source_prefix}-*.bin" if source_prefix else "*.bin"
        self.paths = sorted(self.dir.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no token bin files matching {pattern!r} in {self.dir}")

        self.arrays: list[np.memmap] = []
        self.window_counts: list[int] = []
        for path in self.paths:
            size = path.stat().st_size
            bytes_per_window = seq_len * np.dtype(np.uint16).itemsize
            if size % bytes_per_window != 0:
                raise ValueError(f"{path} size is not a multiple of {seq_len} uint16 tokens")
            windows = size // bytes_per_window
            self.arrays.append(np.memmap(path, dtype=np.uint16, mode="r").reshape(windows, seq_len))
            self.window_counts.append(windows)

        self.offsets = np.cumsum([0, *self.window_counts], dtype=np.int64)
        self.total_windows = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total_windows

    def get_batch(self, indices: np.ndarray, device: str):
        import torch

        batch = np.empty((len(indices), self.seq_len), dtype=np.int64)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            local = indices[mask] - self.offsets[shard_id]
            batch[mask] = self.arrays[int(shard_id)][local]
        return torch.from_numpy(batch).to(device=device, non_blocking=True)


def _build_model(device: str, attn_implementation: str):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    llama_config = LlamaConfig(
        **config.MODEL.to_llama_kwargs(),
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        attn_implementation=attn_implementation,
    )
    model = LlamaForCausalLM(llama_config)
    model.to(device)
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
    return model


def _maybe_compile(model, enabled: bool):
    """torch.compile for throughput. Typically 1.3-2x on this workload at no accuracy
    cost -- the single cheapest MFU win available.

    Wrapped in try/except on purpose: compilation can fail on some driver/CUDA
    combinations, and a 12-hour run must not die at step 0 over an optimisation.
    Falling back to eager is slower but correct.

    LIMITATION (learned the hard way): this only guards COMPILATION. Inductor can
    compile cleanly and then fail at RUNTIME inside the generated graph - a real run
    died with "FlashAttention only support fp16 and bf16 data type" thrown from the
    compiled kernel, well past this try/except. Treat compile as opt-in and validate
    it on a short run before trusting it with a long one.
    """
    if not enabled:
        return model
    import torch

    if not hasattr(torch, "compile"):
        print("  torch.compile unavailable on this torch build - running eager", flush=True)
        return model
    try:
        compiled = torch.compile(model)
        print("  torch.compile enabled", flush=True)
        return compiled
    except Exception as exc:  # noqa: BLE001 - never let an optimisation abort training
        print(f"  torch.compile failed ({type(exc).__name__}) - running eager", flush=True)
        return model


def _lr_at_step(step: int, total_steps: int, lr_mode: str, continue_lr: float = 1e-4) -> float:
    train = config.TRAIN
    if lr_mode == "continue":
        # Continued pretraining from an already-annealed checkpoint. v1 finished at
        # min_lr=6e-5, so resuming on that schedule would barely move the weights.
        # A flat LR (default 1e-4, ~17% of v1's 6e-4 peak) is the standard continual-
        # pretraining setting: high enough to learn new data, low enough to limit
        # catastrophic forgetting. No warmup/decay -> LR is a controlled constant.
        return continue_lr
    if lr_mode == "continue_cosine":
        # Multi-epoch continued pretraining aimed at QUALITY, not at isolating a
        # variable. `continue_lr` is the PEAK here (not a constant): short warmup,
        # then cosine decay to 10% of peak across the whole run.
        #
        # Warmup is short (100 steps, ~1% of a 9.4k-step run) because the weights are
        # already trained -- the long 382-step warmup used from scratch exists to
        # protect randomly-initialised weights, which is not the situation here.
        # Decaying to a floor rather than to 0 keeps the last steps productive and
        # leaves the model resumable for a further epoch.
        warm = min(100, max(1, total_steps // 50))
        floor = continue_lr * 0.1
        if step < warm:
            return continue_lr * (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor + cosine * (continue_lr - floor)
    if lr_mode != "scratch":
        raise ValueError(f"unsupported lr_mode={lr_mode!r}")
    warmup_steps = max(1, math.ceil(train.warmup_tokens / train.global_batch_tokens))
    if step < warmup_steps:
        return train.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return train.min_lr + cosine * (train.lr - train.min_lr)


# --- telemetry -------------------------------------------------------------
# Learned empirically on Modal (see the earlier observability study): /proc/stat,
# /proc/meminfo and /sys/fs/cgroup/* are VIRTUALISED inside the sandbox - psutil
# returns all-zero CPU jiffies and a synthetic 1 TiB MemTotal, so host CPU/RAM is
# not measurable from inside or outside the container. nvidia-smi IS a real
# passthrough, and torch.cuda's own allocator counters are reliable. Those two are
# what we record.
N_PARAMS = 125_847_552
_GPU_QUERY = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
              "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem")


def _nvidia_smi() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10)
        v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_util_percent": float(v[0]), "gpu_mem_util_percent": float(v[1]),
                "gpu_mem_used_mib": float(v[2]), "gpu_mem_total_mib": float(v[3]),
                "gpu_power_w": float(v[4]), "gpu_power_limit_w": float(v[5]),
                "gpu_temp_c": float(v[6]), "gpu_sm_clock_mhz": float(v[7]),
                "gpu_mem_clock_mhz": float(v[8])}
    except Exception as exc:  # never let telemetry kill a 12-hour run
        return {"gpu_telemetry_error": type(exc).__name__}


def _torch_mem(device: str) -> dict:
    if device != "cuda":
        return {}
    import torch

    return {
        "torch_mem_alloc_gib": torch.cuda.memory_allocated() / 2**30,
        "torch_mem_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "torch_mem_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def _mfu(tokens_per_sec: float, peak_flops: float) -> float:
    """Model FLOPs Utilisation via the standard 6ND approximation. The headline
    efficiency number: v1 ran ~27%, i.e. 100% 'GPU utilisation' while ~3/4 of the
    silicon's FLOPs did nothing useful."""
    return 100.0 * (tokens_per_sec * 6 * N_PARAMS) / peak_flops if peak_flops else float("nan")


def _peak_flops(device: str) -> float:
    """bf16 dense peak for the attached GPU (used only to normalise MFU)."""
    if device != "cuda":
        return 0.0
    import torch

    name = torch.cuda.get_device_name(0).upper()
    if "H100" in name:
        return 990e12 if "SXM" in name or "HBM3" in name else 756e12
    if "A100" in name:
        return 312e12
    if "L40S" in name:
        return 362e12
    if "A10G" in name:
        return 125e12
    return 312e12


def _environment(device: str) -> dict:
    """One-off run fingerprint - hardware, versions and Modal identifiers - so the
    run's conditions are reconstructable from the log alone."""
    import torch

    env = {
        "event": "environment",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "n_params": N_PARAMS,
        "modal_task_id": os.environ.get("MODAL_TASK_ID", ""),
        "modal_environment": os.environ.get("MODAL_ENVIRONMENT", ""),
        "cpu_count": os.cpu_count(),
    }
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu_name": torch.cuda.get_device_name(0),
                    "gpu_total_gib": p.total_memory / 2**30,
                    "gpu_capability": f"{p.major}.{p.minor}",
                    "gpu_multi_processor_count": p.multi_processor_count,
                    "peak_bf16_tflops": _peak_flops(device) / 1e12})
        env.update(_nvidia_smi())
    return env


def _atomic_save(obj: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    if path.exists():
        prev = path.with_name("ckpt_prev.pt")
        os.replace(path, prev)
    os.replace(tmp, path)


def _checkpoint_path(ckpt_dir: str | os.PathLike[str]) -> Path:
    return Path(ckpt_dir) / "ckpt.pt"


def _load_checkpoint(model, optimizer, ckpt_dir: str | os.PathLike[str], device: str) -> dict | None:
    import torch

    path = _checkpoint_path(ckpt_dir)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # RNG states MUST be restored as CPU uint8 (ByteTensor). The checkpoint above is
    # loaded with map_location=device, so on a GPU box these tensors come back on CUDA;
    # torch.set_rng_state / torch.cuda.set_rng_state then reject them with
    # "RNG state must be a torch.ByteTensor". This crashed the preemption-resume path
    # once (retry after a spot reclaim died here instead of continuing). Coerce to a
    # CPU byte tensor so resume works regardless of where the checkpoint was mapped.
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu", torch.uint8))
    if device == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].to("cpu", torch.uint8))
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    return checkpoint


def _save_checkpoint(
    *,
    model,
    optimizer,
    ckpt_dir: str | os.PathLike[str],
    step: int,
    epoch: int,
    tokens_seen: int,
    total_steps: int,
    commit_fn: CommitFn,
) -> None:
    import torch

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "model_config": asdict(config.MODEL),
        "train_config": asdict(config.TRAIN),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    _atomic_save(payload, _checkpoint_path(ckpt_dir))
    if commit_fn is not None:
        commit_fn()


def _append_metric(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_loss(model, dataset: BinWindowDataset, batch_size: int, device: str, max_batches: int | None = None) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    total_batches = math.ceil(len(dataset) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    with torch.no_grad():
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            stop = min(len(dataset), start + batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            input_ids = dataset.get_batch(indices, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids, labels=input_ids).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


FORGET_SOURCES = ("case-law", "sec", "fineweb-edu")


def _build_forget_datasets(forget_val_root: str, seq_len: int) -> dict:
    """Datasets for measuring catastrophic forgetting.

    These are v1's ORIGINAL held-out val bins. Because v1 used a 99/1 split, neither v1
    nor the continued model ever trained on them, so comparing v1 vs v1-extended here is
    a fair within-family measurement (identical tokenizer, identical data).

    Split per source: the continuation mix is ~75% web / ~25% legal and contains NO sec
    at all, so sec is the domain most exposed to forgetting -- and the one v1 currently
    wins on. An aggregate loss would blur exactly the signal we care about.
    """
    sets: dict[str, BinWindowDataset] = {}
    for src in FORGET_SOURCES:
        try:
            sets[src] = BinWindowDataset(forget_val_root, "val", seq_len, source_prefix=src)
        except FileNotFoundError:
            print(f"  [forget] no bins for source {src!r} - skipping", flush=True)
    if not sets:
        raise FileNotFoundError(f"no forgetting-eval bins found under {forget_val_root}")
    return sets


def _eval_forgetting(model, sets: dict, batch_size: int, device: str, max_batches: int) -> dict:
    """Per-source val loss on v1's original data. Deterministic (always the first N
    batches) so values are directly comparable across steps."""
    out = {}
    for src, ds in sets.items():
        out[f"forget_loss_{src}"] = evaluate_loss(model, ds, batch_size, device, max_batches)
    vals = list(out.values())
    out["forget_loss_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def run(
    *,
    data_root: str,
    ckpt_dir: str,
    init_from_hf: str = "",
    init_from_ckpt: str = "",
    epochs: int = 1,
    lr_mode: str = "scratch",
    continue_lr: float = 1e-4,
    commit_fn: CommitFn = None,
    resume: bool = True,
    eval_batches: int = 50,
    forget_val_root: str = "",
    eval_every: int = 0,
    micro_batch_size: int = 0,      # 0 = use config.TRAIN.micro_batch_size
    compile_model: bool = False,    # torch.compile for throughput
) -> None:
    import torch

    if init_from_hf and init_from_ckpt:
        raise ValueError("set either init_from_hf or init_from_ckpt, not both")

    train_cfg = config.TRAIN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = BinWindowDataset(data_root, "train", train_cfg.seq_len)
    val_data = BinWindowDataset(data_root, "val", train_cfg.seq_len)
    windows_per_step = train_cfg.global_batch_tokens // train_cfg.seq_len
    # MFU fix: micro_batch is a pure throughput knob. v1 ran 32 and used only 26.8 of
    # 40 GB -- a third of the card idle at 27% MFU. Raising it while shrinking
    # grad_accum keeps the GLOBAL batch (and therefore the maths) identical.
    micro_bs = micro_batch_size or train_cfg.micro_batch_size
    grad_accum = windows_per_step // micro_bs
    if windows_per_step % micro_bs != 0:
        raise ValueError(
            f"micro_batch_size ({micro_bs}) must evenly divide windows per global step "
            f"({windows_per_step}) so the global batch is unchanged")

    steps_per_epoch = len(train_data) // windows_per_step
    total_steps = steps_per_epoch * epochs
    model = _build_model(device, attn_implementation="sdpa")
    # torch.compile returns a WRAPPER whose state_dict() keys are prefixed
    # "_orig_mod.". Checkpoints and the HF export must use the UNCOMPILED module or
    # they become unloadable, so keep a reference and do all state_dict work on it.
    base_model = model
    model = _maybe_compile(model, compile_model)
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        fused=(device == "cuda"),
    )

    start_step = 0
    tokens_seen = 0
    checkpoint = _load_checkpoint(base_model, optimizer, ckpt_dir, device) if resume else None
    if checkpoint is not None:
        # Mid-run recovery (preemption/timeout): restore model + optimizer + step.
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        print(f"resumed from step {start_step:,} with {tokens_seen:,} tokens seen", flush=True)
    elif init_from_hf:
        # Continue from a model published on the HuggingFace Hub rather than from a
        # checkpoint on a Modal Volume. Needed when training runs under a DIFFERENT
        # Modal account than the one holding the original volume: the Hub copy is
        # account-independent, so this avoids shuttling a 720 MiB checkpoint between
        # accounts. Weights only - optimizer and step counter are reset, exactly as
        # in the init_from_ckpt path.
        from transformers import AutoModelForCausalLM

        src = AutoModelForCausalLM.from_pretrained(init_from_hf, torch_dtype=torch.float32)
        missing, unexpected = base_model.load_state_dict(src.state_dict(), strict=True), None
        n = sum(p.numel() for p in src.parameters())
        print(f"initialized weights from HF repo {init_from_hf} ({n:,} params); "
              f"optimizer + step counter reset", flush=True)
        del src
    elif init_from_ckpt:
        # Continued pretraining: load ONLY the weights from a base checkpoint and start
        # a fresh schedule over the new data. Optimizer state and the step counter are
        # deliberately reset — the base run's Adam moments and step index describe a
        # finished cosine schedule over different data, so carrying them over would both
        # corrupt the new schedule and (since base step 3889 > this run's total_steps)
        # make the training loop a no-op.
        base = torch.load(init_from_ckpt, map_location=device)
        base_model.load_state_dict(base["model"], strict=True)
        print(f"initialized weights from {init_from_ckpt} "
              f"(base step {base.get('step')}, {base.get('tokens_seen', 0):,} tokens seen); "
              f"optimizer + step counter reset", flush=True)
        del base

    metrics_path = Path(ckpt_dir) / "metrics.jsonl"
    started = time.time()

    eval_interval = eval_every if eval_every > 0 else train_cfg.eval_every_steps
    peak_flops = _peak_flops(device)
    env = _environment(device)
    env.update({"total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                "epochs": epochs, "lr_mode": lr_mode, "peak_lr": continue_lr,
                "micro_batch": micro_bs, "grad_accum": grad_accum,
                "global_batch_tokens": train_cfg.global_batch_tokens,
                "train_windows": len(train_data), "val_windows": len(val_data),
                "compiled": bool(compile_model)})
    _append_metric(metrics_path, env)
    print(json.dumps(env, sort_keys=True, default=str), flush=True)

    # Catastrophic-forgetting probe on v1's ORIGINAL held-out data.
    forget_sets = _build_forget_datasets(forget_val_root, train_cfg.seq_len) if forget_val_root else {}

    # STEP-0 BASELINE, recorded before a single weight update. Without this the
    # forgetting curve has no true zero and any later drift is uninterpretable.
    if start_step == 0:
        baseline = {
            "event": "eval",
            "step": 0,
            "baseline": True,
            "val_loss": evaluate_loss(model, val_data, micro_bs, device,
                                      max_batches=eval_batches),
            "tokens_seen": 0,
            "wall_time": time.time() - started,
        }
        if forget_sets:
            baseline.update(_eval_forgetting(model, forget_sets, micro_bs,
                                             device, eval_batches))
        baseline["perplexity"] = math.exp(min(20.0, baseline["val_loss"]))
        _append_metric(metrics_path, baseline)
        print(json.dumps(baseline, sort_keys=True), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    current_epoch = -1
    permutation: np.ndarray | None = None
    for global_step in range(start_step, total_steps):
        epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch
        if epoch != current_epoch or permutation is None:
            epoch_rng = np.random.default_rng(train_cfg.seed + epoch)
            permutation = epoch_rng.permutation(len(train_data))
            current_epoch = epoch
        batch_indices = permutation[
            step_in_epoch * windows_per_step : (step_in_epoch + 1) * windows_per_step
        ]

        lr = _lr_at_step(global_step, total_steps, lr_mode, continue_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        step_start = time.time()
        for accum_idx in range(grad_accum):
            micro_indices = batch_indices[
                accum_idx * micro_bs : (accum_idx + 1) * micro_bs
            ]
            input_ids = train_data.get_batch(micro_indices, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids, labels=input_ids).loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu()) * grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += train_cfg.global_batch_tokens
        completed_step = global_step + 1

        if completed_step % train_cfg.log_every_steps == 0 or completed_step == 1:
            elapsed = time.time() - started
            step_time = max(1e-6, time.time() - step_start)
            row = {
                "step": completed_step,
                "lr": lr,
                "train_loss": step_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_seen": tokens_seen,
                "tokens_per_sec": train_cfg.global_batch_tokens / step_time,
                "step_time_s": step_time,
                "wall_time": elapsed,
                "epoch": epoch,
                "micro_batch": micro_bs,
                "grad_accum": grad_accum,
            }
            tps = train_cfg.global_batch_tokens / step_time
            row["mfu_percent"] = _mfu(tps, peak_flops)
            row.update(_nvidia_smi())
            row.update(_torch_mem(device))
            _append_metric(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        should_eval = completed_step % eval_interval == 0 or completed_step == total_steps
        if should_eval:
            val_loss = evaluate_loss(model, val_data, micro_bs, device, max_batches=eval_batches)
            row = {
                "event": "eval",
                "step": completed_step,
                "val_loss": val_loss,
                "perplexity": math.exp(min(20.0, val_loss)),
                "tokens_seen": tokens_seen,
                "wall_time": time.time() - started,
            }
            if forget_sets:
                # per-source loss on v1's original held-out data -> the forgetting curve
                row.update(_eval_forgetting(model, forget_sets, micro_bs,
                                            device, eval_batches))
            row.update(_nvidia_smi())
            _append_metric(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        should_ckpt = completed_step % train_cfg.ckpt_every_steps == 0 or completed_step == total_steps
        if should_ckpt:
            _save_checkpoint(
                model=base_model,
                optimizer=optimizer,
                ckpt_dir=ckpt_dir,
                step=completed_step,
                epoch=epoch,
                tokens_seen=tokens_seen,
                total_steps=total_steps,
                commit_fn=commit_fn,
            )
            print(f"checkpoint saved at step {completed_step:,}", flush=True)
