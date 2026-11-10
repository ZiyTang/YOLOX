"""Fine-tune the exhibition model using the existing local YOLOX Trainer."""

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"), default="yolox-l")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "dataset_iee_exhibition")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "YOLOX_outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, choices=(8, 4, 2), default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    model_name = args.variant.replace("-", "_") + "_iee_expo"
    if args.checkpoint is None:
        args.checkpoint = ROOT.parent / "YOLOX_weights" / (args.variant.replace("-", "_") + ".pth")
    sys.path.insert(0, str(ROOT))
    import yolox
    if Path(yolox.__file__).resolve() != ROOT / "yolox/__init__.py":
        raise RuntimeError(f"Another YOLOX copy was imported: {yolox.__file__}")
    import numpy as np
    import torch
    import torchvision
    from yolox.exp import get_exp
    from yolox.utils import configure_module

    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA PyTorch build is required; CPU training is not used.")
    if torch.__version__.split("+")[0] != "2.2.2" or torch.version.cuda != "11.8":
        raise RuntimeError("Use the configured PyTorch 2.2.2 CUDA 11.8 build.")
    if args.epochs < 20:
        raise ValueError("Use at least 20 epochs to retain the 5-epoch warmup and final 15 no-augmentation epochs.")
    data = json.loads((args.data_dir / "annotations/instances_iee_exhibition.json").read_text(encoding="utf-8"))
    if [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])] != ["hole", "groove"]:
        raise RuntimeError("The exhibition dataset must contain hole and groove in that order.")
    if len(data["images"]) != 268 or len(data["annotations"]) != 170:
        raise RuntimeError("The reviewed 268-image, 170-box exhibition dataset was expected.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    configure_module()
    torch.cuda.set_device(0)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    run = args.output_dir.resolve() / datetime.now().strftime("iee_expo_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    convolution = torch.nn.Conv2d(3, 8, 3).cuda()
    sample = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    convolution(sample).square().mean().backward()
    assert torch.isfinite(sample.grad).all()
    keep = torchvision.ops.nms(torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], device="cuda", dtype=torch.float32),
                               torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    assert keep.tolist() == [0]
    del convolution, sample, keep
    torch.cuda.synchronize()
    write_json(run / "gpu_check.json", {"torch": torch.__version__, "torchvision": torchvision.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "convolution_backward": True, "gpu_nms": True})
    exp_path = ROOT / "exps/example/custom" / (model_name + ".py")

    def make_trainer(batch, name, resume=False):
        exp = get_exp(str(exp_path), None)
        exp.data_dir = str(args.data_dir.resolve())
        exp.output_dir = str(run)
        exp.max_epoch = args.epochs
        options = argparse.Namespace(fp16=True, batch_size=batch, experiment_name=name,
                    resume=resume, ckpt=str((args.resume if resume else args.checkpoint).resolve()),
                    start_epoch=None, cache=None, occupy=False, logger="tensorboard")
        return exp.get_trainer(options)

    chosen = None
    for batch in (b for b in (8, 4, 2) if b <= args.batch_size):
        trainer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            trainer = make_trainer(batch, f"smoke_batch_{batch}")
            trainer.before_train()
            trainer.epoch = 0
            trainer.before_epoch()
            start = time.perf_counter()
            for iteration in range(3):
                trainer.iter = iteration
                trainer.train_one_iter()
                if not torch.isfinite(torch.tensor(float(trainer.meter["total_loss"].latest))):
                    raise RuntimeError("GPU smoke test produced a non-finite loss.")
            torch.cuda.synchronize()
            smoke = {"result": "PASS", "batch_size": batch, "iterations": 3,
                     "elapsed_seconds": time.perf_counter() - start,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2}
            write_json(run / "smoke.json", smoke)
            chosen = batch
            input_size = list(trainer.exp.input_size)
            fp32_no_aug = getattr(trainer.exp, "fp32_no_aug", False)
        except torch.cuda.OutOfMemoryError:
            write_json(run / f"smoke_batch_{batch}_oom.json", {"batch_size": batch, "result": "OUT_OF_MEMORY"})
        finally:
            if trainer is not None:
                trainer.after_train()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError(f"{args.variant.upper()} did not fit even at batch size 2.")
    configuration = {"yolox_root": str(ROOT), "data_dir": str(args.data_dir.resolve()),
                     "pretrained_checkpoint": str(args.checkpoint.resolve()), "epochs": args.epochs,
                     "batch_size": chosen, "input_size": input_size, "fp16": True, "ema": True,
                     "fp32_no_aug": fp32_no_aug,
                     "validation_split": None, "classes": ["hole", "groove"], "seed": 42}
    write_json(run / "training_config.json", configuration)
    if args.smoke_only:
        print(json.dumps({"run": str(run), "smoke": smoke}))
        return
    trainer = make_trainer(chosen, model_name, resume=args.resume is not None)
    trainer.train()
    final = Path(trainer.file_name) / (model_name + ".pth")
    if not final.is_file() or trainer.completed_epochs != args.epochs:
        raise RuntimeError("Training did not complete; final weights are unavailable.")
    write_json(run / "completed.json", {"epochs": trainer.completed_epochs, "checkpoint": str(final),
                                       "weights": "final_ema", "validation_performed": False})
    print(json.dumps({"run": str(run), "checkpoint": str(final), "epochs": trainer.completed_epochs}))


if __name__ == "__main__":
    main()
