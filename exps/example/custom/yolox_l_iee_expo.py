"""Two-class YOLOX-L fine-tuning without a validation split."""

from pathlib import Path

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_l_iee_expo"
        self.num_classes = 2
        self.depth = self.width = 1.0
        self.data_dir = str(Path(__file__).resolve().parents[4] / "dataset_iee_exhibition")
        self.train_ann = "instances_iee_exhibition.json"
        self.input_size = self.test_size = (640, 640)
        self.multiscale_range = 0
        self.data_num_workers = 0
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.save_history_ckpt = False
        self.seed = 42

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, name="images",
                           img_size=self.input_size, preproc=TrainTransform(max_labels=50,
                           flip_prob=self.flip_prob, hsv_prob=self.hsv_prob), cache=cache, cache_type=cache_type)

    def get_evaluator(self, *args, **kwargs):
        return None

    def get_trainer(self, args):
        # Keep training-only dependencies out of the CPU export path.
        import json
        import math
        from loguru import logger
        import torch
        from yolox.core import Trainer

        class ExhibitionTrainer(Trainer):
            def __init__(self, exp, args):
                self.completed_epochs = 0
                super().__init__(exp, args)

            def resume_train(self, model):
                if not self.args.resume:
                    checkpoint = torch.load(self.args.ckpt, map_location="cpu")["model"]
                    current = model.state_dict()
                    expected = {f"head.cls_preds.{scale}.{kind}" for scale in range(3) for kind in ("weight", "bias")}
                    mismatched = {key for key in current if key in checkpoint and current[key].shape != checkpoint[key].shape}
                    if set(current) != set(checkpoint) or mismatched != expected:
                        raise RuntimeError(f"Unexpected pretraining mismatch: {sorted(mismatched)}")
                    report = {"checkpoint": str(Path(self.args.ckpt).resolve()), "mode": "fine_tune",
                              "loaded_tensors": len(current) - len(expected), "skipped_classification_tensors": sorted(expected)}
                    (Path(self.file_name) / "pretrained_load.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    del checkpoint
                return super().resume_train(model)

            def after_iter(self):
                if not math.isfinite(float(self.meter["total_loss"].latest)):
                    raise RuntimeError("Non-finite training loss; stopping without publishing final weights.")
                super().after_iter()

            def after_epoch(self):
                self.save_ckpt("latest")
                self.completed_epochs = self.epoch + 1
                progress = {"completed_epochs": self.completed_epochs, "total_epochs": self.max_epoch,
                            "batch_size": self.args.batch_size, "validation_performed": False}
                (Path(self.file_name) / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

            def after_train(self):
                if hasattr(self, "tblogger"):
                    self.tblogger.flush()
                    self.tblogger.close()
                if self.completed_epochs != self.max_epoch:
                    logger.info("Training incomplete; final exhibition weights were not published.")
                    return
                state = {key: value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()
                         for key, value in self.ema_model.ema.state_dict().items()}
                output = Path(self.file_name) / (self.exp.exp_name + ".pth")
                temporary = output.with_suffix(".pth.tmp")
                torch.save({"model": state, "epoch": self.completed_epochs, "num_classes": 2,
                            "class_names": ["hole", "groove"], "dataset": "iee_exhibition", "weights": "final_ema",
                            "validation_performed": False}, temporary)
                temporary.replace(output)
                logger.info("Completed {} epochs; final EMA saved to {}. No validation AP was measured.", self.completed_epochs, output)

        return ExhibitionTrainer(self, args)
