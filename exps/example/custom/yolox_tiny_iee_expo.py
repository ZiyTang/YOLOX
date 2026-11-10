"""YOLOX-Tiny uses the reviewed exhibition training flow at 416x416."""

from yolox_l_iee_expo import Exp as ExhibitionExp


class Exp(ExhibitionExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_tiny_iee_expo"
        self.depth = 0.33
        self.width = 0.375
        self.input_size = self.test_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.fp32_no_aug = True
