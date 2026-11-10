"""YOLOX-S shares the reviewed IEE-EXPO data and training policy."""

from yolox_l_iee_expo import Exp as ExhibitionExp


class Exp(ExhibitionExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_s_iee_expo"
        self.depth = 0.33
        self.width = 0.5
        self.fp32_no_aug = True
