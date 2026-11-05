"""YOLOX-X shares the reviewed IEE-EXPO data and training policy with L."""

from yolox_l_iee_expo import Exp as ExhibitionExp


class Exp(ExhibitionExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_iee_expo"
        self.depth = 1.33
        self.width = 1.25
