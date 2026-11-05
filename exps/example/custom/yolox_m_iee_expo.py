"""YOLOX-M shares the reviewed IEE-EXPO data and training policy with L."""

from yolox_l_iee_expo import Exp as ExhibitionExp


class Exp(ExhibitionExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_m_iee_expo"
        self.depth = 0.67
        self.width = 0.75
