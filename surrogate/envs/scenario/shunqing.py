from .base import basescenario
import os

HERE = os.path.dirname(__file__)

class shunqing(basescenario):
    r"""Shunqing Scenario

    [ga_ann_for_uds]
    Code: https://github.com/lhmygis/ga_ann_for_uds
    Paper: https://doi.org/10.1016/j.envsoft.2023.105623

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    Shunqing District, Nanchong City, Sichuan Province, China
    Annual precipitation: 1020.8 mm
    Area: 33.02 km2
    131 pipes, 105 manholes, 8 outfalls, 106 subcatchments
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","shunqing.yaml") \
            if config_file is None else config_file
        super.__init__(config_file,swmm_file,global_state,initialize)

