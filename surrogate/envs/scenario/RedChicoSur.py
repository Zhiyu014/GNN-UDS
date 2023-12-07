from .base import basescenario
import os
import numpy as np
HERE = os.path.dirname(__file__)

class RedChicoSur(basescenario):
    r"""RedChicoSur Scenario


    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","RedChicoSur.yaml") \
            if config_file is None else config_file
        super().__init__(config_file,swmm_file,global_state,initialize)


    def get_args(self,directed=False,length=0,order=1,act=False):
        args = super().get_args(directed,length,order)
        if self.global_state:
            args['act_edges'] = self.get_edge_list(list(self.config['action_space'].keys()))
        return args

        
    def controller(self,mode='rand',state=None,setting=None):
        asp = self.config['action_space']
        if mode.lower() == 'rand':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower() == 'off':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        else:
            raise AssertionError("Unknown controller %s"%str(mode))