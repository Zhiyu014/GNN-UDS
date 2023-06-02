# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from envs.environment import env_base
import pyswmm.toolkitapi as tkai

class env_ast(env_base):
    def __init__(self, config, ctrl=True, binary=None):
        super().__init__(config, ctrl, binary)

        # for state and performance
        self.methods.update({'rainfall':self._getGageRainfall,
                             'getlinktype':self._getLinkType})


    # ------ Get necessary Parameters  ----------------------------------------------
    def _getGageRainfall(self,ID):
        # For Cumrainfall state
        return self.sim._model.getGagePrecip(ID,
            tkai.RainGageResults.rainfall.value)

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name