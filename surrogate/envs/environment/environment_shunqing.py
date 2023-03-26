# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from .environment_base import env_base
import pyswmm.toolkitapi as tkai

class env_sq(env_base):
    def __init__(self, config, ctrl=False, binary=None):
        super().__init__(config, ctrl, binary)

        # for state and performance
        self.methods.update({'cumflooding':self._getCumFlooding,
                             'totalinflow':self._getNodeTotalInflow,
                             'totaloutflow':self._getNodeTotalOutflow,
                             'lateral_infow_vol':self._getNodeLateralinflowVol,
                             'rainfall':self._getGageRainfall})


    # ------ Get necessary Parameters  ----------------------------------------------
    def _getCumFlooding(self,ID):
        if ID == "system":
            return self.sim._model.flow_routing_stats()['flooding']
        else:
            return self.sim._model.node_statistics(ID)['flooding_volume']
    
    def _getNodeTotalInflow(self,ID):
        # Cumulative inflow volume
        return self.sim._model.node_inflow(ID)
        
    def _getNodeTotalOutflow(self,ID):
        # Cumulative inflow volume
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.outflow.value)
        
    def _getNodeLateralinflowVol(self,ID):
        # Cumulative lateral inflow volume
        return self.sim._model.node_statistics(ID)['lateral_infow_vol']

    def _getGageRainfall(self,ID):
        # For Cumrainfall state
        return self.sim._model.getGagePrecip(ID,
            tkai.RainGageResults.rainfall.value)
