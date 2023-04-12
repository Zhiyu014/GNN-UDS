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
                             'cuminflow':self._getNodeCumInflow,
                             'totaloutflow':self._getNodeTotalOutflow,
                             'lateral_infow_vol':self._getNodeLateralInflowVol,
                             'rainfall':self._getGageRainfall})


    # ------ Get necessary Parameters  ----------------------------------------------

    # def _getNodeIdList(self):
    #     return self.sim._model.getObjectIDList(tkai.ObjectType.NODE.value)

    def _getFlooding(self,ID):
        # Flooding rate
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.overflow.value)

    def _getCumFlooding(self,ID):
        # Cumulative flooding volume
        if ID == "system":
            return self.sim._model.flow_routing_stats()['flooding']
        else:
            return self.sim._model.node_statistics(ID)['flooding_volume']
    
    def _getNodeTotalInflow(self,ID):
        # Inflow rate
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.totalinflow.value)
    
    def _getNodeCumInflow(self,ID):
        # Cumulative inflow volume
        return self.sim._model.node_inflow(ID)
                
    def _getNodeTotalOutflow(self,ID):
        # Outflow rate * step
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.outflow.value) * self.config['control_interval'] * 60
        
    def _getNodeLateralInflow(self,ID):
        # Lateral inflow rate
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.newLatFlow.value)

    def _getNodeLateralInflowVol(self,ID):
        # Cumulative lateral inflow volume
        return self.sim._model.node_statistics(ID)['lateral_infow_vol']

    def _getGageRainfall(self,ID):
        # For Cumrainfall state
        return self.sim._model.getGagePrecip(ID,
            tkai.RainGageResults.rainfall.value)
