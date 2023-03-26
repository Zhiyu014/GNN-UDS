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
        self.methods.update({'cumflooding':self._getCumFlooding,
                             'totalinflow':self._getNodeTotalInflow,
                             'lateral_infow_vol':self._getNodeLateralinflowVol,
                             'rainfall':self._getGageRainfall,
                             'getlinktype':self._getLinkType,
                             'setting':self._getLinkSetting})


    # ------ Get necessary Parameters  ----------------------------------------------
    def _getCumFlooding(self,ID):
        if ID == "system":
            return self.sim._model.flow_routing_stats()['flooding']
        else:
            return self.sim._model.node_statistics(ID)['flooding_volume']
    
    def _getNodeTotalInflow(self,ID):
        # Cumulative inflow volume
        if ID == 'system':
            stats = self.sim._model.flow_routing_stats()
            return sum([v for k,v in stats.items() if k.endswith('inflow')])
        else:
            return self.sim._model.node_inflow(ID)

    def _getNodeLateralinflowVol(self,ID):
        # Cumulative lateral inflow volume
        return self.sim._model.node_statistics(ID)['lateral_infow_vol']

    def _getGageRainfall(self,ID):
        # For Cumrainfall state
        return self.sim._model.getGagePrecip(ID,
            tkai.RainGageResults.rainfall.value)

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name

    def _getLinkSetting(self,_linkid):
        return self.sim._model.getLinkResult(_linkid,
                                            tkai.LinkResults.setting.value)