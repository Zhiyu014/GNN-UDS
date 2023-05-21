from envs.environment import env_base
import pyswmm.toolkitapi as tkai

class env_rcs(env_base):
    def __init__(self, config, ctrl=True, binary=None):
        super().__init__(config, ctrl, binary)

        # for state and performance
        self.methods.update({'totaloutflow':self._getNodeTotalOutflow,
                             'type':self._getLinkType,
                             'setting':self._getLinkSetting,
                             'flow':self._getLinkFlow,
                             })


    # ------ Get necessary Parameters  ----------------------------------------------
    def _getNodeTotalOutflow(self,ID):
        # Outflow rate * step
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.outflow.value) * self.config['interval'] * 60/1000

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name

    def _getLinkSetting(self,_linkid):
        return self.sim._model.getLinkResult(_linkid,
                                            tkai.LinkResults.setting.value)
    
    def _getLinkFlow(self,_linkid):
        return self.sim._model.getLinkResult(_linkid,
                                            tkai.LinkResults.newFlow.value) * self.config['interval'] * 60/1000
    