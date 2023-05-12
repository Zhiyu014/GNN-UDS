from envs.environment import env_base
import pyswmm.toolkitapi as tkai

class env_rcs(env_base):
    def __init__(self, config, ctrl=True, binary=None):
        super().__init__(config, ctrl, binary)

        # for state and performance
        self.methods.update({'totaloutflow':self._getNodeTotalOutflow,
                             'getlinktype':self._getLinkType,
                             'setting':self._getLinkSetting})


    # ------ Get necessary Parameters  ----------------------------------------------
    def _getNodeTotalOutflow(self,ID):
        # Outflow rate * step
        return self.sim._model.getNodeResult(ID,tkai.NodeResults.outflow.value) * self.config['control_interval'] * 60/1000

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name

    def _getLinkSetting(self,_linkid):
        return self.sim._model.getLinkResult(_linkid,
                                            tkai.LinkResults.setting.value)