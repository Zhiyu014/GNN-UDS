from .base import basescenario
import os
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.section_lists import NODE_SECTIONS
from itertools import product

HERE = os.path.dirname(__file__)

class astlingen(basescenario):
    r"""Astlingen Scenario

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    Objectives are the following:
    1. Minimization of accumulated CSO volume
    2. Minimization of CSO to the creek more than the river
    3. Maximizing flow to the WWTP
    4. Minimizing roughness of control.

    Performance is measured as the following:
    1. *2 for CSO volume (doubled for flow into the creek)
    2. *(-1) for the flow to the WWTP
    3. *(0.01) for the roughness of the control

    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","astlingen.yaml") \
            if config_file is None else config_file
        super().__init__(config_file,swmm_file,global_state,initialize)
        
    def objective(self, seq = False):
        __object = np.zeros(seq) if seq else 0.0
        perfs = self.performance(seq = max(seq,1) + 1 if seq else 2)
        for i,(idx,attr,weight) in enumerate(self.config['performance_targets']):
            if attr == 'cuminflow' and idx != 'Out_to_WWTP':
                __object += np.abs(np.diff(perfs[:,i],axis=0)) * weight
            else:
                __object += perfs[1:,i] * weight
        return __object
     
    def reward(self,norm=False):
        # Calculate the target error in the recent step based on cumulative values
        __reward = 0.0
        __sumnorm = 0.0
        for ID, attribute, weight in self.config["reward"]:
            if self.env._isFinished:
                __cumvolume = self.data_log[attribute][ID][-1]
            else:
                __cumvolume = self.env.methods[attribute](ID)
            # Recent volume has been logged
            if len(self.data_log[attribute][ID]) > 1:
                __volume = __cumvolume - self.data_log[attribute][ID][-2]
            else:
                __volume = __cumvolume

            if attribute == "totalinflow" and ID not in ["Out_to_WWTP","system"]:
                if len(self.data_log[attribute][ID]) > 2:
                    __prevolume = self.data_log[attribute][ID][-2] - self.data_log[attribute][ID][-3]
                elif len(self.data_log[attribute][ID]) == 2:
                    __prevolume = self.data_log[attribute][ID][-2]
                else:
                    __prevolume = 0
                __volume = abs(__volume - __prevolume)
            # __weight = self.penalty_weight[ID]
            if ID == 'system':
                __sumnorm += __volume * weight
            else:
                __reward += __volume * weight
        if norm:
            return - __reward/(__sumnorm + 1e-5)
        else:
            return - __reward

    def objective_pred(self,preds,state):
        preds,_ = preds
        q_w = preds[...,-1]
        q_in = np.concatenate([state[:,-1:,:,1],preds[...,1]],axis=1)
        flood = [q_w[...,self.elements['nodes'].index(idx)].sum(axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        inflow = [np.abs(np.diff(q_in[...,self.elements['nodes'].index(idx)],axis=1)).sum(axis=1) * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' not in idx]
        outflow = [q_in[:,1:,self.elements['nodes'].index(idx)].sum(axis=1) * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' in idx]
        return sum(flood) + sum(inflow) + sum(outflow)
    
    def objective_pred_tf(self,preds,state):
        import tensorflow as tf
        preds,_ = preds
        q_w = preds[...,-1]
        q_in = tf.concat([state[:,-1:,:,1],preds[...,1]],axis=1)
        flood = [tf.reduce_sum(q_w[...,self.elements['nodes'].index(idx)],axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        inflow = [tf.reduce_sum(tf.abs(tf.experimental.numpy.diff(q_in[...,self.elements['nodes'].index(idx)],axis=1)),axis=1) * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' not in idx]
        outflow = [tf.reduce_sum(q_in[:,1:,self.elements['nodes'].index(idx)],axis=1) * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' in idx]
        return tf.reduce_sum(flood,axis=0) + tf.reduce_sum(inflow,axis=0) + tf.reduce_sum(outflow,axis=0)

    def get_action_space(self,act='rand'):
        asp = self.config['action_space'].copy()
        if '3' in act:
            asp = {k:tuple([v[ac] for ac in ac3]) 
            for (k,v),ac3 in zip(asp.items(),[[0,1,-1],[0,3,-1],[0,1,-1],[0,2,-1]])}
        return asp
    
    def get_action_table(self,act='rand'):
        asp = self.get_action_space(act)
        actions = product(*[range(len(v)) for v in asp.values()])
        actions = {act:[v[a] for a,v in zip(act,asp.values())] for act in actions}
        return actions

    def get_args(self,directed=False,length=0,order=1,act=False):
        args = super().get_args(directed,length,order)

        # Rainfall timeseries & events files
        if not os.path.isfile(args['rainfall']['rainfall_timeseries']):
            args['rainfall']['rainfall_timeseries'] = os.path.join(HERE,'config',args['rainfall']['rainfall_timeseries']+'.csv')
        if not os.path.isfile(args['rainfall']['rainfall_events']):
            args['rainfall']['rainfall_events'] = os.path.join(HERE,'config',args['rainfall']['rainfall_events']+'.csv')
        # if not os.path.isfile(args['rainfall']['training_events']):
        #     args['rainfall']['training_events'] = os.path.join(HERE,'config',args['rainfall']['training_events']+'.csv')

        inp = read_inp_file(self.config['swmm_input'])
        args['area'] = np.array([inp.CURVES[node.Curve].points[0][1] if sec == 'STORAGE' else 0.0
                                  for sec in NODE_SECTIONS for node in getattr(inp,sec,dict()).values()])

        if self.global_state:
            args['act_edges'] = self.get_edge_list(list(self.config['action_space'].keys()))
        if act and self.config['act']:
            args['action_space'] = self.get_action_space(act)
            if not act.startswith('conti'):
                args['action_table'] = self.get_action_table(act)
        return args

    def controller(self,mode='rand',state=None,settimg=None):
        asp = self.config['action_space']
        asp3 = {k:[v[ac] for ac in ac3]
                 for (k,v),ac3 in zip(asp.items(),[[0,1,-1],[0,3,-1],[0,1,-1],[0,2,-1]])}
        if mode.lower() == 'rand3':
            return [table[np.random.randint(0,3)] for table in asp3.values()]
        elif mode.lower().startswith('rand'):
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'bc':
            return [table[1] for table in asp3.values()]
        elif mode.lower() == 'efd':
            state_idxs = {k:self.elements['nodes'].index(k.replace('V','T')) for k in asp}
            depth = {k:state[idx,0] for k,idx in state_idxs.items()}
            setting = {k:1 for k in asp}
            if max(depth.values())<1:
                setting = {k:1 for k in asp}
            for k in asp:
                setting[k] = 2 * int(depth[k] >= max(depth.values())) +\
                    0 * int(depth[k] <= min(depth.values())) +\
                        1 * (1-int(depth[k] >= max(depth.values()))) * (1-int(depth[k] <= min(depth.values())))
            setting = [v[setting[k]] for k,v in asp3.items()]
            return setting
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
        
