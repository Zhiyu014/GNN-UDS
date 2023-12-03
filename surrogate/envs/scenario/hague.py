from .base import basescenario
import os
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.section_lists import NODE_SECTIONS

HERE = os.path.dirname(__file__)

class hague(basescenario):
    r"""Hague Scenario

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
    1. Minimization of flooding
    2. Avoid flooding at ponds
    3. Minimization of TSS (not included here)

    Performance is measured as the following:
    1. *2 for CSO volume (doubled for flow into the creek)
    2. *(-1) for the flow to the WWTP
    3. *(0.01) for the roughness of the control

    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","hague.yaml") \
            if config_file is None else config_file
        super.__init__(config_file,swmm_file,global_state,initialize)

        
    # TODO
    def objective(self, seq = False):
        __object = np.zeros(seq) if seq else 0.0
        __object += self.flood(seq).squeeze().sum(axis=-1)
        perfs = self.performance(seq)
        for i,(_,attr,target) in enumerate(self.config['performance_targets']):
            if attr == 'depthN':
                __object += np.abs(perfs[:,i] if seq else perfs[i] - target)
            else:
                __object += perfs[:,i] * target if seq else perfs[i] * target
        return __object
     
     
    def objective_pred(self,preds,state):
        h,q_w = preds[...,0],preds[...,-1]
        flood = [q_w[...,self.elements['nodes'].index(idx)].sum(axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        depth = [np.abs(h[...,self.elements['nodes'].index(idx)]-target).sum(axis=1)
                for idx,attr,target in self.config['performance_targets'] if attr == 'depthN']
        return q_w.sum(axis=-1).sum(axis=-1) + sum(flood) + sum(depth)
    
    def objective_pred_tf(self,preds,state):
        import tensorflow as tf
        h,q_w = preds[...,0],preds[...,-1]
        flood = [tf.reduce_sum(q_w[...,self.elements['nodes'].index(idx)],axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        depth = [tf.reduce_sum(tf.abs(h[...,self.elements['nodes'].index(idx)]-target),axis=1)
                for idx,attr,target in self.config['performance_targets'] if attr == 'depthN']
        return tf.reduce_sum(tf.reduce_sum(q_w,axis=-1),axis=-1) + tf.reduce_sum(flood,axis=0) + tf.reduce_sum(depth,axis=0)    

    def get_action_space(self,act='rand'):
        asp = self.config['action_space'].copy()
        return asp

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
        return args


    # TODO: ctrl mode
    def controller(self,state=None,mode='rand'):
        asp = self.config['action_space']
        if mode.lower() == 'rand':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'off':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'half':
            return [table[1] for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
        
