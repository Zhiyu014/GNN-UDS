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
        # __object = np.zeros(seq) if seq else 0.0
        __object = []
        perfs = self.performance(seq = max(seq,1) + 1 if seq else 2)
        for i,(idx,attr,weight) in enumerate(self.config['performance_targets']):
            if attr == 'cuminflow' and idx != 'Out_to_WWTP':
                # __object += np.abs(np.diff(perfs[:,i],axis=0)) * weight
                __object += [np.abs(np.diff(perfs[:,i],axis=0)).squeeze() * weight]
            else:
                # __object += perfs[1:,i] * weight
                __object += [perfs[1:,i].squeeze() * weight]
        # return __object
        return np.array(__object).sum(axis=-1) if seq else np.array(__object)
         
    def objective_pred(self,preds,states,settings,gamma=None,norm=False,keepdim=False):
        preds,_ = preds
        state,_ = states
        q_w = preds[...,-1]
        q_in = np.concatenate([state[:,-1:,:,1],preds[...,1]],axis=1)
        flood = [q_w[...,self.elements['nodes'].index(idx)] * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        inflow = [np.abs(np.diff(q_in[...,self.elements['nodes'].index(idx)],axis=1)) * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' not in idx]
        outflow = [q_in[:,1:,self.elements['nodes'].index(idx)] * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' in idx]
        obj = np.stack(flood + outflow + inflow,axis=1)
        gamma = np.ones(preds.shape[1]) if gamma is None else np.array(gamma,dtype=np.float32)
        obj = (obj*gamma).sum(axis=-1) if not keepdim else np.transpose(obj*gamma,(0,2,1))
        if norm:
            __norm = (state[...,-1].sum(axis=-1).sum(axis=-1)+1e-5)[:,None]
            obj /= __norm if not keepdim else __norm[:,None]
        return obj
    
    def objective_pred_tf(self,preds,states,settings,gamma=None,norm=False):
        import tensorflow as tf
        preds,_ = preds
        state,_ = states
        q_w = preds[...,-1]
        q_in = tf.concat([state[:,-1:,:,1],preds[...,1]],axis=1)
        flood = [q_w[...,self.elements['nodes'].index(idx)] * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        # inflow = [tf.abs(tf.reduce_sum(preds[...,self.elements['nodes'].index(idx),1],axis=-1,keepdims=True)-\
        #                  tf.reduce_sum(state[...,self.elements['nodes'].index(idx),1],axis=-1,keepdims=True)) * weight
        #                  for idx,attr,weight in self.config['performance_targets']
        #                  if attr == 'cuminflow' and 'WWTP' not in idx]
        # inflow = [tf.repeat(inf,preds.shape[1],axis=-1)/preds.shape[1] for inf in inflow]
        # inflow = [tf.abs(tf.experimental.numpy.diff(q_in[...,self.elements['nodes'].index(idx)],axis=1)) * weight
        #         for idx,attr,weight in self.config['performance_targets']
        #             if attr == 'cuminflow' and 'WWTP' not in idx]
        outflow = [q_in[:,1:,self.elements['nodes'].index(idx)] * weight
                for idx,attr,weight in self.config['performance_targets']
                    if attr == 'cuminflow' and 'WWTP' in idx]
        obj = tf.reduce_sum(flood,axis=0) + tf.reduce_sum(outflow,axis=0)
        gamma = tf.ones((preds.shape[1],)) if gamma is None else tf.convert_to_tensor(gamma,dtype=tf.float32)
        obj = tf.reduce_sum(obj*gamma,axis=-1)
        if norm:
            obj /= (tf.reduce_sum(tf.reduce_sum(state[...,-1],axis=-1),axis=-1)+1e-5)
        return obj

    def get_obj_norm(self,norm_y,norm_e=None,perfs=None):
        nodes = self.elements['nodes']
        targets = self.config['performance_targets']
        fl = [norm_y[...,nodes.index(idx),-1] * weight
              for idx,attr,weight in targets if attr == 'cumflooding']
        infl = [norm_y[...,nodes.index(idx),1] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and 'WWTP' not in idx]
        outfl = [norm_y[...,nodes.index(idx),1] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and 'WWTP' in idx]
        return np.stack(fl + outfl + infl,axis=-1)

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

    def get_args(self,directed=False,length=0,order=1,graph_base=0,act=False,dec=False):
        args = super().get_args(directed,length,order,graph_base)

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
                args['action_shape'] = np.array(list(args['action_table'].keys())).max(axis=0)+1
            else:
                args['action_shape'] = len(args['action_space'])
            # For multi-agent
            if dec:
                args['n_agents'] = len(self.config['site'])
                state = [s[0] for s in self.config['states']]
                args['observ_space'] = [[state.index(o) for o in v['states']]
                                        for v in self.config['site'].values()]
                # args['action_shape'] = np.array(list(args['action_table'].keys())).max(axis=0)+1
            else:
                args['n_agents'] = 1
                args['observ_space'] = self.config['states']
                # args['action_shape'] = len(args['action_table'])
        return args

    def controller(self,mode='rand',state=None,setting=None):
        asp = self.config['action_space']
        asp3 = {k:[v[ac] for ac in ac3]
                 for (k,v),ac3 in zip(asp.items(),[[0,1,-1],[0,3,-1],[0,1,-1],[0,2,-1]])}
        if mode.lower() == 'rand3':
            return [table[np.random.randint(0,3)] for table in asp3.values()]
        elif mode.lower().startswith('rand'):
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'bc' or mode.lower() == 'default':
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
        elif mode.lower() == 'safe':
            return setting
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
        
