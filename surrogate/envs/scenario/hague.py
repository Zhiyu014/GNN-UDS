from .base import basescenario
import os
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import networkx as nx
from itertools import combinations
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
    1. Minimization of system flooding
    2. Avoid flooding at pond st1
    3. Avoid large depths at pond F134101 which may cause upstream flooding
    4. Depth targets for pond st1 & F134101
    5. Minimization of TSS (not included here)

    Performance is measured as the following:
    1. 1 for system flooding volume, depth targets
    2. 1000 for st1 flooding, F134101 large depth

    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","hague.yaml") \
            if config_file is None else config_file
        super().__init__(config_file,swmm_file,global_state,initialize)

        
    # TODO
    def objective(self, seq = False):
        # __object = np.zeros(seq) if seq else 0.0
        # __object += self.flood(seq).squeeze().sum(axis=-1)
        __object = []
        perfs = self.performance(seq if seq else 1)
        for i,(_,attr,target,weight) in enumerate(self.config['performance_targets']):
            if attr == 'head':
                if weight == 1000:
                    __object += [(perfs[:,i]>target)*weight]
                else:
                    __object += [np.abs(perfs[:,i] - target)*weight]
            else:
                __object += [(perfs[:,i] - target)*weight]
        # return __object
        return np.array(__object).sum(axis=-1) if seq else np.array(__object).squeeze()
     
    def objective_pred(self,preds,states,settings,gamma=None,keepdim=False):
        preds,edge_preds = preds
        h,q_w,fl = preds[...,0],preds[...,-1],edge_preds[...,-1]
        nodes,links,targets = self.elements['nodes'],self.elements['links'],self.config['performance_targets']
        flood = [q_w.sum(axis=-1) * weight
                for idx,attr,_,weight in targets 
                if attr == 'cumflooding' and idx=='nodes']
        pondfl = [q_w[...,nodes.index(idx)] * weight
                  for idx,attr,_,weight in targets
                  if attr == 'cumflooding' and idx in nodes]
        outflow = [fl[...,links.index(idx)] * weight
                  for idx,attr,_,weight in targets
                  if attr == 'flow_vol' and idx in links]
        depth = [np.abs(h[...,nodes.index(idx)]-target) * weight
                for idx,attr,target,weight in targets
                if attr == 'head' and weight < 1000]
        exced = [(h[...,nodes.index(idx)]>target)*weight
                for idx,attr,target,weight in targets
                if attr == 'head' and weight == 1000]
        obj = np.stack(flood + pondfl + outflow + depth + exced,axis=1)
        gamma = np.ones(preds.shape[1]) if gamma is None else np.array(gamma,dtype=np.float32)
        obj *= gamma
        return obj.sum(axis=-1) if not keepdim else np.transpose(obj,(0,2,1))
    
    def objective_pred_tf(self,preds,states,settings,gamma=None):
        import tensorflow as tf
        preds,edge_preds = preds
        h,q_w,fl = preds[...,0],preds[...,-1],edge_preds[...,-1]
        nodes,links,targets = self.elements['nodes'],self.elements['links'],self.config['performance_targets']
        # flood = [tf.reduce_sum(q_w,axis=-1) * weight
        #         for idx,attr,_,weight in targets
        #           if attr == 'cumflooding' and idx=='nodes']
        pondfl = [q_w[...,nodes.index(idx)] * weight
                for idx,attr,_,weight in targets
                  if attr == 'cumflooding' and idx in nodes]
        outflow = [fl[...,links.index(idx)] * weight
                   for idx,attr,_,weight in targets
                     if attr == 'flow_vol' and idx in links]
        # depth = [tf.abs(h[...,nodes.index(idx)]-target) * weight
        #         for idx,attr,target,weight in targets
        #           if attr == 'head' and weight < 1000]
        # exced = [tf.cast(h[...,nodes.index(idx)]>target,tf.float32) * weight
        #         for idx,attr,target,weight in targets
        #           if attr == 'head' and weight == 1000]
        # obj = tf.reduce_sum(flood,axis=0) + tf.reduce_sum(pondfl,axis=0)+ tf.reduce_sum(outflow,axis=0) + tf.reduce_sum(depth,axis=0) + tf.reduce_sum(exced,axis=0)
        obj = tf.reduce_sum(pondfl,axis=0) + tf.reduce_sum(outflow,axis=0)
        gamma = tf.ones((preds.shape[1],)) if gamma is None else tf.convert_to_tensor(gamma,dtype=tf.float32)
        obj = tf.reduce_sum(obj*gamma,axis=-1)
        return obj

    def get_action_space(self,act='rand'):
        asp = self.config['action_space'].copy()
        return asp

    def get_args(self,directed=False,length=0,order=1,graph_base=0,act=False):
        args = super().get_args(directed,length,order,graph_base)

        # Rainfall timeseries & events files
        if not os.path.isfile(args['rainfall']['rainfall_timeseries']):
            args['rainfall']['rainfall_timeseries'] = os.path.join(HERE,'config',args['rainfall']['rainfall_timeseries']+'.csv')
        if not os.path.isfile(args['rainfall']['rainfall_events']):
            args['rainfall']['rainfall_events'] = os.path.join(HERE,'config',args['rainfall']['rainfall_events']+'.csv')
        # if not os.path.isfile(args['rainfall']['training_events']):
        #     args['rainfall']['training_events'] = os.path.join(HERE,'config',args['rainfall']['training_events']+'.csv')

        inp = read_inp_file(self.config['swmm_input'])
        args['area'] = np.array([node.Curve[0] if sec == 'STORAGE' else 0.0
                                  for sec in NODE_SECTIONS for node in getattr(inp,sec,dict()).values()])
        args['pump'] = np.array([inp['CURVES'][link.Curve].points[0][1]*60/1000 if sec == 'PUMPS' else 0.0
                                  for sec in LINK_SECTIONS if sec in inp for link in getattr(inp,sec,dict()).values()])
        args['offset'] = np.array([getattr(link,'Offset',0)+getattr(link,'InOffset',0)
                   for sec in LINK_SECTIONS if sec in inp for link in getattr(inp,sec,dict()).values()])

        if self.global_state:
            args['act_edges'] = self.get_edge_list(list(self.config['action_space'].keys()))
        return args

    def controller(self,mode='rand',state=None,setting=None):
        asp = self.config['action_space']
        if mode.lower() == 'rand' or mode.lower() == 'default':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        elif mode.lower() == 'off':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'half':
            return [table[1] for table in asp.values()]
        elif mode.lower() == 'safe':
            return setting
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
    