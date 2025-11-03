from .base import basescenario
import os
import numpy as np
import torch as th
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

        
    def objective(self, seq = False):
        # __object = np.zeros(seq) if seq else 0.0
        # __object += self.flood(seq).squeeze().sum(axis=-1)
        __object = []
        perfs = self.performance(seq = max(seq,1) + 1 if seq else 2)
        for i,(_,attr,target,weight) in enumerate(self.config['performance_targets']):
            if attr == 'head':
                if weight == 1000:
                    __object += [(perfs[1:,i]>target)*weight]
                else:
                    __object += [np.abs(perfs[1:,i] - target)*weight]
            elif attr == 'flow_vol' and weight > 0:
                __object += [np.abs(np.diff(perfs[:,i],axis=0)) * weight]
            else:
                __object += [(perfs[1:,i] - target)*weight]
        # return __object
        return np.array(__object).sum(axis=-1) if seq else np.array(__object).squeeze()
     
    def objective_pred(self,preds,states,settings,gamma=None,keepdim=False):
        preds,edge_preds = preds
        _,edge_state = states
        h,q_w = preds[...,0],preds[...,-1]
        fl = np.concatenate([edge_state[:,-1:,:,-2],edge_preds[...,-1]],axis=1)
        nodes,links,targets = self.elements['nodes'],self.elements['links'],self.config['performance_targets']
        flood = [q_w.sum(axis=-1) * weight
                for idx,attr,_,weight in targets 
                if attr == 'cumflooding' and idx=='nodes']
        pondfl = [q_w[...,nodes.index(idx)] * weight
                  for idx,attr,_,weight in targets
                  if attr == 'cumflooding' and idx in nodes]
        outflow = [fl[:,1:,links.index(idx)] * weight
                  for idx,attr,_,weight in targets
                  if attr == 'flow_vol' and weight<0]
        # depth = [np.abs(h[...,nodes.index(idx)]-target) * weight
        #         for idx,attr,target,weight in targets
        #         if attr == 'head' and weight < 1000]
        # exced = [(h[...,nodes.index(idx)]>target)*weight
        #         for idx,attr,target,weight in targets
        #         if attr == 'head' and weight == 1000]
        # obj = np.stack(flood + pondfl + depth + exced,axis=1)
        inflow = [np.abs(np.diff(fl[...,links.index(idx)],axis=1)) * weight
                for idx,attr,_,weight in self.config['performance_targets']
                    if attr == 'flow_vol' and weight>0]
        obj = np.stack(flood + pondfl + outflow + inflow,axis=1)
        gamma = np.ones(preds.shape[1]) if gamma is None else np.array(gamma,dtype=np.float32)
        obj *= gamma
        return obj.sum(axis=-1) if not keepdim else np.transpose(obj,(0,2,1))
    
    def objective_pred_th(self,preds,states,settings,gamma=None,keepdim=False):
        preds,edge_preds = preds
        _,edge_state = states
        h,q_w = preds[...,0],preds[...,-1]
        fl = th.concat([edge_state[:,-1:,:,-2],edge_preds[...,-1]],dim=1)
        nodes,links,targets = self.elements['nodes'],self.elements['links'],self.config['performance_targets']
        flood = [q_w.sum(dim=-1) * weight
                for idx,attr,_,weight in targets
                  if attr == 'cumflooding' and idx=='nodes']
        pondfl = [q_w[...,nodes.index(idx)] * weight
                for idx,attr,_,weight in targets
                  if attr == 'cumflooding' and idx in nodes]
        outflow = [fl[:,1:,links.index(idx)] * weight
                   for idx,attr,_,weight in targets
                     if attr == 'flow_vol' and weight<0]
        # depth = [tf.abs(h[...,nodes.index(idx)]-target) * weight
        #         for idx,attr,target,weight in targets
        #           if attr == 'head' and weight < 1000]
        # exced = [tf.cast(h[...,nodes.index(idx)]>target,tf.float32) * weight
        #         for idx,attr,target,weight in targets
        #           if attr == 'head' and weight == 1000]
        inflow = [th.abs(th.diff(fl[...,links.index(idx)],dim=1)) * weight
                for idx,attr,_,weight in self.config['performance_targets']
                    if attr == 'flow_vol' and weight>0]
        obj = th.stack(flood + pondfl + outflow + inflow, dim=1)
        gamma = th.ones((preds.shape[1],)).to(device=obj.device) if gamma is None else th.tensor(gamma).to(device=obj.device)
        obj = (obj*gamma).sum(axis=-1) if not keepdim else (obj*gamma).permute(0,2,1)
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
        args['area'] = np.array([node.data[0] if sec == 'STORAGE' else 0.0
                                  for sec in NODE_SECTIONS for node in getattr(inp,sec,dict()).values()])
        args['pump'] = np.array([inp['CURVES'][link.curve_name].points[0][1]*60/1000 if sec == 'PUMPS' else 0.0
                                  for sec in LINK_SECTIONS if sec in inp for link in getattr(inp,sec,dict()).values()])
        args['offset'] = np.array([getattr(link,'Offset',0)+getattr(link,'InOffset',0)
                   for sec in LINK_SECTIONS if sec in inp for link in getattr(inp,sec,dict()).values()])
        act_edges = self.get_edge_list(list(self.config['action_space'].keys()))
        act_edges = [np.where((args['edges']==act_edge).all(1))[0]
                        for act_edge in act_edges]
        act_edges = [i for e in act_edges for i in e]
        args['act_edges'] = sorted(list(set(act_edges)),key=act_edges.index)
        if act and self.config['act']:
            args['action_space'] = self.get_action_space(act)
        return args

    def controller(self,mode='rand',state=None,setting=None):
        asp = self.config['action_space']
        if mode.lower() == 'rand':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'off' or mode.lower() == 'default':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'half':
            return [table[1] for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        elif mode.lower() == 'safe':
            return setting
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
    