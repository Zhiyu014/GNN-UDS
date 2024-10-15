from .base import basescenario
import os
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.section_lists import NODE_SECTIONS
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
        __object += [self.flood(seq).squeeze().sum(axis=-1)]
        perfs = self.performance(seq)
        for i,(_,attr,target) in enumerate(self.config['performance_targets']):
            if attr == 'depthN':
                if type(target) is str:
                    target,weight = eval(target.split(',')[0]),eval(target.split(',')[1])
                    # __object += (perfs[:,i]>target)*weight if seq else (perfs[i]>target)*weight
                    __object += [(perfs[:,i]>target)*weight if seq else (perfs[i]>target)*weight]
                else:
                    # __object += np.abs(perfs[:,i] - target if seq else perfs[i] - target)
                    __object += [np.abs(perfs[:,i] - target if seq else perfs[i] - target)]
            else:
                # __object += perfs[:,i] * target if seq else perfs[i] * target
                __object += [perfs[:,i] * target if seq else perfs[i] * target]
        # return __object
        return np.array(__object).sum(axis=-1)
     
    def objective_pred(self,preds,state):
        preds,_ = preds
        h,q_w = preds[...,0],preds[...,-1]
        flood = [q_w[...,self.elements['nodes'].index(idx)].sum(axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        depth = [np.abs(h[...,self.elements['nodes'].index(idx)]-target).sum(axis=1)
                for idx,attr,target in self.config['performance_targets']
                if attr == 'depthN' and type(target) is not str]
        exced = [eval(target.split(',')[1])*(h[...,self.elements['nodes'].index(idx)]>eval(target.split(',')[0])).sum(axis=1)
                for idx,attr,target in self.config['performance_targets']
                if attr == 'depthN' and type(target) is str]
        # return q_w.sum(axis=-1).sum(axis=-1) + sum(flood) + sum(depth) + sum(exced)
        return np.array([q_w.sum(axis=-1).sum(axis=-1)] + flood + depth + exced).T
    
    def objective_pred_tf(self,preds,state):
        import tensorflow as tf
        preds,_ = preds
        h,q_w = preds[...,0],preds[...,-1]
        flood = [tf.reduce_sum(q_w[...,self.elements['nodes'].index(idx)],axis=1) * weight
                for idx,attr,weight in self.config['performance_targets'] if attr == 'cumflooding']
        depth = [tf.reduce_sum(tf.abs(h[...,self.elements['nodes'].index(idx)]-target),axis=1)
                for idx,attr,target in self.config['performance_targets'] if attr == 'depthN' and type(target) is not str]
        exced = [tf.reduce_sum(eval(target.split(',')[1])*tf.cast(h[...,self.elements['nodes'].index(idx)]>eval(target.split(',')[0]),tf.float32),axis=1)
                for idx,attr,target in self.config['performance_targets'] if attr == 'depthN' and type(target) is str]
        return tf.reduce_sum(tf.reduce_sum(q_w,axis=-1),axis=-1) + tf.reduce_sum(flood,axis=0) + tf.reduce_sum(depth,axis=0) + tf.reduce_sum(exced,axis=0)   

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

        if self.global_state:
            args['act_edges'] = self.get_edge_list(list(self.config['action_space'].keys()))
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
        else:
            raise AssertionError("Unknown controller %s"%str(mode))
        
    def get_edge_adj(self,directed=False,length=0,order=1):
        edges = self.get_edge_list(length=bool(length))
        X = nx.MultiDiGraph() if directed else nx.MultiGraph()
        if length:
            edges,lengths = edges
            l_std = np.std(lengths)
        for i,(u,v) in enumerate(edges):
            if length:
                X.add_edge(u,v,edge=i,length=lengths[i])
            else:
                X.add_edge(u,v,edge=i)
        EX = nx.DiGraph() if directed else nx.Graph()
        for n in X.nodes():
            if directed:
                for _,_,u in X.in_edges(n,data=True):
                    for _,_,v in X.out_edges(n,data=True):
                        EX.add_edge(u['edge'],v['edge'])
                        if length:
                            EX[u['edge']][v['edge']].update(length = (u['length']+v['length'])/2)
            else:
                for (_,_,u),(_,_,v) in combinations(X.edges(n,data=True),2):
                    EX.add_edge(u['edge'],v['edge'])
                    if length:
                        EX[u['edge']][v['edge']].update(length = (u['length']+v['length'])/2)

        n_edge = edges.shape[0]
        adj = np.zeros((n_edge,n_edge))
        for n in range(n_edge):
            if length:
                p_l = nx.single_source_dijkstra_path_length(EX,n,weight='length',cutoff=length)
                for a,l in p_l.items():
                    adj[n,a] = np.exp(-(l/(l_std+1e-5))**2)
            else:
                for a in list(nx.dfs_preorder_nodes(EX,n,order)):
                    adj[n,a] = 1
        return adj