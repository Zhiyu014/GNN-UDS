from .base import basescenario
import os
import numpy as np
import torch as th
from swmm_api import read_inp_file
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import networkx as nx
from itertools import combinations,groupby,product
HERE = os.path.dirname(__file__)
KWperHP = 0.7457
ft_m = 0.3048
cfs_cms = 0.0283168
ene_conv = 8.814
secs_per_hour = 3600.0

class chaohu(basescenario):
    r"""Chaohu Scenario

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
    1. Tank flooding
    2. System flooding
    3. CSO
    4. Depth to encourage storage use
    5. Pumping energy

    Performance is measured as the following:
    1. 50 if tank flooding
    2. 2 for flooding volume
    3. 1 for CSO volume
    4. (TODO) 20 for depth use
    5. (TODO) 0.1 for pumping energy
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","chaohu.yaml") \
            if config_file is None else config_file
        super().__init__(config_file,swmm_file,global_state,initialize)

        inp = read_inp_file(self.config['swmm_input'])
        self.hmin = np.array([getattr(node,'elevation',0) for sec in NODE_SECTIONS
                              for node in getattr(inp,sec,dict()).values()])
        self.hmax = np.array([getattr(node,'depth_max',0)+getattr(node,'depth_surcharge',0) for sec in NODE_SECTIONS
                              for node in getattr(inp,sec,dict()).values()]) + self.hmin
        self.pumps = {k:(inp.PUMPS[k].from_node,inp.PUMPS[k].to_node) for k in self.config['action_space']}
        
    # TODO
    def objective(self, seq = False, keepdim = False):
        __object = []
        # __object += [self.flood(seq).squeeze().sum(axis=-1)]  # move sum flood in performance
        perfs = self.performance(seq = max(seq,1) + 1 if seq else 2)
        # _is_raining = False
        for i,(ID,attr,target) in enumerate(self.config['performance_targets']):
            # __value = perfs[:,i] if seq else perfs[i]
            __value = perfs[:,i] if attr == 'setting' else perfs[1:,i]
            # if attr == 'rainfall':
            #     _is_raining = sum(self.data_log[attr][ID][target:])>0
            # if attr == 'depthN':
            #     __object -= __value/self.hmax[i] * target if _is_raining else (1-__value/self.hmax[i]) * target
            if attr == 'cumflooding' and ID.endswith('storage'):
                # __object += (__value>0) * target
                __object += [(__value>0).squeeze() * target]
            elif attr == 'setting':
                __object += [np.abs(np.diff(__value,axis=0)).squeeze() * target]
            else:
                # __object += __value * target
                __object += [__value.squeeze() * target]
        # return __object
        if seq:
            return np.array(__object).sum(axis=-1) if not keepdim else np.array(__object).T
        else:
            return np.array(__object)
     
     
    def objective_pred(self,preds,states,settings,gamma=None,keepdim=False):
        preds,edge_preds = preds
        h,q_in,q_w,q = preds[...,0],preds[...,1],preds[...,-1],edge_preds[...,-1]
        nodes = self.elements['nodes']
        targets = self.config['performance_targets']
        flood = [q_w.sum(axis=-1) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx=='nodes']
        penal = [(q_w[...,nodes.index(idx)]>0) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx.endswith('storage')]
        outflow = [q_in[...,nodes.index(idx)] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and weight>0]
        wwtp = [q_in[...,nodes.index(idx)] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and weight<0]
        energy = [self.get_energy(h,q,idx) * weight
                for idx,attr,weight in targets if attr == 'cumpumpenergy']
        # Control Roughness
        # asp = list(self.config['action_space'])
        # _,edge_state = states
        # sett = np.concatenate([edge_state[:,-1:,[links.index(idx) for idx,attr,_  in targets if attr == 'setting'],-1],
        #                        settings[...,[asp.index(idx) for idx,attr,_  in targets if attr == 'setting']]],axis=1)
        # rough = [np.abs(np.diff(sett[...,i],axis=1)*gamma).sum(axis=1) * weight
        #          for i,weight in enumerate([weight for _,attr,weight in targets if attr == 'setting'])]
        obj = np.stack(flood + penal + outflow + wwtp + energy,axis=1)
        gamma = np.ones(preds.shape[1]) if gamma is None else np.array(gamma,dtype=np.float32)
        obj = (obj*gamma).sum(axis=-1) if not keepdim else np.transpose(obj*gamma,(0,2,1))
        return obj
    
    def objective_pred_th(self,preds,states,settings,gamma=None,keepdim=False):
        preds,edge_preds = preds
        h,q_in,q_w,q = preds[...,0],preds[...,1],preds[...,-1],edge_preds[...,-1]
        nodes = self.elements['nodes']
        targets = self.config['performance_targets']
        flood = [q_w.sum(dim=-1) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx=='nodes']
        penal = [(q_w[...,nodes.index(idx)]>0).type(th.float32) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx.endswith('storage')]
        outflow = [q_in[...,nodes.index(idx)] * weight
                   for idx,attr,weight in targets if attr == 'cuminflow' and weight>0]
        wwtp = [q_in[...,nodes.index(idx)] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and weight<0]
        # Energy consumption (kWh): refer from swmm engine link_getPower in link.c
        energy = [self.get_energy(h,q,idx) * weight
                for idx,attr,weight in targets if attr == 'cumpumpenergy']
        # Control Roughness
        # asp = list(self.config['action_space'])
        # _,edge_state = states
        # sett = tf.concat([edge_state[:,-1:,[links.index(idx) for idx,attr,_  in targets if attr == 'setting'],-1],
        #                   settings[...,[asp.index(idx) for idx,attr,_  in targets if attr == 'setting']]],axis=1)
        # rough = [tf.reduce_sum(tf.abs(tf.experimental.numpy.diff(sett[...,i],axis=1))*gamma,axis=1) * weight
        #          for i,weight in enumerate([weight for _,attr,weight in targets if attr == 'setting'])]
        obj = th.stack(flood + penal + outflow + wwtp + energy,dim=1)
        gamma = th.ones(preds.shape[1]).to(device=obj.device) if gamma is None else th.tensor(gamma).to(device=obj.device)
        obj = (obj*gamma).sum(axis=-1) if not keepdim else (obj*gamma).permute(0,2,1)
        return obj
    
    def get_energy(self,h,q,idx):
        # Energy consumption (kWh): refer from swmm engine link_getPower in link.c
        nidx0 = self.elements['nodes'].index(self.pumps[idx][0])
        nidx1 = self.elements['nodes'].index(self.pumps[idx][1])
        lidx = self.elements['links'].index(idx)
        lib = th if isinstance(h,th.Tensor) else np
        # Head difference between pump nodes
        if self.config['global_state'][0][-1] == 'head':
            head_diff = lib.abs(h[..., nidx0] - h[..., nidx1]) / ft_m
        else:
            head_diff = lib.abs(
                self.hmin[nidx0] + h[..., nidx0] -
                self.hmin[nidx1] - h[..., nidx1]
            ) / ft_m
        # Discharge flow
        discharge = lib.abs(q[..., lidx]) / cfs_cms
        # Power calculation
        power_kwh = (head_diff * discharge) / ene_conv * KWperHP / secs_per_hour
        return power_kwh
    
    def norm_obj(self,obj,states,inverse=False):
        __norm = states[0][...,-1].sum(axis=-1).sum(axis=-1)
        while __norm.ndim < obj.ndim:
            __norm = np.expand_dims(__norm,-1) if isinstance(__norm,np.ndarray) else th.unsqueeze(__norm,-1)
        return obj*(__norm+1e-5) if inverse else obj/(__norm+1e-5)

    def get_obj_norm(self,norm_y,norm_e,perfs):
        nodes,links = self.elements['nodes'],self.elements['links']
        targets = self.config['performance_targets']
        perfs = perfs.squeeze().sum(axis=-1)
        fl = [np.array([perfs.max(),0]) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx=='nodes']
        pen = [(norm_y[...,nodes.index(idx),-1]>0).astype(np.float32) * weight
                for idx,attr,weight in targets if attr == 'cumflooding' and idx.endswith('storage')]
        outfl = [norm_y[...,nodes.index(idx),1] * weight
                 for idx,attr,weight in targets if attr == 'cuminflow' and weight>0]
        wwtp = [norm_y[...,nodes.index(idx),1] * weight
                for idx,attr,weight in targets if attr == 'cuminflow' and weight<0]
        ene = [norm_e[...,links.index(idx),-2]/cfs_cms *\
                max(np.abs(self.hmin[nodes.index(self.pumps[idx][0])] - self.hmax[nodes.index(self.pumps[idx][1])]),
                    np.abs(self.hmax[nodes.index(self.pumps[idx][0])] - self.hmin[nodes.index(self.pumps[idx][1])]))\
                    /ft_m/ene_conv * KWperHP/secs_per_hour * weight
               for idx,attr,weight in targets if attr == 'cumpumpenergy']
        return np.stack(fl + pen + outfl + wwtp + ene,axis=-1)

    def get_action_table(self,act='rand'):
        asp = self.config['action_space'].copy()
        if act.endswith('bin'):
            actions = {act:list(act) for act in product(*asp.values())}
        else:
            asp = {k:[0]+[i+1 for i,_ in enumerate(v)]
                    for k,v in groupby(asp.keys(),key=lambda x:x[:4])}
            actions = {act:[[1]*a+[0]*(len(v)-1-a) for a,v in zip(act,asp.values())] for act in product(*asp.values())}
            actions = {k:[v for va in values for v in va] for k,values in actions.items()}
            if '2' in act:
                actions = {(k[0]*3+k[1],k[2]*3+k[3]):v for k,v in actions.items()}
        return actions
    
    def get_action_space(self,act='rand'):
        if 'rand' in act:
            asp = self.config['action_space'].copy()
            asps = {k:list(product(*[asp[p] for p in v]))
                    for k,v in groupby(asp.keys(),key=lambda x:x[:4])}
            asps = {k:[va for va in v if va[0]>=va[1]] if len(v[0])==2 else v
                    for k,v in asps.items()}
            return asps
        else:
            return self.config['action_space']

    def get_args(self,directed=False,length=0,order=1,graph_base=0,act=False,dec=False):
        args = super().get_args(directed,length,order,graph_base)

        inp = read_inp_file(self.config['swmm_input'])
        args['area'] = np.array([inp['CURVES'][node.data].points[0][1] if sec == 'STORAGE' else 0.0
                                  for sec in NODE_SECTIONS if sec in inp for node in getattr(inp,sec,dict()).values()])
        args['pump'] = np.array([inp['CURVES'][link.curve_name].points[0][1]*60/1000 if sec == 'PUMPS' else 0.0
                                  for sec in LINK_SECTIONS if sec in inp for link in getattr(inp,sec,dict()).values()])
        args['pump_in'] = (-np.clip(args['node_edge'],-1,0)*args['pump']).sum(axis=1)
        args['pump_out'] = (np.clip(args['node_edge'],0,1)*args['pump']).sum(axis=1)

        act_edges = self.get_edge_list(list(self.config['action_space'].keys()))
        act_edges = [np.where((args['edges']==act_edge).all(1))[0]
                        for act_edge in act_edges]
        act_edges = [i for e in act_edges for i in e]
        args['act_edges'] = sorted(list(set(act_edges)),key=act_edges.index)
        if act and self.config['act'] and not act.startswith('conti'):
            args['action_space'] = self.get_action_space(act)
            args['action_table'] = self.get_action_table(act)
            # For multi-agent
            args['action_shape'] = np.array(list(args['action_table'].keys())).max(axis=0)+1
            if dec:
                args['n_agents'] = len(self.config['site'])
                state = [s[0] for s in self.config['states']]
                args['observ_space'] = [[state.index(o) for o in v['states']]
                                        for v in self.config['site'].values()]
                # args['action_shape'] = np.array(list(args['action_table'].keys())).max(axis=0)+1
            else:
                args['n_agents'] = 1
                args['observ_space'] = self.config['states']
        return args
    
    def controller(self,mode='rand',state=None,setting=None):
        asp = self.get_action_space(mode)
        if mode.lower().startswith('rand'):
            return [t for table in asp.values() for t in table[np.random.randint(0,len(table))]]
        elif mode.lower().startswith('conti'):
            return [np.random.uniform(min(table),max(table)) for table in asp.values()]
        elif mode.lower() == 'off' or mode.lower() == 'default':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        elif mode.lower() == 'hc' or mode.lower() == 'safe':
            state_idxs = {k:self.elements['nodes'].index(k.split('-')[0]+'-storage') for k in asp}
            depth = {k:state[idx,0] for k,idx in state_idxs.items()}
            thres = self.config[mode.lower()+'_thresholds']
            setting = [min(max(sett,int(h>thres[k][1])),1-int(h<thres[k][0]))
                        for (k,h),sett in zip(depth.items(),setting)]
            return setting
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