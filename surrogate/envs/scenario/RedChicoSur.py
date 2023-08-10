from pystorms.scenarios import scenario
from envs.environment import env_rcs
import os
import yaml
import numpy as np
import networkx as nx
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import datetime
from collections import deque
from itertools import combinations
HERE = os.path.dirname(__file__)

class RedChicoSur(scenario):
    r"""RedChicoSur Scenario


    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","RedChicoSur.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(os.path.dirname(HERE),"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        if initialize:
            self.env = env_rcs(self.config, ctrl=True)
        
        # self.penalty_weight = {ID: weight
        #                        for ID, _, weight in \
        #                            self.config["performance_targets"]}
        self.global_state = global_state # If use global state as input

        # initialize logger
        self.initialize_logger()


    def step(self, settings = None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and self.config.get('interval') is not None:
            advance_seconds = self.config['interval'] * 60
        # if actions is not None:
        #     actions = self._convert_actions(actions)

        done = self.env.step(settings, advance_seconds = advance_seconds)
        
        # Log the states, targets, and actions
        if log:
            self._logger()

        # Log the performance
        __performance = []
        for typ, attribute, weight in self.config["performance_targets"]:
            if typ in ['nodes','links','subcatchments']:
                features = self.elements[typ]
                __volume = [self.env.methods[attribute](ID) for ID in features]
                if 'cum' in attribute:
                    __lastvolume = [self.data_log[attribute][ID][-2] if len(self.data_log[attribute][ID])>1 else 0 for ID in features]
                    __perf = np.array(__volume) - np.array(__lastvolume) * weight
                else:
                    __perf = np.array(__volume) * weight
            else:
                __volume = self.env.methods[attribute](typ)
                if 'cum' in attribute:
                    __lastvolume = self.data_log[attribute][typ][-2] if len(self.data_log[attribute][typ])>1 else 0
                    __perf = __volume - __lastvolume * weight
                else:
                    __perf = __volume * weight
            __performance.append(__perf)
        __performance = np.array(__performance).T if typ in ['nodes','links','subcatchments'] else np.array(__performance)

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()
        return done

    def state_full(self, seq = False, typ='nodes'):
        # seq should be smaller than the whole event length
        attrs = [item for item in self.config['global_state'] if item[0]==typ]
        if seq:
            __state = np.array([[self.data_log[attr][ID][-seq:]
        if self.env._isFinished else [0.0]*(seq-1-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][-seq:-1] + [self.env.methods[attr](ID)]
                for ID in self.elements[typ]] for typ,attr in attrs])
        else:
            __state = np.array([[self.data_log[attr][ID][-1]
                if self.env._isFinished else self.env.methods[attr](ID)
                for ID in self.elements[typ]] for typ,attr in attrs])

        if seq:
            __last = np.array([[self.data_log[attr][ID][-seq-1:-1]
                if len(self.data_log[attr][ID]) > seq 
                else [0.0]*(seq-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][:-1]
                for ID in self.elements[typ]] 
                if 'cum' in attr 
                else [[0.0]*seq for _ in self.elements[typ]]
                for typ,attr in attrs])
        else:
            __last = np.array([[self.data_log[attr][ID][-2]
                if 'cum' in attr and len(self.data_log[attr][ID]) > 1 else 0
                for ID in self.elements[typ]] for typ,attr in attrs])
        state = (__state - __last).T
        return state

    def state(self, seq = False):
        # Observe from the environment
        if self.global_state:
            state = self.state_full(seq)
            return state
        if self.env._isFinished:
            if seq:
                __state = [list(self.data_log[attribute][ID])[-seq:]
                for ID,attribute in self.config["states"]]
            else:
                __state = [self.data_log[attribute][ID][-1]
            for ID,attribute in self.config["states"]]
        else:
            __state = self.env._state()
            if seq:
                __state = [list(self.data_log[attribute][ID])[-seq:-1] + [__state[idx]]
                for idx,(ID,attribute) in enumerate(self.config["states"])]  
        state = []
        for idx,(ID,attribute) in enumerate(self.config["states"]):
            if 'cum' not in attribute:
                __value = __state[idx]
                if seq:
                    # ensure length seq and fill with 0
                    __value = [0.0] * (seq-len(__value)) + __value
            else:
                if seq:
                    __value = self.data_log[attribute][ID][-seq-1:-seq] + __state[idx]  # length: seq+1
                    __value = [0.0]*(seq+1-len(__value)) + __value
                    __value = np.diff(__value)
                else:
                    __value = __state[idx]
                    if len(self.data_log[attribute][ID]) > 1:
                        __value -= self.data_log[attribute][ID][-2]
            state.append(np.asarray(__value))
        state = np.asarray(state).T if seq else np.asarray(state)
        return state

    def performance(self, seq = False, metric = 'recent'):
        shape = [len(self.elements[typ]) if typ in ['nodes','links','subcatchments'] else 1
                    for typ,_,_ in self.config['performance_targets']]
        shape = (max(shape),len(shape)) if max(shape)>1 else (len(shape),)
        if not seq:
            if len(self.data_log['performance_measure']) > 0:
                return super().performance(metric)
            else:
                return np.zeros(shape=shape)
        else:
            perf = self.data_log['performance_measure'][-seq:]
            default = np.zeros(shape=shape)
            perf = [default for _ in range(seq-len(perf))] + perf
            return np.array(perf)

    def flood(self, seq = False):
        if seq:
            __flood = np.array([[self.data_log[attr][ID][-seq:]
        if self.env._isFinished else [0.0]*(seq-1-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][-seq:-1] + [self.env.methods[attr](ID)]
                for ID in self.elements[typ]] for typ,attr in self.config['flood']])
        else:
            __flood = np.array([[self.data_log[attr][ID][-1]
                if self.env._isFinished else self.env.methods[attr](ID)
                for ID in self.elements[typ]] for typ,attr in self.config['flood']])

        if seq:
            __last = np.array([[self.data_log[attr][ID][-seq-1:-1]
                if len(self.data_log[attr][ID]) > seq 
                else [0.0]*(seq-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][:-1]
                for ID in self.elements[typ]] 
                if 'cum' in attr 
                else [[0.0]*seq for _ in self.elements[typ]]
                for typ,attr in self.config['flood']])
        else:
            __last = np.array([[self.data_log[attr][ID][-2]
                if 'cum' in attr and len(self.data_log[attr][ID]) > 1 else 0
                for ID in self.elements[typ]] for typ,attr in self.config['flood']])
        flood = (__flood - __last).T
        return flood
    
    def reset(self,swmm_file=None, global_state=False,seq=False):
        # clear the data log and reset the environment
        if swmm_file is not None:
            self.config["swmm_input"] = swmm_file
        if not hasattr(self,'env') or swmm_file is not None:
            self.env = env_rcs(self.config, ctrl=True)
        else:
            _ = self.env.reset()

        self.global_state = global_state
        self.initialize_logger()
        state = self.state(seq)
        return state
        
    def initialize_logger(self, config=None,maxlen=None):
        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [] if maxlen is None else deque(maxlen=maxlen),
            "simulation_time": [] if maxlen is None else deque(maxlen=maxlen),
        }
        config = self.config if config is None else config

        if self.global_state:
            self.elements = {typ:self.get_features(typ) for typ,_ in config['global_state']}
            for typ,attribute in config['global_state']:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                for ID in self.elements[typ]:
                    self.data_log[attribute][ID] =  [] if maxlen is None else deque(maxlen=maxlen)
        else:
            for ID, attribute in config["states"]:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                self.data_log[attribute][ID] =  [] if maxlen is None else deque(maxlen=maxlen)

        # Data logger for storing _performance & _state data
        for ID, attribute, _ in config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            if ID in ['nodes','links','subcatchments']:
                for idx in self.get_features(ID):
                    self.data_log[attribute][idx] = [] if maxlen is None else deque(maxlen=maxlen)
            
        for typ, attribute in config['flood']:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            for ID in self.elements[typ]:
                self.data_log[attribute][ID] = [] if maxlen is None else deque(maxlen=maxlen)

    def _logger(self):
        super()._logger()

    def get_args(self,directed=False,length=0,order=1):
        args = self.config.copy()
        
        nodes = self.get_features('nodes')
        if not hasattr(self,'env') or self.env._isFinished:
            inp = read_inp_file(self.config['swmm_input'])
            args['is_outfall'] = np.array([1 if sec == 'OUTFALLS' else 0 for sec in NODE_SECTIONS
                                           for _ in getattr(getattr(inp,sec,[]),'values',[])])
            args['hmax'] = np.array([getattr(node,'MaxDepth',0) for sec in NODE_SECTIONS
                                      for node in getattr(getattr(inp,sec,[]),'values',[])])
            # It seems that a tidal outfall may have large full depth hmax above the maximum tide level
            # args['hmax'] = np.array([node.MaxDepth for node in list(inp.JUNCTIONS.values())+list(inp.STORAGE.values())])
        else:
            args['is_outfall'] = np.array([self.env._is_Outfall(node) for node in nodes])
            args['hmax'] = np.array([self.env.methods['fulldepth'](node) for node in nodes])

        # state shape
        args['state_shape'] = (len(nodes),len([k for k,_ in self.config['global_state'] if k == 'nodes'])) if self.global_state else len(args['states'])

        if self.global_state:
            args['edges'] = self.get_edge_list()
            links = self.get_features('links')
            inp = read_inp_file(self.config['swmm_input'])
            args['ehmax'] = np.array([inp.XSECTIONS[link].Geom1 for link in links])

            args['adj'] = self.get_adj(directed,length,order)
            args['edge_adj'] = self.get_edge_adj(directed,length,order)
            args['node_edge'] = self.get_node_edge()
            args['node_index'] = self.get_node_index(directed)  # n_edge,n_edge
            args['edge_index'] = self.get_edge_index(directed)  # n_node,n_node
            args['act_edges'] = self.get_edge_list(list(self.config['action_space'].keys()))
        return args


    # TODO: getters Use pyswmm api
    def get_features(self,kind='nodes',no_out=False):
        inp = read_inp_file(self.config['swmm_input'])
        labels = {'nodes':NODE_SECTIONS,'links':LINK_SECTIONS}
        features = []
        for label in labels[kind]:
            if label not in inp or (no_out and label == 'OUTFALLS'):
                continue                
            if no_out and kind == 'links':
                features += [k for k,v in getattr(inp,label).items()
                             if getattr(v,'ToNode') not in inp['OUTFALLS']]
            else:
                features += list(getattr(inp,label))
        return features
    
    def get_edge_list(self,links=None,length=False):
        inp = read_inp_file(self.config['swmm_input'])
        nodes = self.get_features('nodes')
        if links is not None:
            links = [link for label in LINK_SECTIONS if label in inp for k,link in getattr(inp,label).items() if k in links]
        else:
            links = [link for label in LINK_SECTIONS if label in inp for link in getattr(inp,label).values()]
        edges,lengths = [],[]
        for link in links:
            if link.FromNode in nodes and link.ToNode in nodes:
                edges.append((nodes.index(link.FromNode),nodes.index(link.ToNode)))
                lengths.append(getattr(link,'Length',0.0))
        if length:
            return np.array(edges),np.array(lengths)
        else:
            return np.array(edges)

    def get_adj(self,directed=False,length=0,order=1):
        edges = self.get_edge_list(length=bool(length))
        X = nx.DiGraph() if directed else nx.Graph()
        if length:
            edges,lengths = edges
            l_std = np.std(lengths)
            n_node = edges.max()+1
            adj = np.zeros((n_node,n_node))
            for (u,v),l in zip(edges,lengths):
                X.add_edge(u,v,length=l)
            for n in range(n_node):
                path_length = nx.single_source_dijkstra_path_length(X,n,weight='length',cutoff=length)
                for a,l in path_length.items():
                    adj[n,a] = np.exp(-(l/(l_std+1e-5))**2)
        else:
            for u,v in edges:
                X.add_edge(u,v)
            n_node = edges.max()+1
            adj = np.zeros((n_node,n_node))
            for n in range(n_node):
                for a in list(nx.dfs_preorder_nodes(X,n,order)):
                    adj[n,a] = 1
                    if not directed:
                        adj[a,n] = 1
        return adj

    def get_edge_adj(self,directed=False,length=0,order=1):
        edges = self.get_edge_list(length=bool(length))
        X = nx.DiGraph() if directed else nx.Graph()
        if length:
            edges,lengths = edges
            l_std = np.std(lengths)
        for i,(u,v) in enumerate(edges):
            X.add_edge(u,v,edge=i)
            if length:
                X[u][v].update(length=lengths[i])
        EX = nx.DiGraph() if directed else nx.Graph()
        for n in X.nodes():
            if directed:
                for a,b in X.in_edges(n):
                    for c,d in X.out_edges(n):
                        u,v = X[a][b],X[c][d]
                        EX.add_edge(u['edge'],v['edge'])
                        if length:
                            EX[u['edge']][v['edge']].update(length = (u['length']+v['length'])/2)
            else:
                for (a,b),(c,d) in combinations(X.edges(n),2):
                    u,v = X[a][b],X[c][d]
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

    # For NodeEdge fusion
    def get_node_edge(self):
        nodes = self.get_features('nodes')
        edges = self.get_edge_list()
        node_edge = np.zeros((len(nodes),len(edges)))
        for i,(u,v) in enumerate(edges):
            node_edge[u,i] += 1
            node_edge[v,i] += -1
        return node_edge
    
    # For ECCConv
    def get_node_index(self,directed=False):
        edges = self.get_edge_list()
        nodes_in,nodes_out = {},{}
        for i,(u,v) in enumerate(edges):
            nodes_in[u] = nodes_in[u] + [i] if u in nodes_in else [i]
            nodes_out[v] = nodes_out[v] + [i] if v in nodes_out else [i]
        n_edge = edges.shape[0]
        node_idx = np.zeros((n_edge,n_edge))
        for n in range(edges.max()+1):
            if directed:
                for a in nodes_in.get(n,[]):
                    for b in nodes_out.get(n,[]):
                        node_idx[a,b] = n
            else:
                for a,b in combinations(nodes_in.get(n,[])+nodes_out.get(n,[]),2):
                    node_idx[a,b] = n
        return node_idx.astype(int)

    # For ECCConv
    def get_edge_index(self,directed=False):
        edges = self.get_edge_list()
        n_node = edges.max()+1
        edge_idx = np.zeros((n_node,n_node))
        for i,(u,v) in enumerate(edges):
            edge_idx[u,v] = i
            if not directed:
                edge_idx[v,u] = i
        return edge_idx.astype(int)

    # predictive functions
    def save_hotstart(self,hsf_file=None):
        # Save the current state in a .hsf file.
        if hsf_file is None:
            ct = self.env.methods['simulation_time']()
            hsf_file = '%s.hsf'%ct.strftime('%Y-%m-%d-%H-%M')
            hsf_file = os.path.join(os.path.dirname(self.config['swmm_input']),
            self.config['prediction']['hsf_dir'],hsf_file)
        if os.path.exists(os.path.dirname(hsf_file)) == False:
            os.mkdir(os.path.dirname(hsf_file))
        return self.env.save_hotstart(hsf_file)

    def create_eval_file(self,hsf_file=None):
        ct = self.env.methods['simulation_time']()
        inp = read_inp_file(self.config['swmm_input'])

        # Set the simulation time & hsf options
        inp['OPTIONS']['START_DATE'] = inp['OPTIONS']['REPORT_START_DATE'] = ct.date()
        inp['OPTIONS']['START_TIME'] = inp['OPTIONS']['REPORT_START_TIME'] = ct.time()
        inp['OPTIONS']['END_DATE'] = (ct + datetime.timedelta(minutes=self.config['prediction']['eval_horizon'])).date()
        inp['OPTIONS']['END_TIME'] = (ct + datetime.timedelta(minutes=self.config['prediction']['eval_horizon'])).time()
        
        if hsf_file is not None:
            if 'FILES' not in inp:
                inp['FILES'] = FilesSection()
            inp['FILES']['USE HOTSTART'] = hsf_file

        # Output the eval file
        eval_inp_file = os.path.join(os.path.dirname(self.config['swmm_input']),
                                self.config['prediction']['eval_dir'],
                                self.config['prediction']['suffix']+os.path.basename(self.config['swmm_input']))
        if os.path.exists(os.path.dirname(eval_inp_file)) == False:
            os.mkdir(os.path.dirname(eval_inp_file))
        inp.write_file(eval_inp_file)

        return eval_inp_file

    def get_eval_file(self):
        assert not self.env._isFinished, 'Simulation already finished'
        hsf_file = self.save_hotstart()
        eval_file = self.create_eval_file(hsf_file)
        return eval_file

    # TODO: get predictive runoff & states
    # Problems: There are runoff errors in hotstart file
    # Possible solution: use an independent inp with only subcs to collect lateral inflow and simulate from scratch in each step
    # def predict(self,settings=None):
    #     eval_file = self.get_eval_file()

    def get_subc_inp(self):
        inp = read_inp_file(self.config['swmm_input'])
        for k in LINK_SECTIONS+['XSECTIONS']:
            if k in inp:
                inp.pop(k)
        file = self.config['swmm_input'].strip('.inp')+'_subc.inp'
        inp.write_file(file)
        return file

        
        
    def controller(self,state=None,mode='rand'):
        asp = self.config['action_space']
        if mode.lower() == 'rand':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower() == 'off':
            return [table[0] for table in asp.values()]
        elif mode.lower() == 'on':
            return [table[-1] for table in asp.values()]
        else:
            raise AssertionError("Unknown controller %s"%str(mode))