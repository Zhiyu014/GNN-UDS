from pystorms.scenarios import scenario
from envs.environment import env_rcs
import os
import yaml
import numpy as np
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
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}
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
            features = self.get_features(typ)
            __cumvolume = [self.env.methods[attribute](ID) for ID in features]
            __lastvolume = [self.data_log[attribute][ID][-2] if len(self.data_log[attribute][ID])>1 else 0 for ID in features]
            __perf = np.array(__cumvolume) - np.array(__lastvolume) * weight
            __performance.append(__perf)

        # Record the _performance  (N,1)
        self.data_log["performance_measure"].append(np.array(__performance).T)

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
        if not seq:
            if len(self.data_log['performance_measure']) > 0:
                return super().performance(metric)
            else:
                return np.zeros((len(self.elements['nodes']),len(self.config['performance_targets'])))
        else:
            perf = self.data_log['performance_measure'][-seq:]
            default = np.zeros((len(self.elements['nodes']),len(self.config['performance_targets'])))
            perf = [default for _ in range(seq-len(perf))] + perf
            return np.array(perf)

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
        # Data logger for storing _performance & _state data
        for ID, attribute, _ in config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            if ID in ['nodes','links','subcatchments']:
                for idx in self.get_features(ID):
                    self.data_log[attribute][idx] = [] if maxlen is None else deque(maxlen=maxlen)
            
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

    def _logger(self):
        super()._logger()

    def get_args(self):
        args = self.config.copy()
        
        nodes = self.get_features('nodes')
        if not hasattr(self,'env') or self.env._isFinished:
            inp = read_inp_file(self.config['swmm_input'])
            args['hmax'] = np.array([inp.JUNCTIONS[node].MaxDepth if node in inp.JUNCTIONS else 0 for node in nodes])
        else:
            args['hmax'] = np.array([self.env.methods['fulldepth'](node) for node in nodes])
        
        # state shape
        args['state_shape'] = (len(nodes),len([k for k,_ in self.config['global_state'] if k == 'nodes'])) if self.global_state else len(args['states'])

        if self.global_state:
            args['edges'] = self.get_edge_list()
            links = self.get_features('links')
            inp = read_inp_file(self.config['swmm_input'])
            args['ehmax'] = np.array([inp.XSECTIONS[link].Geom1 for link in links])

            args['adj'] = self.get_adj()
            args['edge_adj'] = self.get_edge_adj()
            args['node_edge'] = self.get_node_edge()
            args['edge_state_shape'] = (len(args['edges']),len([k for k,_ in self.config['global_state'] if k == 'links']))
            
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
    
    def get_edge_list(self,links=None):
        inp = read_inp_file(self.config['swmm_input'])
        nodes = self.get_features('nodes')
        if links is not None:
            links = [link for label in LINK_SECTIONS if label in inp for k,link in getattr(inp,label).items() if k in links]
        else:
            links = [link for label in LINK_SECTIONS if label in inp for link in getattr(inp,label).values()]
        edges = []
        for link in links:
            if link.FromNode in nodes and link.ToNode in nodes:
                edges.append((nodes.index(link.FromNode),nodes.index(link.ToNode)))
        return np.array(edges)

    def get_adj(self):
        edges = self.get_edge_list()
        adj = np.eye(edges.max()+1)
        for u,v in edges:
            adj[u,v] += 1
            adj[v,u] += 1
        return adj

    def get_node_edge(self):
        nodes = self.get_features('nodes')
        edges = self.get_edge_list()
        node_edge = np.zeros((len(nodes),len(edges)))
        for i,(u,v) in enumerate(edges):
            node_edge[u,i] += 1
            node_edge[v,i] += -1
        return node_edge
    
    def get_edge_adj(self):
        edges = self.get_edge_list()
        nodes = {}
        for i,(u,v) in enumerate(edges):
            for n in [u,v]:
                if n not in nodes:
                    nodes[n] = []
                else:
                    nodes[n].append(i)
        adj = np.eye(len(edges))
        for v in nodes.values():
            for a,b in combinations(v,2):
                adj[a,b] += 1
        return adj


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