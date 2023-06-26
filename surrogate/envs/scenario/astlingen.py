from pystorms.scenarios import scenario
from envs.environment import env_ast
import os
import yaml
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import datetime
from functools import reduce
from itertools import product
from collections import deque
from itertools import combinations
from mpc import run_ea

HERE = os.path.dirname(__file__)

class astlingen(scenario):
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
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(os.path.dirname(HERE),"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        if initialize:
            self.env = env_ast(self.config, ctrl=True)
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}
        self.global_state = global_state # If use global state as input

        # initialize logger
        self.initialize_logger()


    def step(self, actions=None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and 'interval' in self.config:
            advance_seconds = self.config['interval'] * 60
        # if actions is not None:
        #     actions = self._convert_actions(actions)

        done = self.env.step(actions, advance_seconds = advance_seconds)
        
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


        # Record the _performance
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

    def reset(self,swmm_file=None, global_state=True,seq=False):
        # clear the data log and reset the environment
        if swmm_file is not None:
            self.config["swmm_input"] = swmm_file
        if not hasattr(self,'env') or swmm_file is not None:
            self.env = env_ast(self.config, ctrl=True)
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
            "setting": {}
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

        for ID in config["action_space"].keys():
            self.data_log["setting"][ID] =  [] if maxlen is None else deque(maxlen=maxlen)
        
        # for ID, attribute, _ in config["reward"]:
        #     if attribute not in self.data_log.keys():
        #         self.data_log[attribute] = {}
        #     self.data_log[attribute][ID] =  [] if maxlen is None else deque(maxlen=maxlen)

    def _logger(self):
        super()._logger()


    def get_action_table(self,if_mac):
        action_table = {}
        actions = [len(v) for v in self.config['action_space'].values()]
        site_combs = product(*[range(act) for act in actions])
        for idx,site in enumerate(site_combs):
            if if_mac:
                action_table[site] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
            else:
                action_table[(idx,)] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
        return action_table

    def get_args(self):
        args = self.config.copy()
        # Rainfall timeseries & events files
        if not os.path.isfile(args['rainfall']['rainfall_timeseries']):
            args['rainfall']['rainfall_timeseries'] = os.path.join(HERE,'config',args['rainfall']['rainfall_timeseries']+'.csv')
        if not os.path.isfile(args['rainfall']['rainfall_events']):
            args['rainfall']['rainfall_events'] = os.path.join(HERE,'config',args['rainfall']['rainfall_events']+'.csv')
        # if not os.path.isfile(args['rainfall']['training_events']):
        #     args['rainfall']['training_events'] = os.path.join(HERE,'config',args['rainfall']['training_events']+'.csv')

        nodes = self.get_features('nodes')
        if not hasattr(self,'env') or self.env._isFinished:
            inp = read_inp_file(self.config['swmm_input'])
            args['hmax'] = np.array([node.MaxDepth for node in list(inp.JUNCTIONS.values())+list(inp.STORAGE.values())])
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


    # getters
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
        
        # Set the Control Rules
        # inp['CONTROLS'] = Control.create_section()
        # for i in range(self.config['prediction']['control_horizon']//self.config['control_interval']):
        #     time = round(self.config['control_interval']/60*(i+1),2)
        #     conditions = [Control._Condition('IF','SIMULATION','TIME', '<', str(time))]
        #     actions = []
        #     for idx,k in enumerate(self.config['action_space']):
        #         logic = 'THEN' if idx == 0 else 'AND'
        #         kind = self.env.methods['getlinktype'](k)
        #         action = Control._Action(logic,kind,k,'SETTING','=',str('1.0'))
        #         actions.append(action)
        #     inp['CONTROLS'].add_obj(Control('P%s'%(i+1),conditions,actions,priority=5-i))
    

        # Output the eval file
        eval_inp_file = os.path.join(os.path.dirname(self.config['swmm_input']),
                                self.config['prediction']['eval_dir'],
                                self.config['prediction']['suffix']+os.path.basename(self.config['swmm_input']))
        if os.path.exists(os.path.dirname(eval_inp_file)) == False:
            os.mkdir(os.path.dirname(eval_inp_file))
        inp.write_file(eval_inp_file)

        return eval_inp_file

    def get_eval_file(self):
        if self.env._isFinished:
            print('Simulation already finished')
            return None
        else:
            hsf_file = self.save_hotstart()
            eval_file = self.create_eval_file(hsf_file)
            return eval_file

    def get_current_setting(self):
        if len(self.data_log['setting']) > 0 :
            setting = [self.data_log["setting"][ID][-1]
            for ID in self.config["action_space"]]
        else:
            setting = [self.env.methods["setting"](ID)
            for ID in self.config["action_space"]]
        return setting

    def controller(self,state=None,mode='rand'):
        asp = self.config['action_space']
        if mode.lower() == 'rand':
            return [table[np.random.randint(0,len(table))] for table in asp.values()]
        elif mode.lower() == 'bc':
            return [table[1] for table in asp.values()]
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
            setting = [v[setting[k]] for k,v in asp.items()]
            return setting
        elif mode.lower() == 'mpc':
            eval_file = self.get_eval_file()
            setting = self.get_current_setting()
            setting = run_ea(eval_file,arg,setting)
            return setting
        else:
            raise AssertionError("Unknown controller %s"%str(mode))