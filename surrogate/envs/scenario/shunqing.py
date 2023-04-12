from pystorms.scenarios import scenario
from envs.environment import env_sq
import os
import yaml
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import datetime
from collections import deque

HERE = os.path.dirname(__file__)

class shunqing(scenario):
    r"""Shunqing Scenario

    [ga_ann_for_uds]
    Code: https://github.com/lhmygis/ga_ann_for_uds
    Paper: https://doi.org/10.1016/j.envsoft.2023.105623

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    Shunqing District, Nanchong City, Sichuan Province, China
    Annual precipitation: 1020.8 mm
    Area: 33.02 km2
    131 pipes, 105 manholes, 8 outfalls, 106 subcatchments
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"..","config","shunqing.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(os.path.dirname(HERE),"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        if initialize:
            self.env = env_sq(self.config, ctrl=True)
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}
        self.global_state = global_state # If use global state as input

        # initialize logger
        self.initialize_logger()


    def step(self, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and self.config.get('control_interval') is not None:
            advance_seconds = self.config['control_interval'] * 60
        # if actions is not None:
        #     actions = self._convert_actions(actions)

        done = self.env.step(None, advance_seconds = advance_seconds)
        
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

    def state_full(self, seq = False):
        # seq should be smaller than the whole event length
        if seq:
            __state = np.array([[self.data_log[attr][ID][-seq:]
        if self.env._isFinished else [0.0]*(seq-1-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][-seq:-1] + [self.env.methods[attr](ID)]
                for ID in self.get_features(typ)] for typ,attr in self.config['global_state']])
        else:
            __state = np.array([[self.data_log[attr][ID][-1]
                if self.env._isFinished else self.env.methods[attr](ID)
                for ID in self.get_features(typ)] for typ,attr in self.config['global_state']])

        if seq:
            __last = np.array([[self.data_log[attr][ID][-seq-1:-1]
                if len(self.data_log[attr][ID]) > seq 
                else [0.0]*(seq-len(self.data_log[attr][ID][:-1])) + self.data_log[attr][ID][:-1]
                for ID in self.get_features(typ)] 
                if 'cum' in attr or 'vol' in attr
                else [[0.0]*seq for _ in self.get_features(typ)]
                for typ,attr in self.config['global_state']])
        else:
            __last = np.array([[self.data_log[attr][ID][-2]
                if ('cum' in attr or 'vol' in attr) and len(self.data_log[attr][ID]) > 1 else 0
                for ID in self.get_features(typ)] for typ,attr in self.config['global_state']])
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
            if 'cum' not in attribute and 'vol' not in attribute:
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
            return super().performance(metric)
        else:
            perf = self.data_log['performance_measure'][-seq:]
            default = np.zeros((len(self.get_features('nodes')),len(self.config['performance_targets'])))
            perf = [default for _ in range(seq-len(perf))] + perf
            return np.array(perf)

    def reset(self,swmm_file=None, global_state=False,seq=False):
        # clear the data log and reset the environment
        if swmm_file is not None:
            self.config["swmm_input"] = swmm_file
        if not hasattr(self,'env') or swmm_file is not None:
            self.env = env_sq(self.config, ctrl=True)
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
                    self.data_log[attribute][idx] =  [] if maxlen is None else deque(maxlen=maxlen)
            
        if self.global_state:
            for typ,attribute in config['global_state']:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                for ID in self.get_features(typ):
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
        # state shape
        args['state_shape'] = (len(self.get_features('nodes')),len(self.config['global_state'])) if self.global_state else len(args['states'])
        
        nodes = self.get_features('nodes')
        inp = read_inp_file(self.config['swmm_input'])
        args['hmax'] = np.array([inp.JUNCTIONS[node].MaxDepth for node in nodes])

        if self.global_state:
            args['edges'] = self.get_edge_list()
        return args


    # TODO: getters Use pyswmm api
    def get_features(self,kind='nodes',no_out=True):
        inp = read_inp_file(self.config['swmm_input'])
        labels = {'nodes':NODE_SECTIONS,'links':LINK_SECTIONS}
        features = []
        for label in labels[kind]:
            if no_out and label == 'OUTFALLS':
                continue
            elif label in inp:
                features += list(getattr(inp,label))            
        return features
    
    def get_edge_list(self):
        inp = read_inp_file(self.config['swmm_input'])
        nodes = self.get_features('nodes')
        edges = []
        for label in LINK_SECTIONS:
            if label in inp:
                edges += [(nodes.index(link.FromNode),nodes.index(link.ToNode))
                 for link in getattr(inp,label).values() if link.FromNode in nodes and link.ToNode in nodes]
        return np.array(edges)

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

        
        
