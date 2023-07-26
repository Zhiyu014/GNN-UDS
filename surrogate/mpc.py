from emulator import Emulator # Emulator should be imported before env
from utilities import get_inp_files
from swmm_api import read_inp_file,read_rpt_file
from itertools import product
import os,time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import argparse,yaml
from envs import get_env
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
HERE = os.path.dirname(__file__)

def parser(config=None):
    parser = argparse.ArgumentParser(description='mpc')
    parser.add_argument('--env',type=str,default='astlingen',help='set drainage scenarios')
    
    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')
    parser.add_argument('--pop_size',type=int,default=32,help='number of population')
    parser.add_argument('--use_current',action="store_true",help='if use current setting as initial')
    parser.add_argument('--sampling',type=float,default=0.4,help='sampling rate')
    parser.add_argument('--crossover',nargs='+',type=float,default=[1.0,3.0],help='crossover rate')
    parser.add_argument('--mutation',nargs='+',type=float,default=[1.0,3.0],help='mutation rate')
    parser.add_argument('--termination',nargs='+',type=str,default=['n_eval','256'],help='Iteration termination criteria')
    
    parser.add_argument('--surrogate',action='store_true',help='if use surrogate for dynamic emulation')
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')

    parser.add_argument('--result_dir',type=str,default='./result/',help='path of the control results')
    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        hyp = {k:v for k,v in hyps[args.env].items() if hasattr(args,k)}
        parser.set_defaults(**hyp)
    args = parser.parse_args()
    config = {k:v for k,v in args.__dict__.items() if v!=hyp.get(k)}
    for k,v in config.items():
        if 'dir' in k:
            setattr(args,k,os.path.join(hyp[k],v))
    args.termination[-1] = eval(args.termination[-1]) if args.termination[0] != 'time' else args.termination[-1]

    print('MPC configs: {}'.format(args))
    return args,config

class mpc_problem(Problem):
    def __init__(self,args,eval_file=None,margs=None):
        self.args = args
        self.file = eval_file
        if margs is not None:
            self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            self.emul.load(margs.model_dir)
            self.state,self.runoff,self.edge_state = margs.state,margs.runoff,getattr(margs,"edge_state",None)
        self.n_act = len(args.action_space)
        self.actions = [{i:v for i,v in enumerate(val)}
                         for val in args.action_space.values()]
        self.step = args.interval
        self.eval_hrz = args.prediction['eval_horizon']
        self.n_step = args.prediction['control_horizon']//args.control_interval
        self.r_step = args.control_interval//args.interval
        self.n_var = self.n_act*self.n_step
        self.n_obj = 1
            
        super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                         xl = np.array([0 for _ in range(self.n_var)]),
                         xu = np.array([len(v)-1 for _ in range(self.n_step)
                            for v in self.actions]),
                         vtype=int)

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_act))
        y = np.concatenate([np.repeat(y[i:i+1,:],self.r_step,axis=0) for i in range(self.n_step)])
        if y.shape[0] < self.eval_hrz // self.step:
            y = np.concatenate([y,np.repeat(y[-1:,:],self.eval_hrz // self.step-y.shape[0],axis=0)],axis=0)

        env = get_env(self.args.env)(swmm_file = self.file)
        done = False
        idx = 0
        perf = 0
        while not done and idx < y.shape[0]:
            yi = y[idx]
            done = env.step([self.actions[i][act] for i,act in enumerate(yi)])
            perf += env.performance().sum()
            idx += 1
        return perf
    
    def pred_emu(self,y):
        y = y.reshape((-1,self.n_step,self.n_act))
        settings = np.stack([np.vectorize(self.actions[i].get)(y[...,i]) for i in range(self.n_act)],axis=-1)
        settings = np.concatenate([np.repeat(settings[:,i:i+1,...],self.r_step,axis=1) for i in range(self.n_step)],axis=1)
        if settings.shape[1] < self.runoff.shape[0]:
            # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
            settings = np.concatenate([settings,np.repeat(settings[:,-1:,:],self.runoff.shape[0]-settings.shape[1],axis=1)],axis=1)
        state,runoff = np.repeat(np.expand_dims(self.state,0),y.shape[0],axis=0),np.repeat(np.expand_dims(self.runoff,0),y.shape[0],axis=0)
        edge_state = np.repeat(np.expand_dims(self.edge_state,0),y.shape[0],axis=0) if self.edge_state is not None else None
        preds = self.emul.predict(state,runoff,settings,edge_state)
        q_w = preds[0][...,-1] if self.emul.use_edge else preds[...,-1]
        return q_w.sum(axis=1).sum(axis=1)

        
    def _evaluate(self,x,out,*args,**kwargs):        
        if hasattr(self,'emul'):
            out['F'] = self.pred_emu(x)
        else:
            pool = mp.Pool(self.args.processes)
            res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
            pool.close()
            pool.join()
            F = [r.get() for r in res]
            out['F'] = np.array(F)

def initialize(x0,xl,xu,pop_size,prob):
    x0 = np.reshape(x0,-1)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [np.random.randint(xl[idx],xu[idx]+1) if np.random.random()<prob else x for idx,x in enumerate(x0)]
        population.append(xi)
    return np.array(population)

def get_runoff(env,event,rate=False):
    _ = env.reset(event,global_state=True)
    runoffs = []
    done = False
    while not done:
        done = env.step()
        if rate:
            runoff = [[env.env._getNodeLateralInflow(node)
                        if not env.env._isFinished else 0.0]
                       for node in env.elements['nodes']]
        else:
            runoff = env.state()[...,-1:]
        runoffs.append(runoff)
    runoff = np.array(runoffs)
    return runoff

def pred_simu(y,file,args,r=None):
    n_step = args.prediction['control_horizon']//args.control_interval
    r_step = args.control_interval//args.interval
    e_hrz = args.prediction['eval_horizon'] // args.interval
    actions = list(args.action_space.values())

    y = y.reshape(n_step,len(args.action_space))
    y = np.concatenate([np.repeat(y[i:i+1,:],r_step,axis=0) for i in range(n_step)])
    if y.shape[0] < e_hrz:
        y = np.concatenate([y,np.repeat(y[-1:,:], e_hrz - y.shape[0],axis=0)],axis=0)
    
    env = get_env(args.env_name)(swmm_file = file)
    done = False
    idx = 0
    perf = []
    while not done and idx < y.shape[0]:
        if args.prediction['no_runoff']:
            for node,ri in zip(env.elements['nodes'],r[idx]):
                env.env._setNodeInflow(node,ri)
        done = env.step([actions[i][int(act)] for i,act in enumerate(y[idx])])
        perf.append(env.performance())
        idx += 1
    return np.array(perf)

def run_ea(args,margs=None,eval_file=None,setting=None):
    if margs is not None:
        prob = mpc_problem(args,margs=margs)
    elif eval_file is not None:
        prob = mpc_problem(args,eval_file=eval_file)
    else:
        raise AssertionError('No margs or file claimed')

    if args.use_current and setting is not None:
        setting *= prob.n_step
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling)
    else:
        sampling = IntegerRandomSampling()
    crossover = SBX(*args.crossover,vtype=int,repair=RoundingRepair())
    mutation = PM(*args.mutation,vtype=int,repair=RoundingRepair())
    termination = get_termination(*args.termination)

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    
    res = minimize(prob,
                   method,
                   termination = termination,
                   verbose = True)
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)

    # Multiple solutions with the same performance
    # Choose the minimum changing
    # if res.X.ndim == 2:
    #     X = res.X[:,:prob.n_pump]
    #     chan = (X-np.array(settings)).sum(axis=1)
    #     ctrls = res.X[chan.argmin()]
    # else:
    ctrls = res.X
    ctrls = ctrls.reshape((prob.n_step,prob.n_act)).tolist()
    return ctrls[0]
    

if __name__ == '__main__':
    args,config = parser('config.yaml')

    # de = {'env':'astlingen',
    #       'processes':5,
    #       'pop_size':64,
    #       'sampling':0.4,
    #       'termination':['n_eval','256'],
    #       'surrogate':True,
    #       'model_dir':'./model/astlingen/10s_20k_act_edge_res_norm_flood',
    #       'result_dir':'./results/astlingen/10s_20k_mpc_edge_res_norm_flood'}
    # for k,v in de.items():
    #     setattr(args,k,v)

    env = get_env(args.env)()
    env_args = env.get_args()
    for k,v in env_args.items():
        setattr(args,k,v)
    args.act = 'mpc'

    events = get_inp_files(env.config['swmm_input'],env.config['rainfall'])

    if args.surrogate:
        hyps = yaml.load(open('config.yaml','r'),yaml.FullLoader)
        margs = argparse.Namespace(**hyps[args.env])
        margs.model_dir = args.model_dir
        known_hyps = yaml.load(open(os.path.join(margs.model_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in list(known_hyps.items())+list(env_args.items()):
            if k == 'model_dir':
                continue
            setattr(margs,k,v)
        margs.use_edge = margs.use_edge or margs.edge_fusion

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))

    # for event in events:
    event = events[0]
    name = os.path.basename(event).strip('.inp')
    t0 = time.time()
    if args.surrogate:
        runoff = get_runoff(env,event)
        horizon = args.prediction['eval_horizon']//args.interval
        runoff = np.stack([np.concatenate([runoff[idx:idx+horizon],np.tile(np.zeros_like(s),(max(idx+horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                            for idx,s in enumerate(runoff)])
    t1 = time.time()
    state = env.reset(event,global_state=True,seq=margs.seq_in if args.surrogate else False)
    perf = env.performance(seq=margs.seq_in if args.surrogate else False)
    states,perfs = [state[-1] if args.surrogate else state],[perf[-1] if args.surrogate else perf]

    edge_state = env.state_full(typ='links',seq=margs.seq_in if args.surrogate else False)
    edge_states = [edge_state[-1] if args.surrogate else edge_state]
    
    setting = [1 for _ in env.config['action_space']]
    settings = [setting]
    done,i = False,0
    while not done:
        if i*args.interval % args.control_interval == 0:
            if args.surrogate:
                if margs.if_flood:
                    f = (perf>0).astype(int)
                    f = np.eye(2)[f].squeeze(-2)
                    state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
                margs.state = state
                margs.runoff = runoff[i]
                if margs.use_edge:
                    margs.edge_state = edge_state
                setting = run_ea(args,margs=margs,setting=setting)
            else:
                eval_file = env.get_eval_file()
                setting = run_ea(args,eval_file=eval_file,setting=setting)
            done = env.step(setting)
        else:
            done = env.step()
        state = env.state(seq=margs.seq_in if args.surrogate else False)
        perf = env.performance(seq=margs.seq_in if args.surrogate else False)
        edge_state = env.state_full(margs.seq_in if args.surrogate else False,'links')
        states.append(state[-1] if args.surrogate else state)
        perfs.append(perf[-1] if args.surrogate else perf)
        edge_states.append(edge_state[-1] if args.surrogate else edge_state)
        settings.append(setting)
        i += 1
        print('Simulation time: %s'%env.data_log['simulation_time'][-1])            
    print('Runoff : {} s Simulation: {} s per step'.format(t1-t0,(time.time()-t1)/i))
    
    item = 'emul' if args.surrogate else 'simu'
    np.save(os.path.join(args.result_dir,name + '_%s_state.npy'%item),np.array(states))
    np.save(os.path.join(args.result_dir,name + '_%s_perf.npy'%item),np.array(perfs))
    np.save(os.path.join(args.result_dir,name + '_%s_settings.npy'%item),np.array(settings))
    np.save(os.path.join(args.result_dir,name + '_%s_edge_states.npy'%item),np.array(edge_states))



