import os,time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from emulator import Emulator # Emulator should be imported before env
from predictor import Predictor
from utils.utilities import get_inp_files
import pandas as pd
import multiprocessing as mp
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from scipy.optimize import minimize as scioptminimize
import argparse,yaml
from envs import get_env
from pymoo.optimize import minimize as pymoominimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling,random_by_bounds
from pymoo.operators.sampling.lhs import LatinHypercubeSampling,sampling_lhs
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.binx import BX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
HERE = os.path.dirname(__file__)

class TerminationCollection(TerminationCollection):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _update(self, algorithm):
        return max([termination.update(algorithm) for termination in self.terminations])

def parser(config=None):
    parser = argparse.ArgumentParser(description='mpc')
    parser.add_argument('--env',type=str,default='astlingen',help='set drainage scenarios')
    parser.add_argument('--directed',action='store_true',help='if use directed graph')
    parser.add_argument('--length',type=float,default=0,help='adjacency range')
    parser.add_argument('--order',type=int,default=1,help='adjacency order')
    parser.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
    parser.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
    parser.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')
    parser.add_argument('--swmm_step',type=int,default=30,help='routing step for swmm inp files')

    parser.add_argument('--setting_duration',type=int,default=5,help='setting duration')
    parser.add_argument('--control_interval',type=int,default=5,help='control interval')
    parser.add_argument('--horizon',type=int,default=60,help='control horizon')
    parser.add_argument('--act',type=str,default='rand',help='what control actions')

    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')
    parser.add_argument('--pop_size',type=int,default=32,help='number of population')
    parser.add_argument('--use_current',action="store_true",help='if use current setting as initial')
    parser.add_argument('--sampling',type=float,default=0.4,help='sampling rate')
    parser.add_argument('--learning_rate',type=float,default=0.1,help='learning rate of gradient')
    parser.add_argument('--crossover',nargs='+',type=float,default=[1.0,3.0],help='crossover rate')
    parser.add_argument('--mutation',nargs='+',type=float,default=[1.0,3.0],help='mutation rate')
    parser.add_argument('--termination',nargs='+',type=str,default=['n_eval','256'],help='Iteration termination criteria')
    
    parser.add_argument('--surrogate',action='store_true',help='if use surrogate for dynamic emulation')
    parser.add_argument('--predict',action='store_true',help='if use predictor for emulation')
    parser.add_argument('--gradient',action='store_true',help='if use gradient-based optimization')
    parser.add_argument('--lbfgsb',action='store_true',help='if use Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded optimization')
    parser.add_argument('--cross_entropy',type=float,default=0,help=' if use cross-entropy method and the kept percentage')
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')
    parser.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')
    parser.add_argument('--result_dir',type=str,default='./result/',help='path of the control results')

    # Only used to keep the same condition to test internal model efficiency
    parser.add_argument('--keep',type=str,default='False',help='if keep the default trajectory')
    
    # stochastic MPC for surrogate-based internal model
    parser.add_argument('--stochastic',type=int,default=0,help='number of stochastic scenarios')
    parser.add_argument('--error',type=float,default=0.0,help='error range of stochastic scenarios')
    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        hyp = {k:v for k,v in hyps[args.env].items() if hasattr(args,k)}
        parser.set_defaults(**hyp)
    if args.act.endswith('bin'):
        parser.set_defaults(**{'crossover':[0.5,2],'mutation':[0.5,0.3]})
    args = parser.parse_args()
    config = {k:v for k,v in args.__dict__.items() if v!=hyp.get(k)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyp[k],v))
    for i in range(len(args.termination)//2):
        args.termination[2*i+1] = eval(args.termination[2*i+1]) if args.termination[2*i] not in ['time','soo'] else args.termination[2*i+1]

    print('MPC configs: {}'.format(args))
    return args,config

def get_runoff(env,event,rate=False,tide=False):
    _ = env.reset(event,global_state=True)
    runoffs = []
    t0 = env.env.methods['simulation_time']()
    done = False
    while not done:
        done = env.step()
        if rate:
            runoff = np.array([[env.env._getNodeLateralinflow(node)
                        if not env.env._isFinished else 0.0]
                       for node in env.elements['nodes']])
        else:
            runoff = env.state_full()[...,-1:]
        if tide:
            ti = env.state_full()[...,:1]
            runoff = np.concatenate([runoff,ti],axis=-1)
        runoffs.append(runoff)
    ts = [t0]+env.data_log['simulation_time'][:-1]
    runoff = np.array(runoffs)
    return ts,runoff

def pred_simu(y,file,args,r=None,act=True):
    # n_step = args.prediction['control_horizon']//args.setting_duration
    # r_step = args.setting_duration//args.interval
    # e_hrz = args.prediction['eval_horizon'] // args.interval
    # actions = list(args.action_space.values())

    # y = y.reshape(n_step,len(args.action_space))
    # y = np.concatenate([np.repeat(y[i:i+1,:],r_step,axis=0) for i in range(n_step)])
    # if y.shape[0] < e_hrz:
    #     y = np.concatenate([y,np.repeat(y[-1:,:], e_hrz - y.shape[0],axis=0)],axis=0)
    
    env = get_env(args.env_name)(swmm_file = file)
    done,idx = False,0
    if getattr(args,'log') is not None:
        env.data_log.update({k:v for k,v in args.log.items() if 'cum' not in k})
    # perf = []
    while not done and idx < (y.shape[0] if act else args.prediction['eval_horizon']):
        if args.prediction['no_runoff']:
            for node,ri in zip(env.elements['nodes'],r[idx]):
                env.env._setNodeInflow(node,ri)
        # done = env.step([actions[i][int(act)] for i,act in enumerate(y[idx])])
        done = env.step([sett for sett in y[idx]] if act else None)
        # perf.append(env.flood())
        idx += 1
    # return np.array(perf)
    return env.objective(idx)

class mpc_problem(Problem):
    def __init__(self,args,margs=None):
        self.args = args
        if margs is not None:
            tf.keras.backend.clear_session()    # to clear backend occupied models
            if args.predict:
                self.emul = Predictor(margs.recurrent,margs)
            else:
                self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            self.emul.load(margs.model_dir)
            self.stochastic = getattr(args,"stochastic",False)
        self.step = args.interval
        self.eval_hrz = args.prediction['eval_horizon']
        self.n_step = args.prediction['control_horizon']//args.setting_duration
        self.r_step = args.setting_duration//args.interval
        self.n_obj = 1
        self.env = get_env(args.env)(initialize=False)
        if args.act.startswith('conti'):
            self.n_act = len(args.action_space)
            self.n_var = self.n_act*self.n_step
            super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                            xl = np.array([min(v) for _ in range(self.n_step)
                                            for v in args.action_space.values()]),
                            xu = np.array([max(v) for _ in range(self.n_step)
                                           for v in args.action_space.values()]),
                            vtype=float)
        else:
            self.actions = args.action_table
            self.n_act = np.array(list(self.actions)).shape[-1]
            self.n_var = self.n_act*self.n_step
            # self.actions = [{i:v for i,v in enumerate(val)}
            #                 for val in args.action_space.values()]            
            super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                            xl = np.zeros(self.n_var),
                            # xu = np.array([len(v)-1 for _ in range(self.n_step)
                            #     for v in self.actions]),
                            xu = np.array([v for _ in range(self.n_step)
                                for v in np.array(list(self.actions.keys())).max(axis=0)]),
                            vtype=bool if args.act.endswith('bin') else int)
            # print(self.n_var,self.xu)
    
    def load_state(self,state,runoff,edge_state):
        self.state,self.runoff,self.edge_state = state,runoff,edge_state
    
    def load_file(self,eval_file,log=None,runoff_rate=None):
        self.file,self.runoff_rate,self.log = eval_file,runoff_rate,log

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_act))
        y = np.repeat(y,self.r_step,axis=0)
        if y.shape[0] < self.eval_hrz // self.step:
            y = np.concatenate([y,np.repeat(y[-1:,:],self.eval_hrz // self.step-y.shape[0],axis=0)],axis=0)

        _ = self.env.reset(swmm_file = self.file)
        if getattr(self,'log') is not None:
            self.env.data_log.update({k:v for k,v in self.log.items() if 'cum' not in k})
        done,idx = False,0
        # perf = 0
        while not done and idx < y.shape[0]:
            if self.args.prediction['no_runoff']:
                for node,ri in zip(self.env.elements['nodes'],self.runoff_rate[idx]):
                    self.env.env._setNodeInflow(node,ri)
            yi = y[idx]
            # done = env.step([act if self.args.act.startswith('conti') else self.actions[i][act] for i,act in enumerate(yi)])
            done = self.env.step(yi if self.args.act.startswith('conti') else self.actions[tuple(yi.astype(int))])
            # perf += env.performance().sum()
            idx += 1
        return self.env.objective(idx).sum()
    
    def pred_emu(self,y):
        y = y.reshape((-1,self.n_step,self.n_act))
        pop_size = y.shape[0]
        # settings = y if self.args.act.startswith('conti') else np.stack([np.vectorize(self.actions[i].get)(y[...,i]) for i in range(self.n_act)],axis=-1)
        settings = y if self.args.act.startswith('conti') else np.apply_along_axis(lambda x:self.actions.get(tuple(x)),-1,y.astype(int))
        settings = np.repeat(settings,self.r_step,axis=1)
        if settings.shape[1] < self.eval_hrz // self.step:
            # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
            settings = np.concatenate([settings,np.repeat(settings[:,-1:,:],self.eval_hrz // self.step - settings.shape[1],axis=1)],axis=1)
        if self.stochastic:
            settings = np.repeat(settings,self.stochastic,axis=0)
            runoff = np.tile(self.runoff,(pop_size,)+tuple([1 for _ in range(self.runoff.ndim-1)]))
        else:
            runoff = np.repeat(np.expand_dims(self.runoff,0),pop_size,axis=0)
        state = np.repeat(np.expand_dims(self.state,0),settings.shape[0],axis=0)
        edge_state = np.repeat(np.expand_dims(self.edge_state,0),settings.shape[0],axis=0)
        preds = self.emul.predict(state,runoff,settings,edge_state)
        if self.args.predict:
            objs = preds.numpy().sum(axis=-1).sum(axis=-1)
            if not getattr(self.emul,'norm',False):
                objs = self.env.norm_obj(objs,[state,edge_state],inverse=True)
        else:
            objs = self.env.objective_pred(preds,[state,edge_state],settings).sum(axis=-1)
        return np.array([objs[i*self.stochastic:(i+1)*self.stochastic].mean() for i in range(pop_size)]) if self.stochastic else objs
        
    def _evaluate(self,x,out,*args,**kwargs):        
        if hasattr(self,'emul'):
            out['F'] = self.pred_emu(x)+1e-6
        else:
            pool = mp.Pool(self.args.processes)
            res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
            pool.close()
            pool.join()
            F = [r.get() for r in res]
            out['F'] = np.array(F)+1e-6

    def pred(self,x):
        if hasattr(self,'emul'):
            return self.pred_emu(x)+1e-6
        else:
            pool = mp.Pool(self.args.processes)
            res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
            pool.close()
            pool.join()
            F = [r.get() for r in res]
            return np.array(F)+1e-6

class BestCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["time"] = []
        self.t0 = time.time()

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())
        self.data["time"].append(time.time()-self.t0)

def initialize(x0,xl,xu,pop_size,prob,conti=False):
    x0 = np.reshape(x0,-1)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [np.random.uniform(xl[idx],xu[idx]) if conti else np.random.randint(xl[idx],xu[idx]+1)\
               if np.random.random()<prob else x for idx,x in enumerate(x0)]
        population.append(xi)
    return np.array(population)

def run_ea(prob,args,setting=None):
    print('Running genetic algorithm')
    # if margs is not None:
    #     prob = mpc_problem(args,margs=margs)
    # elif eval_file is not None:
    #     prob = mpc_problem(args,eval_file=eval_file)
    # else:
    #     raise AssertionError('No margs or file claimed')

    if args.use_current and setting is not None:
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling,args.act.startswith('conti'))
    else:
        sampling = LatinHypercubeSampling() if args.act.startswith('conti') else IntegerRandomSampling()
    if args.act.endswith('bin'):
        crossover = BX(*args.crossover,vtype=bool)
        mutation = BitflipMutation(*args.mutation,vtype=bool)
    else:
        crossover = SBX(*args.crossover,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
        mutation = PM(*args.mutation,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
    if len(args.termination) > 2:
        terms = {}
        for val in args.termination:
            if val in ['n_eval','n_gen','fmin','time','soo']:
                term = val
                terms[term] = {} if val =='soo' else None
            else:
                if isinstance(terms[term],dict):
                    terms[term][val.split('-')[0]] = eval(val.split('-')[1])
                else:
                    terms[term] = val
        termination = []
        for k,v in terms.items():
            termination.append(get_termination(k,**v) if isinstance(v,dict) else get_termination(k,v))
        termination = TerminationCollection(*termination)
    else:
        termination = get_termination(*args.termination)

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    print('Minimizing')
    res = pymoominimize(prob,
                        method,
                        termination = termination,
                        callback=BestCallback(),
                        verbose = True)
    # Multiple solutions with the same performance
    # Choose the minimum changing
    # if res.X.ndim == 2:
    #     X = res.X[:,:prob.n_pump]
    #     chan = (X-np.array(settings)).sum(axis=1)
    #     ctrls = res.X[chan.argmin()]
    # else:
    ctrls = res.X
    ctrls = ctrls.reshape((prob.n_step,prob.n_act))
    if not args.act.startswith('conti'):
        # ctrls = np.stack([np.vectorize(prob.actions[i].get)(ctrls[...,i]) for i in range(prob.n_act)],axis=-1)
        ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls.astype(int))
    print("Best solution found: %s" % ctrls.tolist())
    print("Function value: %s" % res.F)

    vals = res.algorithm.callback.data["best"]
    nfuns = args.pop_size*np.arange(1,len(vals)+1)
    times = res.algorithm.callback.data["time"]
    # if margs is not None:
    #     del prob.emul
    # del prob
    # gc.collect()
    return ctrls.tolist(),vals,nfuns,times

# TODO: cross-entropy method
def run_ce(prob,args,setting=None):
    print('Running cross entropy')
    def sample_conti(mu,sig,n=1):
        return np.array([truncnorm.rvs((prob.xl-mu)/sig,(prob.xu-mu)/sig)*sig+mu for _ in range(n)])
    def sample_disc(pr,n=1):
        pr = [np.array(pri)/np.sum(pri) for pri in pr]
        return np.array([np.random.choice(np.arange(pri.shape[0]),p=pri,size=n) for pri in pr]).T
    
    # Initialize
    if args.act.startswith('conti'):
        mu,sig = np.random.uniform(low=prob.xl,high=prob.xu,size=prob.n_var),np.random.uniform(low=0.001,size=prob.n_var)
    else:
        pr = [np.random.uniform(low=0.001,size=int(xui)+1) for xui in prob.xu]

    def if_terminate(item,vm,rec):
        if item == 'n_gen':
            return rec[0] >= vm
        elif item == 'obj':
            return rec[-2] <= vm  # TODO
        elif item == 'time':
            return rec[1] >= vm
        
    t0 = time.time()
    recs = [[0,0,1e6,1e6,1e6]]
    print('=======================================================================')
    print(' n_gen |      time     |     f_lam     |     f_min     |     g_min     ')
    print('=======================================================================')
    obj = np.random.uniform(size=(args.pop_size,))
    while not if_terminate(*args.termination,recs[-1]):
        x = sample_conti(mu,sig,args.pop_size) if args.act.startswith('conti') else sample_disc(pr,args.pop_size)
        obj = prob.pred(x)
        rec = [recs[-1][0]+1, time.time()-t0-recs[-1][1], 
               obj.mean(), 
               obj.min(), 
               min(obj.argsort()[int(args.pop_size*args.cross_entropy)],recs[-1][1])]
        if obj.min() < recs[-1][2]:
            ctrls = np.clip(x[obj.argmin()],prob.xl,prob.xu)
            ctrls = ctrls.reshape((prob.n_step,prob.n_act))
            if not args.act.startswith('conti'):
                # ctrls = np.stack([np.vectorize(prob.actions[i].get)(ctrls[...,i]) for i in range(prob.n_act)],axis=-1)
                ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls)
            ctrls = ctrls.tolist()
        # formulate a new distribution with elites
        x_new = x[obj<=rec[-1]]
        if args.act.startswith('conti'):
            mu,sig = x_new.mean(axis=0),x_new.std(axis=0)
        else:
            pr = [np.bincount(x_ni) for x_ni in x_new.T]
        
        log = ''.join([str(round(r,4)).center(7 if i == 0 else 15) + '|' for i,r in enumerate(rec)])
        print(log[:-1])
        rec[1] = time.time() - t0
        recs.append(rec)
    # print('Initial solution: ',sampling.reshape((-1,prob.n_step,prob.n_act)).tolist())
    print('Best solution: ',ctrls)
    vals,nfuns,times = np.array(recs)[1:,-2],np.arange(1,len(vals)+1)*args.pop_size,np.array(recs)[1:,1]
    return ctrls,vals,nfuns,times

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper
    
class mpc_problem_gr:
    def __init__(self,args,margs):
        self.args = args
        tf.keras.backend.clear_session()    # to clear backend occupied models
        if args.predict:
            self.emul = Predictor(margs.recurrent,margs)
        else:
            self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
        self.emul.load(margs.model_dir)
        # self.load_state(margs)
        self.cross_entropy = bool(getattr(args,"cross_entropy",False))
        self.stochastic = getattr(args,"stochastic",False)
        self.asp = list(args.action_space.values())
        self.n_act = len(self.asp)
        self.step = args.interval
        self.eval_hrz = args.prediction['eval_horizon']
        self.n_step = args.prediction['control_horizon']//args.setting_duration
        self.r_step = args.setting_duration//args.interval
        self.env = get_env(args.env)(initialize=False)
        self.optimizer = Adam(getattr(args,"learning_rate",1e-3))
        self.pop_size = getattr(args,'pop_size',64)
        self.n_var = self.n_act*self.n_step
        self.xl = np.array([min(v) for _ in range(self.n_step)
                            for v in self.asp])
        self.xu = np.array([max(v) for _ in range(self.n_step)
                            for v in self.asp])

    def load_state(self,state,runoff,edge_state):
        self.state,self.runoff,self.edge_state = [tf.convert_to_tensor(x) for x in [state,runoff,edge_state]]

    def initialize(self,sampling):
        if not hasattr(self,'y'):
            self.y = tf.Variable(sampling)
        else:
            self.y.assign(sampling)
        self.train_vars = [self.y]

    def initialize_distr(self,sampling):
        if not hasattr(self,'distr'):
            self.distr = tfd.TruncatedNormal(loc=tf.Variable(sampling[0,:]),scale=tf.Variable(sampling[1,:]),
                                             low=self.xl,high=self.xu)
        else:
            self.distr.loc.assign(sampling[0,:])
            self.distr.scale.assign(sampling[1,:])
        self.train_vars = [self.distr.loc,self.distr.scale]

    # TODO: distr.sample does not work in autograph
    # @tf.function
    @call_counter
    def pred_fit(self):
        # assert getattr(self,'y',None) is not None
        if self.stochastic:
            runoff = tf.cast(tf.tile(self.runoff,(self.pop_size,)+tuple([1 for _ in range(self.runoff.ndim-1)])),tf.float32)
        else:
            runoff = tf.cast(tf.repeat(tf.expand_dims(self.runoff,0),self.pop_size,axis=0),tf.float32)
        state = tf.cast(tf.repeat(tf.expand_dims(self.state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32)
        edge_state = tf.cast(tf.repeat(tf.expand_dims(self.edge_state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(self.train_vars)
            if self.cross_entropy:
                # distr = tfd.TruncatedNormal(loc=tf.clip_by_value(self.train_vars[0],self.xl,self.xu),
                #                             scale=tf.clip_by_value(self.train_vars[1],0,np.nan),
                #                             low=self.xl,high=self.xu)
                # self.x = distr.sample(self.pop_size)
                self.train_vars[0].assign(tf.clip_by_value(self.train_vars[0],self.xl,self.xu))
                self.train_vars[1].assign(tf.clip_by_value(self.train_vars[1],1e-3,np.inf))
                self.y = self.distr.sample(self.pop_size)
            settings = tf.reshape(tf.clip_by_value(self.y,self.xl,self.xu),(-1,self.n_step,self.n_act))
            settings = tf.cast(tf.repeat(settings,self.r_step,axis=1),tf.float32)
            if settings.shape[1] < self.eval_hrz // self.step:
                # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
                settings = tf.concat([settings,tf.repeat(settings[:,-1:,:],self.eval_hrz // self.step-settings.shape[1],axis=1)],axis=1)
            if self.stochastic:
                settings = tf.repeat(settings,self.stochastic,axis=0)
            preds = self.emul.predict_tf(state,runoff,settings,edge_state)
            if not self.args.predict:
                obj = tf.reduce_sum(tf.reduce_sum(preds,axis=-1),axis=-1)
                if not getattr(self.emul,'norm',False):
                    obj = self.env.norm_obj(obj,[state,edge_state],inverse=True)
            else:
                obj = self.env.objective_pred_tf(preds,[state,edge_state],settings)
            if self.stochastic:
                obj = tf.stack([tf.reduce_mean(obj[i*self.stochastic:(i+1)*self.stochastic],axis=0) for i in range(self.pop_size)])
            loss = tf.reduce_mean(obj,axis=0) if self.cross_entropy else obj
        grads = tape.gradient(loss,self.train_vars)
        self.optimizer.apply_gradients(zip(grads,self.train_vars))
        return obj,grads

    @tf.function
    def objective_fn(self,y,runoff,state,edge_state):
        runoff = tf.cast(runoff if self.stochastic else tf.expand_dims(runoff,0),tf.float32)
        state = tf.cast(tf.repeat(tf.expand_dims(state,0),max(self.stochastic,1),axis=0),tf.float32)
        edge_state = tf.cast(tf.repeat(tf.expand_dims(edge_state,0),max(self.stochastic,1),axis=0),tf.float32)
        settings = tf.reshape(y,(-1,self.n_step,self.n_act))
        settings = tf.cast(tf.repeat(settings,self.r_step,axis=1),tf.float32)
        if settings.shape[1] < self.eval_hrz // self.step:
            # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
            settings = tf.concat([settings,tf.repeat(settings[:,-1:,:],self.eval_hrz // self.step-settings.shape[1],axis=1)],axis=1)
        if self.stochastic:
            settings = tf.repeat(settings,self.stochastic,axis=0)
        preds = self.emul.predict_tf(state,runoff,settings,edge_state)
        if self.args.predict:
            obj = tf.reduce_sum(tf.reduce_sum(preds,axis=-1),axis=-1)
            if not getattr(self.emul,'norm',False):
                obj = self.env.norm_obj(obj,[state,edge_state],inverse=True)
        else:
            obj = self.env.objective_pred_tf(preds,[state,edge_state],settings)
        return tf.reduce_mean(obj)

    @call_counter
    @tf.function
    def gradient_fn(self,y,runoff,state,edge_state):
        y = tf.convert_to_tensor(y)
        with tf.GradientTape() as tape:
            tape.watch(y)
            obj = self.objective_fn(y,runoff,state,edge_state)
        grads = tape.gradient(obj,y)
        return obj,grads

    def gradient_fn_parallel(self,y):
        y = tf.convert_to_tensor(y)
        if self.stochastic:
            runoff = tf.cast(tf.tile(self.runoff,(self.pop_size,)+tuple([1 for _ in range(self.runoff.ndim-1)])),tf.float32)
        else:
            runoff = tf.cast(tf.repeat(tf.expand_dims(self.runoff,0),self.pop_size,axis=0),tf.float32)
        state = tf.cast(tf.repeat(tf.expand_dims(self.state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32)
        edge_state = tf.cast(tf.repeat(tf.expand_dims(self.edge_state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(y)
            settings = tf.tanh(y) * (self.xu - self.xl) / 2 + (self.xu + self.xl) / 2
            settings = tf.reshape(settings,(-1,self.n_step,self.n_act))
            settings = tf.cast(tf.repeat(settings,self.r_step,axis=1),tf.float32)
            if settings.shape[1] < self.eval_hrz // self.step:
                # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
                settings = tf.concat([settings,tf.repeat(settings[:,-1:,:],self.eval_hrz // self.step-settings.shape[1],axis=1)],axis=1)
            if self.stochastic:
                settings = tf.repeat(settings,self.stochastic,axis=0)
            preds = self.emul.predict_tf(state,runoff,settings,edge_state)
            if self.args.predict:
                obj = tf.reduce_sum(tf.reduce_sum(preds,axis=-1),axis=-1)
                if not getattr(self.emul,'norm',False):
                    obj = self.env.norm_obj(obj,[state,edge_state],inverse=True)
            else:
                obj = self.env.objective_pred_tf(preds,[state,edge_state],settings)
            if self.stochastic:
                obj = tf.stack([tf.reduce_mean(obj[i*self.stochastic:(i+1)*self.stochastic],axis=0) for i in range(self.pop_size)])
        grads = tape.gradient(obj,y)
        return obj,grads
    
def run_gr(prob,args,setting=None):
    print('Running gradient inversion')
    # prob = mpc_problem_gr(args,margs)
    if args.use_current and setting is not None:
        sampling = initialize(setting,prob.xl,prob.xu,2 if args.cross_entropy else args.pop_size,args.sampling,conti=True)
    else:
        sampling = sampling_lhs(2 if args.cross_entropy else args.pop_size,prob.n_var,prob.xl,prob.xu)
    def if_terminate(item,vm,rec):
        if item == 'n_gen':
            return rec[0] >= vm
        elif item == 'obj':
            return rec[-2] <= vm # TODO
        elif item == 'grad':
            return rec[2] <= vm
        elif item == 'time':
            return rec[1] >= vm

    t0 = time.time()
    recs = [[0,prob.pred_fit.calls*args.pop_size,0,1e6,1e6,1e6,1e6]]
    print('=======================================================================================')
    print(' n_gen |      time     |     grads     |     f_avg     |     f_min     |     g_min     ')
    print('=======================================================================================')
    prob.initialize_distr(sampling) if args.cross_entropy else prob.initialize(sampling)
    obj = tf.random.uniform((args.pop_size,))
    while not if_terminate(*args.termination,recs[-1]):
        obj,grads = prob.pred_fit()
        rec = [recs[-1][0]+1, 
               prob.pred_fit.calls*args.pop_size-recs[-1][1], 
               time.time()-t0-recs[-1][2], 
               np.mean([grad.numpy().mean() for grad in grads]), 
               obj.numpy().mean(), 
               obj.numpy().min(), 
               min(recs[-1][-1],obj.numpy().min())]
        if obj.numpy().min() < recs[-1][-1]:
            ctrls = prob.y[obj.numpy().argmin()].numpy()
            ctrls = np.clip(ctrls,prob.xl,prob.xu)
            ctrls = ctrls.reshape((prob.n_step,prob.n_act)).tolist()
        log = ''.join([str(round(r,4)).center(7 if i == 0 else 15) + '|' for i,r in enumerate(rec)])
        print(log[:-1])
        rec[1],rec[2] = prob.pred_fit.calls*args.pop_size,time.time()-t0
        recs.append(rec)
    # print('Initial solution: ',sampling.reshape((-1,prob.n_step,prob.n_act)).tolist())
    print('Best solution: ',ctrls)
    vals,nfuns,times = np.array(recs)[1:,-1],np.array(recs)[1:,1]-prob.pred_fit.calls*args.pop_size,np.array(recs)[1:,2]
    return ctrls,vals,nfuns,times

# TODO: normalization in GR and LBFGS
def run_lbfgs(prob,args,setting=None):
    '''
    TODO: GR/BFGS are still local optimization methods, and a global optimization method is needed to combine them. Try SHGO/basinhopping/etc.
    tfp has no callbacks, while scipy.optimize.minimize has
    scipy has L-BFGS-B for bounded optimization, but tfp only has L-BFGS and a tanh transformation is used
    it seems tfp can do batch parallelization with multiple initial values
    needs manual implementation for l-bfbs-b if wanna use GPU parallel of multiple initial values
    '''
    print('Running BFGS inversion')
    if args.use_current and setting is not None:
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling,conti=True)
    else:
        sampling = sampling_lhs(args.pop_size,prob.n_var,prob.xl,prob.xu)
    if args.pop_size > 1:
        t0 = time.time()
        sampling = tf.atanh((tf.constant(sampling,dtype=tf.float32) - (prob.xu + prob.xl) / 2) * 2 / (prob.xu - prob.xl))
        results = tfp.optimizer.lbfgs_minimize(prob.gradient_fn_parallel,
                                               initial_position=sampling,)
        print("Optimization converged: " + str(results.converged.numpy()))
        print("Optimization failed: " + str(results.failed.numpy()))
        vals = results.objective_value.numpy()
        ctrls = results.position.numpy()[vals.argmin()]
        ctrls = np.tanh(ctrls) * (prob.xu - prob.xl) / 2 + (prob.xu + prob.xl) / 2
        vals = np.array([vals.min()]*max(results.num_iterations.numpy(),1))
        times = np.linspace(0,time.time()-t0,max(results.num_iterations.numpy(),1))
    else:
        recs = [[0,prob.gradient_fn.calls,time.time(),1e6]]
        print('===============================================')
        print(' n_gen | n_fun |     time      |       f       ')
        print('===============================================')
        def mycallback(intermediate_result):
            obj = getattr(intermediate_result,'fun',np.nan)
            # TODO: find a way to count calls of gradient_fn & gradient_fn_parallel
            nfev = prob.gradient_fn.calls
            rec = [recs[-1][0]+1, nfev-recs[-1][1], time.time()-recs[-1][2], obj]
        # def mycallback(x,obj=None,*args,**kwargs):
        #     obj = prob.objective_fn(x,prob.runoff,prob.state,prob.edge_state).numpy() if obj is None else obj
        #     rec = [recs[-1][0]+1, time.time()-recs[-1][1], min(obj,recs[-1][-2]), obj]
            log = ''.join([str(round(r,4)).center(7 if i < 1 else 15) + '|' for i,r in enumerate(rec)])
            print(log[:-1])
            rec[1],rec[2] = nfev,time.time()
            recs.append(rec)
        results = scioptminimize(lambda x,r,s,e: tuple([re.numpy() for re in prob.gradient_fn(x,r,s,e)]),
                                 args=(prob.runoff,prob.state,prob.edge_state),
                                 x0=sampling[0],
                                 method='L-BFGS-B',
                                 jac=True,
                                 bounds=list(zip(prob.xl,prob.xu)),
                                 callback=mycallback,
                                 options={"ftol":1e-25,"gtol":1e-10})
        print("Optimization {}".format("successful" if results.success else "failed"))
        ctrls = results.x
        vals = np.array(recs)[1:,-1]
        nfuns = np.array(recs)[1:,1]-recs[0][1]
        times = np.array(recs)[1:,2]-recs[0][2]
    ctrls = ctrls.astype(np.float32).reshape((prob.n_step,prob.n_act)).tolist()
    print('Best solution: ',ctrls)
    return ctrls,vals,nfuns,times

if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','mpc.yaml'))
    mp.set_start_method('spawn', force=True)    # use gpu in multiprocessing
    # ctx = mp.get_context("spawn")
    de = {
        # 'env':'astlingen',
        # 'act':'conti',
        # 'processes':4,
        # 'pop_size':1,
        # # 'sampling':0.4,
        # # 'learning_rate':0.1,
        # 'termination':['n_gen',200],
        # 'surrogate':True,
        # 'predict':False,
        # 'lbfgsb':True,
        # 'rain_dir':'./envs/config/ast_test5_events.csv',
        # 'rain_suffix':'chaohu_testall',
        # 'rain_num':100,
        # 'model_dir':'./model/astlingen/60s_50k_conti_1000ledgef_res_norm_flood_gat_5lyrs',
        # 'result_dir':'./results/astlingen/test',
        }
    # config['rain_dir'] = de['rain_dir']
    for k,v in de.items():
        setattr(args,k,v)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args(args.directed,args.length,args.order,act=args.act)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)
    setattr(args,'elements',env.elements)
    # args.act = 'mpc'

    rain_arg = env.config['rainfall']
    if 'rain_dir' in config:
        rain_arg['rainfall_events'] = args.rain_dir
    if 'rain_suffix' in config:
        rain_arg['suffix'] = args.rain_suffix
    if 'rain_num' in config:
        rain_arg['rain_num'] = args.rain_num
    events = get_inp_files(env.config['swmm_input'],rain_arg,swmm_step=args.swmm_step)

    if args.surrogate:
        hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'),'r'),yaml.FullLoader)
        margs = argparse.Namespace(**hyps[args.env])
        margs.model_dir = args.model_dir
        known_hyps = yaml.load(open(os.path.join(margs.model_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k == 'model_dir':
                continue
            setattr(margs,k,v)
        setattr(margs,'epsilon',args.epsilon)
        env_args = env.get_args(margs.directed,margs.length,margs.order)
        for k,v in env_args.items():
            setattr(margs,k,v)
        args.prediction['eval_horizon'] = args.prediction['control_horizon'] = margs.seq_out * args.interval
    else:
        args.prediction['eval_horizon'] = args.prediction['control_horizon'] = args.horizon * args.interval

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))

    results = pd.DataFrame(columns=['rr time','fl time','perf','objective'])
    item = 'emul' if args.surrogate else 'simu'
    # events = ['./envs/network/astlingen/astlingen_08_22_2007_21.inp']
    for event in events:
        name = os.path.basename(event).strip('.inp')
        if os.path.exists(os.path.join(args.result_dir,name + '_%s_state.npy'%item)):
            continue
        t0 = time.time()
        if args.surrogate:
            ts,runoff = get_runoff(env,event,tide=args.tide)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            horizon = args.prediction['eval_horizon']//args.interval
            runoff = np.stack([np.concatenate([runoff[idx:idx+horizon],np.tile(np.zeros_like(s),(max(idx+horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
        elif args.prediction['no_runoff']:
            ts,runoff_rate = get_runoff(env,event,True)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            horizon = args.prediction['eval_horizon']//args.interval
            runoff_rate = np.stack([np.concatenate([runoff_rate[idx:idx+horizon],np.tile(np.zeros_like(s),(max(idx+horizon-runoff_rate.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff_rate)])


        t1 = time.time()
        print('Runoff time: {} s'.format(t1-t0))
        opt_times = []
        state = env.reset(event,global_state=True,seq=margs.seq_in if args.surrogate else False)
        if args.surrogate and margs.if_flood:
            flood = env.flood(seq=margs.seq_in)
        states = [state[-1] if args.surrogate else state]
        perfs,objects = [env.flood()],[env.objective()]

        edge_state = env.state_full(typ='links',seq=margs.seq_in if args.surrogate else False)
        edge_states = [edge_state[-1] if args.surrogate else edge_state]
        
        # setting = [1 for _ in args.action_space]
        setting = [env.controller('default') for _ in range(args.prediction['control_horizon']//args.setting_duration)]
        settings = [env.controller(args.keep,states[0],setting[0]) if args.keep != 'False' else setting[0]]

        if args.surrogate and (args.lbfgsb or args.gradient):
            prob = mpc_problem_gr(args,margs)
        else:
            prob = mpc_problem(args,margs=margs if args.surrogate else None)

        done,i,j,valss,nfunss,timess = False,0,0,[],[],[]
        while not done:
            if i*args.interval % args.control_interval == 0:
                t2 = time.time()
                setting = setting[j+1:] + setting[-1:] * (j+1)
                if args.surrogate:
                    state[...,1] = state[...,1] - state[...,-1]
                    if margs.if_flood:
                        f = (flood>0).astype(float)
                        # f = np.eye(2)[f].squeeze(-2)
                        state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
                    t = env.env.methods['simulation_time']()
                    r = runoff[int(tss.asof(t)['Index'])]
                    if args.error > 0:
                        std = np.array([ri*args.error*i/r.shape[0] for i,ri in enumerate(r)])
                        if args.stochastic:
                            err = np.array([np.random.uniform(-std,std) for _ in range(args.stochastic)])
                            r = np.abs(np.tile(r,(args.stochastic,)+tuple([1 for _ in range(r.ndim)])) + err)
                        else:
                            r += np.random.uniform(-std,std)
                    # margs.state = state
                    # margs.runoff = r
                    prob.load_state(state,r,edge_state)
                    if args.gradient:
                        setting,vals,nfuns,times = run_gr(prob,args,setting=setting)
                    elif args.lbfgsb:
                        setting,vals,nfuns,times = run_lbfgs(prob,args,setting=setting)
                    elif args.cross_entropy:
                        setting,vals,nfuns,times = run_ce(prob,args,setting=setting)
                    else:
                        setting,vals,nfuns,times = run_ea(prob,args,setting=setting)
                else:
                    eval_file = env.get_eval_file(args.prediction['no_runoff'])
                    if args.prediction['no_runoff']:
                        t = env.env.methods['simulation_time']()
                        rr = runoff_rate[int(tss.asof(t)['Index']),...,0]
                    prob.load_file(eval_file,env.data_log,rr if args.prediction['no_runoff'] else None)
                    if args.cross_entropy:
                        setting,vals = run_ce(prob,args,setting=setting)
                    else:
                        setting,vals = run_ea(prob,args,setting=setting)
                valss.append(vals)
                nfunss.append(nfuns)
                timess.append(times)
                t3 = time.time()
                print('Optimization time: {} s'.format(t3-t2))
                opt_times.append(t3-t2)
                j = 0
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[j]) if args.keep == 'False' else settings[0]
            elif i*args.interval % args.setting_duration == 0:
                j += 1
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[j]) if args.keep == 'False' else settings[0]
            done = env.step(sett)
            state = env.state_full(seq=margs.seq_in if args.surrogate else False)
            if args.surrogate and margs.if_flood:
                flood = env.flood(seq=margs.seq_in)
            edge_state = env.state_full(margs.seq_in if args.surrogate else False,'links')
            states.append(state[-1] if args.surrogate else state)
            perfs.append(env.flood())
            objects.append(env.objective())
            edge_states.append(edge_state[-1] if args.surrogate else edge_state)
            settings.append(sett)
            i += 1
            print('Simulation time: %s'%env.data_log['simulation_time'][-1])            
        
        np.save(os.path.join(args.result_dir,name + '_%s_state.npy'%item),np.stack(states))
        np.save(os.path.join(args.result_dir,name + '_%s_perf.npy'%item),np.stack(perfs))
        np.save(os.path.join(args.result_dir,name + '_%s_object.npy'%item),np.array(objects))
        np.save(os.path.join(args.result_dir,name + '_%s_settings.npy'%item),np.array(settings))
        np.save(os.path.join(args.result_dir,name + '_%s_edge_states.npy'%item),np.stack(edge_states))
        np.save(os.path.join(args.result_dir,name + '_%s_vals.npy'%item),np.array(valss))
        np.save(os.path.join(args.result_dir,name + '_%s_nfuns.npy'%item),np.array(nfunss))
        np.save(os.path.join(args.result_dir,name + '_%s_times.npy'%item),np.array(timess))

        results.loc[name] = [t1-t0,np.mean(opt_times),np.stack(perfs).sum(),np.stack(objects).sum()]
    results.to_csv(os.path.join(args.result_dir,'results_%s.csv'%item))

