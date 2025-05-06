import os,time,math
import torch as th
from einops import rearrange,repeat,reduce
from emulator import Emulator # Emulator should be imported before env
# from predictor import Predictor
from utils.utilities import get_inp_files
import pandas as pd
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize as scioptminimize
import argparse,yaml
from envs import get_env
from pymoo.optimize import minimize as pymoominimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling
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
    parser.add_argument('--lag',action='store_true',help='if consider optimization time lag, only work when swmm_step==1')

    parser.add_argument('--seed',type=int,default=42,help='random seed')
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
    parser.add_argument('--method',type=str,default='descent',help='optimizer: descent lbfgs trust-constr')
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
        if tide is not False:
            ti = env.state_full()[...,0] * tide
            runoff = np.concatenate([runoff,np.expand_dims(ti,axis=-1)],axis=-1)
        runoffs.append(runoff)
    ts = [t0]+env.data_log['simulation_time'][:-1]
    runoff = np.array(runoffs)
    return ts,runoff

def pred_simu(y,file,args,r=None,act=True):
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

class mpc_ga(Problem):
    def __init__(self,args,margs=None):
        self.args = args
        if margs is not None:
            # if args.predict:
            #     self.emul = Predictor(margs.recurrent,margs)
            # else:
            self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            self.emul.load(margs.model_dir)
            self.stochastic = getattr(args,"stochastic",False)
        self.step = args.interval
        self.eval_hrz = args.prediction['eval_horizon']
        self.n_step = args.prediction['control_horizon']//args.setting_duration
        self.r_step = args.setting_duration//args.interval
        self.env = get_env(args.env)(initialize=False)
        if args.act.startswith('conti'):
            self.n_act = len(args.action_space)
            self.n_var = self.n_act*self.n_step
            super().__init__(n_var=self.n_var, n_obj=1,
                            xl = np.array([min(v) for _ in range(self.n_step)
                                            for v in args.action_space.values()]),
                            xu = np.array([max(v) for _ in range(self.n_step)
                                           for v in args.action_space.values()]),
                            vtype=float)
        else:
            self.actions = args.action_table
            self.n_act = np.array(list(self.actions)).shape[-1]
            self.n_var = self.n_act*self.n_step         
            super().__init__(n_var=self.n_var, n_obj=1,
                            xl = np.zeros(self.n_var),
                            xu = np.array([v for _ in range(self.n_step)
                                for v in np.array(list(self.actions.keys())).max(axis=0)]),
                            vtype=bool if args.act.endswith('bin') else int)

    def load_state(self,state,runoff,edge_state):
        self.state,self.runoff,self.edge_state = state,runoff,edge_state
    
    def load_file(self,eval_file,log=None,runoff_rate=None):
        self.file,self.runoff_rate,self.log = eval_file,runoff_rate,log

    def obj_simu(self,y):
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
            done = self.env.step(yi if self.args.act.startswith('conti') else self.actions[tuple(yi.astype(int))])
            idx += 1
        return self.env.objective(idx).sum()
    
    def pre_state(self,y,state,runoff,edge_state):
        y = y.reshape((-1,self.n_step,self.n_act))
        pop_size = y.shape[0]
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
        return settings,state,runoff,edge_state

    def predict(self,settings,state,runoff,edge_state):
        if self.eval_hrz > self.margs.seq_out * self.args.interval:
            state,edge_state = state[:,-self.margs.seq_in:,...],edge_state[:,-self.margs.seq_in:,...]
            predss = []
            for idx in range(self.eval_hrz//self.margs.seq_out):
                ri = runoff[:,idx*self.margs.seq_out:(idx+1)*self.margs.seq_out,...]
                sett = settings[:,idx*self.margs.seq_out:(idx+1)*self.margs.seq_out,:]
                if self.margs.if_flood and idx > 0:
                    f = (perf>0).astype(np.float32)
                    state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
                preds = self.emul.predict(state,ri,sett,edge_state)
                state,perf = np.concatenate([preds[0][...,:-2],ri],axis=-1),preds[0][...,-1:]
                ae = self.emul.get_edge_action(sett)
                edge_state = np.concatenate([preds[1],ae],axis=-1)
                predss.append(preds)
        else:
            return self.emul.predict(state,runoff,settings,edge_state)   

    def obj_emu(self,y,state,runoff,edge_state):
        settings,state,runoff,edge_state = self.pre_state(y,state,runoff,edge_state)
        preds = self.predict(settings,state,runoff,edge_state)
        if self.args.predict:
            objs = preds.numpy().sum(axis=-1).sum(axis=-1)
            if not getattr(self.emul,'norm',False):
                objs = self.env.norm_obj(objs,[state,edge_state],inverse=True)
        else:
            objs = self.env.objective_pred(preds,[state,edge_state],settings).sum(axis=-1)
        if self.stochastic:
            objs = np.stack([np.reduce_mean(objs[i*self.stochastic:(i+1)*self.stochastic])
                              for i in range(y.shape[0])])
        return objs

    def _evaluate(self,x,out,*args,**kwargs):        
        if hasattr(self,'emul'):
            # out['F'] = self.pred_emu(x)+1e-6
            out['F'] = self.obj_emu(x,self.runoff,self.state,self.edge_state).numpy()+1e-6
        else:
            pool = mp.Pool(self.args.processes)
            res = [pool.apply_async(func=self.obj_simu,args=(xi,)) for xi in x]
            pool.close()
            pool.join()
            F = [r.get() for r in res]
            out['F'] = np.array(F)+1e-6

    def pred(self,x):
        if hasattr(self,'emul'):
            # return self.pred_emu(x)+1e-6
            return self.obj_emu(x,self.runoff,self.state,self.edge_state).numpy()+1e-6
        else:
            pool = mp.Pool(self.args.processes)
            res = [pool.apply_async(func=self.obj_simu,args=(xi,)) for xi in x]
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
        self.data["sol"] = []

    def notify(self, algorithm):
        vals = algorithm.pop.get("F")
        self.data["best"].append(vals.min())
        self.data["time"].append(time.time()-self.t0)
        self.data["sol"].append(algorithm.pop.get("X")[vals.argmin()])    

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
    ctrls = res.X
    ctrls = ctrls.reshape((prob.n_step,prob.n_act))
    if not args.act.startswith('conti'):
        ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls.astype(int))
    print("Best solution found: %s" % ctrls.tolist())
    print("Function value: %s" % res.F)

    vals = res.algorithm.callback.data["best"]
    nfuns = args.pop_size*np.arange(1,len(vals)+1)
    times = res.algorithm.callback.data["time"]
    sols = res.algorithm.callback.data["sol"]
    sols = np.array(sols).reshape((-1,prob.n_step,prob.n_act))
    if not args.act.startswith('conti'):
        sols = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,sols.astype(int))
    return ctrls.tolist(),vals,nfuns,times,sols

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper
    
class mpc_gr:
    def __init__(self,args,margs,load_model=False):
        self.args = args
        self.margs = margs
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        if load_model:
            # if args.processes > 1:
            #     tf.config.experimental.set_virtual_device_configuration(
            #         gpus[0],
            #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)
            #         for _ in range(args.processes)])
            self.load_model(margs,args.predict)
        self.cross_entropy = bool(getattr(args,"cross_entropy",False))
        self.stochastic = getattr(args,"stochastic",False)
        self.asp = list(args.action_space.values())
        self.n_act = len(self.asp)
        self.step = args.interval
        self.eval_hrz = args.prediction['eval_horizon']
        self.n_step = args.prediction['control_horizon']//args.setting_duration
        self.r_step = args.setting_duration//args.interval
        self.env = get_env(args.env)(initialize=False)
        self.pop_size = getattr(args,'pop_size',64)
        self.n_var = self.n_act*self.n_step
        self.bounded = args.method in ['l-bfgs-b','trust-constr']
        self.xl = np.array([min(v) for _ in range(self.n_step)
                            for v in self.asp])
        self.xu = np.array([max(v) for _ in range(self.n_step)
                            for v in self.asp])

    def load_model(self,margs,predict=False):
        self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
        self.emul.load(margs.model_dir)

    def load_state(self,state,runoff,edge_state):
        self.state,self.runoff,self.edge_state = [th.tensor(x).type(th.float32).to(self.device) for x in [state,runoff,edge_state]]

    def get_state(self):
        return tuple([getattr(self,item,None) for item in ['state','runoff','edge_state']])

    def pre_state(self,y,state,runoff,edge_state):
        n_pop = settings.shape[0]
        settings = rearrange(y,'b (s a) -> b s a',s=self.n_step,a=self.n_act)
        settings = th.tensor(repeat(settings,'b s a -> b (s r) a',r=self.r_step)).type(th.float32).to(self.device)
        if settings.shape[1] < self.eval_hrz // self.step:
            # Expand settings to match runoff in temporal exis (control_horizon --> eval_horizon)
            settings = th.concat([settings,repeat(settings[:,-1:,:],'b s a -> b (s r) a',r=self.eval_hrz//self.step-settings.shape[1])],dim=1)
        if self.stochastic:
            settings = repeat(settings,'b s a -> (b sto) s a', sto=self.stochastic)
        runoff = repeat(runoff,'sto s d -> (b sto) s d' if self.stochastic else 's d -> b s d',b=n_pop)
        state = repeat(state,'s d -> b s d',b=n_pop*max(self.stochastic,1))
        edge_state = repeat(edge_state,'s d -> b s d',b=n_pop*max(self.stochastic,1))
        return settings,state,runoff,edge_state

    def predict(self,settings,state,runoff,edge_state):
        if self.eval_hrz > self.margs.seq_out * self.args.interval:
            state,edge_state = state[:,-self.margs.seq_in:,...],edge_state[:,-self.margs.seq_in:,...]
            predss = []
            for idx in range(self.eval_hrz//self.margs.seq_out):
                ri = runoff[:,idx*self.margs.seq_out:(idx+1)*self.margs.seq_out,...]
                sett = settings[:,idx*self.margs.seq_out:(idx+1)*self.margs.seq_out,:]
                if self.margs.if_flood and idx > 0:
                    f = (perf>0).type(th.float32)
                    state = th.concat([state[...,:-1],f,state[...,-1:]],dim=-1)
                preds = self.emul.predict(state,ri,sett,edge_state)
                state,perf = th.concat([preds[0][...,:-2],ri],dim=-1),preds[0][...,-1:]
                ae = self.emul.get_edge_action(sett,True)
                edge_state = th.concat([preds[1],ae],dim=-1)
                predss.append(preds)
            return [th.concat([preds[i] for preds in predss],dim=1) for i in range(2)]
        else:
            return self.emul.predict_tf(state,runoff,settings,edge_state)

    @call_counter
    def objective_fn(self,y,state,runoff,edge_state):
        settings,state,runoff,edge_state = self.pre_state(y,state,runoff,edge_state)
        preds = self.predict(settings,state,runoff,edge_state)
        if self.args.predict:
            obj = reduce(preds,'b s d -> b', 'sum')
            if not getattr(self.emul,'norm',False):
                obj = self.env.norm_obj(obj,[state,edge_state],inverse=True)
        else:
            obj = self.env.objective_pred(preds,[state,edge_state],settings)
        if self.stochastic:
            obj = th.stack([obj[i*self.stochastic:(i+1)*self.stochastic].mean()
                              for i in range(y.shape[0])],dim=0)
        return obj.squeeze()

    @call_counter
    def gradient_fn(self,y,state,runoff,edge_state):
        y = th.tensor(y).type(th.float32).to(self.device)
        if not self.bounded:
            y = self.project(y)
        self.emul.model.zero_grad()
        obj = self.objective_fn(y,state,runoff,edge_state)
        grads = th.autograd.grad(obj,y,retain_graph=True)[0]
        return obj,grads

    def gradient(self,*args):
        objs,grads = self.gradient_fn(*args)
        return objs.numpy(),grads.numpy()
    
    @call_counter
    def hessp_fn(self,y,p,state,runoff,edge_state):
        y,p = th.tensor(y).type(th.float32).to(self.device),th.tensor(p).type(th.float32).to(self.device)
        self.emul.model.zero_grad()
        _,grads = self.gradient_fn(y,state,runoff,edge_state)
        hvp = th.autograd.grad(grads,y,grad_outputs=p.detach())[0]
        return hvp

    def hessp(self,*args):
        return self.hessp_fn(*args).numpy()

    def project(self,y,inverse=False):
        # return tf.clip_by_value(y,self.xl,self.xu)
        if inverse:
            y = (y-self.xl)/(self.xu-self.xl)
            return th.log(y/(1-y+1e-6))
        else:
            return th.sigmoid(y)*(self.xu-self.xl)+self.xl

    @property
    def calls(self):
        return self.pred_fit.calls + self.objective_fn.calls + self.gradient_fn.calls + self.hessp_fn.calls

def run_ntopt(prob,args,setting=None):
    '''
    l-bfgs-b or trust-constr
    '''
    print(f'Running {args.method} optimization')
    sampling = sampling_lhs(args.pop_size,prob.n_var,prob.xl,prob.xu).tolist()
    if args.use_current and setting is not None:
        sampling = np.array([np.reshape(setting,-1)] + sampling[:args.pop_size-1])

    recs,sols = [[0,prob.calls,time.time(),1e6]],[]
    print('===============================================')
    print(' n_gen | n_fun |     time      |       f       ')
    print('===============================================')
    def mycallback(intermediate_result):
        sols.append(intermediate_result.x if prob.bounded else prob.project(intermediate_result.x).numpy())
        obj = getattr(intermediate_result,'fun',np.nan)
        nfev = prob.calls
        rec = [recs[-1][0]+1, nfev-recs[-1][1], time.time()-recs[-1][2], obj]
        log = ''.join([str(round(r,4)).center(7 if i < 2 else 15) + '|' for i,r in enumerate(rec)])
        print(log[:-1])
        rec[1],rec[2] = nfev,time.time()
        recs.append(rec)
    res = []
    for i,x0 in enumerate(sampling):
        results = scioptminimize(lambda *arg: tuple([re.numpy() for re in prob.gradient_fn(*arg)]),
                                    x0=x0 if prob.bounded else prob.project(x0,True),
                                    args=prob.get_state(),
                                    method=args.method,
                                    jac=True,
                                    hessp=lambda *arg: prob.hessp_fn(*arg).numpy(),
                                    bounds=list(zip(prob.xl,prob.xu)),
                                    callback=mycallback,
                                    options={k:v for k,v in zip(args.termination[0::2],args.termination[1::2])},
                                    )
        res.append(results)
    idx = np.argmin([r.fun for r in res])
    results = res[idx]
    print("Optimization {}, Best run {} Objective {}".format("successful" if results.success else "failed",idx,results.fun))
    ctrls = results.x if prob.bounded else prob.project(results.x).numpy()
    vals = np.array(recs)[1:,-1]
    nfuns = np.array(recs)[1:,1]-recs[0][1]
    times = np.array(recs)[1:,2]-recs[0][2]
    ctrls = ctrls.astype(np.float32).reshape((prob.n_step,prob.n_act)).tolist()
    sols = np.array(sols,dtype=np.float32).reshape((-1,prob.n_step,prob.n_act))
    print('Best solution: ',ctrls)
    return ctrls,vals,nfuns,times,sols


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','mpc.yaml'))
    mp.set_start_method('spawn', force=True)    # use gpu in multiprocessing
    de = {
        # 'env':'astlingen',
        # 'act':'conti',
        # 'processes':2,
        # 'pop_size':16,
        # # 'sampling':0.4,
        # # 'learning_rate':0.1,
        # # 'termination':['n_gen',5,'time','00:05:00'],
        # 'surrogate':True,
        # 'gradient': True,
        # 'predict':False,
        # 'method':'l-bfgs-b',
        # 'use_current':True,
        # 'rain_dir':'./envs/config/ast_test2007_events.csv',
        # 'rain_suffix':'chaohu_testall',
        # 'rain_num':100,
        # 'swmm_step':1,
        # 'lag':True,
        # 'horizon':60,
        # 'model_dir':'./model/astlingen/60s_50k_conti_1000ledgef_res_norm_flood_gat_5lyrs',
        # 'result_dir':'./results/astlingen/60s_conti_lbfgsb16fl_2007',
        }
    # config['rain_dir'] = de['rain_dir']
    for k,v in de.items():
        setattr(args,k,v)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args(args.directed,args.length,args.order,act=args.act)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)
    setattr(args,'elements',env.elements)

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
    args.prediction['eval_horizon'] = args.prediction['control_horizon'] = args.horizon * args.interval

    if args.surrogate and args.gradient:
        prob = mpc_gr(args,margs,load_model=True)
    else:
        prob = mpc_ga(args,margs=margs if args.surrogate else None)

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
            ts,runoff = get_runoff(env,event,tide=args.tide and args.is_outfall)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            horizon = args.prediction['eval_horizon']//args.interval
            runoff = np.stack([np.concatenate([runoff[idx:idx+horizon],np.tile(np.zeros_like(s),(max(idx+horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
        elif args.prediction['no_runoff']:
            ts,runoff_rate = get_runoff(env,event,True,tide=args.tide and args.is_outfall)
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

        done,i,j,valss,nfunss,timess,solss = False,0,0,[],[],[],[]
        while not done:
            if i*args.interval % args.control_interval == 0:
                t2 = time.time()
                setting = setting[j+1:] + setting[-1:] * (j+1)
                if args.surrogate:
                    state[...,1] = state[...,1] - state[...,-1]
                    if margs.if_flood:
                        f = (flood>0).astype(float)
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
                        setting,vals,nfuns,times,sols = run_ntopt(prob,args,setting=setting)
                    else:
                        setting,vals,nfuns,times,sols = run_ea(prob,args,setting=setting)
                else:
                    eval_file = env.get_eval_file(args.prediction['no_runoff'])
                    if args.prediction['no_runoff']:
                        t = env.env.methods['simulation_time']()
                        rr = runoff_rate[int(tss.asof(t)['Index']),...,0]
                    prob.load_file(eval_file,env.data_log,rr if args.prediction['no_runoff'] else None)
                    setting,vals,nfuns,times,sols = run_ea(prob,args,setting=setting)
                valss.append(vals)
                nfunss.append(nfuns)
                timess.append(times)
                solss.append(sols[:,:args.control_interval//args.setting_duration,:])
                t3 = time.time()
                print('Optimization time: {} s'.format(t3-t2))
                opt_times.append(t3-t2)
                j = 0
                lag = (t3-t2)/60/args.interval
                prev_sett = sett.copy() if i > 0 else settings[0].copy()
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[j]) if args.keep == 'False' else settings[0]
            elif i*args.interval % args.setting_duration == 0:
                j += 1
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[j]) if args.keep == 'False' else settings[0]
            real_sett = prev_sett if args.lag and i*args.interval % args.control_interval < int(lag) else sett
            done = env.step(real_sett,
                            lag_seconds = (lag%1)*args.interval*60 if args.lag and i*args.interval % args.control_interval == int(lag) else None)
            state = env.state_full(seq=margs.seq_in if args.surrogate else False)
            if args.surrogate and margs.if_flood:
                flood = env.flood(seq=margs.seq_in)
            edge_state = env.state_full(margs.seq_in if args.surrogate else False,'links')
            states.append(state[-1] if args.surrogate else state)
            perfs.append(env.flood())
            objects.append(env.objective())
            edge_states.append(edge_state[-1] if args.surrogate else edge_state)
            settings.append(real_sett)
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
        np.save(os.path.join(args.result_dir,name + '_%s_sols.npy'%item),np.array(solss))

        results.loc[name] = [t1-t0,np.mean(opt_times),np.stack(perfs).sum(),np.stack(objects).sum()]
    results.to_csv(os.path.join(args.result_dir,'results_%s.csv'%item))

