import os,time,math
import torch as th
import torch.nn.functional as F
from einops import rearrange,repeat,reduce
from emulator import Emulator # Emulator should be imported before env
# from predictor import Predictor
from utils.utilities import get_inp_files
import pandas as pd
import torch.multiprocessing as mp
from utils.reinmax import reinmax as reinmax_determ
from reinmax import reinmax
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

class TerminationOrCollection(TerminationCollection):
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

    parser.add_argument('--sample_interval',type=int,default=10,help='sampling interval')
    parser.add_argument('--ctrl_step',type=int,default=5,help='setting duration')
    parser.add_argument('--horizon',type=int,default=60,help='control horizon')
    parser.add_argument('--act',type=str,default='rand',help='what control actions')
    parser.add_argument('--lag',action='store_true',help='if consider optimization time lag, only work when swmm_step==1')

    parser.add_argument('--seed',type=int,default=42,help='random seed')
    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')
    parser.add_argument('--pop_size',type=int,default=32,help='number of population')
    parser.add_argument('--use_current',action="store_true",help='if use current setting as initial')
    parser.add_argument('--sampling',type=float,default=0.4,help='sampling rate')
    parser.add_argument('--crossover',nargs='+',type=float,default=[1.0,3.0],help='crossover rate')
    parser.add_argument('--mutation',nargs='+',type=float,default=[1.0,3.0],help='mutation rate')
    parser.add_argument('--termination',nargs='+',type=str,default=['n_eval','256'],help='Iteration termination criteria')
    
    parser.add_argument('--surrogate',action='store_true',help='if use surrogate for dynamic emulation')
    parser.add_argument('--predict',action='store_true',help='if use predictor for emulation')
    parser.add_argument('--batch_size',type=int,default=32,help='number of batch size')
    parser.add_argument('--gradient',action='store_true',help='if use gradient-based optimization')
    parser.add_argument('--method',type=str,default='gr',help='optimizer: gr l-bfgs-b trust-constr')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate for gradient-based optimization')
    parser.add_argument('--tau',type=float,default=1.0,help='softmax temperature for discrete gradient optimization')
    parser.add_argument('--warm_eps',type=float,default=0.01,help='extra probability above uniform for current discrete warm-start action')
    parser.add_argument('--logit_bound',type=float,default=0.1,help='bound for discrete logits in scipy optimization')
    parser.add_argument('--du',type=float,default=0.0,help='bound for continuous control increments in gradient MPC')
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
    n_term = len(args.termination)
    conds = ['maxls','ftol','gtol','maxcor'] if args.gradient else ['n_eval','n_gen','fmin']
    for i,j in zip(range(n_term-1),range(1,n_term)):
        args.termination[j] = eval(args.termination[j]) if args.termination[i] in conds else args.termination[j]
    # for i in range(len(args.termination)//2):
    #     args.termination[2*i+1] = eval(args.termination[2*i+1]) if args.termination[2*i] not in ['time','soo'] else args.termination[2*i+1]
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

def pred_simu(y,file,args,r=None,act=True,keepdim=False):
    env = get_env(args.env_name)(swmm_file = file)
    done,idx = False,0
    if getattr(args,'log') is not None:
        env.data_log.update({k:v for k,v in args.log.items() if 'cum' not in k})
    # perf = []
    while not done and idx < (y.shape[0] if act else args.horizon):
        if args.prediction['no_runoff']:
            for node,ri in zip(env.elements['nodes'],r[idx]):
                env.env._setNodeInflow(node,ri)
        # done = env.step([actions[i][int(act)] for i,act in enumerate(y[idx])])
        done = env.step([sett for sett in y[idx]] if act else None)
        # perf.append(env.flood())
        idx += 1
    # return np.array(perf)
    return env.objective(idx,keepdim=keepdim)

class mpc_ga(Problem):
    def __init__(self,args,margs=None):
        self.args = args
        if margs is not None:
            # if args.predict:
            #     self.emul = Predictor(margs.recurrent,margs)
            # else:
            self.emul = Emulator(margs)
            self.emul.load()
            pop_size,batch_size = getattr(args,'pop_size',128),getattr(margs,'batch_size',128)
            self.batch_size = min(pop_size,batch_size)
            self.emul.set_indices(self.batch_size)
            self.stochastic = getattr(args,"stochastic",False)
            self.margs = margs
        self.step = args.interval
        self.horizon = getattr(args,'horizon',60)
        self.n_step = self.horizon//args.ctrl_step
        self.r_step = args.ctrl_step//args.interval
        self.pop_size = getattr(args,'pop_size',128)
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
        if y.shape[0] < self.horizon // self.step:
            y = np.concatenate([y,np.repeat(y[-1:,:],self.horizon // self.step-y.shape[0],axis=0)],axis=0)

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
        settings = y if self.args.act.startswith('conti') else np.apply_along_axis(lambda x:self.actions.get(tuple(x)),-1,y.astype(int))
        settings = np.repeat(settings,self.r_step,axis=1)
        if self.stochastic:
            settings = np.repeat(settings,self.stochastic,axis=0)
            runoff = np.tile(self.runoff,(self.pop_size,)+tuple([1 for _ in range(self.runoff.ndim-1)]))
        else:
            runoff = np.repeat(np.expand_dims(self.runoff,0),self.pop_size,axis=0)
        state = np.repeat(np.expand_dims(self.state,0),self.pop_size,axis=0)
        edge_state = np.repeat(np.expand_dims(self.edge_state,0),self.pop_size,axis=0)
        return settings,state,runoff,edge_state

    @th.compile(fullgraph=True)
    def predict(self,settings,state,runoff,edge_state):
        if self.horizon > self.margs.seq * self.args.interval:
            x,ex = state[:,-self.margs.seq:,...],edge_state[:,-self.margs.seq:,...]
            predss,edge_preds = [],[]
            for idx in range(self.horizon//self.margs.seq):
                ri = runoff[:,idx*self.margs.seq:(idx+1)*self.margs.seq,...]
                sett = settings[:,idx*self.margs.seq:(idx+1)*self.margs.seq,:]
                preds = self.emul.predict(x,ex,ri,sett,constr=False)
                if self.margs.if_flood:
                    f = (preds[0][...,-1:]>0).type(th.float32)
                    x = th.concat([preds[0][...,:-1],f,ri],dim=-1)
                else:
                    x = th.concat([preds[0],ri],dim=-1)
                ae = self.emul.get_edge_action(sett,True)
                ex = th.concat([preds[1],ae],dim=-1)
                predss.append(x)
                edge_preds.append(ex)
            predss,edge_predss = th.concat(predss,dim=1)[...,:-1],th.concat(edge_preds,dim=1)
            predss = self.emul.constrain(predss,state[:,-1,:,0])
            qw = self.emul.get_flood(predss,runoff[...,0])
            predss = th.concat([predss,qw],dim=-1)
            return predss,edge_predss
        else:
            return self.emul.predict(state,edge_state,runoff,settings,constr=True)
        
    def obj_emu(self,y,state,runoff,edge_state):
        n_pop = y.shape[0]
        if n_pop < self.pop_size:
            y = np.concatenate([y,np.repeat(y[:1,...],self.pop_size-n_pop,axis=0)],axis=0)
        settings,state,runoff,edge_state = self.pre_state(y,state,runoff,edge_state)
        if self.batch_size < self.pop_size:
            ninfer = self.pop_size//self.batch_size
            dats = [np.split(dat,ninfer) for dat in [settings,state,runoff,edge_state]]
            preds = []
            for dat in zip(*dats):
                pred = self.predict(*[self.emul.to_tensor(d) for d in dat])
                pred = [p.detach().cpu().numpy() for p in pred]
                preds.append(pred)
            preds = np.concatenate([p[0] for p in preds],axis=0),np.concatenate([p[1] for p in preds],axis=0)
        else:
            preds = self.predict(*[self.emul.to_tensor(dat) for dat in [settings,state,runoff,edge_state]])
            preds = preds.cpu().numpy() if self.args.predict else (preds[0].detach().cpu().numpy(),preds[1].detach().cpu().numpy())
        if self.args.predict:
            objs = preds.cpu().numpy().sum(axis=-1).sum(axis=-1)
            if not getattr(self.emul,'norm',False):
                objs = self.env.norm_obj(objs,[state,edge_state],inverse=True)
        else:
            objs = self.env.objective_pred(preds,
                                           [state,edge_state],settings).sum(axis=-1)
        if self.stochastic:
            objs = np.stack([np.mean(objs[i*self.stochastic:(i+1)*self.stochastic],axis=0)
                              for i in range(self.pop_size)])
        return objs[:n_pop]

    def _evaluate(self,x,out,*args,**kwargs):        
        if hasattr(self,'emul'):
            # out['F'] = self.pred_emu(x)+1e-6
            out['F'] = self.obj_emu(x,self.state,self.runoff,self.edge_state)+1e-6
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
            return self.obj_emu(x,self.state,self.runoff,self.edge_state)+1e-6
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
        self.t0 = time.perf_counter()
        self.data["sol"] = []

    def notify(self, algorithm):
        vals = algorithm.pop.get("F")
        self.data["best"].append(vals.min())
        self.data["time"].append(time.perf_counter()-self.t0)
        self.data["sol"].append(algorithm.pop.get("X")[vals.argmin()])    

def run_ea(prob,args,setting=None):
    print('Running genetic algorithm')
    if args.act.startswith('conti'):
        sampling = sampling_lhs(args.pop_size,prob.n_var,prob.xl,prob.xu)
    else:
        sampling = sampling_lhs(args.pop_size,prob.n_var,prob.xl-0.5,prob.xu+0.5).round(0).astype(int)
        # sampling = np.random.randint(prob.xl,prob.xu+1,size=(args.pop_size,prob.n_var))
    if args.use_current and setting is not None:
        sampling = np.concatenate([np.reshape(setting,(1,-1)), sampling[:args.pop_size-int(args.pop_size>1)]],axis=0)
    if args.act.endswith('bin'):
        crossover = BX(*args.crossover,vtype=bool)
        mutation = BitflipMutation(*args.mutation,vtype=bool)
    else:
        crossover = SBX(*args.crossover,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
        mutation = PM(*args.mutation,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
    if len(args.termination) > 2:
        terms = {}
        for val in args.termination[1:]:
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
        termination = TerminationOrCollection(*termination) if 'or' in args.termination[0] else TerminationCollection(*termination)
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
        self.pop_size = getattr(args,'pop_size',1)
        if load_model:
            self.load_model(margs,self.pop_size if args.method == 'gr' else 1)
        self.asp = [np.array(ap) for ap in args.action_space.values()]
        self.n_act = len(self.asp)
        self.horizon = getattr(args,'horizon',60)
        self.n_step = self.horizon//args.ctrl_step
        self.r_step = args.ctrl_step//args.interval
        self.env = get_env(args.env)(initialize=False)
        self.conti = args.act.startswith('conti')
        self.du = max(float(getattr(args,'du',0.0)),0.0)
        self.delta_control = self.conti and self.du > 0
        if self.conti:
            self.n_var = self.n_act*self.n_step
            self.bounded = args.method in ['l-bfgs-b','trust-constr']
            self.u_min = np.array([min(v) for _ in range(self.n_step)
                                   for v in self.asp],dtype=np.float32)
            self.u_max = np.array([max(v) for _ in range(self.n_step)
                                   for v in self.asp],dtype=np.float32)
            self.xl = np.full(self.n_var,-self.du,dtype=np.float32) if self.delta_control else self.u_min.copy()
            self.xu = np.full(self.n_var,self.du,dtype=np.float32) if self.delta_control else self.u_max.copy()
            self.bounds = list(zip(self.xl,self.xu))
            self.ctrl_base = th.zeros(self.n_var,dtype=th.float32,device=self.device)
            self.stochastic = False
        else:
            '''
            Discrete gradient optimization uses logits over each actuator's
            action options. The forward pass is hard argmax through ReinMax,
            matching the deterministic controls applied by MPC, while the
            backward pass uses ReinMax's surrogate gradient. warm_eps controls
            the initial preference for the current action above uniform, tau
            controls backward smoothness, and logit_bound keeps logits near
            argmax boundaries so line-search steps can flip discrete actions.
            stochastic mode samples multiple rollouts in forward passes,
            and is recommended in gr (first-order), not lbfgsb (second-order).
            '''
            self.eps = getattr(args,'warm_eps',0.01)
            self.tau = getattr(args,'tau',1.0)
            self.opts = [len(ap) for ap in self.asp] * self.n_step
            self.n_var = sum(self.opts)
            self.asp *= self.n_step
            self.asp_th = [th.tensor(ap,device=self.device,dtype=th.float32) for ap in self.asp]
            self.logit_bound = getattr(args,'logit_bound',0.1)
            self.bounds = [(-self.logit_bound,self.logit_bound)]*self.n_var
            self.stochastic = getattr(args,"stochastic",False) # Used for stochastic discrete sampling

    def load_model(self,margs,batch=1):
        self.emul = Emulator(margs)
        self.emul.load()
        self.emul.set_indices(batch)

    def load_state(self,state,runoff,edge_state):
        self.state,self.runoff,self.edge_state = [th.tensor(x,dtype=th.float32,device=self.device) for x in [state,runoff,edge_state]]


    def set_delta_base(self,setting=None):
        if setting is None:
            first = self.u_max.reshape(self.n_step,self.n_act)[0]
        else:
            setting = np.asarray(setting,dtype=np.float32)
            first = setting.reshape(self.n_step,self.n_act)[0] if setting.size != self.n_act else setting.reshape(self.n_act)
        base = np.tile(first,(self.n_step,1)).reshape(-1)
        base = np.clip(base,self.u_min,self.u_max).astype(np.float32)
        self.ctrl_base = th.tensor(base,dtype=th.float32,device=self.device)
        self.xl = np.maximum(-self.du,self.u_min-base).astype(np.float32)
        self.xu = np.minimum(self.du,self.u_max-base).astype(np.float32)
        self.bounds = list(zip(self.xl,self.xu))

    def get_state(self):
        return tuple([getattr(self,item,None) for item in ['state','runoff','edge_state','ctrl_base']])

    def pre_state(self,y: th.Tensor, state: th.Tensor, runoff: th.Tensor,edge_state: th.Tensor,
                  ctrl_base: th.Tensor):
        if self.delta_control:
            y = y + ctrl_base
        elif self.conti and not self.bounded:
            y = self.project(y)
        if not self.conti:
            if self.stochastic:
                y = y.tile(self.stochastic,1)
            y = self.ste(y)
        settings = y.reshape(-1,self.n_step,self.n_act)
        settings = settings.tile(1,self.r_step,1)
        state,edge_state,runoff = state.unsqueeze(0),edge_state.unsqueeze(0),runoff.unsqueeze(0)
        if self.stochastic:
            state,edge_state,runoff = [dat.tile(self.stochastic,1,1,1) for dat in [state,edge_state,runoff]]
        if self.args.method == 'gr': # multi-start in run_gr
            state,edge_state,runoff = [dat.tile(self.pop_size,1,1,1) for dat in [state,edge_state,runoff]]
        return settings,state,runoff,edge_state

    @th.compile(fullgraph=True)
    def predict(self,settings,state,runoff,edge_state):
        if self.horizon > self.margs.seq * self.args.interval:
            x,ex = state[:,-self.margs.seq:,...],edge_state[:,-self.margs.seq:,...]
            predss,edge_preds = [],[]
            for idx in range(self.horizon//self.margs.seq):
                ri = runoff[:,idx*self.margs.seq:(idx+1)*self.margs.seq,...]
                sett = settings[:,idx*self.margs.seq:(idx+1)*self.margs.seq,:]
                preds = self.emul.predict(x,ex,ri,sett,constr=False)
                if self.margs.if_flood:
                    f = (preds[0][...,-1:]>0).type(th.float32)
                    x = th.concat([preds[0][...,:-1],f,ri],dim=-1)
                else:
                    x = th.concat([preds[0],ri],dim=-1)
                ae = self.emul.get_edge_action(sett,True)
                ex = th.concat([preds[1],ae],dim=-1)
                predss.append(x)
                edge_preds.append(ex)
            predss,edge_predss = th.concat(predss,dim=1)[...,:-1],th.concat(edge_preds,dim=1)
            predss = self.emul.constrain(predss,state[:,-1,:,0])
            qw = self.emul.get_flood(predss,runoff[...,0])
            predss = th.concat([predss,qw],dim=-1)
            return predss,edge_predss
        else:
            return self.emul.predict(state,edge_state,runoff,settings,constr=True)

    @th.compile(fullgraph=True)
    def objective_fn(self,y,state,runoff,edge_state,ctrl_base=None):
        settings,state,runoff,edge_state = self.pre_state(y,state,runoff,edge_state,ctrl_base)
        preds = self.predict(settings,state,runoff,edge_state)
        if self.args.predict:
            obj = preds.sum(dim=-1).sum(dim=-1)
            if not getattr(self.emul,'norm',False):
                obj = self.env.norm_obj(obj,[state,edge_state],inverse=True)
        else:
            obj = self.env.objective_pred_th(preds,[state,edge_state],settings).sum(dim=-1)
        if self.stochastic:
            obj = obj.reshape(-1,self.stochastic).mean(dim=1)
        return obj

    @th.compile(dynamic=False)
    def gradient_fn(self,y,*args):
        self.emul.model.zero_grad()
        obj = self.objective_fn(y[None,:],*args).squeeze()
        grads = th.autograd.grad(obj,y,retain_graph=True)[0]
        return obj,grads

    @call_counter
    def gradient(self,y,*args):
        if not isinstance(y,th.Tensor):
            y = th.tensor(y,requires_grad=True,dtype=th.float32,device=self.device)
        objs,grads = self.gradient_fn(y,*args)
        return objs.detach().cpu().numpy(),grads.detach().cpu().numpy()
    
    @th.compile(dynamic=False)
    def hessp_fn(self,y,p,*args):
        self.emul.model.zero_grad()
        _,grads = self.gradient_fn(y,*args)
        hvp = th.autograd.grad(grads,y,grad_outputs=p.detach())[0]
        return hvp

    @call_counter
    def hessp(self,y,p,*args):
        y,p = th.tensor(y).type(th.float32).to(self.device),th.tensor(p).type(th.float32).to(self.device)
        if self.conti and not self.bounded and not self.delta_control:
            y = self.project(y)
        return self.hessp_fn(y,p,*args).detach().cpu().numpy()

    def project(self,y,inverse=False):
        if inverse:
            y = (y-self.xl)/(self.xu-self.xl)
            return th.log(y/(1-y+1e-6))
        else:
            return th.sigmoid(y)*(self.xu-self.xl)+self.xl

    def control_from_var(self,y):
        if self.conti:
            if self.delta_control:
                y = np.clip(np.asarray(y,dtype=np.float32),self.xl,self.xu)
                return y + self.ctrl_base.detach().cpu().numpy()
            elif self.bounded:
                return np.asarray(y,dtype=np.float32)
            else:
                return self.project(y).detach().cpu().numpy()
        else:
            return self.ste(y)

    def initial_guess(self,setting):
        if self.conti:
            if self.delta_control:
                setting = np.asarray(setting,dtype=np.float32).reshape(self.n_step,self.n_act)
                setting = np.concatenate([setting[1:],setting[-1:]],axis=0) # repeat last step for the whole horizon
                du = setting.reshape(-1) - self.ctrl_base.detach().cpu().numpy()
                return np.clip(du,self.xl,self.xu).astype(np.float32)
            return np.reshape(setting,-1)
        setting = np.asarray(setting,dtype=np.float32)
        if setting.size == self.n_var:
            return setting.reshape(-1)
        setting = setting.reshape(self.n_step,self.n_act,-1)
        logits = np.empty((self.n_step,sum(self.opts[:self.n_act])),dtype=np.float32)
        start = 0
        for act,(ap,opt) in enumerate(zip(self.asp[:self.n_act],self.opts[:self.n_act])):
            ap = np.asarray(ap,dtype=np.float32).reshape(opt,-1)
            dist = np.linalg.norm(setting[:,act,None,:]-ap[None,:,:],axis=-1)
            idx = np.argmin(dist,axis=-1)
            prob = np.full((self.n_step,opt),1.0/opt,dtype=np.float32)
            eps = np.clip(self.eps,0.0,1.0-1.0/opt) if opt > 1 else 0.0
            prob[np.arange(self.n_step),idx] += eps
            prob += (1.0-prob.sum(axis=-1,keepdims=True))/opt
            logits[:,start:start+opt] = np.log(prob) - np.log(prob).mean(axis=-1,keepdims=True)
            start += opt
        return logits.reshape(-1).clip(-self.logit_bound,self.logit_bound)

    def ste(self,y):
        '''
        Straight through estimator (STE) for discrete variables
        TODO: need to consider temperature tau for the gradient scale
        '''
        if isinstance(y,np.ndarray):
            y = np.split(y,np.cumsum(self.opts)[:-1])
            yhard = [np.take(ap,np.argmax(yi,axis=-1),-1)
                     for yi,ap in zip(y,self.asp)]
            return np.stack(yhard,axis=-1)
        else:
            y = th.split(y,self.opts,dim=-1)
            yste = [reinmax(yi,self.tau)[0] if self.stochastic else reinmax_determ(yi,self.tau)[0] for yi in y]
            # yste = [th.softmax(yi/self.tau, dim=-1) for yi in y]
            # yhard = [F.one_hot(yi.argmax(dim=-1),opt)
            #             for yi,opt in zip(y,self.opts)]
            # yste = [ys - ys.detach() + yh # constant
            #             for ys,yh in zip(yste,yhard)]
            yste = [(ys * ap).sum(dim=-1)
                     for ys,ap in zip(yste,self.asp_th)]
            return th.stack(yste,dim=-1)

    @property
    def calls(self):
        return self.gradient.calls + self.hessp.calls

def sample_initials(prob,args,setting=None):
    if prob.conti:
        sampling = np.asarray(sampling_lhs(args.pop_size,prob.n_var,prob.xl,prob.xu),
                              dtype=np.float32)
    else:
        sampling = np.random.uniform(-prob.logit_bound,prob.logit_bound,
                                     size=(args.pop_size,prob.n_var)).astype(np.float32)

    if args.use_current and setting is not None:
        x0 = np.asarray(prob.initial_guess(setting),dtype=np.float32)
        sampling = np.concatenate([x0[None, :], sampling[:args.pop_size-int(args.pop_size>1)]],axis=0)
    return sampling

def parse_time_limit(value):
    if value is None:
        return None
    if isinstance(value,(int,float)):
        return float(value)
    value = str(value)
    if ':' not in value:
        return float(value)
    vals = [float(v) for v in value.split(':')]
    if len(vals) == 3:
        return vals[0]*3600 + vals[1]*60 + vals[2]
    if len(vals) == 2:
        return vals[0]*60 + vals[1]
    return vals[0]

def parse_gr_termination(args):
    term = {
        'min_gen':20,
        'max_gen':100,
        'max_time':None,
        'patience':10,
        'ftol':1e-3,
    }
    vals = list(getattr(args,'termination',[]))
    for key,value in zip(vals[0::2],vals[1::2]):
        key = str(key)
        if key in ['n_gen','max_gen','n_eval']:
            term['max_gen'] = int(value)
        elif key in ['time','max_time']:
            term['max_time'] = parse_time_limit(value)
        elif key in ['min_gen','patience']:
            term[key] = int(value)
        elif key in ['ftol']:
            term[key] = float(value)
    term['min_gen'] = max(0,term['min_gen'])
    term['max_gen'] = max(1,term['max_gen'])
    term['min_gen'] = min(term['min_gen'],term['max_gen'])
    term['patience'] = max(1,term['patience'])
    return term

def run_gr(prob,args,setting=None):
    '''
    TODO:
    First-order gradient-based optimization using Pytorch
    '''
    if prob.delta_control:
        prob.set_delta_base(setting)
    sampling = sample_initials(prob,args,setting)
    term = parse_gr_termination(args)

    fun,idx,ini = np.inf,0,0
    sol = sampling[0]
    stale = 0
    start_time = time.perf_counter()
    recs,sols = [[0,prob.calls,0.0,1e6,1e6,np.nan,0]],[sampling[0]]
    print('====================================================================================')
    print(' n_gen | n_fun |     time      |       f       |     best      |    |g|     | stale ')
    print('====================================================================================')
    y = prob.project(sampling[:args.pop_size],True) if prob.conti and not prob.bounded and not prob.delta_control else sampling[:args.pop_size]
    y = th.tensor(y, requires_grad=True, dtype=th.float32, device = prob.device)
    optim = th.optim.Adam([y],lr=getattr(args,"lr",0.01))
    while True:
        prob.emul.model.zero_grad()
        if prob.delta_control:
            with th.no_grad():
                y.clamp_(th.tensor(prob.xl,dtype=th.float32,device=prob.device),
                         th.tensor(prob.xu,dtype=th.float32,device=prob.device))
        obj = prob.objective_fn(y, *prob.get_state()).reshape(-1)
        optim.zero_grad()
        obj.sum().backward()
        obj = obj.detach().cpu().numpy()
        obj_min = float(obj.min())
        gen = recs[-1][0] + 1
        elapsed = time.perf_counter() - start_time
        if obj_min < fun - term['ftol']:
            idx,ini,fun = gen,int(obj.argmin()),obj_min
            sol = y.detach().cpu().numpy()[ini]
            sols.append(sol)
            stale = 0
        else:
            stale += 1
        grad_norms = y.grad.detach().reshape(y.shape[0],-1).norm(dim=-1)
        grad_norm = grad_norms[int(obj.argmin())].item()
        optim.step()
        if prob.delta_control:
            with th.no_grad():
                y.clamp_(th.tensor(prob.xl,dtype=th.float32,device=prob.device),
                         th.tensor(prob.xu,dtype=th.float32,device=prob.device))
        # if not prob.conti:
        #     y.data.clamp_(-prob.logit_bound,prob.logit_bound)
        rec = [gen, prob.calls, elapsed, obj_min, fun, grad_norm, stale]
        log = ''.join([str(round(r,4)).center(7 if i < 2 else 15) + '|' for i,r in enumerate(rec)])
        print(log)
        recs.append(rec)
        stop_patience = gen >= term['min_gen'] and stale >= term['patience']
        stop_time = term['max_time'] is not None and elapsed >= term['max_time']
        if gen >= term['max_gen'] or stop_time or stop_patience:
            break
    print("Best iter {} initial {} Objective {}".format(idx,ini,fun))
    ctrls = prob.control_from_var(sol)
    ctrls = ctrls.astype(np.float32).reshape((prob.n_step,prob.n_act)).tolist()
    print('Best solution: ',ctrls)
    recs = np.array(recs)
    vals = recs[1:,3]
    nfuns = recs[1:,1]-recs[0,1]
    times = recs[1:,2]
    sols = np.array(sols,dtype=np.float32).reshape((len(sols),prob.n_step,-1))
    return ctrls,vals,nfuns,times,sols

def run_ntopt(prob,args,setting=None):
    '''
    l-bfgs-b or trust-constr via scipy.optimize.minimize
    '''
    print(f'Running {args.method} optimization')
    if prob.delta_control:
        prob.set_delta_base(setting)
    sampling = sample_initials(prob,args,setting)

    recs,sols = [[0,prob.calls,time.perf_counter(),1e6]],[sampling[0]]
    print('===============================================')
    print(' n_gen | n_fun |     time      |       f       ')
    print('===============================================')
    def mycallback(intermediate_result):
        # sols.append(intermediate_result.x if prob.bounded else prob.project(intermediate_result.x).numpy())
        obj = getattr(intermediate_result,'fun',np.nan)
        nfev = prob.calls
        rec = [recs[-1][0]+1, nfev-recs[-1][1], time.perf_counter()-recs[-1][2], obj]
        log = ''.join([str(round(r,4)).center(7 if i < 2 else 15) + '|' for i,r in enumerate(rec)])
        print(log[:-1])
        rec[1],rec[2] = nfev,time.perf_counter()
        recs.append(rec)
    res = []
    for _,x0 in enumerate(sampling):
        if prob.conti and not prob.bounded and not prob.delta_control:
            x0 = prob.project(x0,True).numpy()
        results = scioptminimize(prob.gradient,
                                 x0 = x0,
                                 args=prob.get_state(),
                                 method=args.method,
                                 jac=True,
                                 hessp=prob.hessp,
                                 bounds=prob.bounds,
                                 callback=mycallback,
                                 options={k:v for k,v in zip(args.termination[0::2],args.termination[1::2])},
                                 )
        res.append(results)
        if args.pop_size == 1 and (results.success and len(recs) > 1):
            break
    idx = np.argmin([r.fun for r in res])
    results = res[idx]
    print("Optimization {}, Best run {} Objective {}".format("successful" if results.success else "failed",idx,results.fun))
    ctrls = prob.control_from_var(results.x)
    ctrls = ctrls.astype(np.float32).reshape((prob.n_step,prob.n_act)).tolist()
    print('Best solution: ',ctrls)
    vals = np.array(recs)[1:,-1]
    nfuns = np.array(recs)[1:,1]-recs[0][1]
    times = np.array(recs)[1:,2]-recs[0][2]
    sols.append(results.x)
    sols = np.array(sols,dtype=np.float32).reshape((len(sols),prob.n_step,-1))
    return ctrls,vals,nfuns,times,sols

if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','mpc.yaml'))
    mp.set_start_method('spawn', force=True)    # use gpu in multiprocessing
    de = {
        # 'env':'astlingen',
        # 'act':'rand3',
        # 'processes':1,
        # 'pop_size':1,
        # # 'sampling':0.4,
        # # 'termination':['or','time','00:05:00','soo','ftol-0.001'],
        # # 'termination':['ftol',1e-3,'maxls',30],
        # 'termination':['n_gen',100],
        # 'surrogate':True,
        # 'batch_size':1,
        # 'gradient': True,
        # 'predict':False,
        # 'method':'gr',
        # 'lr':0.01,
        # 'use_current':True,
        # 'stochastic':8,
        # 'rain_dir':'./envs/config/ast_test2007_events.csv',
        # # 'rain_suffix':'chaohu_testall',
        # # 'rain_num':100,
        # 'swmm_step':1,
        # 'lag':True,
        # 'horizon':120,
        # 'model_dir':'./model/astlingen/120s_gat_5ly_floodwei_gradnorm',
        # 'result_dir':'./results/astlingen/test',
        }
    for k,v in de.items():
        setattr(args,k,v)
        config[k] = v

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
        model_dir = os.path.dirname(margs.model_dir) if os.path.isfile(margs.model_dir) else margs.model_dir
        known_hyps = yaml.load(open(os.path.join(model_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k == 'model_dir':
                continue
            setattr(margs,k,v)
        setattr(margs,'epsilon',args.epsilon)
        env_args = env.get_args(margs.directed,margs.length,margs.order)
        for k,v in env_args.items():
            setattr(margs,k,v)
        if 'batch_size' in config:
            setattr(margs, 'batch_size', args.batch_size)

    if args.surrogate and args.gradient:
        prob = mpc_gr(args,margs,load_model=True)
    else:
        prob = mpc_ga(args,margs=margs if args.surrogate else None)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))

    results = pd.DataFrame(columns=['rr time','fl time','perf','objective'])
    for event in events:
        name = os.path.basename(event).strip('.inp')
        if os.path.exists(os.path.join(args.result_dir,name + '_state.npy')):
            continue
        t0 = time.perf_counter()
        if args.surrogate:
            ts,runoff = get_runoff(env,event,tide=args.tide and args.is_outfall)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            runoff = np.stack([np.concatenate([runoff[idx:idx+args.horizon],np.tile(np.zeros_like(s),(max(idx+args.horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
        elif args.prediction['no_runoff']:
            ts,runoff_rate = get_runoff(env,event,True,tide=args.tide and args.is_outfall)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            runoff_rate = np.stack([np.concatenate([runoff_rate[idx:idx+args.horizon],np.tile(np.zeros_like(s),(max(idx+args.horizon-runoff_rate.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff_rate)])


        t1 = time.perf_counter()
        print('Runoff time: {} s'.format(t1-t0))
        opt_times = []
        state = env.reset(event,global_state=True,seq=margs.seq if args.surrogate else False)
        if args.surrogate and margs.if_flood:
            flood = env.flood(seq=margs.seq)
        states = [state[-1] if args.surrogate else state]
        perfs,objects = [env.flood()],[env.objective()]

        edge_state = env.state_full(typ='links',seq=margs.seq if args.surrogate else False)
        edge_states = [edge_state[-1] if args.surrogate else edge_state]
        
        # setting = [1 for _ in args.action_space]
        setting = [env.controller('default') for _ in range(args.horizon//args.ctrl_step)]
        settings = [env.controller(args.keep,states[0],setting[0]) if args.keep != 'False' else setting[0]]

        done,i,j,valss,nfunss,timess,solss = False,0,0,[],[],[],[]
        while not done:
            if i*args.interval % args.sample_interval == 0:
                t2 = time.perf_counter()
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
                    prob.load_state(state,r,edge_state)
                    carry_over = j if hasattr(prob,'delta_control') and prob.delta_control else j+1
                    setting = setting[carry_over:] + setting[-1:] * carry_over
                    if args.gradient:
                        if not prob.conti: # Warm start with logits of previous horizon for discrete action space
                            setting = np.concatenate([sols[-1][j+1:], sols[-1][-1:]*(j+1)],axis=0) if i>0 else None
                        if args.method == 'gr':
                            setting,vals,nfuns,times,sols = run_gr(prob,args,setting=setting)
                        else:
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
                solss.append(sols)
                t3 = time.perf_counter()
                print('Optimization time: {} s'.format(t3-t2))
                opt_times.append(t3-t2)
                j = 0
                lag = (t3-t2)/60/args.interval
                prev_sett = sett.copy() if i > 0 else settings[0].copy()
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[0]) if args.keep == 'False' else settings[0]
            elif i*args.interval % args.ctrl_step == 0:
                j += 1
                sett = env.controller('safe',state[-1] if args.surrogate else state,setting[j]) if args.keep == 'False' else settings[0]
            real_sett = prev_sett if args.lag and i*args.interval % args.sample_interval < int(lag) else sett
            done = env.step(real_sett,
                            lag_seconds = (lag%1)*args.interval*60 if args.lag and i*args.interval % args.sample_interval == int(lag) else None)
            state = env.state_full(seq=margs.seq if args.surrogate else False)
            if args.surrogate and margs.if_flood:
                flood = env.flood(seq=margs.seq)
            edge_state = env.state_full(margs.seq if args.surrogate else False,'links')
            states.append(state[-1] if args.surrogate else state)
            perfs.append(env.flood())
            objects.append(env.objective())
            edge_states.append(edge_state[-1] if args.surrogate else edge_state)
            settings.append(real_sett)
            i += 1
            print('Simulation time: %s'%env.data_log['simulation_time'][-1])            
        
        np.save(os.path.join(args.result_dir,name + '_state.npy'),np.stack(states))
        np.save(os.path.join(args.result_dir,name + '_perf.npy'),np.stack(perfs))
        np.save(os.path.join(args.result_dir,name + '_object.npy'),np.array(objects))
        np.save(os.path.join(args.result_dir,name + '_settings.npy'),np.array(settings))
        np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),np.stack(edge_states))
        np.savez(os.path.join(args.result_dir,name + '_vals.npz'),*[np.array(vals) for vals in valss])
        np.savez(os.path.join(args.result_dir,name + '_nfuns.npz'),*[np.array(nfuns) for nfuns in nfunss])
        np.savez(os.path.join(args.result_dir,name + '_times.npz'),*[np.array(timss) for timss in timess])
        np.savez(os.path.join(args.result_dir,name + '_sols.npz'),*[np.array(sols) for sols in solss])

        results.loc[name] = [t1-t0,np.mean(opt_times),np.stack(perfs).sum(),np.stack(objects).sum()]
    results.to_csv(os.path.join(args.result_dir,'results.csv'))
