from emulator import Emulator # Emulator should be imported before env
from utilities import get_inp_files
import pandas as pd
import os,time,gc
import multiprocessing as mp
import numpy as np
from scipy.stats import truncnorm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow_probability import distributions as tfd
import argparse,yaml
from envs import get_env
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling,random_by_bounds
from pymoo.operators.sampling.lhs import LatinHypercubeSampling,sampling_lhs
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
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
    parser.add_argument('--gradient',action='store_true',help='if use gradient-based optimization')
    parser.add_argument('--cross_entropy',type=float,default=0,help=' if use cross-entropy method and the kept percentage')
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')
    parser.add_argument('--epsilon',type=float,default=0.1,help='the depth threshold of flooding')
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
    args = parser.parse_args()
    config = {k:v for k,v in args.__dict__.items() if v!=hyp.get(k)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyp[k],v))
    for i in range(len(args.termination)//2):
        args.termination[2*i+1] = eval(args.termination[2*i+1]) if args.termination[2*i] != 'time' else args.termination[2*i+1]

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
            runoff = env.state()[...,-1:]
        if tide:
            ti = env.state()[...,:1]
            runoff = np.concatenate([runoff,ti],axis=-1)
        runoffs.append(runoff)
    ts = [t0]+env.data_log['simulation_time'][:-1]
    runoff = np.array(runoffs)
    return ts,runoff

def pred_simu(y,file,args,r=None):
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
    while not done and idx < y.shape[0]:
        if args.prediction['no_runoff']:
            for node,ri in zip(env.elements['nodes'],r[idx]):
                env.env._setNodeInflow(node,ri)
        # done = env.step([actions[i][int(act)] for i,act in enumerate(y[idx])])
        done = env.step([sett for sett in y[idx]])
        # perf.append(env.flood())
        idx += 1
    # return np.array(perf)
    return env.objective(idx)

class mpc_problem(Problem):
    def __init__(self,args,margs=None):
        self.args = args
        if margs is not None:
            tf.keras.backend.clear_session()    # to clear backend occupied models
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
                            xl = np.array([0 for _ in range(self.n_var)]),
                            # xu = np.array([len(v)-1 for _ in range(self.n_step)
                            #     for v in self.actions]),
                            xu = np.array([v for _ in range(self.n_step)
                                for v in np.array(list(self.actions.keys())).max(axis=0)]),
                            vtype=int)
            # print(self.n_var,self.xu)
    
    def load_state(self,state,runoff,edge_state=None):
        self.state,self.runoff,self.edge_state = state,runoff,edge_state
    
    def load_file(self,eval_file,log=None,runoff_rate=None):
        self.file,self.runoff_rate,self.log = eval_file,runoff_rate,log

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_act))
        y = np.repeat(y,self.r_step,axis=0)
        if y.shape[0] < self.eval_hrz // self.step:
            y = np.concatenate([y,np.repeat(y[-1:,:],self.eval_hrz // self.step-y.shape[0],axis=0)],axis=0)

        env = get_env(self.args.env)(swmm_file = self.file)
        if getattr(self,'log') is not None:
            env.data_log.update({k:v for k,v in self.log.items() if 'cum' not in k})
        done,idx = False,0
        # perf = 0
        while not done and idx < y.shape[0]:
            if self.args.prediction['no_runoff']:
                for node,ri in zip(env.elements['nodes'],self.runoff_rate[idx]):
                    env.env._setNodeInflow(node,ri)
            yi = y[idx]
            # done = env.step([act if self.args.act.startswith('conti') else self.actions[i][act] for i,act in enumerate(yi)])
            done = env.step(yi if self.args.act.startswith('conti') else self.actions[tuple(yi)])
            # perf += env.performance().sum()
            idx += 1
        return env.objective(idx).sum()
    
    def pred_emu(self,y):
        y = y.reshape((-1,self.n_step,self.n_act))
        pop_size = y.shape[0]
        # settings = y if self.args.act.startswith('conti') else np.stack([np.vectorize(self.actions[i].get)(y[...,i]) for i in range(self.n_act)],axis=-1)
        settings = y if self.args.act.startswith('conti') else np.apply_along_axis(lambda x:self.actions.get(tuple(x)),-1,y)
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
        edge_state = np.repeat(np.expand_dims(self.edge_state,0),settings.shape[0],axis=0) if self.edge_state is not None else None
        preds = self.emul.predict(state,runoff,settings,edge_state)
        # env = get_env(self.args.env)(initialize=False)
        objs = self.env.objective_pred(preds if self.emul.use_edge else [preds,None],state).sum(axis=-1)
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

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())

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
        setting *= prob.n_step
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling,args.act.startswith('conti'))
    else:
        sampling = LatinHypercubeSampling() if args.act.startswith('conti') else IntegerRandomSampling()
    crossover = SBX(*args.crossover,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
    mutation = PM(*args.mutation,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
    if len(args.termination) > 2:
        termination = TerminationCollection(*[get_termination(*args.termination[i*2:(i+1)*2]) for i in range(len(args.termination)//2)])
    else:
        termination = get_termination(*args.termination)

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    print('Minimizing')
    res = minimize(prob,
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
        ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls)
    print("Best solution found: %s" % ctrls.tolist())
    print("Function value: %s" % res.F)

    vals = res.algorithm.callback.data["best"]
    # if margs is not None:
    #     del prob.emul
    # del prob
    # gc.collect()
    return ctrls.tolist(),vals

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
            return rec[1] <= vm  # TODO
        elif item == 'time':
            return rec[2] >= vm
        
    t0,rec,vals = time.time(),[0,1e6,time.time()],[]
    print('=======================================================')
    print(' n_gen |     f_lam     |     f_min     |     g_min     ')
    print('=======================================================')
    obj = np.random.uniform(size=(args.pop_size,))
    while not if_terminate(*args.termination,rec):
        x = sample_conti(mu,sig,args.pop_size) if args.act.startswith('conti') else sample_disc(pr,args.pop_size)
        obj = prob.pred(x)
        rec[0] += 1
        if obj.min() < rec[1]:
            ctrls = np.clip(x[obj.argmin()],prob.xl,prob.xu)
            ctrls = ctrls.reshape((prob.n_step,prob.n_act))
            if not args.act.startswith('conti'):
                # ctrls = np.stack([np.vectorize(prob.actions[i].get)(ctrls[...,i]) for i in range(prob.n_act)],axis=-1)
                ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls)
            ctrls = ctrls.tolist()
        # formulate a new distribution with elites
        rec[1] = min(obj.argsort()[int(args.pop_size*args.cross_entropy)],rec[1])
        x_new = x[obj<=rec[1]]
        if args.act.startswith('conti'):
            mu,sig = x_new.mean(axis=0),x_new.std(axis=0)
        else:
            pr = [np.bincount(x_ni) for x_ni in x_new.T]
        
        rec[2] = time.time() - t0
        vals.append(obj.min())
        log = str(rec[0]).center(7)+'|'
        log += str(round(obj.mean(),4)).center(15)+'|'
        log += str(round(obj.min(),4)).center(15)+'|'
        log += str(round(rec[1],4)).center(15)
        print(log)
    # print('Initial solution: ',sampling.reshape((-1,prob.n_step,prob.n_act)).tolist())
    print('Best solution: ',ctrls)
    return ctrls,vals

class mpc_problem_gr:
    def __init__(self,args,margs):
        self.args = args
        tf.keras.backend.clear_session()    # to clear backend occupied models
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
        self.optimizer = Adam(getattr(args,"learning_rate",1e-3))
        self.pop_size = getattr(args,'pop_size',64)
        self.n_var = self.n_act*self.n_step
        self.xl = np.array([min(v) for _ in range(self.n_step)
                            for v in self.asp])
        self.xu = np.array([max(v) for _ in range(self.n_step)
                            for v in self.asp])
        
    def load_state(self,state,runoff,edge_state=None):
        self.state,self.runoff,self.edge_state = state,runoff,edge_state

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
    def pred_fit(self):
        # assert getattr(self,'y',None) is not None
        if self.stochastic:
            runoff = tf.cast(tf.tile(self.runoff,(self.pop_size,)+tuple([1 for _ in range(self.runoff.ndim-1)])),tf.float32)
        else:
            runoff = tf.cast(tf.repeat(tf.expand_dims(self.runoff,0),self.pop_size,axis=0),tf.float32)
        state = tf.cast(tf.repeat(tf.expand_dims(self.state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32)
        edge_state = tf.cast(tf.repeat(tf.expand_dims(self.edge_state,0),self.pop_size*max(self.stochastic,1),axis=0),tf.float32) if self.edge_state is not None else None
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
            env = get_env(self.args.env)(initialize=False)
            obj = env.objective_pred_tf(preds if self.emul.use_edge else [preds,None],state)
            if self.stochastic:
                obj = tf.stack([tf.reduce_mean(obj[i*self.stochastic:(i+1)*self.stochastic],axis=0) for i in range(self.pop_size)])
            loss = tf.reduce_mean(obj,axis=0) if self.cross_entropy else obj
        grads = tape.gradient(loss,self.train_vars)
        self.optimizer.apply_gradients(zip(grads,self.train_vars)) # How to regulate y in (xl,xu)
        return obj,grads

def run_gr(prob,args,setting=None):
    print('Running gradient inversion')
    # prob = mpc_problem_gr(args,margs)
    if args.use_current and setting is not None:
        setting *= prob.n_step
        sampling = initialize(setting,prob.xl,prob.xu,2 if args.cross_entropy else args.pop_size,args.sampling,conti=True)
    else:
        sampling = sampling_lhs(2 if args.cross_entropy else args.pop_size,prob.n_var,prob.xl,prob.xu)
    def if_terminate(item,vm,rec):
        if item == 'n_gen':
            return rec[0] >= vm
        elif item == 'obj':
            return rec[1] <= vm # TODO
        elif item == 'grad':
            return rec[2] <= vm
        elif item == 'time':
            return rec[3] >= vm

    t0,rec,vals = time.time(),[0,1e6,1e6,time.time()],[]
    print('=======================================================================')
    print(' n_gen |     grads     |     f_avg     |     f_min     |     g_min     ')
    print('=======================================================================')
    prob.initialize_distr(sampling) if args.cross_entropy else prob.initialize(sampling)
    obj = tf.random.uniform((args.pop_size,))
    while not if_terminate(*args.termination,rec):
        obj,grads = prob.pred_fit()
        rec[0] += 1
        if obj.numpy().min() < rec[1]:
            ctrls = prob.y[obj.numpy().argmin()].numpy()
            ctrls = np.clip(ctrls,prob.xl,prob.xu)
            ctrls = ctrls.reshape((prob.n_step,prob.n_act)).tolist()
        rec[1],rec[2] = min(rec[1],obj.numpy().min()),np.mean([grad.numpy().mean() for grad in grads])
        rec[3] = time.time() - t0
        vals.append(obj.numpy().min())
        log = str(rec[0]).center(7)+'|'
        log += str(round(rec[2],4)).center(15)+'|'
        log += str(round(obj.numpy().mean(),4)).center(15)+'|'
        log += str(round(obj.numpy().min(),4)).center(15)+'|'
        log += str(round(rec[1],4)).center(15)
        print(log)
    # print('Initial solution: ',sampling.reshape((-1,prob.n_step,prob.n_act)).tolist())
    print('Best solution: ',ctrls)
    return ctrls,vals


if __name__ == '__main__':
    args,config = parser('config.yaml')
    # mp.set_start_method('spawn', force=True)    # use gpu in multiprocessing
    ctx = mp.get_context("spawn")
    # de = {'env':'astlingen',
    #       'act':'rand3',
    #       'processes':5,
    #       'pop_size':128,
    #       'sampling':0.4,
    #       'learning_rate':0.1,
    #       'termination':['n_gen',200],
    #       'surrogate':True,
    #       'gradient':True,
    #       'rain_dir':'./envs/config/ast_test5_events.csv',
    #       'model_dir':'./model/astlingen/30s_20k_3act_1000ledgef_res_norm_flood_gat_2tcn',
    #       'result_dir':'./results/astlingen/30s_20k_3actgmpc_1000ledgef_res_norm_flood_gat_2tcn'}
    # config['rain_dir'] = de['rain_dir']
    # for k,v in de.items():
    #     setattr(args,k,v)

    env = get_env(args.env)()
    env_args = env.get_args(args.directed,args.length,args.order,args.act)
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
    events = get_inp_files(env.config['swmm_input'],rain_arg)

    if args.surrogate:
        hyps = yaml.load(open('config.yaml','r'),yaml.FullLoader)
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
        margs.use_edge = margs.use_edge or margs.edge_fusion
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
        setting = env.controller('default')
        if args.keep != 'False':
            setting = env.controller(args.keep,state,setting)
        settings = [setting]

        if args.surrogate and args.gradient:
            prob = mpc_problem_gr(args,margs)
        else:
            prob = mpc_problem(args,margs=margs if args.surrogate else None)

        done,i,valss = False,0,[]
        while not done:
            if i*args.interval % args.control_interval == 0:
                t2 = time.time()
                if args.surrogate:
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
                    # if margs.use_edge:
                        # margs.edge_state = edge_state
                    prob.load_state(state,r,edge_state if margs.use_edge else None)
                    if args.gradient:
                        setting,vals = run_gr(prob,args,setting=setting)
                    elif args.cross_entropy:
                        setting,vals = run_ce(prob,args,setting=setting)
                    else:
                        setting,vals = run_ea(prob,args,setting=setting)
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
                t3 = time.time()
                print('Optimization time: {} s'.format(t3-t2))
                opt_times.append(t3-t2)
                # Only used to keep the same condition to test internal model efficiency
                # setting = [env.controller(mode='bc')
                #             for _ in range(args.prediction['control_horizon']//args.setting_duration)]
                j = 0
            elif i*args.interval % args.setting_duration == 0:
                j += 1
            done = env.step(setting[j]) if args.keep == 'False' else env.step(settings[0])
            state = env.state(seq=margs.seq_in if args.surrogate else False)
            if args.surrogate and margs.if_flood:
                flood = env.flood(seq=margs.seq_in)
            edge_state = env.state_full(margs.seq_in if args.surrogate else False,'links')
            states.append(state[-1] if args.surrogate else state)
            perfs.append(env.flood())
            objects.append(env.objective())
            edge_states.append(edge_state[-1] if args.surrogate else edge_state)
            settings.append(setting[j])
            i += 1
            print('Simulation time: %s'%env.data_log['simulation_time'][-1])            
        
        np.save(os.path.join(args.result_dir,name + '_%s_state.npy'%item),np.stack(states))
        np.save(os.path.join(args.result_dir,name + '_%s_perf.npy'%item),np.stack(perfs))
        np.save(os.path.join(args.result_dir,name + '_%s_object.npy'%item),np.array(objects))
        np.save(os.path.join(args.result_dir,name + '_%s_settings.npy'%item),np.array(settings))
        np.save(os.path.join(args.result_dir,name + '_%s_edge_states.npy'%item),np.stack(edge_states))
        np.save(os.path.join(args.result_dir,name + '_%s_vals.npy'%item),np.array(valss))

        results.loc[name] = [t1-t0,np.mean(opt_times),np.stack(perfs).sum(),np.stack(objects).sum()]
    results.to_csv(os.path.join(args.result_dir,'results_%s.csv'%item))

