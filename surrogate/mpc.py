from envs import get_env
from swmm_api import read_inp_file,read_rpt_file
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
import yaml
HERE = os.path.dirname(__file__)

class mpc_problem(Problem):
    def __init__(self,eval_file,config):
        self.config = config
        self.file = eval_file
        self.n_act = len(config['action_space'])
        self.actions = list(config['action_space'].values())
        self.n_step = config['prediction']['control_horizon']//config['control_interval']
        self.n_var = self.n_act*self.n_step
        self.n_obj = 1
            
        super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                         xl = np.array([0 for _ in range(self.n_var)]),
                         xu = np.array([len(v)-1 for _ in range(self.n_step)
                            for v in self.actions]),
                         vtype=int)

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_act)).tolist()
        # eval_file = update_controls(self.file,self.config.action_space,k,y)
        # rpt_file,_ = swmm5_run(eval_file,create_out=False)

        env = get_env()(swmm_file = self.file)
        done = False
        idx = 0
        perf = 0
        while not done:
            yi = y[idx] if idx < len(y) else y[-1]
            done = env.step([self.actions[i][act] for i,act in enumerate(yi)])
            perf += env.performance().sum()
            idx += 1
        return perf
        
    def _evaluate(self,x,out,*args,**kwargs):        
        pool = mp.Pool(self.config.processes)
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

# TODO: ea args
def run_ea(eval_file,args,setting=None):
    prob = mpc_problem(eval_file,args)
    if setting is not None:
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling[-1])
    else:
        sampling = eval(args.sampling[0])()
    crossover = eval(args.crossover[0])(vtype=int,repair=RoundingRepair())
    mutation = eval(args.mutation[0])(*args.mutation[1:],vtype=int,repair=RoundingRepair())

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    
    res = minimize(prob,
                   method,
                   termination = args.termination,
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
    ctrls = ctrls.reshape((prob.n_step,prob.n_pump)).tolist()
    return ctrls[0]