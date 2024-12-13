from utils.utilities import get_inp_files
import pandas as pd
import os,time,gc
import multiprocessing as mp
import numpy as np
import argparse,yaml
from envs import get_env
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling,random_by_bounds
from pymoo.operators.sampling.lhs import LatinHypercubeSampling,sampling_lhs
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
HERE = os.path.dirname(__file__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    parser.add_argument('--act',type=str,default='rand',help='what control actions')

    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')
    parser.add_argument('--pop_size',type=int,default=32,help='number of population')
    parser.add_argument('--sampling',type=float,default=0.4,help='sampling rate')
    parser.add_argument('--crossover',nargs='+',type=float,default=[1.0,3.0],help='crossover rate')
    parser.add_argument('--mutation',nargs='+',type=float,default=[1.0,3.0],help='mutation rate')
    parser.add_argument('--termination',nargs='+',type=str,default=['n_eval','256'],help='Iteration termination criteria')
    
    parser.add_argument('--result_dir',type=str,default='./results/',help='path of the control results')
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
    args.termination[-1] = eval(args.termination[-1]) if args.termination[0] not in ['time','soo','moo'] else args.termination[-1]

    print('MaxRed configs: {}'.format(args))
    return args,config

class mpc_problem(Problem):
    def __init__(self,args,eval_file=None):
        self.args = args
        self.file = eval_file
        self.n_act = len(args.action_space)
        self.step = args.interval
        self.n_step = args.prediction['control_horizon']//args.setting_duration
        self.r_step = args.setting_duration//args.interval
        self.n_var = self.n_act*self.n_step
        self.n_obj = 1
        if args.act.startswith('conti'):
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
            super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                            xl = np.array([0 for _ in range(self.n_var)]),
                            # xu = np.array([len(v)-1 for _ in range(self.n_step)
                            #     for v in self.actions]),
                            xu = np.array([v for _ in range(self.n_step)
                                for v in np.array(list(self.actions.keys())).max(axis=0)]),
                            vtype=int)

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_act))
        y = np.repeat(y,self.r_step,axis=0)

        env = get_env(self.args.env)(swmm_file = self.file)
        state = env.reset(self.file,global_state=True)
        done = False
        idx = 0
        # perf = 0
        while not done and idx < y.shape[0]:
            if self.args.prediction['no_runoff']:
                for node,ri in zip(env.elements['nodes'],self.args.runoff_rate[idx]):
                    env.env._setNodeInflow(node,ri)
            if idx % self.args.setting_duration == 0:
                yi = y[idx]
                sett = yi if self.args.act.startswith('conti') else self.actions[tuple(yi)]
                # sett = np.array(env.controller('safe',state,sett)).astype(float)
            # done = env.step([act if self.args.act.startswith('conti') else self.actions[i][act] for i,act in enumerate(yi)])
            done = env.step(sett)
            state = env.state_full()
            # perf += env.performance().sum()
            idx += 1
        return env.objective(idx).sum()
 
    def _evaluate(self,x,out,*args,**kwargs):
        pool = mp.Pool(self.args.processes)
        res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
        pool.close()
        pool.join()
        F = [r.get() for r in res]
        out['F'] = np.array(F)

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

if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))
    # mp.set_start_method('spawn', force=True)    # use gpu in multiprocessing
    ctx = mp.get_context("spawn")

    env = get_env(args.env)()
    env_args = env.get_args(args.directed,args.length,args.order,args.act)
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


    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))

    for event in events:
        name = os.path.basename(event).strip('.inp')
        if os.path.exists(os.path.join(args.result_dir,name + '_state.npy')):
            continue
        t0 = time.time()
        ts,runoff_rate = get_runoff(env,event,True)
        tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
        tss.index = pd.to_datetime(tss.index)
        t1 = time.time()
        print('Runoff time: {} s'.format(t1-t0))

        args.prediction['eval_horizon'] = args.prediction['control_horizon'] = runoff_rate.shape[0] * args.interval
        args.prediction['no_runoff'] = False

        prob = mpc_problem(args,eval_file=event)

        sampling = LatinHypercubeSampling() if args.act.startswith('conti') else IntegerRandomSampling()
        crossover = SBX(*args.crossover,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
        mutation = PM(*args.mutation,vtype=float if args.act.startswith('conti') else int,repair=None if args.act.startswith('conti') else RoundingRepair())
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
                    # callback=BestCallback(),
                    verbose = True)
        print("Best solution found: %s" % res.X)
        print("Function value: %s" % res.F)

        ctrls = res.X
        ctrls = ctrls.reshape((prob.n_step,prob.n_act))
        if not args.act.startswith('conti'):
            # ctrls = np.stack([np.vectorize(prob.actions[i].get)(ctrls[...,i]) for i in range(prob.n_act)],axis=-1)
            ctrls = np.apply_along_axis(lambda x:prob.actions.get(tuple(x)),-1,ctrls)
        ctrls = np.repeat(ctrls,prob.r_step,axis=0)

        state = env.reset(event)
        states = [state]
        perfs,objects = [env.flood()],[env.objective()]
        edge_state = env.state_full(False,typ='links')
        edge_states = [edge_state]
        setting = [1 for _ in args.action_space]
        settings = [setting]
        done,idx = False,0
        while not done:
            if idx % args.setting_duration == 0:
                setting = ctrls[idx] if idx<ctrls.shape[0] else ctrls[-1]
                setting = np.array(env.controller('safe',state,setting)).astype(float)
            done = env.step(setting)
            state = env.state_full()
            edge_state = env.state_full(False,'links')
            states.append(state)
            perfs.append(env.flood())
            objects.append(env.objective())
            edge_states.append(edge_state)
            settings.append(setting)
            idx += 1

        np.save(os.path.join(args.result_dir,name + '_state.npy'),np.stack(states))
        np.save(os.path.join(args.result_dir,name + '_perf.npy'),np.stack(perfs))
        np.save(os.path.join(args.result_dir,name + '_object.npy'),np.array(objects))
        np.save(os.path.join(args.result_dir,name + '_settings.npy'),np.array(settings))
        np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),np.stack(edge_states))
        # np.save(os.path.join(args.result_dir,name + '_vals.npy'),np.array(valss))