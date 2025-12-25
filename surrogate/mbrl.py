import os,yaml,shutil
import torch.multiprocessing as mp
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import warnings
warnings.filterwarnings("ignore")
import torch as th
from torch.utils.tensorboard import SummaryWriter
from emulator import Emulator
from dataloader import DataGenerator
from memory import Memory
from agent import get_agent
from mpc import get_runoff
from envs import get_env
from utils.utilities import get_inp_files
import pandas as pd,matplotlib.pyplot as plt
import argparse,time, datetime as dt
# 5 runs with different seeds
# seeds = [42, 123, 246, 357, 489]
HERE = os.path.dirname(__file__)

def parser(config):
    parser = argparse.ArgumentParser(description='surrogate')

    parser.add_argument('--env',type=str,default='astlingen',help='set drainage scenarios')
    parser.add_argument('--directed',action='store_true',help='if use directed graph')
    parser.add_argument('--length',type=float,default=0,help='adjacency range')
    parser.add_argument('--order',type=int,default=1,help='adjacency order')
    parser.add_argument('--graph_base',type=int,default=0,help='if use node(1) or edge(2) based graph structure')

    # control args
    parser.add_argument('--ctrl_step',type=int,default=5,help='number of the rainfall events')
    parser.add_argument('--act',type=str,default='rand',help='what control actions')
    parser.add_argument('--mac',action="store_true",help='if use multi-agent action space')
    parser.add_argument('--dec',action="store_true",help='if use dec-pomdp observation space')

    # pretrain args
    parser.add_argument('--pretrain',action="store_true",help='if pretrain the agent')
    parser.add_argument('--expert_dir',type=str,default='./envs/data/',help='path of the expert data')

    # rl args
    parser.add_argument('--train',action="store_true",help='if train')
    parser.add_argument('--seed',type=int,default=42,help='random seed')
    parser.add_argument('--episodes',type=int,default=1000,help='training episodes')
    parser.add_argument('--update_batch',type=int,default=64,help='training batch size')
    parser.add_argument('--batch_size',type=int,default=128,help='modelling batch size')
    parser.add_argument('--limit',type=int,default=23,help='maximum capacity 2^n of the buffer')
    parser.add_argument('--tune_gap',type=int,default=0,help='finetune the model per sample gap')
    parser.add_argument('--sample_gap',type=int,default=0,help='sample data with swmm per sample gap')
    parser.add_argument('--start_gap',type=int,default=100,help='start updating agent after start gap')
    parser.add_argument('--eval_gap',type=int,default=10,help='evaluate the agent per eval_gap')
    parser.add_argument('--save_gap',type=int,default=1000,help='save the agent per gap')

    # rollout args
    parser.add_argument('--data_dir',type=str,default='./envs/data/',help='path of the initial data')
    parser.add_argument('--branch',action="store_true",help='if use model-free branched sampling')
    parser.add_argument('--horizon',type=int,default=60,help='prediction & control horizon')
    parser.add_argument('--model_based',action="store_true",help='if use model-based sampling')
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')
    parser.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')
    parser.add_argument('--error',type=float,default=0.0,help='the modelling error noises')
    parser.add_argument('--epochs',type=int,default=100,help='model finetune times per episode')

    # agent args
    parser.add_argument('--agent',type=str,default='SAC',help='agent name')
    parser.add_argument('--conv',action="store_true",help='if use full state info')
    parser.add_argument('--dim',type=int,default=128,help='number of decision-making channels')
    parser.add_argument('--nly',type=int,default=3,help='number of graphconv layers')
    parser.add_argument('--agent_dir',type=str,default='./agent/',help='path of the agent')
    parser.add_argument('--load_agent',action="store_true",help='if load agents')
   
    # agent update args
    parser.add_argument('--repeats',type=int,default=1,help='agent update times per episode')
    parser.add_argument('--norm',action="store_true",help='if use reward normalization')
    parser.add_argument('--scale',type=float,default=1.0,help='reward scaling factor')
    parser.add_argument('--gamma',type=float,default=0.98,help='discount factor')
    parser.add_argument('--act_lr',type=float,default=1e-4,help='actor learning rate')
    parser.add_argument('--cri_lr',type=float,default=1e-3,help='critic learning rate')
    parser.add_argument('--tau',type=float,default=0.005,help='target update interval')
    parser.add_argument('--en_disc',type=float,default=1.0,help='target entropy discount factor of SAC')
    parser.add_argument('--log_alpha',type=float,default=-1.0,help='start log alpha of SAC')

    # testing scenario args: rain and result dir not useful here
    parser.add_argument('--test',action="store_true",help='if test')
    parser.add_argument('--skip_eval',action="store_true",help='if skip evaluate in RL and vali actors together')
    parser.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
    parser.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
    parser.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')
    parser.add_argument('--swmm_step',type=int,default=1,help='routing step for swmm inp files')
    parser.add_argument('--processes',type=int,default=1,help='parallel simulation')
    parser.add_argument('--result_dir',type=str,default='./results/',help='path of the results')

    args = parser.parse_args()
    hyps = yaml.load(open(config,'r'),yaml.FullLoader)
    hyps = {k:v for k,v in hyps[args.env].items() if hasattr(args,k)}
    parser.set_defaults(**hyps)
    args = parser.parse_args()

    config = {k:v for k,v in args.__dict__.items() if v!=hyps.get(k,v)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyps[k],v))

    print('MBRL configs: {}'.format(args))
    return args,config

def interact_steps(args,event,runoff,ctrl=True,params=None,train=False,device='cpu'):
    if ctrl:
        args.device = device # cpu-based interactions --> 2x speed than gpu
        agent = get_agent(args.agent)(args.action_shape,len(args.observ_space),args,act_only=True)
        if isinstance(params,dict):
            agent.actor.load_state_dict({k:agent.to_tensor(v) for k,v in params['actor'].items()})
            agent.set_norm(**params['norm'])
        else:
            agent.load(params)
    tss,runoff = runoff
    env = get_env(args.env)(swmm_file=event)
    r_step = args.ctrl_step//args.interval
    elements,attrs = args.elements,args.attrs
    observs = [th.LongTensor([(elements[typ].index(idx), attrs[typ].index(attr))
                            for idx,attr in args.states if attr in attrs[typ]])
                            for typ in ['nodes','links']]
    state = env.state_full(seq=r_step)
    if getattr(args,"if_flood",False):
        flood = env.flood(seq=r_step)
    # Fixed the bug of initial state non
    edge_state = env.state_full(r_step,'links')
    setting = np.array(env.controller('default'))
    states,perfs,settings,rains,edge_states,objects = [],[],[],[],[],[]
    done,i,times = False,0,[]
    # Reminder: args.branch and train
    while not done and (i < args.horizon + r_step if args.branch and train else True):
        if ctrl and i*args.interval % args.ctrl_step == 0:
            t0 = time.perf_counter()
            state[...,1] = state[...,1] - state[...,-1]
            if getattr(args,"if_flood",False):
                f = (flood>0).astype(float)
                state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
            t = env.env.methods['simulation_time']()
            b = runoff[int(tss.asof(t)['Index'])][:r_step]
            x_norm,b_norm,e_norm = [agent.normalize(agent.to_tensor(dat),item) if dat is not None else None
                                    for dat,item in zip([state,b,edge_state],'xbe')]
            if args.conv:
                x_norm,e_norm = [th.stack([dat[...,idx].mean(dim=0) if 'cum' in attr or '_vol' in attr else dat[-1,:,idx]
                                        for idx,attr in enumerate(attrs[items])],dim=-1)
                                        for dat,items in zip([x_norm,e_norm],['nodes','links'])]
                b_norm = th.concat([b_norm[...,:1].mean(dim=0),b_norm[-1,:,1:]],dim=-1)
                observ = [th.concat([x_norm,b_norm],dim=-1),e_norm]
            else:
                r_norm = agent.normalize(agent.to_tensor(env.rainfall(seq=r_step)),'r')
                observ = th.concat([r_norm] + [dat[:,obs[:,0],obs[:,1]]
                                               for dat,obs in zip([x_norm,e_norm],observs)
                                               if len(obs)>0],dim=-1)
                observ = th.stack([observ[:,i].mean(dim=-1) if 'cum' in attr or '_vol' in attr else observ[-1,i]
                                for i,(_,attr) in enumerate(args.states)],dim=-1)
            action = agent.control(observ,train)
            setting = agent.convert_action_to_setting(action).squeeze().cpu().numpy()
            times.append(time.perf_counter()-t0)
        done = env.step([float(sett) for sett in setting.tolist()])
        state = env.state_full(seq=r_step)
        if getattr(args,"if_flood",False):
            flood = env.flood(seq=r_step)
        edge_state = env.state_full(r_step,'links')
        states.append(state[-1])
        perfs.append(env.flood())
        objects.append(env.objective())
        edge_states.append(edge_state[-1])
        settings.append(setting)
        rains.append(env.rainfall())
        i += 1
    env.initialize_logger()
    if train or not args.skip_eval:
        return [np.array(dat) for dat in [states,perfs,settings,rains,edge_states,objects,times]]
    else:
        print(f"Finish eval {params}: {os.path.basename(event).strip('.inp')}",flush=True)
        return np.array(objects).sum()

class rl_ctrl:
    def __init__(self,args,margs=None,act_only=False):
        if margs is not None:
            self.emul = Emulator(margs,act_only = getattr(args,"tune_gap",0)==0)
            self.emul.load()
            self.emul.set_indices(getattr(args,'batch_size',128))
        self.args = args
        self.n_step = args.horizon//args.ctrl_step
        self.r_step = args.ctrl_step//args.interval
        self.agent = get_agent(args.agent)(args.action_shape,len(args.observ_space),args,act_only=act_only)
        self.conv = getattr(args,'conv',False)
        self.attrs,self.states = args.attrs,args.states
        observs = [[(args.elements[typ].index(idx), args.attrs[typ].index(attr))
                    for idx,attr in args.states if attr in args.attrs[typ]]
                    for typ in ['nodes','links']]
        self.observs = [self.to_tensor(th.LongTensor(observ)) for observ in observs]
        self.env = get_env(args.env)(initialize=False)
        self.agent_dir = getattr(args,"agent_dir",os.path.join(HERE,"agent",args.env))

        act_edges = getattr(args,"act_edges",[])
        sett = np.zeros(getattr(args,"edge_state_shape",(29,3))[0])
        sett[act_edges] = range(1,len(act_edges)+1)
        self.sett = self.to_tensor(th.LongTensor(sett))
        self.act_edges = self.to_tensor(th.LongTensor(act_edges))
        
    def finetune(self,dats):
        x,a,b,y = [self.emul.to_tensor(dat) if dat is not None else dat for dat in dats[:4]]
        ex,ey = [self.emul.to_tensor(dat) for dat in dats[6:8]]
        x,b,y,ex,ey = [self.emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],list('xbye')+['ey'])]
        model_loss = self.emul.fit_eval(x,a,b,y,ex,ey)
        return np.sum([los.detach().cpu().numpy() for los in model_loss])

    @th.compile(fullgraph=True)
    def rollout(self,data,ctrl=True,train=True):
        x,_,b,_ = data[:4]
        r = th.concat(data[4:6],dim=1)
        ex = data[6]
        sett0 = th.index_select(ex[...,-1],-1,self.act_edges)
        xs,exs,settings = [x],[ex],[sett0]
        for i in range(self.n_step):
            bi,ri = b[:,i*self.r_step:(i+1)*self.r_step,:],r[:,i*self.r_step:(i+1)*self.r_step,:]
            if ctrl:
                x_norm,e_norm,b_norm = [self.agent.normalize(dat,item) if dat is not None else None
                                        for dat,item in zip([x,ex,bi if self.conv else ri],'xeb' if self.conv else 'xer')]
                s_norm = self.get_observ(x_norm,e_norm,b_norm)
                action = self.agent.control(s_norm,train=train,batch=True)
                setting = self.agent.convert_action_to_setting(action)
                setting = setting[:,th.newaxis,:].repeat(1,self.r_step,1)
            else:
                setting = data[1][:,i*self.r_step:(i+1)*self.r_step,:]
            preds = self.emul.predict(x,ex,bi,setting,constr=False)
            if self.emul.if_flood:
                x = th.concat([preds[0][...,:-1],(preds[0][...,-1:]>0).type(th.float32),bi],dim=-1)
            else:
                x = th.concat([preds[0],bi],dim=-1)
            ex = th.concat([preds[1],self.get_edge_action(setting)],dim=-1)
            xs.append(x)
            exs.append(ex)
            settings.append(setting)
        xs,exs,settings = [th.concat(dat,dim=1) for dat in [xs,exs,settings]]
        xs,bs = self.emul.constrain(xs[...,:-1],xs[:,0,:,0]),xs[...,-1:]
        perfs = self.emul.get_flood(xs,bs[...,0])
        xs = th.concat([xs[...,:-1] if self.emul.if_flood else xs,bs],dim=-1)
        xs[...,1] += xs[...,-1]
        return xs,perfs,settings,r,exs

    @th.no_grad()
    def state_split_trajs(self,trajs,rollout=False):
        shape = (-1,self.n_step+1,self.r_step) if rollout else (1,-1,self.r_step)
        trajs = [traj.reshape(shape + tuple(traj.shape[2 if rollout else 1:])) for traj in trajs]
        seq = self.r_step*2 if self.conv else self.r_step
        if self.conv:
            trajs = [traj.repeat(2,axis=1)[:,1:-1,...] for traj in trajs]
            shape = (-1,self.n_step,seq) if rollout else (1,-1,seq)
            trajs = [traj.reshape(shape+tuple(traj.shape[3:])) for traj in trajs]
        xs,perfs,settings,rains,exs = trajs
        states = (xs[:,:-1,...].reshape((-1,seq)+tuple(xs.shape[-2:])),
                xs[:,1:,...].reshape((-1,seq)+tuple(xs.shape[-2:])))
        perfs = (perfs[:,:-1,...].reshape((-1,seq)+tuple(perfs.shape[-2:])),
                perfs[:,1:,...].reshape((-1,seq)+tuple(perfs.shape[-2:])))
        edge_states = (exs[:,:-1,...].reshape((-1,seq)+tuple(exs.shape[-2:])),
                        exs[:,1:,...].reshape((-1,seq)+tuple(exs.shape[-2:])))
        settings = settings[:,1:,...].reshape((-1,seq)+tuple(settings.shape[-1:]))
        rx = rains[:,:-1,...].reshape((-1,seq)+tuple(rains.shape[-1:]))
        ry = rains[:,1:,...].reshape((-1,seq)+tuple(rains.shape[-1:]))
        return states,perfs,settings,rx,ry,edge_states

    @th.no_grad()
    def get_trans(self,train_dats):
        x,settings,b,y,rx,ry,ex,ey = train_dats[:8]
        x_norm,b_norm,y_norm,rx,ry = [self.normalize(dat,item) for dat,item in zip([x,b,y,rx,ry],'xbyrr')]
        ex_norm,ey_norm = [self.normalize(dat,item) for dat,item in zip([ex,ey],['e','ey'])]
        b0,b1 = b_norm[:,:self.r_step,...],b_norm[:,self.r_step:,...]
        x0,x1 = x_norm[:,-self.r_step:,...],th.concat([y_norm[:,:self.r_step,:,:-1],b0],dim=-1)
        settings = settings[:,:1,:].repeat(1,self.r_step,1)
        ex0,ex1 = ex_norm[:,-self.r_step:,...],ey_norm[:,:self.r_step,...]
        # Get edge action and concat into ex1
        ex1 = th.concat([ex1,self.get_edge_action(settings)],dim=-1)
        r0,r1 = rx[:,-self.r_step:,...],ry[:,:self.r_step,...]
        s,s_ = self.get_observ(x0,ex0,b0 if self.conv else r0),self.get_observ(x1,ex1,b1 if self.conv else r1)
        # Get action info from settings
        a = self.agent.convert_setting_to_action(settings[:,0,:])
        # Get reward from env as -obj_pred
        states = (x[:,-self.r_step:,...],ex[:,-self.r_step:,...])
        preds = (y[:,:self.r_step,...],ey[:,:self.r_step,...])
        obj = self.env.objective_pred_th(preds,states,settings).sum(dim=-1)
        r = - self.env.norm_obj(obj,states) * self.args.scale if self.args.norm else - obj * self.args.scale
        return s,a,r,s_

    # TODO: calculate rollout return and derive gradients for policy/value, refer to MAAC/SVG/Dreamer paper
    # TODO: update rollout return during each online control step? But may lose batch-mean gradient
    def rollout_return(self,data):
        traj = self.rollout(data)
        # get preds,states,settings from trajs
        # obj = self.env.objective_pred_th(preds,states,settings)
        # r = - self.env.norm_obj(obj,states,g=True) * self.args.scale if self.args.norm else - obj * self.args.scale
        # Use Straight-Through Estimator for random action distribution
        pass

    def get_edge_action(self,a):
        a = th.concat([th.ones_like(a[...,:1]),a],dim=-1)
        return th.index_select(a, -1, self.sett).unsqueeze(dim=-1)
        
    def get_observ(self,x,e,b):
        # TODO: calculate mean instead of sum for vol or cum to avoid step-accum
        if self.conv:
            x,e = [th.stack([dat[...,i].mean(dim=1) if 'cum' in attr or '_vol' in attr else dat[:,-1,:,i]
                                    for i,attr in enumerate(self.attrs[items])],dim=-1)
                                        for dat,items in zip([x,e],['nodes','links'])]
            b = th.concat([b[...,:1].mean(dim=1),b[:,-1,:,1:]],dim=-1)
            s = [th.concat([x,b],dim=-1),e]
        else:
            s = th.concat([b] + [dat[...,obs[:,0],obs[:,1]] for dat,obs in zip([x,e],self.observs) if len(obs)>0],dim=-1)
            s = th.stack([s[...,i].mean(dim=-1) if 'cum' in attr or '_vol' in attr else s[...,-1,i]
                                for i,(_,attr) in enumerate(self.states)],dim=-1)
        return s

    def to_tensor(self,dat):
        return self.agent.to_tensor(dat)
    
    def normalize(self,dat,item):
        return self.agent.normalize(dat,item)
    
    def set_norm(self,*args,**kwargs):
        self.agent.set_norm(*args,**kwargs)

    def save(self,epoch=None):
        return self.agent.save(epoch)

    def load(self,epoch=None):
        return self.agent.load(epoch)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args,config = parser(os.path.join(HERE,'utils','policy.yaml'))

    # train_de = {
        # 'agent':'SAC',
        # 'train':True,
        # 'pretrain':False,'expert_dir':'./envs/data/astlingen/1s_efd_rain50/','epochs':500,
        # 'env':'chaohu','ctrl_step':10,
        # 'act':'rand',
        # 'mac':True,
        # 'dec':False,
        # 'nly':5,
        # 'data_dir':'./envs/data/chaohu/1s_rand64_rain50_smalltank/',
        # 'branch':True,'horizon':120,
        # 'sample_gap':0,
        # 'model_based':True,'model_dir':'./model/chaohu/10s_3ly_smalltank/retrain',
        # 'batch_size':8,
        # 'episodes':101,
        # 'limit':21,
        # 'tune_gap':0,
        # 'norm':True,'scale':1.0,
        # 'gamma':0.92,
        # 'conv':False,
        # 'eval_gap':10,'start_gap':0,
        # 'agent_dir': './agent/chaohu/rand_mbrl_disc01_098_50k',
        # 'load_agent':False,
        # 'processes':1,
        # 'skip_eval':True,
        # 'swmm_step':10,
        # 'rain_num':8,
        # 'rain_dir':'sh_train50_events.csv',
    # }
    # for k,v in train_de.items():
    #     setattr(args,k,v)
    #     config[k] = v
    
    # test_de = {
    #     'test':True,
    #     'env':'chaohu','ctrl_step':10, 'act':'rand',
    #     'rain_dir': 'sh_train50_events.csv',
    #     'skip_eval':True, 'eval_gap':20000,
    #     'agent_dir': './agent/chaohu/rand_mbrl_disc01_098_50k_seed123',
    #     'data_dir': './envs/data/chaohu/1s_rand64_rain50_smalltank',
    #     'processes':4,
    #     'swmm_step':10,
    #     'result_dir':'./results/astlingen/rand3_mbrl_sac',
    # }
    # for k,v in test_de.items():
    #     setattr(args,k,v)
    #     config[k] = v

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args(args.directed,args.length,args.order,args.graph_base,act=args.act,dec=args.dec)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)

    if args.train:
        # Model args
        if args.model_based or args.sample_gap == 0:
            hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'),'r'),yaml.FullLoader)
            margs = argparse.Namespace(**hyps[args.env])
            margs.model_dir = args.model_dir
            known_hyps = yaml.load(open(os.path.join(margs.model_dir,'parser.yaml'),'r'),yaml.FullLoader)
            for k,v in known_hyps.items():
                if '_dir' in k:
                    setattr(margs,k,os.path.join(hyps[args.env][k],v))
                    continue
                setattr(margs,k,v)
            setattr(margs,'epsilon',args.epsilon)
            setattr(args,'seq',margs.seq)
            assert margs.seq == args.ctrl_step//args.interval
            setattr(args,'if_flood',bool(margs.if_flood))
            setattr(args,'data_dir',margs.data_dir)
            config['if_flood'] = args.if_flood
            config['data_dir'] = os.path.basename(args.data_dir)
            if args.if_flood:
                args.attrs['nodes'] = args.attrs['nodes'][:-1] + ['if_flood'] + args.attrs['nodes'][-1:]
            env_args = env.get_args(margs.directed,margs.length,margs.order,margs.graph_base)
            for k,v in env_args.items():
                setattr(margs,k,v)

        if not os.path.exists(args.agent_dir):
            os.mkdir(args.agent_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.agent_dir,'parser.yaml'),'w'))

        # Rainfall args
        print("Get training events runoff")
        hyp = yaml.load(open(os.path.join(args.data_dir,'parser.yaml'),'r'),yaml.FullLoader)
        rain_arg = env.config['rainfall']
        if 'rain_dir' in hyp:
            rain_arg['rainfall_events'] = os.path.join('./envs/config/',hyp['rain_dir'])
        if 'rain_suffix' in hyp:
            rain_arg['suffix'] = hyp['rain_suffix']
        if 'rain_num' in hyp:
            rain_arg['rain_num'] = hyp['rain_num']
        events = get_inp_files(env.config['swmm_input'],rain_arg,swmm_step = args.swmm_step)
        if os.path.exists(os.path.join(args.agent_dir,'train_runoff.npz')):
            res = [np.load(os.path.join(args.agent_dir,'train_runoff_ts.npz'),allow_pickle=True),
                   np.load(os.path.join(args.agent_dir,'train_runoff.npz'),allow_pickle=True)]
            res = [(ts,runoff) for ts,runoff in zip(res[0].values(),res[1].values())]
        else:
            with mp.Pool(args.processes) as pool:
                res = [pool.apply_async(func=get_runoff,args=(env,event,False,args.tide and args.is_outfall,))
                    for event in events]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
            np.savez(os.path.join(args.agent_dir,'train_runoff_ts.npz'),*[np.array(r[0]) for r in res])
            np.savez(os.path.join(args.agent_dir,'train_runoff.npz'),*[np.array(r[1]) for r in res])
        runoffs = []
        for ts,runoff in res:
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            seq = args.ctrl_step//args.interval
            runoff = np.stack([np.concatenate([runoff[idx:idx+seq],np.tile(np.zeros_like(s),(max(idx+seq-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
            runoffs.append([tss,runoff])
        print("Finish training events runoff")

        # Real data for sampling base points
        dG = DataGenerator(env.config,args.data_dir,args)
        dG.load(args.data_dir)
        ctrl = rl_ctrl(args,margs=margs if args.model_based else None)
        if args.load_agent:
            ctrl.load()
        params = {'norm':dG.get_norm(),}
        ctrl.set_norm(**params['norm'])
        # Virtual data buffer for model-based rollout trajs
        dGv = Memory(args.limit,args.conv,args.agent_dir)
        if args.load_agent and not ctrl.agent.on_policy:
            dGv.load()
        n_events = int(max(dG.event_id))+1
        train_ids = np.load(os.path.join(args.data_dir,'train_id.npy'))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        train_events,test_events = [events[ix] for ix in train_ids],[events[ix] for ix in test_ids]
        train_returns,train_losses,train_objss,test_objss,secs,num_trans,dmts = [],[],[],[],[],[],[]
        if args.model_based and args.tune_gap > 0:
            model_losses,test_losses = [],[]
        log_dir = "logs/agent/"
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir)

        if args.pretrain:
            # TODO: mix policy & expert data/loss in RL, gradually reduce the IL weight
            print("Start imitation learning")
            dGi = DataGenerator(env.config,args.expert_dir,args)
            dGi.load(args.expert_dir)
            train_idxs = dGi.get_data_idxs(train_ids,args.ctrl_step,concat=False)
            for epoch in range(args.epochs):
                t = time.perf_counter()
                train_dats = dGi.prepare_batch(np.concatenate(train_idxs,axis=0),
                                               args.ctrl_step,args.batch_size,args.ctrl_step,continuous=ctrl.agent.on_policy)
                trans = ctrl.get_trans([ctrl.to_tensor(dat) if dat is not None else dat for dat in train_dats])
                trans += (ctrl.to_tensor(train_dats[-2]),)
                if args.agent == 'PPO':
                    trans = ctrl.agent.update_trajs(trans)
                pretrain_loss = ctrl.agent.update(trans,pretrain=True)
                print("{}/{} Finish imitation update: {:.2f}s ".format(epoch,args.epochs,time.perf_counter()-t) +\
                       "Mean loss:"+ (len(pretrain_loss)*" {:.2f}").format(*pretrain_loss))
                writer.add_scalar('Imitation value loss', pretrain_loss[0], epoch)
                writer.add_scalar('Imitation actor loss', pretrain_loss[1], epoch)
                writer.add_scalar('Imitation entropy', pretrain_loss[2], epoch)
            # retain expert data dGi in memory buffer dGv
            trajs = [[getattr(dGi,item)[dat]
                      for dat in train_idxs if len(dat)>0]
                      for item in ['states','perfs','settings','rains','edge_states']]
            for traj in zip(*trajs):
                states,perfs,settings,rx,ry,edge_states = ctrl.state_split_trajs(traj)
                x,b,y = dGi.state_split_batch(states,perfs)
                ex,ey = dGi.edge_state_split_batch(edge_states)
                trans = ctrl.get_trans([ctrl.to_tensor(dat) for dat in [x,settings,b,y,rx,ry,ex,ey]])
                trans += (ctrl.to_tensor(np.eye(x.shape[0])[-1]),)
                if args.agent == 'PPO':
                    trans = ctrl.agent.update_trajs(trans)
                dGv.update(trans)
            dGi.clear()

        for episode in range(args.episodes):
            setattr(args,"episode",episode)
            sec,t = [],time.perf_counter()

            # TODO: Model fine-tuning, need real model and current policy to collect data
            if args.model_based and args.tune_gap > 0 and episode > args.start_gap and episode % args.tune_gap == 0:
                print(f"{episode}/{args.episodes} Start model finetuning")
                if args.sample_gap == 0:
                    print(f"    Model finetuning: model-free sampling first")
                    # ctrl.save()
                    pool = mp.Pool(args.processes)
                    res = [pool.apply_async(func=interact_steps,args=(event,runoffs[idx],True,params,False,'cpu',))
                        for idx,event in zip(train_ids,train_events)]
                    pool.close()
                    pool.join()
                    res = [r.get() for r in res]
                    trajs = [np.concatenate([r[i] for r in res],axis=0) for i in range(5)]
                    trajs.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(train_ids,res)],axis=-1))
                    dG.update(trajs)
                train_idxs = dG.get_data_idxs(train_ids,margs.seq)
                test_idxs = dG.get_data_idxs(test_ids,margs.seq)
                ctrl.emul.set_norm(**dG.get_norm())
                k = 1 + min(dG.update_num / (dG.cur_capa - dG.update_num),5)
                epochs,batch_size = int(k*args.epochs),int(k*args.batch_size)
                for epoch in range(epochs):
                    t1 = time.perf_counter()
                    train_dats = dG.prepare_batch(train_idxs,margs.seq,batch_size,interval=args.ctrl_step)
                    model_loss = ctrl.finetune(train_dats)
                    model_losses.append(model_loss)
                    test_dats = dG.prepare_batch(test_idxs,margs.seq,batch_size,interval=args.ctrl_step)
                    test_loss = ctrl.finetune(test_dats)
                    test_losses.append(test_loss)
                    print("    Episode {} fine-tuning epoch {}/{}: {:.2f}s Train loss: {:.4f} Test loss: {:.4f}"\
                          .format(episode,epoch,epochs,time.perf_counter()-t1,model_losses[-1],test_losses[-1]))
                sec.append(time.perf_counter()-t)
                t = time.perf_counter()
                print("{}/{} Finish model fine-tuning: {:.2f}s Mean loss: Train {:.4f} Test {:.4f}"\
                      .format(episode,args.episodes,sec[-1],np.mean(model_losses[-epochs:]),np.mean(test_losses[-epochs:])))
                writer.add_scalar('Model fine-tuning train loss', np.mean(model_losses[-epochs:]), episode)
                writer.add_scalar('Model fine-tuning test loss', np.mean(test_losses[-epochs:]), episode)

            # Model-free sampling
            if args.sample_gap > 0 and episode % args.sample_gap == 0:
                print(f"{episode}/{args.episodes} Start model-free sampling")
                params['actor'] = {k:v.cpu() for k,v in ctrl.agent.actor.state_dict().items()}
                # TODO: Branched model-free RL
                if args.branch:
                    train_idxs = dG.get_data_idxs(train_ids,args.ctrl_step*2,args.horizon)
                    idxs = np.arange(0,train_idxs.shape[0],args.ctrl_step)
                    wei = dG.get_flood_weight(args.horizon)[train_idxs][idxs]
                    idxs = np.random.choice(idxs,args.batch_size,replace=False,p=wei/wei.sum(),)
                    train_id = train_idxs[idxs] - args.ctrl_step
                    env.config['prediction']['suffix'] = 'mfrl_eval_'
                    # identify the hsf file by rain, time and repeat
                    event_id,rept = dG.event_id[train_id].astype(int),dG.rept[train_id]
                    r = dG.states.take(train_id[:,None,None] +\
                                        np.arange(args.horizon+args.ctrl_step)[:,None] +\
                                              np.arange(args.ctrl_step),axis=0)[...,-1:]
                    mfbr_events,mfbr_runoffs = [],[]
                    for i,(eid,(rep,ts),ri) in enumerate(zip(event_id,rept,r)):
                        env.config['swmm_input'] = events[eid]
                        ts = dt.datetime.fromtimestamp(ts)
                        if env.config['rainfall'].get('duplicate_rain',False):
                            gage = events[eid].split('_')[-1].strip('.inp')
                            hsf = f'{rep}-{gage}-{ts.strftime("%Y-%m-%d-%H-%M")}.hsf'
                        else:
                            hsf = f'{rep}-{ts.strftime("%Y-%m-%d-%H-%M")}.hsf'
                        hsf = os.path.join(os.path.abspath(args.data_dir),'hsf/',hsf)
                        mfbr_events.append(env.create_eval_file(hsf,ct=ts,eval_file='%s.inp'%i))
                        tss = pd.DataFrame.from_dict({'Time':np.arange(ts,ts + dt.timedelta(minutes=args.horizon+args.ctrl_step),
                                                                       dt.timedelta(minutes=args.interval)),
                                                      'Index':np.arange(args.horizon+args.ctrl_step)}).set_index('Time')
                        tss.index = pd.to_datetime(tss.index)
                        mfbr_runoffs.append((tss,ri))
                else:
                    train_id = np.random.choice(train_ids,args.rain_num,replace=False)
                with mp.Pool(args.processes) as pool:
                    res = [pool.apply_async(func=interact_steps,
                                            args=(args,mfbr_events[i] if args.branch else events[idx],
                                                  mfbr_runoffs[i] if args.branch else runoffs[idx],
                                                  True, params,True,'cpu',))
                        for i,idx in enumerate(train_id)]
                    pool.close()
                    pool.join()
                    res = [r.get() for r in res]
                # res = [interact_steps(args,mfbr_events[i] if args.branch else events[idx],
                #                       mfbr_runoffs[i] if args.branch else runoffs[idx],True, params,True,'cpu',)
                #                       for i,idx in enumerate(train_id)]
                if not args.branch:
                    trajs = [[r[i][args.horizon:] for r in res] for i in range(5)]
                trajs = [[r[i] if r[i].shape[0] % ctrl.r_step == 0 else \
                          np.concatenate([r[i],np.repeat(r[i][-1:],ctrl.r_step - r[i].shape[0] % ctrl.r_step,axis=0)],axis=0)
                           for r in res] for i in range(5)]
                returns,n_trans = [],0
                for traj in zip(*trajs):
                    states,perfs,settings,rx,ry,edge_states = ctrl.state_split_trajs(traj)
                    x,b,y = dG.state_split_batch(states,perfs)
                    ex,ey = dG.edge_state_split_batch(edge_states)
                    trans = ctrl.get_trans([ctrl.to_tensor(dat) for dat in [x,settings,b,y,rx,ry,ex,ey]])
                    returns.append(trans[2].cpu().numpy().sum())
                    trans += (ctrl.to_tensor(np.eye(x.shape[0])[-1]),)
                    dGv.update(trans)
                    n_trans += trans[0].shape[0]
                if args.agent == 'PPO':
                    trans = ctrl.agent.update_trajs([getattr(dGv,item) for item in dGv.items])
                    dGv.clear()
                    dGv.update(trans)
                num_trans.append(n_trans)
                train_objss.append(np.array([np.sum(r[5]) for r in res]))
                if np.mean(train_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in train_objss[:-1]]):
                    ctrl.save('retrain-train' if args.load_agent else 'train')
                dmts.append(np.array([np.sum(r[-1]) for r in res]))
                sec.append(time.perf_counter()-t)
                t = time.perf_counter()
                print("{}/{} Finish model-free sampling: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(train_objss[-1])),flush=True)
                writer.add_scalar('Model-free training objectives', np.mean(train_objss[-1]), episode)
                writer.add_scalar('Model-free training return', np.mean(returns), episode)
                train_returns.append(np.mean(returns))

            # Model-based sampling
            if args.model_based or args.sample_gap == 0:
                print(f"{episode}/{args.episodes} Start model-based sampling")
                train_idxs = dG.get_data_idxs(train_ids,args.horizon)
                train_dats = dG.prepare_batch(train_idxs,args.horizon,args.batch_size,args.ctrl_step,weight=True,trim=True)
                with th.no_grad():
                    trajs_v = ctrl.rollout([ctrl.to_tensor(dat) if dat is not None else dat for dat in train_dats[:-2]],ctrl=True)
                # convert trajectories to dG dats and to transitions
                states,perfs,settings,rx,ry,edge_states = ctrl.state_split_trajs(trajs_v,True)
                x,b,y = dG.state_split_batch(states,perfs)
                ex,ey = dG.edge_state_split_batch(edge_states)
                dats = [x,settings,b,y,rx,ry,ex,ey]
                dones = np.tile(np.eye(ctrl.n_step - int(ctrl.conv))[-1],args.batch_size)
                trans = ctrl.get_trans(dats) + (ctrl.to_tensor(dones),)
                returns = trans[2].cpu().numpy().sum()/args.batch_size
                if args.agent == 'PPO':
                    trans = ctrl.agent.update_trajs(trans)
                dGv.update(trans)
                num_trans.append(trans[0].shape[0])
                sec.append(time.perf_counter()-t)
                t = time.perf_counter()
                print("{}/{} Finish model-based sampling: {:.2f}s".format(episode,args.episodes,sec[-1]),flush=True)
                writer.add_scalar('Rollout return', returns, episode)
                train_returns.append(returns)

            # Model-free update
            if (ctrl.agent.on_policy or episode > args.start_gap) and dGv.cur_capa > 0:
                print(f"{episode}/{args.episodes} Start model-free update")
                # TODO: according to update-to-data ratio papers, probably need repeats>12 to reuse every data
                repeats = int((1 + dGv.cur_capa / dGv.limit) * args.repeats)
                train_loss = []
                for _ in range(repeats):
                    trans = dGv.sample(args.update_batch,continuous=ctrl.agent.on_policy)
                    loss = ctrl.agent.update(trans)
                    train_loss.append(loss)
                sec.append(time.perf_counter()-t)
                t = time.perf_counter()
                train_loss = np.mean(train_loss,axis=0)
                if isinstance(train_loss,np.ndarray):
                    print("{}/{} Finish model-free update: {:.2f}s Mean loss:".format(episode,args.episodes,sec[-1])+ (len(train_loss)*" {:.2f}").format(*train_loss),flush=True)
                    writer.add_scalar('Value loss', train_loss[0], episode)
                    writer.add_scalar('Policy loss', train_loss[1], episode)
                    writer.add_scalar('Entropy', train_loss[2], episode)
                    if args.agent.upper() == 'SAC':
                        writer.add_scalar('Alpha', train_loss[-1], episode)
                else:
                    print("{}/{} Finish model-free update: {:.2f}s Mean loss: {:.2f}".format(episode,args.episodes,sec[-1],train_loss),flush=True)
                    writer.add_scalar('Value loss', train_loss, episode)
                    writer.add_scalar('Epsilon', max(0.1,getattr(args,"epsilon_decay",0.9996)**episode), episode)
                train_losses.append(train_loss)

            # Evaluate the model in several episodes
            if episode > args.start_gap and args.eval_gap > 0 and episode % args.eval_gap == 0:
                if args.skip_eval:
                    ctrl.agent.actor.save(os.path.join(ctrl.agent_dir,'actors'),name=episode)
                else:
                    print(f"{episode}/{args.episodes} Start model-free interaction")
                    # ctrl.save()
                    params['actor'] = {k:v.cpu() for k,v in ctrl.agent.actor.state_dict().items()}
                    with mp.Pool(args.processes) as pool:
                        res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],True,params,False,'cpu',))
                                for idx,event in zip(test_ids,test_events)]
                        pool.close()
                        pool.join()
                        res = [r.get() for r in res]
                    # res = [interact_steps(args,event,runoffs[idx],True,params,False,) for idx,event in zip(test_ids,test_events)]
                    trajs = [np.concatenate([r[i] for r in res],axis=0) for i in range(5)]
                    trajs.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(test_ids,res)],axis=-1))
                    if args.tune_gap > 0:
                        dG.update(trajs)
                    test_objss.append(np.array([np.sum(r[5]) for r in res]))
                    sec.append(time.perf_counter()-t)
                    t = time.perf_counter()
                    print("{}/{} Finish model-free interaction: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(test_objss[-1])))
                    writer.add_scalar('Testing objectives', np.mean(test_objss[-1]), episode)
                    if np.mean(test_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in test_objss[:-1]]):
                        ctrl.save('retrain-test' if args.load_agent else 'test')
            secs.append(sec)

            if ctrl.agent.on_policy and (args.model_based or (episode+1) % args.sample_gap == 0):
                dGv.clear()
            # Save the agent
            if episode > args.start_gap and episode % args.save_gap == 0:
                cwd = os.path.join(ctrl.agent_dir, 'retrain') if args.load_agent else ctrl.agent_dir
                if not os.path.exists(cwd):
                    os.mkdir(cwd)
                ctrl.save('retrain-%s'%episode if args.load_agent else episode)
                if args.skip_eval:
                    ctrl.agent.save_norm(os.path.join(ctrl.agent_dir,'actors'))
                if args.model_based and args.tune_gap > 0:
                    np.save(os.path.join(cwd,'model_loss.npy'),np.array(model_losses))
                if args.sample_gap > 0:
                    np.save(os.path.join(cwd,'train_objs.npy'),np.array(train_objss))
                np.save(os.path.join(cwd,'train_returns.npy'),np.array(train_returns))
                np.save(os.path.join(cwd,'train_loss.npy'),train_losses)
                if not args.skip_eval:
                    np.save(os.path.join(cwd,'test_objs.npy'),np.array(test_objss))
                if len(dmts) > 0: np.savez(os.path.join(cwd,'decision-making_time.npz'),*dmts)
                np.savez(os.path.join(cwd,'time.npz'),*[np.array(sec) for sec in secs])
                np.save(os.path.join(cwd,'num_trans.npy'),np.array(num_trans))
                if not ctrl.agent.on_policy:
                    dGv.save(cwd)
        ctrl.save('retrain' if args.load_agent else None)
        if args.skip_eval:
            ctrl.agent.save_norm(os.path.join(ctrl.agent_dir,'actors'))
        cwd = os.path.join(ctrl.agent_dir, 'retrain') if args.load_agent else ctrl.agent_dir
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        if args.model_based and args.tune_gap > 0:
            np.save(os.path.join(cwd,'model_loss.npy'),np.array(model_losses))
            plt.plot(model_losses,label='model_loss',alpha=0.5)
            plt.savefig(os.path.join(cwd,'model_loss.png'),dpi=300)
            plt.clf()
        train_losses = np.array(train_losses)
        np.save(os.path.join(cwd,'train_loss.npy'),train_losses)
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,8))
        for ax,item,losses in zip([ax1,ax2,ax3,ax4][:train_losses.shape[-1]],
                                  ['value loss','policy loss','entropy','alpha'][:train_losses.shape[-1]],
                                  train_losses.T):
            ax.plot(losses,label=item)
            ax.set_xlabel('Episode')
            ax.set_ylabel(item)
        fig.savefig(os.path.join(cwd,'train_loss.png'),dpi=300)
        plt.clf()
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,12))
        if args.sample_gap > 0:
            ax1.plot(np.mean(train_objss,axis=-1),label='train_objs')
            np.save(os.path.join(cwd,'train_objs.npy'),np.array(train_objss))
        if not args.skip_eval:
            ax2.plot(np.mean(test_objss,axis=-1),label='test_objs')
            np.save(os.path.join(cwd,'test_objs.npy'),np.array(test_objss))
        ax3.plot(train_returns,label='sample returns')
        np.save(os.path.join(cwd,'train_returns.npy'),np.array(train_returns))
        fig.savefig(os.path.join(cwd,'objectives.png'),dpi=300)
        plt.clf()
        if len(dmts) > 0: np.savez(os.path.join(cwd,'decision-making_time.npz'),*dmts)
        np.savez(os.path.join(cwd,'time.npz'),*[np.array(sec) for sec in secs])
        np.save(os.path.join(cwd,'num_trans.npy'),np.array(num_trans))
        if not ctrl.agent.on_policy:
            dGv.save(cwd)

    if args.test:
        known_hyps = yaml.load(open(os.path.join(args.agent_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k in config or k == 'skip_eval':
                continue
            elif k.endswith('_dir'):
                v = os.path.join(getattr(args,k),v)
            setattr(args,k,v)

        cwd = args.agent_dir if args.skip_eval else args.result_dir
        if not args.skip_eval:
            if not os.path.exists(cwd):
                os.mkdir(cwd)
            yaml.dump(data=config,stream=open(os.path.join(cwd,'parser.yaml'),'w'))

        # Rainfall args
        print("Get training events runoff")
        rain_arg = env.config['rainfall']
        if 'rain_dir' in config:
            rain_arg['rainfall_events'] = os.path.join('./envs/config/',config['rain_dir'])
        if 'rain_suffix' in config:
            rain_arg['suffix'] = config['rain_suffix']
        if 'rain_num' in config:
            rain_arg['rain_num'] = config['rain_num']
        events = get_inp_files(env.config['swmm_input'],rain_arg,swmm_step=args.swmm_step)
        if os.path.exists(os.path.join(cwd,'train_runoff.npz')):
            res = [np.load(os.path.join(cwd,'train_runoff_ts.npz'),allow_pickle=True),
                   np.load(os.path.join(cwd,'train_runoff.npz'),allow_pickle=True)]
            res = [(ts,runoff) for ts,runoff in zip(res[0].values(),res[1].values())]
        else:
            with mp.Pool(args.processes) as pool:
                res = [pool.apply_async(func=get_runoff,args=(env,event,False,args.tide and args.is_outfall,))
                    for event in events]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
            np.savez(os.path.join(cwd,'train_runoff_ts.npz'),*[np.array(r[0]) for r in res])
            np.savez(os.path.join(cwd,'train_runoff.npz'),*[np.array(r[1]) for r in res])
        runoffs = []
        for ts,runoff in res:
            # Use mp to get runoff
            # ts,runoff = get_runoff(env,event,tide=args.tide)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            seq = args.ctrl_step//args.interval
            runoff = np.stack([np.concatenate([runoff[idx:idx+seq],np.tile(np.zeros_like(s),(max(idx+seq-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
            runoffs.append([tss,runoff])
        print("Finish testing events runoff")

        # Test the agent
        if args.skip_eval:
            train_ids = np.load(os.path.join(args.data_dir,'train_id.npy'))
            test_ids = [ev for ev in range(len(events)) if ev not in train_ids]
            events,runoffs = [events[idx] for idx in test_ids],[runoffs[idx] for idx in test_ids]
            args.agent_dir = os.path.join(args.agent_dir,'actors')
            epochs = sorted([int(f.split('.')[0][5:]) for f in os.listdir(args.agent_dir)
                             if f.endswith('.pt') and f.startswith('actor')])
            epochs = [ep for ep in epochs if ep%args.eval_gap==0]
        else:
            epochs = [None]
        ress = []
        for epoch in epochs:
            # res = [interact_steps(args,event,runoffs[idx],True,epoch,False,'cpu',) for epoch in epochs for idx,event in enumerate(events)]
            t = time.perf_counter()
            with mp.Pool(args.processes) as pool:
                res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],True,epoch,False,'cpu',))
                       for idx,event in enumerate(events)]
                pool.close()
                pool.join()
                ress.append([r.get() for r in res])
            print("Finish testing epoch {}: {:.2f}s".format(epoch if epoch is not None else 'final',time.perf_counter()-t),flush=True)
        if args.skip_eval:
            np.save(os.path.join(cwd,'test_objs.npy'),np.array(ress))
        else:
            for event,(states,perfs,settings,rains,edge_states,objects,times) in zip(events,ress[0]):
                name = os.path.basename(event).strip('.inp')
                np.save(os.path.join(cwd,name + '_state.npy'),states)
                np.save(os.path.join(cwd,name + '_perf.npy'),perfs)
                np.save(os.path.join(cwd,name + '_object.npy'),objects)
                np.save(os.path.join(cwd,name + '_settings.npy'),settings)
                np.save(os.path.join(cwd,name + '_edge_states.npy'),edge_states)
                np.save(os.path.join(cwd,name + '_times.npy'),times)