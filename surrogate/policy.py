import os,yaml
import multiprocessing as mp
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')
from dataloader import DataGenerator
from emulator import Emulator
from agent import Actor
from mpc import get_runoff
from envs import get_env
from utils.utilities import get_inp_files
from utils.memory import RandomMemory
from functools import reduce
import pandas as pd
import argparse,time
import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')
    # env args
    parser.add_argument('--env',type=str,default='astlingen',help='set drainage scenarios')
    parser.add_argument('--directed',action='store_true',help='if use directed graph')
    parser.add_argument('--length',type=float,default=0,help='adjacency range')
    parser.add_argument('--order',type=int,default=1,help='adjacency order')
    # control args
    parser.add_argument('--setting_duration',type=int,default=5,help='setting duration')
    parser.add_argument('--act',type=str,default='rand',help='what control actions')
    parser.add_argument('--mac',action="store_true",help='if use multi-agent')
    # surrogate args
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')
    parser.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')
    # agent network args
    parser.add_argument('--seq_in',type=int,default=1,help='recurrent information for agent')
    parser.add_argument('--seq_out',type=int,default=1,help='prediction horizon')
    parser.add_argument('--conv',type=str,default='GATconv',help='convolution type')
    parser.add_argument('--recurrent',type=str,default='Conv1D',help='recurrent type')
    parser.add_argument('--use_edge',action="store_true",help='if use edge data')
    parser.add_argument('--net_dim',type=int,default=128,help='number of decision-making channels')
    parser.add_argument('--n_layer',type=int,default=3,help='number of decision-making layers')
    parser.add_argument('--conv_dim',type=int,default=128,help='number of graphconv channels')
    parser.add_argument('--n_sp_layer',type=int,default=3,help='number of graphconv layers')
    parser.add_argument('--hidden_dim',type=int,default=128,help='number of recurrent channels')
    parser.add_argument('--n_tp_layer',type=int,default=3,help='number of recurrent layers')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')

    # agent training args
    parser.add_argument('--train',action="store_true",help='if train')
    parser.add_argument('--episodes',type=int,default=1000,help='training episode')
    parser.add_argument('--gamma',type=float,default=0.98,help='discount factor')
    parser.add_argument('--batch_size',type=int,default=256,help='training batch size')
    parser.add_argument('--learning_rate',type=float,default=1e-4,help='actor learning rate')
    parser.add_argument('--save_gap',type=int,default=100,help='save the agent per gap')
    parser.add_argument('--agent_dir',type=str,default='./agent/',help='path of the agent')
    parser.add_argument('--load_agent',action="store_true",help='if load agents')

    # testing scenario args
    parser.add_argument('--test',action="store_true",help='if test')
    parser.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
    parser.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
    parser.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')
    parser.add_argument('--processes',type=int,default=1,help='parallel simulation')
    parser.add_argument('--eval_gap',type=int,default=10,help='evaluate the agent per eval_gap')
    parser.add_argument('--control_interval',type=int,default=5,help='number of the rainfall events')
    parser.add_argument('--result_dir',type=str,default='./results/',help='path of the results')

    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        parser.set_defaults(**hyps[args.env])
    args = parser.parse_args()

    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k,v)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyps[args.env][k],v))

    print('Policy configs: {}'.format(args))
    return args,config

def interact_steps(env,args,event,runoff,ctrl=None,train=False):
    # with tf.device('/cpu:0'):
    if ctrl is None:
        args.load_agent = True
        ctrl = Actor(args.action_shape,args.observ_space,args,act_only=True)
    # trajs = []
    tss,runoff = runoff
    state = env.reset(event,env.global_state,args.seq_in if args.recurrent else False)
    if args.if_flood:
        flood = env.flood(seq=args.seq_in)
    states = [state[-1]]
    perfs,objects = [env.flood()],[env.objective()]
    edge_state = env.state_full(args.seq_in,'links')
    edge_states = [edge_state[-1]]
    setting = env.controller('default')
    settings = [setting]
    done,i = False,0
    while not done:
        if i*args.interval % args.control_interval == 0:
            state[...,1] = state[...,1] - state[...,-1]
            if args.if_flood:
                f = (flood>0).astype(float)
                # f = np.eye(2)[f].squeeze(-2)
                state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
            t = env.env.methods['simulation_time']()
            b = runoff[int(tss.asof(t)['Index'])]
            # traj = [state]
            setting = ctrl.control([state,b,edge_state if args.use_edge else None],train)
            setting = setting.astype(np.float32).tolist()
            # if on_policy:
            #     action,log_probs = action
            j = 0
        elif i*args.interval % args.setting_duration == 0:
            j += 1
        sett = env.controller('safe',state[-1],setting[j])
        done = env.step(sett)
        state = env.state_full(seq=args.seq_in)
        if args.if_flood:
            flood = env.flood(seq=args.seq_in)
        edge_state = env.state_full(args.seq_in,'links')
        states.append(state[-1])
        perfs.append(env.flood())
        objects.append(env.objective())
        edge_states.append(edge_state[-1])
        settings.append(sett)        
        i += 1
        # reward = env.reward(norm = True)
        # rewards += reward
        # traj += [action,reward,state,done]
        # if on_policy:
        #     value = f.criticize(traj[0])
        #     traj += [log_probs,value]
        # trajs.append(traj)
    # perf = env.performance('cumulative')
    # if train:
    #     print('Training Reward at event {0}: {1}'.format(os.path.basename(event),rewards))
    #     print('Training Score at event {0}: {1}'.format(os.path.basename(event),perf))
    # else:
    #     print('Evaluation Score at event {0}: {1}'.format(os.path.basename(event),perf))
    return [np.array(dat) for dat in [states,edge_states,settings,perfs,objects]]


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','policy.yaml'))

    train_de = {
        # 'train':True,
        # 'env':'astlingen',
        # 'length':501,
        # 'act':'conti',
        # 'model_dir':'./model/astlingen/60s_50k_conti_1000ledgef_res_norm_flood_gat_5lyrs/',
        # 'batch_size':2,
        # 'episodes':10000,
        # 'seq_in':60,'seq_out':60,
        # 'use_edge':True,
        # 'conv':'GAT',
        # 'recurrent':'Conv1D',
        # 'eval_gap':10,
        # 'agent_dir': './agent/astlingen/60s_10k_conti',
        # 'rain_dir':'./envs/config/ast_test1_events.csv',
        # 'test':True,
        # 'result_dir':'./results/astlingen/60s_10k_conti_policy2007',
        }
    for k,v in train_de.items():
        setattr(args,k,v)
        config[k] = v

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args(args.directed,args.length,args.order,act=args.act,mac=args.mac)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)

    # Evaluation rainfall args
    print("Get runoff")
    rain_arg = env.config['rainfall']
    if 'rain_dir' in config:
        rain_arg['rainfall_events'] = args.rain_dir
    if 'rain_suffix' in config:
        rain_arg['suffix'] = args.rain_suffix
    if 'rain_num' in config:
        rain_arg['rain_num'] = args.rain_num
    events = get_inp_files(env.config['swmm_input'],rain_arg)
    pool = mp.Pool(args.processes)
    res = [pool.apply_async(func=get_runoff,args=(env,event,False,args.tide,))
     for event in events]
    pool.close()
    pool.join()
    res = [r.get() for r in res]
    runoffs = []
    for ts,runoff in res:
        # Use mp to get runoff
        # ts,runoff = get_runoff(env,event,tide=args.tide)
        tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
        tss.index = pd.to_datetime(tss.index)
        horizon = args.prediction['eval_horizon']//args.interval
        runoff = np.stack([np.concatenate([runoff[idx:idx+horizon],np.tile(np.zeros_like(s),(max(idx+horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                            for idx,s in enumerate(runoff)])
        runoffs.append([tss,runoff])
    print("Finish runoff")

    if args.train:
        # Model args
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
        setattr(args,'if_flood',margs.if_flood)
        config['if_flood'] = args.if_flood
        env_args = env.get_args(margs.directed,margs.length,margs.order)
        for k,v in env_args.items():
            setattr(margs,k,v)
        margs.use_edge = margs.use_edge or margs.edge_fusion

        dG = DataGenerator(env,margs.data_dir,args)
        dG.load(margs.data_dir)
        if not os.path.exists(args.agent_dir):
            os.mkdir(args.agent_dir)

        ctrl = Actor(args.action_shape,args.observ_space,args,act_only=False,margs=margs)
        ctrl.set_norm(*dG.get_norm())
        yaml.dump(data=config,stream=open(os.path.join(args.agent_dir,'parser.yaml'),'w'))

        seq = max(args.seq_in,args.seq_out) if args.recurrent else 0
        n_events = int(max(dG.event_id))+1
        train_ids = np.load(os.path.join(margs.model_dir,'train_id.npy'))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        train_idxs = dG.get_data_idxs(train_ids,seq)
        test_idxs = dG.get_data_idxs(test_ids,seq)

        t0 = time.time()
        train_losses,test_objss,secs = [],[],[0]
        for episode in range(args.episodes):
            # Training
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size)
            x,b = train_dats[0],train_dats[2]
            if margs.use_edge:
                ex = train_dats[-2]
            train_loss = ctrl.update(x,b,ex if args.use_edge else None)
            train_losses.append(train_loss)

            secs.append(time.time()-t0)
            # Log output
            log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f}".format(episode,args.episodes,secs[-1]-secs[-2],train_loss)
            print(log)

            if episode % args.eval_gap == 0:
                ctrl.save()
                pool = mp.Pool(args.processes)
                res = [pool.apply_async(func=interact_steps,args=(env,args,event,runoff,))
                 for event,runoff in zip(events,runoffs)]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
                # res = [interact_steps(env,args,event,runoff,ctrl)
                #         for event,runoff in zip(events,runoffs)]
                test_objs = np.array([r[-1].sum() for r in res]).sum()
                test_objss.append(test_objs)
                secs.append(time.time()-t0)
                print("Eval {}/{}  {:.4f}s Eval objs: {}".format(episode,args.episodes,secs[-1]-secs[-2],test_objs))
                if test_objs < min([1e6]+test_objss[:-1]):
                    ctrl.save(os.path.join(args.agent_dir,'test'))

            if train_loss < min([1e6]+train_losses[:-1]):
                ctrl.save(os.path.join(args.agent_dir,'train'))
            if episode > 0 and episode % args.save_gap == 0:
                ctrl.save(os.path.join(args.agent_dir,'%s'%episode))
        
        ctrl.save(args.agent_dir)
        np.save(os.path.join(args.agent_dir,'train_id.npy'),np.array(train_ids))
        np.save(os.path.join(args.agent_dir,'test_id.npy'),np.array(test_ids))
        np.save(os.path.join(args.agent_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(args.agent_dir,'test_objs.npy'),np.array(test_objss))
        np.save(os.path.join(args.agent_dir,'time.npy'),np.array(secs[1:]))
        plt.plot(train_losses,label='train')
        plt.savefig(os.path.join(args.agent_dir,'train.png'),dpi=300)
        plt.plot(test_objss,label='test')
        plt.savefig(os.path.join(args.agent_dir,'test.png'),dpi=300)

    if args.test:
        known_hyps = yaml.load(open(os.path.join(args.agent_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k in ['agent_dir','act']:
                continue
            setattr(args,k,v)        
        args.load_agent = True
        env_args = env.get_args(args.directed,args.length,args.order)
        for k,v in env_args.items():
            if k == 'act':
                v = v and args.act != 'False' and args.act
            setattr(args,k,v)
        ctrl = Actor(args.action_shape,args.observ_space,args,act_only=True)
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))
        # pool = mp.Pool(args.processes)
        # res = [pool.apply_async(func=interact_steps,args=(env,args,event,runoff,ctrl,))
        #  for event,runoff in zip(events,runoffs)]
        # pool.close()
        # pool.join()
        # res = [r.get() for r in res]
        for event,runoff in zip(events,runoffs):
            states,edge_states,settings,perfs,objs = interact_steps(env,args,event,runoff,ctrl)
            # states,edge_states,settings,objs = r
            name = os.path.basename(event).strip('.inp')
            np.save(os.path.join(args.result_dir,name + '_runoff.npy'),runoff[-1])
            np.save(os.path.join(args.result_dir,name + '_state.npy'),states)
            np.save(os.path.join(args.result_dir,name + '_settings.npy'),settings)
            np.save(os.path.join(args.result_dir,name + '_perf.npy'),perfs)
            np.save(os.path.join(args.result_dir,name + '_objs.npy'),objs)
            np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),edge_states)

