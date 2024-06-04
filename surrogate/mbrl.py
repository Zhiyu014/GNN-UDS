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
from agent import Actor,Agent
from mpc import get_runoff
from envs import get_env
from utils.utilities import get_inp_files
from utils.memory import RandomMemory
from functools import reduce
import pandas as pd
import argparse,time
HERE = os.path.dirname(__file__)

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')

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
    parser.add_argument('--horizon',type=int,default=60,help='prediction & control horizon')
    parser.add_argument('--conv',type=str,default='GATconv',help='convolution type')
    parser.add_argument('--recurrent',type=str,default='Conv1D',help='recurrent type')
    parser.add_argument('--use_edge',action="store_true",help='if use edge data')
    parser.add_argument('--net_dim',type=int,default=128,help='number of decision-making channels')
    parser.add_argument('--n_layer',type=int,default=3,help='number of decision-making layers')
    parser.add_argument('--conv_dim',type=int,default=128,help='number of graphconv channels')
    parser.add_argument('--n_sp_layer',type=int,default=3,help='number of graphconv layers')
    parser.add_argument('--hidden_dim',type=int,default=128,help='number of recurrent channels')
    parser.add_argument('--n_tp_layer',type=int,default=3,help='number of recurrent layers')
    parser.add_argument('--dueling',action="store_true",help='if use dueling network')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')

    # agent training args
    parser.add_argument('--train',action="store_true",help='if train')
    parser.add_argument('--episodes',type=int,default=1000,help='training episode')
    parser.add_argument('--repeats',type=int,default=5,help='training repeats per episode')
    parser.add_argument('--gamma',type=float,default=0.98,help='discount factor')
    parser.add_argument('--batch_size',type=int,default=256,help='training batch size')
    parser.add_argument('--max_capacity',type=int,default=2**19,help='training batch size')
    parser.add_argument('--act_lr',type=float,default=1e-4,help='actor learning rate')
    parser.add_argument('--cri_lr',type=float,default=1e-3,help='critic learning rate')
    parser.add_argument('--update_interval',type=float,default=0.005,help='target update interval')
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
        hyp = {k:v for k,v in hyps.items() if hasattr(args,k)}
        parser.set_defaults(**hyp)
    args = parser.parse_args()

    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k,v)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyps[args.env][k],v))

    print('MBRL configs: {}'.format(args))
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
            b = runoff[int(tss.asof(t)['Index'])][:args.seq_in]
            # traj = [state]
            setting = ctrl.control([state,b,edge_state if args.use_edge else None],train)
            setting = setting.astype(np.float32).tolist()
            # if on_policy:
            #     action,log_probs = action
        #     j = 0
        # elif i*args.interval .format args.setting_duration == 0:
        #     j += 1
        sett = env.controller('safe',state[-1],setting)
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
    return [np.array(dat) for dat in [states,perfs,settings,edge_states,objects]]


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','policy.yaml'))

    train_de = {
        'train':True,
        'env':'astlingen',
        'length':501,
        'act':'conti',
        'model_dir':'./model/astlingen/5s_20k_conti_500ledgef_res_norm_flood_gat/',
        'batch_size':2,
        'episodes':1000,
        'seq_in':5,'horizon':60,
        'setting_duration':5,
        'use_edge':True,
        'conv':'GAT',
        'recurrent':'Conv1D',
        'eval_gap':10,
        'agent_dir': './agent/astlingen/5s_10k_conti_mbrl',
        'processes':2,
        'test':False,
        'rain_dir':'./envs/config/ast_test1_events.csv',
        'result_dir':'./results/astlingen/60s_10k_conti_policy2007',
        }
    for k,v in train_de.items():
        setattr(args,k,v)
        config[k] = v

    env = get_env(args.env)()
    env_args = env.get_args(act=args.act,mac=args.mac)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)

    if args.train:
        if not os.path.exists(args.agent_dir):
            os.mkdir(args.agent_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.agent_dir,'parser.yaml'),'w'))

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
        
        # Rainfall args
        # margs.data_dir = './envs/data/astlingen/1s_edge_rand3128_rain50/act1/'
        print("Get training events runoff")
        hyp = yaml.load(open(os.path.join(margs.data_dir,'parser.yaml'),'r'),yaml.FullLoader)
        rain_arg = env.config['rainfall']
        if 'rain_dir' in hyp:
            rain_arg['rainfall_events'] = os.path.join('./envs/config/',hyp['rain_dir'])
        if 'rain_suffix' in hyp:
            rain_arg['suffix'] = hyp['rain_suffix']
        if 'rain_num' in hyp:
            rain_arg['rain_num'] = hyp['rain_num']
        events = get_inp_files(env.config['swmm_input'],rain_arg)
        if os.path.exists(os.path.join(args.agent_dir,'train_runoff.npy')):
            res = [np.load(os.path.join(args.agent_dir,'train_runoff_ts.npy'),allow_pickle=True),
                   np.load(os.path.join(args.agent_dir,'train_runoff.npy'),allow_pickle=True)]
            res = [(ts,runoff) for ts,runoff in zip(res[0],res[1])]
        else:
            pool = mp.Pool(args.processes)
            res = [pool.apply_async(func=get_runoff,args=(env,event,False,args.tide,))
            for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
            np.save(os.path.join(args.agent_dir,'train_runoff_ts.npy'),np.array([r[0] for r in res]))
            np.save(os.path.join(args.agent_dir,'train_runoff.npy'),np.array([r[1] for r in res]))
        runoffs = []
        for ts,runoff in res:
            # Use mp to get runoff
            # ts,runoff = get_runoff(env,event,tide=args.tide)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            seq = args.setting_duration//args.interval
            runoff = np.stack([np.concatenate([runoff[idx:idx+seq],np.tile(np.zeros_like(s),(max(idx+seq-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                for idx,s in enumerate(runoff)])
            runoffs.append([tss,runoff])
        print("Finish training events runoff")

        dG = DataGenerator(env,margs.data_dir,args)
        dG.load(margs.data_dir)
        ctrl = Agent(args.action_shape,args.observ_space,args,act_only=False,margs=margs)
        ctrl.set_norm(*dG.get_norm())
        seq = max(args.seq_in,args.horizon) if args.recurrent else 0
        n_events = int(max(dG.event_id))+1
        train_ids = np.load(os.path.join(margs.model_dir,'train_id.npy'))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        train_events,test_events = [events[ix] for ix in train_ids],[events[ix] for ix in test_ids]

        train_losses,eval_losses,train_objss,test_objss,secs = [],[],[],[],[0]
        for episode in range(args.episodes):
            # Model-free sampling
            t1 = time.time()
            print(f"{episode}/{args.episodes} Start model-free sampling")
            if args.processes > 1:
                args.load_agent = True
                ctrl.save()
                pool = mp.Pool(args.processes)
                res = [pool.apply_async(func=interact_steps,args=(env,args,event,runoffs[idx],None,True,))
                       for idx,event in zip(train_ids,train_events)]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
            else:
                res = [interact_steps(env,args,event,runoffs[idx],ctrl=ctrl,train=True)
                    for idx,event in zip(train_ids,train_events)]
            data = [np.concatenate([r[i] for r in res],axis=0) for i in range(4)]
            data.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(train_ids,res)],axis=-1))
            dG.update(data)
            train_objss.append(np.array([np.sum(r[-1]) for r in res]))
            if np.mean(train_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in train_objss[:-1]]):
                if not os.path.exists(os.path.join(args.agent_dir,'train')):
                    os.mkdir(os.path.join(args.agent_dir,'train'))
                ctrl.save(os.path.join(ctrl.agent_dir,'train'))
            t2 = time.time()
            print("{}/{} Finish model-free sampling: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,t2-t1,np.mean(train_objss[-1])))

            # Model-based sampling
            print(f"{episode}/{args.episodes} Start model-based sampling")
            train_idxs = dG.get_data_idxs(seq,train_ids)
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,return_idx=True)
            x,b,idxs = train_dats[0],train_dats[2],train_dats[-1]
            if margs.use_edge:
                ex = train_dats[-3]
            xs,exs,settings,perfs = ctrl.rollout(x,b,ex)
            xs[...,1] += xs[...,-1]
            if margs.if_flood:
                xs = np.concatenate([xs[...,:-2],xs[...,-1:]],axis=-1)
            idxs = np.repeat(idxs,args.horizon+1)
            dG.update([xs,perfs,settings,exs,idxs])
            t3 = time.time()
            print("{}/{} Finish model-based sampling: {:.2f}s".format(episode,args.episodes,t3-t2))

            # Model-free update
            print(f"{episode}/{args.episodes} Start model-free update")
            for _ in range(args.repeats):
                train_idxs = dG.get_data_idxs(seq,train_ids)
                train_dats = dG.prepare_batch(train_idxs,args.seq_in*2,args.batch_size,args.setting_duration)
                x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
                x,b,y = [ctrl.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]    
                s,a = [x[:,-args.seq_in:,...],b[:,:args.seq_in,...]],a[:,0:1,...]
                s_ = [y[:,:args.seq_in,...],b[:,args.seq_in:,...]]
                if margs.use_edge:
                    ex,ey = [ctrl.normalize(dat,'e') for dat in train_dats[-2:]]
                    s += [ex[:,-args.seq_in:,...]]
                    ae = ctrl.emul.get_edge_action(tf.repeat(a,args.setting_duration,axis=1),True)
                    s_ += [tf.concat([ey[:,:args.seq_in,...],ae],axis=-1)]
                train_loss = ctrl.update_eval(s,a,s_,train=False)
                train_losses.append(train_loss)
            t4 = time.time()
            print("{}/{} Finish model-free update: {:.2f}s Mean loss: {:.2f} {:.2f} {:.2f}".format(episode,args.episodes,t4-t3,*np.mean(train_losses[-args.repeats:],axis=0)))

            # Model-free validation
            print(f"{episode}/{args.episodes} Start model-free validation")
            for _ in range(args.repeats):
                test_idxs = dG.get_data_idxs(seq,test_ids)
                test_dats = dG.prepare_batch(test_idxs,args.seq_in*2,args.batch_size,args.setting_duration)
                x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
                x,b,y = [ctrl.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]    
                s,a = [x[:,-args.seq_in:,...],b[:,:args.seq_in,...]],a[:,0:1,...]
                s_ = [y[:,:args.seq_in,...],b[:,args.seq_in:,...]]
                if margs.use_edge:
                    ex,ey = [ctrl.normalize(dat,'e') for dat in test_dats[-2:]]
                    s += [ex[:,-args.seq_in:,...]]
                    ae = ctrl.emul.get_edge_action(tf.repeat(a,args.setting_duration,axis=1),True)
                    s_ += [tf.concat([ey[:,:args.seq_in,...],ae],axis=-1)]
                eval_loss = ctrl.update_eval(s,a,s_,train=False)
                eval_losses.append(eval_loss)
            t5 = time.time()
            print("{}/{} Finish model-free validation: {:.2f}s Mean loss: {:.2f} {:.2f} {:.2f}".format(episode,args.episodes,t5-t4,*np.mean(eval_losses[-args.repeats:],axis=0)))

            # Evaluate the model in several episodes
            if episode % args.eval_gap == 0:
                print(f"{episode}/{args.episodes} Start model-free interaction")
                if args.processes > 1:
                    ctrl.save()
                    pool = mp.Pool(args.processes)
                    res = [pool.apply_async(func=interact_steps,args=(env,args,event,runoffs[idx],None,False,))
                           for idx,event in zip(test_ids,test_events)]
                    pool.close()
                    pool.join()
                    res = [r.get() for r in res]
                else:
                    res = [interact_steps(env,args,event,runoffs[idx],ctrl,train=False)
                        for idx,event in zip(test_ids,test_events)]
                # data = [np.concatenate([r[i] for r in res],axis=0) for i in range(4)]
                test_objss.append(np.array([np.sum(r[-1]) for r in res]))
                t6 = time.time()
                print("{}/{} Finish model-free interaction: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,t6-t5,np.mean(test_objss[-1])))
                if np.mean(test_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in test_objss[:-1]]):
                    if not os.path.exists(os.path.join(args.agent_dir,'test')):
                        os.mkdir(os.path.join(args.agent_dir,'test'))
                    ctrl.save(os.path.join(ctrl.agent_dir,'test'))
                secs.append([t2-t1,t3-t2,t4-t3,t5-t4,t6-t5])
            else:
                secs.append([t2-t1,t3-t2,t4-t3,t5-t4,0])

            if episode % args.save_gap == 0:
                ctrl.save()
            ctrl.target_update_func(episode)
        ctrl.save()
        dG.save(args.agent_dir)
        np.save(os.path.join(ctrl.agent_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(ctrl.agent_dir,'eval_loss.npy'),np.array(eval_losses))
        np.save(os.path.join(ctrl.agent_dir,'train_objs.npy'),np.array(train_objss))
        np.save(os.path.join(ctrl.agent_dir,'test_objs.npy'),np.array(test_objss))
        np.save(os.path.join(ctrl.agent_dir,'time.npy'),np.array(secs))