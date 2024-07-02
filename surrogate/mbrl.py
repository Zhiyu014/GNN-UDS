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
from agent import Actor,Agent
from mpc import get_runoff
from envs import get_env
from utils.utilities import get_inp_files
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
    parser.add_argument('--limit',type=int,default=23,help='maximum capacity 2^n of the buffer')
    parser.add_argument('--act_lr',type=float,default=1e-4,help='actor learning rate')
    parser.add_argument('--cri_lr',type=float,default=1e-3,help='critic learning rate')
    parser.add_argument('--update_interval',type=float,default=0.005,help='target update interval')
    parser.add_argument('--sample_gap',type=int,default=0,help='sample data with swmm per sample gap')
    parser.add_argument('--start_gap',type=int,default=100,help='start updating agent after start gap')
    parser.add_argument('--save_gap',type=int,default=100,help='save the agent per gap')
    parser.add_argument('--agent_dir',type=str,default='./agent/',help='path of the agent')
    parser.add_argument('--load_agent',action="store_true",help='if load agents')

    # testing scenario args: rain and result dir not useful here
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
            setting = env.controller('safe',state[-1],setting)
        done = env.step([float(sett) for sett in setting.tolist()])
        state = env.state_full(seq=args.seq_in)
        if args.if_flood:
            flood = env.flood(seq=args.seq_in)
        edge_state = env.state_full(args.seq_in,'links')
        states.append(state[-1])
        perfs.append(env.flood())
        objects.append(env.objective())
        edge_states.append(edge_state[-1])
        settings.append(setting)        
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
        'batch_size':64,
        'episodes':1000,
        'seq_in':5,'horizon':60,
        'setting_duration':5,
        'use_edge':True,
        'conv':'GAT',
        'recurrent':'Conv1D',
        'eval_gap':10,'start_gap':0,
        'agent_dir': './agent/astlingen/5s_10k_conti_mbrl',
        'load_agent':False,
        'processes':2,
        # 'test':False,
        # 'rain_dir':'./envs/config/ast_test1_events.csv',
        # 'result_dir':'./results/astlingen/60s_10k_conti_policy2007',
        }
    for k,v in train_de.items():
        setattr(args,k,v)
        config[k] = v

    env = get_env(args.env)(initialize=False)
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
        # events = get_inp_files(env.config['swmm_input'],rain_arg)
        events = ['./envs/network/astlingen/astlingen_03_05_2006_01.inp', './envs/network/astlingen/astlingen_07_30_2004_21.inp', './envs/network/astlingen/astlingen_01_13_2002_12.inp', './envs/network/astlingen/astlingen_08_12_2003_08.inp', './envs/network/astlingen/astlingen_10_05_2005_16.inp', './envs/network/astlingen/astlingen_04_12_2003_18.inp', './envs/network/astlingen/astlingen_05_27_2004_06.inp', './envs/network/astlingen/astlingen_12_02_2004_23.inp', './envs/network/astlingen/astlingen_12_28_2006_08.inp', './envs/network/astlingen/astlingen_12_13_2006_23.inp', './envs/network/astlingen/astlingen_03_11_2002_09.inp', './envs/network/astlingen/astlingen_08_11_2003_19.inp', './envs/network/astlingen/astlingen_09_16_2006_05.inp', './envs/network/astlingen/astlingen_03_23_2006_08.inp', './envs/network/astlingen/astlingen_06_13_2000_20.inp', './envs/network/astlingen/astlingen_11_15_2003_17.inp', './envs/network/astlingen/astlingen_02_07_2001_07.inp', './envs/network/astlingen/astlingen_04_17_2005_12.inp', './envs/network/astlingen/astlingen_06_29_2002_07.inp', './envs/network/astlingen/astlingen_05_06_2004_19.inp', './envs/network/astlingen/astlingen_08_21_2001_08.inp', './envs/network/astlingen/astlingen_04_30_2001_09.inp', './envs/network/astlingen/astlingen_03_13_2001_16.inp', './envs/network/astlingen/astlingen_07_27_2000_14.inp', './envs/network/astlingen/astlingen_04_27_2005_00.inp', './envs/network/astlingen/astlingen_08_01_2002_11.inp', './envs/network/astlingen/astlingen_11_28_2006_01.inp', './envs/network/astlingen/astlingen_10_29_2004_11.inp', './envs/network/astlingen/astlingen_07_25_2000_01.inp', './envs/network/astlingen/astlingen_09_11_2006_11.inp', './envs/network/astlingen/astlingen_06_01_2005_10.inp', './envs/network/astlingen/astlingen_02_10_2004_00.inp', './envs/network/astlingen/astlingen_03_07_2003_20.inp', './envs/network/astlingen/astlingen_10_25_2000_13.inp', './envs/network/astlingen/astlingen_12_23_2000_19.inp', './envs/network/astlingen/astlingen_08_08_2005_22.inp', './envs/network/astlingen/astlingen_12_15_2006_17.inp', './envs/network/astlingen/astlingen_04_17_2000_07.inp', './envs/network/astlingen/astlingen_11_12_2005_09.inp', './envs/network/astlingen/astlingen_03_07_2006_18.inp', './envs/network/astlingen/astlingen_10_13_2003_15.inp', './envs/network/astlingen/astlingen_09_26_2002_16.inp', './envs/network/astlingen/astlingen_10_28_2000_08.inp', './envs/network/astlingen/astlingen_10_23_2004_17.inp', './envs/network/astlingen/astlingen_06_11_2006_01.inp', './envs/network/astlingen/astlingen_12_16_2004_17.inp', './envs/network/astlingen/astlingen_03_27_2004_11.inp', './envs/network/astlingen/astlingen_01_04_2004_17.inp', './envs/network/astlingen/astlingen_11_17_2001_18.inp', './envs/network/astlingen/astlingen_04_17_2000_22.inp', './envs/network/astlingen/astlingen_08_22_2006_02.inp']
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

        # Real data for sampling base points
        dG = DataGenerator(env,margs.data_dir,args)
        dG.load(margs.data_dir)
        # Virtual data buffer for model-based rollout trajs
        dGv = DataGenerator(env,args=args)
        ctrl = Agent(args.action_shape,args.observ_space,args,act_only=False,margs=margs)
        ctrl.set_norm(*dG.get_norm())
        n_events = int(max(dG.event_id))+1
        train_ids = np.load(os.path.join(margs.model_dir,'train_id.npy'))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        train_events,test_events = [events[ix] for ix in train_ids],[events[ix] for ix in test_ids]

        train_losses,eval_losses,train_objss,test_objss,secs = [],[],[],[],[]
        for episode in range(args.episodes):
            sec,t = [],time.time()
            # Model-free sampling
            if args.sample_gap > 0 and episode % args.sample_gap == 0:
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
                dGv.update(data)
                train_objss.append(np.array([np.sum(r[-1]) for r in res]))
                if np.mean(train_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in train_objss[:-1]]):
                    if not os.path.exists(os.path.join(args.agent_dir,'train')):
                        os.mkdir(os.path.join(args.agent_dir,'train'))
                    ctrl.save(os.path.join(ctrl.agent_dir,'train'))
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-free sampling: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(train_objss[-1])))
                
            # Model-based sampling
            print(f"{episode}/{args.episodes} Start model-based sampling")
            seq = max(args.seq_in,args.horizon) if args.recurrent else 0
            train_idxs = dG.get_data_idxs(train_ids,seq)
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,args.setting_duration,return_idx=True)
            i,rand = 0,np.arange(train_dats[-1].shape[0])
            # Split the trajs within the same event
            while any(np.diff(train_dats[-1])==0):
                np.random.shuffle(rand)
                train_dats = [dat[rand] for dat in train_dats]
                i += 1
                if i > 100:
                    break
            trajs = ctrl.rollout(train_dats[:-1])
            xs,exs,settings,perfs = [traj.numpy().reshape((-1,)+tuple(traj.shape[2:])) for traj in trajs]
            xs[...,1] += xs[...,-1]
            if margs.if_flood:
                xs = np.concatenate([xs[...,:-2],xs[...,-1:]],axis=-1)
            idxs = np.repeat(train_dats[-1],args.horizon+args.seq_in)
            dGv.update([xs,perfs,settings,exs,idxs])
            sec.append(time.time()-t)
            t = time.time()
            print("{}/{} Finish model-based sampling: {:.2f}s".format(episode,args.episodes,sec[-1]))

            if episode > args.start_gap:
                # Model-free update
                print(f"{episode}/{args.episodes} Start model-free update")
                for _ in range(args.repeats):
                    train_idxs = dGv.get_data_idxs(train_ids,args.seq_in,args.seq_in*2)
                    train_dats = dGv.prepare_batch(train_idxs,args.seq_in*2,args.batch_size,args.setting_duration,trim=False)
                    x,settings,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
                    x_norm,b_norm,y_norm = [ctrl.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]    
                    s,settings = [x_norm[:,-args.seq_in:,...],b_norm[:,:args.seq_in,...]],settings[:,0:1,...]
                    s_ = [tf.concat([y_norm[:,:args.seq_in,...,:-1],b_norm[:,:args.seq_in,...]],axis=-1),b_norm[:,args.seq_in:,...]]
                    if margs.use_edge:
                        ex,ey = train_dats[-2:]
                        ex_norm,ey_norm = [ctrl.normalize(dat,'e') for dat in [ex,ey]]
                        s += [ex_norm[:,-args.seq_in:,...]]
                        ae = ctrl.emul.get_edge_action(tf.repeat(settings,args.setting_duration,axis=1),True)
                        s_ += [tf.concat([ey_norm[:,:args.seq_in,...],ae],axis=-1)]
                    # Get reward from env as -obj_pred
                    states = (x[:,-args.seq_in:,...],ex[:,-args.seq_in:,...] if margs.use_edge else None)
                    preds = (y[:,:args.seq_in,...],ey[:,:args.seq_in,...] if margs.use_edge else None)
                    r = - env.objective_pred_tf(preds,states,tf.repeat(settings,args.setting_duration,axis=1))/100
                    # r = tf.clip_by_value(r, -10, 10)
                    a = ctrl.actor.convert_setting_to_action(settings)
                    train_loss = ctrl.update_eval(s,a,r,s_,train=True)
                    train_losses.append([los.numpy() for los in train_loss])
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-free update: {:.2f}s Mean loss: {:.2f} {:.2f} {:.2f}".format(episode,args.episodes,sec[-1],*np.mean(train_losses[-args.repeats:],axis=0)))

                # Model-free validation
                # print(f"{episode}/{args.episodes} Start model-free validation")
                # for _ in range(args.repeats):
                #     test_idxs = dGv.get_data_idxs(test_ids,args.seq_in,args.seq_in*2)
                #     test_dats = dGv.prepare_batch(test_idxs,args.seq_in*2,args.batch_size,args.setting_duration,trim=False)
                #     x,settings,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
                #     x_norm,b_norm,y_norm = [ctrl.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]    
                #     s,settings = [x_norm[:,-args.seq_in:,...],b_norm[:,:args.seq_in,...]],settings[:,0:1,...]
                #     s_ = [tf.concat([y_norm[:,:args.seq_in,...,:-1],b_norm[:,:args.seq_in,...]],axis=-1),b_norm[:,args.seq_in:,...]]
                #     if margs.use_edge:
                #         ex,ey = test_dats[-2:]
                #         ex_norm,ey_norm = [ctrl.normalize(dat,'e') for dat in [ex,ey]]
                #         s += [ex_norm[:,-args.seq_in:,...]]
                #         ae = ctrl.emul.get_edge_action(tf.repeat(settings,args.setting_duration,axis=1),True)
                #         s_ += [tf.concat([ey_norm[:,:args.seq_in,...],ae],axis=-1)]
                #     # Get reward from env as -obj_pred
                #     states = (x[:,-args.seq_in:,...],ex[:,-args.seq_in:,...] if margs.use_edge else None)
                #     preds = (y[:,:args.seq_in,...],ey[:,:args.seq_in,...] if margs.use_edge else None)
                #     r = - env.objective_pred_tf(preds,states,tf.repeat(settings,args.setting_duration,axis=1),norm=True)
                #     a = ctrl.actor.convert_setting_to_action(settings)
                #     eval_loss = ctrl.update_eval(s,a,r,s_,train=False)
                #     eval_losses.append(eval_loss)
                # sec.append(time.time()-t)
                # t = time.time()
                # print("{}/{} Finish model-free validation: {:.2f}s Mean loss: {:.2f} {:.2f} {:.2f}".format(episode,args.episodes,sec[-1],*np.mean(eval_losses[-args.repeats:],axis=0)))

            # Evaluate the model in several episodes
            if episode > args.start_gap and episode % args.eval_gap == 0:
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
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-free interaction: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(test_objss[-1])))
                if np.mean(test_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in test_objss[:-1]]):
                    if not os.path.exists(os.path.join(args.agent_dir,'test')):
                        os.mkdir(os.path.join(args.agent_dir,'test'))
                    ctrl.save(os.path.join(ctrl.agent_dir,'test'))
                secs.append(sec)

            if episode % args.save_gap == 0:
                ctrl.save()
            ctrl.target_update_func(episode)
        ctrl.save()
        dGv.save(args.agent_dir)
        np.save(os.path.join(ctrl.agent_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(ctrl.agent_dir,'eval_loss.npy'),np.array(eval_losses))
        np.save(os.path.join(ctrl.agent_dir,'train_objs.npy'),np.array(train_objss))
        np.save(os.path.join(ctrl.agent_dir,'test_objs.npy'),np.array(test_objss))
        np.save(os.path.join(ctrl.agent_dir,'time.npy'),np.array(secs))