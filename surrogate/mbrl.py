import os,yaml,shutil
import multiprocessing as mp
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpu, True)
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
from emulator import Emulator
from dataloader import DataGenerator
from agent import get_agent
from mpc import get_runoff
from envs import get_env
from utils.utilities import get_inp_files
import pandas as pd,matplotlib.pyplot as plt
import argparse,time

HERE = os.path.dirname(__file__)

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')

    parser.add_argument('--env',type=str,default='astlingen',help='set drainage scenarios')
    parser.add_argument('--directed',action='store_true',help='if use directed graph')
    parser.add_argument('--length',type=float,default=0,help='adjacency range')
    parser.add_argument('--order',type=int,default=1,help='adjacency order')
    parser.add_argument('--graph_base',type=int,default=0,help='if use node(1) or edge(2) based graph structure')

    # control args
    parser.add_argument('--control_interval',type=int,default=5,help='number of the rainfall events')
    parser.add_argument('--setting_duration',type=int,default=5,help='setting duration')
    parser.add_argument('--act',type=str,default='rand',help='what control actions')
    parser.add_argument('--mac',action="store_true",help='if use multi-agent action space')
    parser.add_argument('--dec',action="store_true",help='if use dec-pomdp observation space')

    # rl args
    parser.add_argument('--train',action="store_true",help='if train')
    parser.add_argument('--episodes',type=int,default=1000,help='training episodes')
    parser.add_argument('--batch_size',type=int,default=128,help='training batch size')
    parser.add_argument('--limit',type=int,default=23,help='maximum capacity 2^n of the buffer')
    parser.add_argument('--tune_gap',type=int,default=0,help='finetune the model per sample gap')
    parser.add_argument('--sample_gap',type=int,default=0,help='sample data with swmm per sample gap')
    parser.add_argument('--start_gap',type=int,default=100,help='start updating agent after start gap')
    parser.add_argument('--eval_gap',type=int,default=10,help='evaluate the agent per eval_gap')
    parser.add_argument('--save_gap',type=int,default=1000,help='save the agent per gap')

    # rollout args
    parser.add_argument('--data_dir',type=str,default='./envs/data/',help='path of the training data')
    parser.add_argument('--model_based',action="store_true",help='if use model-based sampling')
    parser.add_argument('--horizon',type=int,default=60,help='prediction & control horizon')
    parser.add_argument('--model_dir',type=str,default='./model/',help='path of the surrogate model')
    parser.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')
    parser.add_argument('--epochs',type=int,default=100,help='model update times per episode')

    # agent args
    parser.add_argument('--agent',type=str,default='SAC',help='agent name')
    parser.add_argument('--conv',type=str,default='GATconv',help='convolution type')
    parser.add_argument('--use_pred',action="store_true",help='if use prediction runoff')
    parser.add_argument('--net_dim',type=int,default=128,help='number of decision-making channels')
    parser.add_argument('--n_layer',type=int,default=3,help='number of decision-making layers')
    parser.add_argument('--conv_dim',type=int,default=128,help='number of graphconv channels')
    parser.add_argument('--n_sp_layer',type=int,default=3,help='number of graphconv layers')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')
    parser.add_argument('--agent_dir',type=str,default='./agent/',help='path of the agent')
    parser.add_argument('--load_agent',action="store_true",help='if load agents')
   
    # agent update args
    parser.add_argument('--repeats',type=int,default=5,help='agent update times per episode')
    parser.add_argument('--norm',action="store_true",help='if use reward normalization')
    parser.add_argument('--scale',type=float,default=1.0,help='reward scaling factor')
    parser.add_argument('--gamma',type=float,default=0.98,help='discount factor')
    parser.add_argument('--en_disc',type=float,default=0.5,help='target entropy discount factor')
    parser.add_argument('--act_lr',type=float,default=1e-4,help='actor learning rate')
    parser.add_argument('--cri_lr',type=float,default=1e-3,help='critic learning rate')
    parser.add_argument('--update_interval',type=float,default=0.005,help='target update interval')
    parser.add_argument('--epsilon_decay',type=float,default=0.9996,help='epsilon decay rate in QMIX')
    parser.add_argument('--noise_std',type=float,default=0.05,help='std of the exploration noise')
    parser.add_argument('--value_tau',type=float,default=0.0,help='value running average tau')

    # testing scenario args: rain and result dir not useful here
    parser.add_argument('--test',action="store_true",help='if test')
    parser.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
    parser.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
    parser.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')
    parser.add_argument('--swmm_step',type=int,default=1,help='routing step for swmm inp files')
    parser.add_argument('--processes',type=int,default=1,help='parallel simulation')
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

def interact_steps(args,event,runoff,ctrl=None,train=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # with tf.device('/cpu:0'):
    if ctrl is None:
        # tf.keras.backend.clear_session()    # to clear backend occupied models
        args.load_agent = True
        ctrl = get_agent(args.agent)(args.action_shape,args.observ_space,args,act_only=True)
    # trajs = []
    tss,runoff = runoff
    env = get_env(args.env)(swmm_file=event)
    state = env.state_full(seq=args.setting_duration)
    if getattr(args,"if_flood",False):
        flood = env.flood(seq=args.setting_duration)
    states = [state[-1]]
    perfs,objects = [env.flood()],[env.objective()]
    edge_state = env.state_full(args.setting_duration,'links')
    edge_states = [edge_state[-1]]
    setting = env.controller('default')
    settings = [setting]
    rains = [env.rainfall()]
    done,i = False,0
    while not done:
        if i*args.interval % args.control_interval == 0:
            state[...,1] = state[...,1] - state[...,-1]
            if getattr(args,"if_flood",False):
                f = (flood>0).astype(float)
                state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
            t = env.env.methods['simulation_time']()
            b = runoff[int(tss.asof(t)['Index'])][:args.setting_duration]
            x_norm,b_norm,e_norm = [ctrl.normalize(dat,item) if dat is not None else None
                                    for dat,item in zip([state,b,edge_state],'xbe')]
            if ctrl.conv:
                x_norm,e_norm = [tf.stack([tf.reduce_sum(dat[...,i],axis=0) if 'cum' in attr or '_vol' in attr else dat[-1,:,i]
                                        for i,attr in enumerate(args.attrs[items])],axis=-1)
                                          for dat,items in zip([x_norm,e_norm],['nodes','links'])]
                if args.use_pred:
                    b_norm = tf.stack([tf.reduce_sum(b_norm[...,i],axis=0) if i==0 else b_norm[-1,:,i]
                                    for i in range(b_norm.shape[-1])],axis=-1)
                observ = [x_norm,b_norm,e_norm] if args.use_pred else [x_norm,e_norm]
            else:
                r_norm = ctrl.normalize(env.rainfall(seq=args.setting_duration),'r')
                observ = tf.stack([x_norm[...,args.elements['nodes'].index(idx),args.attrs['nodes'].index(attr)] if attr in args.attrs['nodes'] 
                                    else e_norm[...,args.elements['links'].index(idx),args.attrs['links'].index(attr)]
                                        for idx,attr in args.states if attr in args.attrs['nodes']+args.attrs['links']],axis=-1)
                observ = tf.concat([r_norm,observ],axis=-1)
                observ = tf.stack([tf.reduce_sum(observ[...,i],axis=-1) if 'cum' in attr or '_vol' in attr else observ[...,-1,i]
                                   for i,(_,attr) in enumerate(args.states)],axis=-1)
            setting = ctrl.control(observ,train).numpy()
            setting = env.controller('safe',state[-1],setting)
        done = env.step([float(sett) for sett in setting.tolist()])
        state = env.state_full(seq=args.setting_duration)
        if getattr(args,"if_flood",False):
            flood = env.flood(seq=args.setting_duration)
        edge_state = env.state_full(args.setting_duration,'links')
        states.append(state[-1])
        perfs.append(env.flood())
        objects.append(env.objective())
        edge_states.append(edge_state[-1])
        settings.append(setting)        
        rains.append(env.rainfall())
        i += 1
    env.initialize_logger()
    return [np.array(dat) for dat in [states,perfs,settings,rains,edge_states,rains,objects]]

if __name__ == '__main__':
    mp.set_start_method('spawn')
    args,config = parser(os.path.join(HERE,'utils','policy.yaml'))

    train_de = {
        # 'agent':'SAC',
        # 'train':True,
        # 'env':'astlingen',
        # 'act':'conti',
        # 'mac':False,
        # 'dec':False,
        # 'model_based':True,'sample_gap':0,'data_dir':'./envs/data/astlingen/1s_edge_conti128_rain50/',
        # 'model_dir':'./model/astlingen/5s_20k_conti_500ledgef_res_norm_flood_gat/',
        # 'batch_size':64,
        # 'episodes':10000,
        # 'limit':20,
        # 'tune_gap':0,
        # 'horizon':60,
        # 'norm':True,'scale':1.0,
        # 'conv':'False','use_pred':True,
        # 'eval_gap':10,'start_gap':0,
        # 'agent_dir': './agent/astlingen/test',
        # 'load_agent':False,
        # 'processes':4,
        # 'rain_dir':'ast_train50_events.csv','swmm_step':10,
        # 'test':False,
        }
    for k,v in train_de.items():
        setattr(args,k,v)
        config[k] = v

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
            setattr(args,'seq_in',margs.seq_in)
            assert margs.seq_in == args.setting_duration//args.interval
            setattr(args,'if_flood',bool(margs.if_flood))
            setattr(args,'data_dir',margs.data_dir)
            config['data_dir'] = args.data_dir
            if args.if_flood:
                args.attrs['nodes'] = args.attrs['nodes'][:-1] + ['if_flood'] + args.attrs['nodes'][-1:]
            env_args = env.get_args(margs.directed,margs.length,margs.order,margs.graph_base)
            for k,v in env_args.items():
                setattr(margs,k,v)
            emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            emul.load(margs.model_dir)

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
        dG = DataGenerator(env.config,args.data_dir,args)
        dG.load(args.data_dir)
        # Virtual data buffer for model-based rollout trajs
        dGv = DataGenerator(env.config,args=args)
        ctrl = get_agent(args.agent)(args.action_shape,args.observ_space,args,act_only=False)
        ctrl.set_norm(*dG.get_norm())
        n_events = int(max(dG.event_id))+1
        train_ids = np.load(os.path.join(margs.model_dir,'train_id.npy') if args.model_based else os.path.join(args.data_dir,'train_id.npy'))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        # train_ids = np.arange(n_events)
        train_events,test_events = [events[ix] for ix in train_ids],[events[ix] for ix in test_ids]

        @tf.function
        def rollout(data,episode):
            # args = argparse.Namespace(**args)
            if hasattr(ctrl,"epsilon"):
                ctrl._epsilon_decay(episode)
            n_step,r_step = args.horizon//args.setting_duration,args.setting_duration//args.interval
            x,a,b,y = data[:4]
            r = tf.concat(data[4:6],axis=1)
            ex = data[-2]
            xs,exs,settings,perfs = [x],[ex],[a[:,:args.seq_in,:]],[y[:,:args.seq_in,:,-1:]]
            for i in range(n_step):
                bi = b[:,i*r_step:(i+1)*r_step,:]
                x_norm,b_norm,e_norm = [ctrl.normalize(dat,item) if dat is not None else None
                                        for dat,item in zip([x,bi,ex],'xbe')]
                if ctrl.conv:
                    x_norm,e_norm = [tf.stack([tf.reduce_sum(dat[...,i],axis=1) if 'cum' in attr or '_vol' in attr else dat[:,-1,:,i]
                                            for i,attr in enumerate(args.attrs[items])],axis=-1)
                                                for dat,items in zip([x_norm,e_norm],['nodes','links'])]
                    if args.use_pred:
                        b_norm = tf.stack([tf.reduce_sum(b_norm[...,i],axis=1) if i==0 else b_norm[:,-1,:,i]
                                        for i in range(b_norm.shape[-1])],axis=-1)
                    s_norm = [x_norm,b_norm,e_norm] if args.use_pred else [x_norm,e_norm]
                else:
                    r_norm = ctrl.normalize(r[:,i*r_step:(i+1)*r_step,:],'r')
                    s_norm = tf.stack([x_norm[...,args.elements['nodes'].index(idx),args.attrs['nodes'].index(attr)] if attr in args.attrs['nodes']
                                    else e_norm[...,args.elements['links'].index(idx),args.attrs['links'].index(attr)]
                                    for idx,attr in args.states if attr in args.attrs['nodes']+args.attrs['links']],axis=-1)
                    s_norm = tf.concat([r_norm,s_norm],axis=-1)
                    s_norm = tf.stack([tf.reduce_sum(s_norm[...,i],axis=-1) if 'cum' in attr or '_vol' in attr else s_norm[...,-1,i]
                                        for i,(_,attr) in enumerate(args.states)],axis=-1)
                setting = ctrl.control(s_norm,train=True,batch=True)
                setting = tf.repeat(setting[:,tf.newaxis,:],r_step,axis=1)
                settings.append(setting)
                preds = emul.predict_tf(x,bi,setting,ex)
                if emul.if_flood:
                    x = tf.concat([preds[0][...,:-2],tf.cast(preds[0][...,-2:-1]>0.5,tf.float32),bi],axis=-1)
                else:
                    x = tf.concat([preds[0][...,:-1],bi],axis=-1)
                ae = emul.get_edge_action(setting,True)
                ex = tf.concat([preds[1],ae],axis=-1)
                xs.append(x)
                exs.append(ex)
                perfs.append(preds[0][...,-1:])
            return [tf.concat(tf.concat(dat,axis=1),axis=0) for dat in [xs,exs,settings,perfs]]+[tf.concat(r,axis=0)]

        model_losses,test_losses,train_losses,train_objss,test_objss,secs = [],[],[],[],[],[]
        log_dir = "logs/agent/"
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir,exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        for episode in range(args.episodes):
            setattr(args,"episode",episode)
            sec,t = [],time.time()

            # TODO: Model fine-tuning, need real model and current policy to collect data
            if args.model_based and args.tune_gap > 0 and episode > args.start_gap and episode % args.tune_gap == 0:
                print(f"{episode}/{args.episodes} Start model finetuning")
                if args.sample_gap == 0:
                    print(f"    Model finetuning: model-free sampling first")
                    args.load_agent = True
                    ctrl.save()
                    pool = mp.Pool(args.processes)
                    res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],None,False,))
                        for idx,event in zip(train_ids,train_events)]
                    pool.close()
                    pool.join()
                    res = [r.get() for r in res]
                    trajs = [np.concatenate([r[i] for r in res],axis=0) for i in range(5)]
                    trajs.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(train_ids,res)],axis=-1))
                    dG.update(trajs)
                seq = max(margs.seq_in,margs.seq_out)
                train_idxs = dG.get_data_idxs(train_ids,seq)
                test_idxs = dG.get_data_idxs(test_ids,seq)
                emul.set_norm(*dG.get_norm())
                k = (1 + min(dG.cur_capa / dG.limit,1))
                epochs,batch_size = int(k * args.epochs), int(k * args.batch_size)
                for epoch in range(epochs):
                    t1 = time.time()
                    train_dats = dG.prepare_batch(train_idxs,seq,batch_size,interval=args.setting_duration,trim=False)
                    x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
                    ex,ey = [dat for dat in train_dats[-2:]]
                    x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
                    model_loss = emul.fit_eval(x,a,b,y,ex,ey)
                    model_losses.append(model_loss.numpy())

                    test_dats = dG.prepare_batch(test_idxs,seq,batch_size,interval=args.setting_duration,trim=False)
                    x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
                    ex,ey = [dat for dat in test_dats[-2:]]
                    x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
                    test_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=False)
                    test_losses.append(np.sum([los.numpy() for los in test_loss]))
                    print("    Episode {} fine-tuning epoch {}/{}: {:.2f}s Train loss: {:.4f} Test loss: {:.4f}"\
                          .format(episode,epoch,epochs,time.time()-t1,model_losses[-1],test_losses[-1]))
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model fine-tuning: {:.2f}s Mean loss: Train {:.4f} Test {:.4f}"\
                      .format(episode,args.episodes,sec[-1],np.mean(model_losses[-epochs:]),np.mean(test_losses[-epochs:])))
                with tf.summary.create_file_writer(log_dir).as_default():
                    tf.summary.scalar('Model fine-tuning train loss', np.mean(model_losses[-epochs:]), step=episode)
                    tf.summary.scalar('Model fine-tuning test loss', np.mean(test_losses[-epochs:]), step=episode)

            # Model-free sampling
            if args.sample_gap > 0 and episode % args.sample_gap == 0:
                print(f"{episode}/{args.episodes} Start model-free sampling")
                args.load_agent = True
                ctrl.save()
                pool = mp.Pool(args.processes)
                res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],None,True,))
                    for idx,event in zip(train_ids,train_events)]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
                trajs = [np.concatenate([r[i] for r in res],axis=0) for i in range(5)]
                trajs.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(train_ids,res)],axis=-1))
                dGv.update(trajs)
                if args.model_based:
                    dG.update(trajs)
                train_objss.append(np.array([np.sum(r[-1]) for r in res]))
                if np.mean(train_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in train_objss[:-1]]):
                    if not os.path.exists(os.path.join(args.agent_dir,'train')):
                        os.mkdir(os.path.join(args.agent_dir,'train'))
                    ctrl.save(os.path.join(ctrl.agent_dir,'train'))
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-free sampling: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(train_objss[-1])))
                with tf.summary.create_file_writer(log_dir).as_default():
                    tf.summary.scalar('Model-free training objectives', np.mean(train_objss[-1]), step=episode)

            # Model-based sampling
            if args.model_based or args.sample_gap == 0:
                print(f"{episode}/{args.episodes} Start model-based sampling")
                seq = max(args.seq_in,args.horizon)
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
                train_dats = [tf.convert_to_tensor(dat) for dat in train_dats]
                trajs_v = rollout(train_dats[:-1],tf.constant(episode))
                # TODO: replace dataloader with memory
                xs,exs,settings,perfs,rains = [traj.numpy().reshape((-1,)+tuple(traj.shape[2:])) for traj in trajs_v]
                xs[...,1] += xs[...,-1]
                if emul.if_flood:
                    xs = np.concatenate([xs[...,:-2],xs[...,-1:]],axis=-1)
                idxs = np.repeat(train_dats[-1],args.horizon+args.seq_in)
                trajs_v = [xs,perfs,settings,rains,exs,idxs]
                dGv.update(trajs_v) # data num: batch * (horizon + seq_in)
                # Debugging reward scale
                # xs,exs,perfs = [dat.reshape((-1,65,)+dat.shape[-2:]) for dat in [xs,exs,perfs]]
                # dats = [dG.state_split_batch((xs[:,i*args.setting_duration:(i+1)*args.setting_duration,...],xs[:,(i+1)*args.setting_duration:(i+2)*args.setting_duration,...]),
                #                              (perfs[:,i*args.setting_duration:(i+1)*args.setting_duration,...],perfs[:,(i+1)*args.setting_duration:(i+2)*args.setting_duration,...]),
                #                              trim=False) for i in range(args.horizon//args.setting_duration)]
                # edge_dats = [dG.edge_state_split_batch((exs[:,i*args.setting_duration:(i+1)*args.setting_duration,...],exs[:,(i+1)*args.setting_duration:(i+2)*args.setting_duration,...]),
                #                              trim=False) for i in range(args.horizon//args.setting_duration)]
                # predss = [(d[-1].astype(np.float32),ed[-1].astype(np.float32)) for d,ed in zip(dats,edge_dats)]
                # statess = [(d[0].astype(np.float32),ed[0].astype(np.float32)) for d,ed in zip(dats,edge_dats)]
                # objs = [env.objective_pred_tf(preds,states,[]) for preds,states,in zip(predss,statess)]
                # rs = np.sum([- env.norm_obj(obj,states,g=True) * args.scale if args.norm else - obj * args.scale 
                #              for obj,states in zip(objs,statess)],axis=0)
                # print('Episode reward:  Max {:.2f} Min {:.2f} Avg {:.2f} Sum {:.2f}'.format(rs.max(),rs.min(),rs.mean(),rs.sum()))
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-based sampling: {:.2f}s".format(episode,args.episodes,sec[-1]))

            # Model-free update
            if episode > args.start_gap:
                print(f"{episode}/{args.episodes} Start model-free update")
                # TODO: normalization running mean and mean std
                ctrl.set_norm(*dGv.get_norm())
                k = (1 + dGv.cur_capa / dGv.limit)
                repeats,batch_size = int(k * args.repeats), int(k * args.batch_size)
                for _ in range(repeats):
                    train_idxs = dGv.get_data_idxs(train_ids,args.setting_duration,args.setting_duration*2)
                    train_dats = dGv.prepare_batch(train_idxs,args.setting_duration*2,batch_size,args.setting_duration,trim=False)
                    x,settings,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
                    x_norm,b_norm,y_norm = [ctrl.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
                    b0,b1 = b_norm[:,:args.setting_duration,...],b_norm[:,args.setting_duration:,...]
                    x0,x1 = x_norm[:,-args.setting_duration:,...],tf.concat([y_norm[:,:args.setting_duration,:,:-1],b0],axis=-1)
                    settings = tf.repeat(settings[:,0:1,:],args.setting_duration,axis=1)
                    ex,ey = train_dats[-2:]
                    ex_norm,ey_norm = [ctrl.normalize(dat,'e') for dat in [ex,ey]]
                    ex0,ex1 = ex_norm[:,-args.setting_duration:,...],ey_norm[:,:args.setting_duration,...]
                    # Get edge action and concat into ex1
                    act_edges = [i for act_edge in args.act_edges for i in np.where((args.edges==act_edge).all(1))[0]]
                    act_edges = sorted(list(set(act_edges)),key=act_edges.index)
                    ae = np.zeros(args.edges.shape[0])
                    ae[act_edges] = range(1,settings.shape[-1]+1)
                    ae = tf.expand_dims(tf.gather(tf.concat([tf.ones_like(settings[...,:1]),settings],axis=-1),tf.cast(ae,tf.int32),axis=-1),axis=-1) 
                    ex1 = tf.concat([ex1,ae],axis=-1)
                    # Reduce temporal dimension and extract observs 
                    if ctrl.conv:
                        x0,x1 = [tf.stack([tf.reduce_sum(xi[...,i],axis=1) if 'cum' in attr or '_vol' in attr else xi[:,-1,:,i]
                                            for i,attr in enumerate(args.attrs['nodes'])],axis=-1) for xi in [x0,x1]]
                        s,s_ = [x0],[x1]
                        if args.use_pred:
                            b0,b1 = [tf.stack([tf.reduce_sum(bi[...,i],axis=1) if i==0 else bi[:,-1,:,i]
                                               for i in range(bi.shape[-1])],axis=-1) for bi in [b0,b1]]
                            s,s_ = s+[b0],s_+[b1]
                        ex0,ex1 = [tf.stack([tf.reduce_sum(ei[...,i],axis=1) if 'cum' in attr or '_vol' in attr else ei[:,-1,:,i]
                                                for i,attr in enumerate(args.attrs['links'])],axis=-1) for ei in [ex0,ex1]]
                        s,s_ = s+[ex0],s_+[ex1]
                    else:
                        r0,r1 = ctrl.normalize(train_dats[4],'r')[:,-args.setting_duration:,...],ctrl.normalize(train_dats[5],'r')[:,:args.setting_duration,...]
                        s,s_ = [tf.stack([xi[...,args.elements['nodes'].index(idx),args.attrs['nodes'].index(attr)] if attr in args.attrs['nodes']
                                           else ei[...,args.elements['links'].index(idx),args.attrs['links'].index(attr)]
                                           for idx,attr in args.states if attr in args.attrs['nodes']+args.attrs['links']],axis=-1)
                                             for xi,ei in zip([x0,x1],[ex0,ex1])]
                        s,s_ = [tf.concat([ri,si],axis=-1) for ri,si in zip([r0,r1],[s,s_])]
                        s,s_ = [tf.stack([tf.reduce_sum(si[...,i],axis=-1) if 'cum' in attr or '_vol' in attr else si[...,-1,i]
                                          for i,(_,attr) in enumerate(args.states)],axis=-1) for si in [s,s_]]
                    # Get reward from env as -obj_pred
                    states = (x[:,-args.setting_duration:,...],ex[:,-args.setting_duration:,...])
                    preds = (y[:,:args.setting_duration,...],ey[:,:args.setting_duration,...])
                    obj = env.objective_pred_tf(preds,states,settings)
                    r = - env.norm_obj(obj,states,g=True) * args.scale if args.norm else - obj * args.scale
                    # r = tf.clip_by_value(r, -10, 10)
                    a = ctrl.convert_setting_to_action(settings[:,0,:])
                    train_loss = ctrl.update_eval(s,a,r,s_,train=True)
                    train_losses.append([los.numpy() for los in train_loss] if isinstance(train_loss,tuple) else train_loss.numpy())
                sec.append(time.time()-t)
                t = time.time()
                loss = np.mean(train_losses[-repeats:],axis=0)
                if isinstance(loss,np.ndarray):
                    print("{}/{} Finish model-free update: {:.2f}s Mean loss:".format(episode,args.episodes,sec[-1])+ (len(loss)*" {:.2f}").format(*loss))
                    with tf.summary.create_file_writer(log_dir).as_default():
                        tf.summary.scalar('Value loss', loss[0], step=episode)
                        if args.agent.upper() == 'SAC':
                            tf.summary.scalar('Alpha', loss[1], step=episode)
                            tf.summary.scalar('Entropy', loss[2], step=episode)
                            tf.summary.scalar('Policy loss', loss[3], step=episode)
                            if hasattr(ctrl,'vnet'):
                                tf.summary.scalar('VNet loss', loss[4], step=episode)
                        else:
                            tf.summary.scalar('Policy loss', loss[1], step=episode)
                else:
                    print("{}/{} Finish model-free update: {:.2f}s Mean loss: {:.2f}".format(episode,args.episodes,sec[-1],loss))
                    with tf.summary.create_file_writer(log_dir).as_default():
                        tf.summary.scalar('Value loss', loss, step=episode)
                        tf.summary.scalar('Epsilon', max(0.1,getattr(args,"epsilon_decay",0.9996)**episode), step=episode)

            # Evaluate the model in several episodes
            if episode > args.start_gap and args.eval_gap > 0 and episode % args.eval_gap == 0:
                print(f"{episode}/{args.episodes} Start model-free interaction")
                ctrl.save()
                pool = mp.Pool(args.processes)
                res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],None,False,))
                        for idx,event in zip(test_ids,test_events)]
                pool.close()
                pool.join()
                res = [r.get() for r in res]
                trajs = [np.concatenate([r[i] for r in res],axis=0) for i in range(5)]
                trajs.append(np.concatenate([[idx]*r[0].shape[0] for idx,r in zip(train_ids,res)],axis=-1))
                dG.update(trajs)
                # data = [np.concatenate([r[i] for r in res],axis=0) for i in range(4)]
                test_objss.append(np.array([np.sum(r[-1]) for r in res]))
                sec.append(time.time()-t)
                t = time.time()
                print("{}/{} Finish model-free interaction: {:.2f}s Mean objs: {:.2f}".format(episode,args.episodes,sec[-1],np.mean(test_objss[-1])))
                with tf.summary.create_file_writer(log_dir).as_default():
                    tf.summary.scalar('Testing objectives', np.mean(test_objss[-1]), step=episode)
                if np.mean(test_objss[-1]) < np.min([1e6]+[np.mean(obj) for obj in test_objss[:-1]]):
                    if not os.path.exists(os.path.join(args.agent_dir,'test')):
                        os.mkdir(os.path.join(args.agent_dir,'test'))
                    ctrl.save(os.path.join(ctrl.agent_dir,'test'))
            secs.append(sec)

            # Save the agent
            if episode >0 and episode % args.save_gap == 0:
                save_dir = os.path.join(args.agent_dir,str(episode))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                ctrl.save(save_dir)
            ctrl.update_func()
        ctrl.save()
        # dGv.save(args.agent_dir)
        if args.model_based and args.tune_gap > 0:
            np.save(os.path.join(ctrl.agent_dir,'model_loss.npy'),np.array(model_losses))
            plt.plot(model_losses,label='model_loss')
            plt.savefig(os.path.join(ctrl.agent_dir,'model_loss.png'),dpi=300)
            plt.clf()
        np.save(os.path.join(ctrl.agent_dir,'train_loss.npy'),np.array(train_losses))
        plt.plot(train_losses,label='train_loss')
        plt.savefig(os.path.join(ctrl.agent_dir,'train_loss.png'),dpi=300)
        plt.clf()
        if args.sample_gap > 0:
            np.save(os.path.join(ctrl.agent_dir,'train_objs.npy'),np.array(train_objss))
            plt.plot(np.mean(train_objss,axis=-1),label='train_objs')
            plt.savefig(os.path.join(ctrl.agent_dir,'train_objs.png'),dpi=300)
            plt.clf()
        np.save(os.path.join(ctrl.agent_dir,'test_objs.npy'),np.array(test_objss))
        plt.plot(np.mean(test_objss,axis=-1),label='test_objs')
        plt.savefig(os.path.join(ctrl.agent_dir,'test_objs.png'),dpi=300)
        plt.clf()
        np.save(os.path.join(ctrl.agent_dir,'time.npy'),np.array(secs))

    if args.test:
        known_hyps = yaml.load(open(os.path.join(args.agent_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k in ['agent_dir','result_dir']:
                continue
            setattr(args,k,v)

        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))

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
        if os.path.exists(os.path.join(args.result_dir,'test_runoff.npy')):
            res = [np.load(os.path.join(args.result_dir,'test_runoff_ts.npy'),allow_pickle=True),
                   np.load(os.path.join(args.result_dir,'test_runoff.npy'),allow_pickle=True)]
            res = [(ts,runoff) for ts,runoff in zip(res[0],res[1])]
        else:
            pool = mp.Pool(args.processes)
            res = [pool.apply_async(func=get_runoff,args=(env,event,False,args.tide,))
                   for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
            np.save(os.path.join(args.result_dir,'test_runoff_ts.npy'),np.array([r[0] for r in res]))
            np.save(os.path.join(args.result_dir,'test_runoff.npy'),np.array([r[1] for r in res]))
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
        print("Finish testing events runoff")

        # Test the agent
        pool = mp.Pool(args.processes)
        res = [pool.apply_async(func=interact_steps,args=(args,event,runoffs[idx],None,False,))
                for idx,event in enumerate(events)]
        pool.close()
        pool.join()
        for event,r in zip(events,res):
            name = os.path.basename(event).strip('.inp')
            states,perfs,settings,rains,edge_states,rains,objects = r.get()
            np.save(os.path.join(args.result_dir,name + '_state.npy'),states)
            np.save(os.path.join(args.result_dir,name + '_perf.npy'),perfs)
            np.save(os.path.join(args.result_dir,name + '_object.npy'),objects)
            np.save(os.path.join(args.result_dir,name + '_settings.npy'),settings)
            np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),edge_states)