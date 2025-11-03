import torch as th
from torch.utils.tensorboard import SummaryWriter
from emulator import Emulator # Emulator should be imported before env
from dataloader import DataGenerator
from utils.utilities import get_inp_files
import argparse,yaml
from envs import get_env
import numpy as np
import os,time,shutil
import pandas as pd
import multiprocessing as mp
from mpc import get_runoff,pred_simu
# from line_profiler import LineProfiler
HERE = os.path.dirname(__file__)
# mixed_float16 causes nan values in log ops (softmax in GAT and bce loss in if_flood)

class Argument(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(Argument, self).__init__(*args, **kwargs)

        # env args
        self.add_argument('--env',type=str,default='shunqing',help='set drainage scenarios')
        self.add_argument('--directed',action='store_true',help='if use directed graph')
        self.add_argument('--length',type=float,default=0,help='adjacency range')
        self.add_argument('--order',type=int,default=1,help='adjacency order')
        self.add_argument('--graph_base',action='store_true',help='if use node edge based graph')
        self.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
        self.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
        self.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')
        self.add_argument('--swmm_step',type=int,default=1,help='routing step for swmm inp files')

        # simulate args
        self.add_argument('--simulate',action="store_true",help='if simulate rainfall events for training data')
        self.add_argument('--data_dir',type=str,default='./envs/data/',help='the sampling data file')
        self.add_argument('--train_event_id',type=str,default='train_id.npy',help='the training event id file')
        self.add_argument('--act',type=str,default='False',help='if and what control actions')
        self.add_argument('--ctrl_step',type=int,default=5,help='setting duration')
        self.add_argument('--processes',type=int,default=1,help='number of simulation processes')
        self.add_argument('--repeats',type=int,default=1,help='number of simulation repeats of each event')
        self.add_argument('--hotstart',action="store_true",help='if save hotstart files')

        # train args
        self.add_argument('--train',action="store_true",help='if train the emulator')
        self.add_argument('--seed',type=int,default=42,help='random seeds')
        self.add_argument('--load_model',action="store_true",help='if use existed model file to further train')
        self.add_argument('--model_dir',type=str,default='./model/',help='the surrogate model weights')
        self.add_argument('--ratio',type=float,default=0.8,help='ratio of training events')
        self.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
        self.add_argument('--epochs',type=int,default=500,help='training epochs')
        self.add_argument('--save_gap',type=int,default=100,help='save model per epochs')
        self.add_argument('--batch_size',type=int,default=256,help='training batch size')
        self.add_argument('--roll',type=int,default=0,help='if rolls out for curriculum learning')
        self.add_argument('--mixed_precision',action="store_true",help='if use mixed precision bf16 for training')
        self.add_argument('--gradnorm',action="store_true",help='if use GradNorm to balance multi learning tasks')

        # network args
        self.add_argument('--nly',type=int,default=3,help='number of spatial layers')
        self.add_argument('--dim',type=int,default=64,help='number of channels in each recurrent layer')
        self.add_argument('--activation',type=str,default='relu',help='activation function')
        self.add_argument('--kernel',type=int,default=3,help='number of channels in each convolution layer')
        self.add_argument('--seq',type=int,default=6,help='input sequential length')
        self.add_argument('--if_flood',action="store_true",help='if classify flooding with layers or not')
        self.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')

        # test args
        self.add_argument('--test',action="store_true",help='if test the emulator')
        self.add_argument('--horizon',type=int,default=60,help='horizon length')
        self.add_argument('--pop_size',type=int,default=1,help='number of parallel control options')
        self.add_argument('--result_dir',type=str,default='./results/',help='the test results')

def parser(config=None):
    parser = Argument(description='surrogate')

    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        parser.set_defaults(**hyps[args.env])
    args = parser.parse_args()

    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k,v)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyps[args.env][k],v))

    print('Training configs: {}'.format(args))
    return args,config

if __name__ == "__main__":
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))

    # simu_de = {'simulate':True,
    #            'env':'RedChicoSur',
    #            'data_dir':'./envs/data/RedChicoSur/act_edge/',
    #            'act':True,
    #            'processes':1,
    #            'repeats':1,
    #            }
    # for k,v in simu_de.items():
    #     setattr(args,k,v)

    # train_de = {'train':True,
    #             'env':'hague',
    #             'order':1,
    #             'data_dir':'./envs/data/hague/1s_conti_rain50/',
    #             'act':'conti',
    #             'model_dir':'./model/hague/test/',
    #             'load_model':False,
    #             'roll':0,
    #             'batch_size':4,
    #             'epochs':50000,
    #             'nly':5,
    #             'seq':60,
    #             'if_flood':False,
    #             'gradnorm':True}
    # for k,v in train_de.items():
    #     setattr(args,k,v)

    # test_de = {'test':True,
    #            'env':'hague',
    #            'act':False,
    #            'model_dir':'./model/hague/12s_20k_res_norm_flood_gcn/',
    #            'resnet':True,
    #            'seq':12,
    #            'if_flood':True,
    #            'balance':False,
    #            'conv':'GCN',
    #            'recurrent':'Conv1D',
    #            'result_dir':'./results/hague/12s_20k_res_norm_flood_gcn/',
    #            'rain_dir':'./envs/config/hg_test_events.csv'}
    # for k,v in test_de.items():
    #     setattr(args,k,v)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args(args.directed,args.length,args.order)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)

    dG = DataGenerator(env.config,args.data_dir,args)
    
    if args.simulate:
        if not os.path.exists(args.data_dir):
            os.mkdir(args.data_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.data_dir,'parser.yaml'),'w'))
        rain_arg = env.config['rainfall']
        if 'rain_dir' in config:
            rain_arg['rainfall_events'] = args.rain_dir
        if 'rain_suffix' in config:
            rain_arg['suffix'] = args.rain_suffix
        if 'rain_num' in config:
            rain_arg['rain_num'] = args.rain_num
        events = get_inp_files(env.config['swmm_input'],rain_arg,swmm_step=args.swmm_step)
        dG.generate(events,processes=args.processes,repeats=args.repeats,act=args.act)
        dG.save(args.data_dir)

    if args.train:
        dG.load(args.data_dir)
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)

        seq = args.seq*args.roll if args.roll > 0 else args.seq
        n_events = int(max(dG.event_id))+1
        if os.path.isfile(os.path.join(args.data_dir,args.train_event_id)):
            train_ids = np.load(os.path.join(args.data_dir,args.train_event_id))
        elif args.load_model:
            train_ids = np.load(os.path.join(args.model_dir,'train_id.npy'))
        else:
            train_ids = np.random.choice(np.arange(n_events),int(n_events*args.ratio),replace=False)
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        train_idxs = dG.get_data_idxs(train_ids,seq)
        test_idxs = dG.get_data_idxs(test_ids,seq)
        # if args.if_flood:
        #     args.poswei = dG.get_flood_poswei()

        emul = Emulator(args)
        if args.load_model:
            args.model_dir = os.path.join(args.model_dir,'retrain')
            emul.load(retrain=args.load_model)
            setattr(emul,'model_dir',args.model_dir)
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            if 'model_dir' in config:
                config['model_dir'] += '/retrain'
        emul.set_norm(*dG.get_norm())
        yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

        t0 = time.time()
        train_losses,test_losses,secs = [],[],[0]
        log_dir = "logs/model/"
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir)
        for epoch in range(args.epochs):
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,interval=args.ctrl_step,trim=False,weight=True)
            x,a,b,y = [emul.to_tensor(dat) for dat in train_dats[:4]]
            ex,ey = [emul.to_tensor(dat) for dat in train_dats[6:8]]
            x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],list('xbye')+['ey'])]
            ini_loss = emul.to_tensor(np.array(train_losses[0])) if epoch > 0 else None
            train_loss = emul.fit_eval(x,a,b,y,ex,ey,ini_loss)
            train_loss = [los.detach().cpu().numpy() for los in train_loss]
            train_losses.append(train_loss)

            test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size,interval=args.ctrl_step,trim=False,weight=True)
            x,a,b,y = [emul.to_tensor(dat) for dat in test_dats[:4]]
            ex,ey = [emul.to_tensor(dat) for dat in test_dats[6:8]]
            x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],list('xbye')+['ey'])]
            test_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=False)
            test_loss = [los.detach().cpu().numpy() for los in test_loss]
            test_losses.append(test_loss)

            if sum(train_loss) < min([1e6]+[sum(los) for los in train_losses[:-1]]):
                emul.save('train')
            if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
                emul.save('test')
            if epoch > 0 and epoch % args.save_gap == 0:
                emul.save('%s'%epoch)
                
            secs.append(time.time()-t0)

            # Log output
            log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],sum(train_loss),sum(test_loss))
            writer.add_scalar('train loss', sum(train_loss), epoch)
            log += " ("
            log += "Node: {:.4f}".format(test_loss[0])
            writer.add_scalar('Node loss', test_loss[0], epoch)
            i = 1
            if args.if_flood:
                log += " if_flood: {:.4f}".format(test_loss[i])
                writer.add_scalar('Flood classification', test_loss[i], epoch)
                i += 1
            log += " Edge: {:.4f})".format(test_loss[i])
            writer.add_scalar('Edge loss', test_loss[i], epoch)
            print(log)

        # save
        emul.save()
        np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
        np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
        np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
        np.save(os.path.join(args.model_dir,'time.npy'),np.array(secs[1:]))
        # plt.plot(train_losses,label='train')
        # plt.plot(np.array(test_losses).sum(axis=1),label='test')
        # plt.legend()
        # plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)

    if args.test:
        known_hyps = yaml.load(open(os.path.join(args.model_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k in ['model_dir','act']:
                continue
            elif k == 'data_dir':
                v = os.path.join(args.data_dir,v)
            setattr(args,k,v)
        env_args = env.get_args(args.directed,args.length,args.order,args.graph_base)
        for k,v in env_args.items():
            if k == 'act':
                v = v and args.act != 'False' and args.act
            setattr(args,k,v)
        emul = Emulator(args.conv,args.resnet,args.recurrent,args)
        emul.load(args.model_dir)
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))
        rain_arg = env.config['rainfall']
        if 'rain_dir' in config:
            rain_arg['rainfall_events'] = args.rain_dir
        if 'rain_suffix' in config:
            rain_arg['suffix'] = args.rain_suffix
        if 'rain_num' in config:
            rain_arg['rain_num'] = args.rain_num
        events = get_inp_files(env.config['swmm_input'],rain_arg,swmm_step=args.swmm_step)
        args.n_step,args.r_step = args.horizon//args.ctrl_step,args.ctrl_step//args.interval

        emu_objss,simu_objss,objss,settingss = [],[],[],[]       
        for event in events:
            name = os.path.basename(event).strip('.inp')

            ts,runoff = get_runoff(env,event)
            tss = pd.DataFrame.from_dict({'Time':ts,'Index':np.arange(len(ts))}).set_index('Time')
            tss.index = pd.to_datetime(tss.index)
            runoff = np.stack([np.concatenate([runoff[idx:idx+args.horizon],
                                               np.tile(np.zeros_like(s),(max(idx+args.horizon-runoff.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                            for idx,s in enumerate(runoff)])

            if args.prediction['no_runoff']:
                ts2,runoff_rate = get_runoff(env,event,True)
                tss2 = pd.DataFrame.from_dict({'Time':ts2,'Index':np.arange(len(ts2))}).set_index('Time')
                tss2.index = pd.to_datetime(tss2.index)
                runoff_rate = np.stack([np.concatenate([runoff_rate[idx:idx+args.horizon],
                                                        np.tile(np.zeros_like(s),(max(idx+args.horizon-runoff_rate.shape[0],0),)+tuple(1 for _ in s.shape))],axis=0)
                                    for idx,s in enumerate(runoff_rate)])

            state = env.reset(event,global_state=True,seq=args.seq_in)
            perf = env.flood(seq=args.seq_in)
            edge_state = env.state_full(typ='links',seq=args.seq_in)
            y = np.array([[env.controller(mode='conti')
                        for _ in range(runoff.shape[0]//args.ctrl_step+args.n_step+1)]
                            for _ in range(args.pop_size)])
            done,i = False,0
            emu_objs,simu_objs,objs,settings = [],[],[],[]
            while not done:
                if i*args.interval % args.ctrl_step == 0:
                    t = env.env.methods['simulation_time']()
                    yi = y[:,i//args.ctrl_step:i//args.ctrl_step+args.n_step,:]
                    setting = np.concatenate([np.repeat(yi[:,idx:idx+1,:],args.r_step,axis=1)
                                            for idx in range(yi.shape[1])],axis=1)
                    if setting.shape[1] < args.horizon // args.interval:
                        setting = np.concatenate([setting,np.repeat(setting[:,-1:,:],args.horizon // args.interval - setting.shape[1],axis=1)],axis=1)
                    settings.append(setting)
                    t0 = time.time()
                    if args.error > 0:
                        r = runoff[int(tss.asof(t)['Index'])]
                        std = np.array([ri*args.error*i/r.shape[0] for i,ri in enumerate(r)])
                        r += np.random.uniform(-std,std)
                        r = np.repeat(np.expand_dims(r,0),args.pop_size,axis=0)
                    else:
                        r = np.repeat(np.expand_dims(runoff[int(tss.asof(t)['Index'])],0),args.pop_size,axis=0)

                    state[...,1] = state[...,1] - state[...,-1]
                    state = np.repeat(np.expand_dims(state,0),args.pop_size,axis=0)
                    perf = np.repeat(np.expand_dims(perf,0),args.pop_size,axis=0)
                    edge_state = np.repeat(np.expand_dims(edge_state,0),args.pop_size,axis=0)
                    if args.horizon > args.seq * args.interval:
                        predss = []
                        s0,e0 = state.copy(),edge_state.copy()
                        for idx in range(args.horizon//args.seq_in):
                            ri = r[:,idx*args.seq:(idx+1)*args.seq,...]
                            sett = setting[:,idx*args.seq:(idx+1)*args.seq,:]
                            if args.if_flood:
                                f = (perf>0).astype(float)
                                state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
                            preds = emul.predict(state,ri,sett,edge_state)
                            state = np.concatenate([preds[0][...,:-2],ri[...,:1]],axis=-1)
                            perf = preds[0][...,-1:]
                            ae = emul.get_edge_action(sett,False)
                            edge_state = np.concatenate([preds[1],ae],axis=-1)
                            predss.append(preds)
                        predss = [np.concatenate([preds[0] for preds in predss],axis=1),
                                  np.concatenate([preds[1] for preds in predss],axis=1)]
                        emu_obj = env.objective_pred(predss,[s0,e0],sett)
                    else:
                        if args.if_flood:
                            f = (perf>0).astype(float)
                            state = np.concatenate([state[...,:-1],f,state[...,-1:]],axis=-1)
                        preds = emul.predict(state,r,setting,edge_state)
                        emu_obj = env.objective_pred(preds,[state,edge_state],setting)
                    t1 = time.time()
                    print('emulation time: %s'%(t1-t0))
                    emu_objs.append(emu_obj.squeeze())

                    eval_file = env.get_eval_file(args.prediction['no_runoff'])
                    if args.prediction['no_runoff']:
                        r = np.repeat(np.expand_dims(runoff_rate[int(tss2.asof(t)['Index'])],0),args.pop_size,axis=0)
                    else:
                        r = [None for _ in range(len(y))]
                    args.log = env.data_log.copy()
                    pool = mp.Pool(args.processes)
                    res = [pool.apply_async(func=pred_simu,args=(sett,eval_file,args,ri[...,0] if args.prediction['no_runoff'] else None,))
                            for sett,ri in zip(setting,r)]
                    pool.close()
                    pool.join()
                    simu_obj = np.stack([r.get() for r in res])
                    print('hsf simu time: %s'%(time.time()-t1))
                    simu_objs.append(simu_obj.squeeze())

                    objs.append(env.objective(args.horizon))
                done = env.step(setting[0,0,:])
                state = env.state_full(seq=args.seq_in)
                perf = env.flood(seq=args.seq_in)
                edge_state = env.state_full(args.seq_in,'links')
                i += 1
            emu_objss.append(np.stack(emu_objs,axis=0))
            simu_objss.append(np.stack(simu_objs,axis=0))
            objss.append(np.stack(objs))
            settingss.append(settings)
        
        emu_objss = np.concatenate([obj[:-args.n_step] for obj in emu_objss],axis=0)
        simu_objss = np.concatenate([obj[:-args.n_step] for obj in simu_objss],axis=0)
        objss = np.concatenate([obj[args.n_step:] for obj in objss],axis=0)
        np.save(os.path.join(args.result_dir,'emu_objs.npy'),emu_objss)
        np.save(os.path.join(args.result_dir,'simu_objs.npy'),simu_objss)
        np.save(os.path.join(args.result_dir,'objs.npy'),objss)
        np.save(os.path.join(args.result_dir,'settings.npy'),np.array(settingss))
