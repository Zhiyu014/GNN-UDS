import torch as th
from torch.utils.tensorboard import SummaryWriter
from emulator import Emulator # Emulator should be imported before env
from dataloader import DataGenerator
from utils.utilities import get_inp_files
import argparse,yaml
from envs import get_env
import numpy as np
import os,time,shutil
import matplotlib.pyplot as plt
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
        self.add_argument('--train_event_id',type=str,default='',help='the training event id file')
        self.add_argument('--act',type=str,default='False',help='if and what control actions')
        self.add_argument('--setting_duration',type=int,default=5,help='setting duration')
        self.add_argument('--processes',type=int,default=1,help='number of simulation processes')
        self.add_argument('--repeats',type=int,default=1,help='number of simulation repeats of each event')

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
        self.add_argument('--kernel',type=int,default=3,help='number of channels in each convolution layer')
        self.add_argument('--seq',type=int,default=6,help='input sequential length')
        self.add_argument('--if_flood',action="store_true",help='if classify flooding with layers or not')
        self.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')

        # test args
        self.add_argument('--test',action="store_true",help='if test the emulator')
        self.add_argument('--result_dir',type=str,default='./results/',help='the test results')
        self.add_argument('--hotstart',action="store_true",help='if use hotstart to test simulation time')

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
    #             'env':'astlingen',
    #             'order':1,
    #             'data_dir':'./envs/data/astlingen/1s_edge_conti128_rain50/',
    #             'act':'conti',
    #             'model_dir':'./model/astlingen/test/',
    #             'load_model':False,
    #             'roll':0,
    #             'batch_size':128,
    #             'epochs':50000,
    #             'nly':3,
    #             'seq':5,
    #             'if_flood':True,
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

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        emul = Emulator(args)
        if args.load_model:
            emul.load()
        emul.set_norm(*dG.get_norm())
        yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

        t0 = time.time()
        train_losses,test_losses,secs = [],[],[0]
        log_dir = "logs/model/"
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir)
        for epoch in range(args.epochs):
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
            x,a,b,y = [th.tensor(dat).to(device) for dat in train_dats[:4]]
            ex,ey = [th.tensor(dat).to(device) for dat in train_dats[6:8]]
            x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
            train_loss = emul.fit_eval(x,a,b,y,ex,ey)
            train_loss = train_loss.detach().cpu().numpy()
            if epoch >= 500:
                train_losses.append(train_loss)

            test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
            x,a,b,y = [th.tensor(dat).to(device) for dat in test_dats[:4]]
            ex,ey = [th.tensor(dat).to(device) for dat in test_dats[6:8]]
            x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
            test_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=False)
            test_loss = [los.detach().cpu().numpy() for los in test_loss]
            if epoch >= 500:
                test_losses.append(test_loss)

            if train_loss < min([1e6]+train_losses[:-1]):
                emul.save('train')
            if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
                emul.save('test')
            if epoch > 0 and epoch % args.save_gap == 0:
                emul.save('%s'%epoch)
                
            secs.append(time.time()-t0)

            # Log output
            log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],train_loss,sum(test_loss))
            log += " ("
            log += "Node: {:.4f}".format(test_loss[0])
            i = 1
            if args.if_flood:
                log += " if_flood: {:.4f}".format(test_loss[i])
                i += 1
            log += " Edge: {:.4f})".format(test_loss[i])
            print(log)
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('Node loss', test_loss[0], epoch)
            i = 1
            if args.if_flood:
                writer.add_scalar('Flood classification', test_loss[i], epoch)
                i += 1
            writer.add_scalar('Edge loss', test_loss[i], epoch)

        # save
        emul.save()
        np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
        np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
        np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
        np.save(os.path.join(args.model_dir,'time.npy'),np.array(secs[1:]))
        plt.plot(train_losses,label='train')
        plt.plot(np.array(test_losses).sum(axis=1),label='test')
        plt.legend()
        plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)

    if args.test:
        known_hyps = yaml.load(open(os.path.join(args.model_dir,'parser.yaml'),'r'),yaml.FullLoader)
        for k,v in known_hyps.items():
            if k in ['model_dir','act']:
                continue
            elif k == 'data_dir':
                v = os.path.join(args.data_dir,v)
            setattr(args,k,v)
        env_args = env.get_args(args.directed,args.length,args.order)
        for k,v in env_args.items():
            if k == 'act':
                v = v and args.act != 'False' and args.act
            setattr(args,k,v)
        dG = DataGenerator(env.config,args.data_dir,args)
        emul = Emulator(args)
        emul.load()
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
        if 'train_event_id' in config and os.path.isfile(os.path.join(args.data_dir,config['train_event_id'])):
            train_ids = np.load(os.path.join(args.data_dir,config['train_event_id']))
            events = [eve for i,eve in enumerate(events) if i not in train_ids]
        for event in events:
            name = os.path.basename(event).strip('.inp')
            if os.path.exists(os.path.join(args.result_dir,name + '_states.npy')):
                states = np.load(os.path.join(args.result_dir,name + '_states.npy'))
                perfs = np.load(os.path.join(args.result_dir,name + '_perfs.npy'))
                if args.act:
                    settings = np.load(os.path.join(args.result_dir,name + '_settings.npy'))
                edge_states = np.load(os.path.join(args.result_dir,name + '_edge_states.npy'))
            else:
                t0 = time.time()
                pre_step = rain_arg.get('pre_time',0) // args.interval
                res = dG.simulate(env,event,act=args.act,hotstart=args.seq*args.hotstart)
                states,perfs,settings = [r[pre_step:] if r is not None else None for r in res[:3]]
                print("{} Simulation time: {}".format(name,time.time()-t0))
                np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
                np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)
                if settings is not None:
                    np.save(os.path.join(args.result_dir,name + '_settings.npy'),settings)
                edge_states = res[-1][pre_step:]
                np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),edge_states)

            states,perfs = [dG.expand_seq(dat,args.seq) for dat in [states,perfs]]
            edge_states = dG.expand_seq(edge_states,args.seq)

            states[...,1] = states[...,1] - states[...,-1]
            r,true = states[args.seq:,...,-1:],states[args.seq:,...,:-1]
            if args.tide:
                t = states[args.seq:,...,0] * args.is_outfall
                r = np.concatenate([r,np.expand_dims(t,axis=-1)],axis=-1)
                
            # states = states[...,:-1]
            if args.if_flood:
                f = (perfs>0).astype(float)
                # f = np.eye(2)[f].squeeze(-2)
                states = np.concatenate([states[...,:-1],f,states[...,-1:]],axis=-1)
                true = np.concatenate([true,f[args.seq:]],axis=-1)
            states = states[:-args.seq]

            edge_true = edge_states[args.seq:,...,:-1]
            edge_states = edge_states[:-args.seq]
            edge_states = edge_states[:,-args.seq:,...]
            edge_true = edge_true[:,:args.seq,...]

            if args.act:
                a = dG.expand_seq(settings,args.seq,zeros=False)
                a = a[args.seq:,:args.seq,...]
            else:
                a = None
            
            t0 = time.time()
            # lp = LineProfiler()
            # lp_wrapper = lp(emul.simulate)
            # pred = lp_wrapper(states,r,a,edge_states)
            # lp.print_stats()
            pred,edge_pred = emul.simulate(states,r,a,edge_states)
            print("{} Emulation time: {}".format(name,time.time()-t0))

            true = np.concatenate([true,perfs[args.seq:,...]],axis=-1)  # cumflooding in performance
            true = true[:,:args.seq,...]

            los_str = "{} Testing loss: (".format(name)
            loss = [emul.mse(emul.normalize(pred[...,:3],'y'),emul.normalize(true[...,:3],'y'))]
            los_str += "Node: {:.4f} ".format(loss[-1])
            if args.if_flood:
                loss += [emul.bce(pred[...,-2:-1],true[...,-2:-1])]
                los_str += "if_flood: {:.4f} ".format(loss[-1])
            loss += [emul.mse(emul.normalize(edge_pred,'e'),emul.normalize(edge_true,'e'))]
            los_str += "Edge: {:.4f})".format(loss[-1])
            print(los_str)

            np.save(os.path.join(args.result_dir,name + '_runoff.npy'),r.astype(np.float32))
            np.save(os.path.join(args.result_dir,name + '_true.npy'),true.astype(np.float32))
            np.save(os.path.join(args.result_dir,name + '_pred.npy'),pred.astype(np.float32))
            edge_true = edge_true[:,:args.seq,...]
            np.save(os.path.join(args.result_dir,name + '_edge_true.npy'),edge_true.astype(np.float32))
            np.save(os.path.join(args.result_dir,name + '_edge_pred.npy'),edge_pred.astype(np.float32))


