from emulator import Emulator # Emulator should be imported before env
from dataloader import DataGenerator
from utils.utilities import get_inp_files
import argparse,yaml
from envs import get_env
import numpy as np
import os,time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
# from line_profiler import LineProfiler
HERE = os.path.dirname(__file__)

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')

    # env args
    parser.add_argument('--env',type=str,default='shunqing',help='set drainage scenarios')
    parser.add_argument('--directed',action='store_true',help='if use directed graph')
    parser.add_argument('--length',type=float,default=0,help='adjacency range')
    parser.add_argument('--order',type=int,default=1,help='adjacency order')
    parser.add_argument('--rain_dir',type=str,default='./envs/config/',help='path of the rainfall events')
    parser.add_argument('--rain_suffix',type=str,default=None,help='suffix of the rainfall names')
    parser.add_argument('--rain_num',type=int,default=1,help='number of the rainfall events')

    # simulate args
    parser.add_argument('--simulate',action="store_true",help='if simulate rainfall events for training data')
    parser.add_argument('--data_dir',type=str,default='./envs/data/',help='the sampling data file')
    parser.add_argument('--act',type=str,default='False',help='if and what control actions')
    parser.add_argument('--setting_duration',type=int,default=5,help='setting duration')
    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')
    parser.add_argument('--repeats',type=int,default=1,help='number of simulation repeats of each event')
    parser.add_argument('--use_edge',action='store_true',help='if models edge attrs')

    # train args
    parser.add_argument('--train',action="store_true",help='if train the emulator')
    parser.add_argument('--load_model',action="store_true",help='if use existed model file to further train')
    parser.add_argument('--edge_fusion',action='store_true',help='if use node-edge fusion model')
    parser.add_argument('--use_adj',action="store_true",help='if use filter to act control')
    parser.add_argument('--model_dir',type=str,default='./model/',help='the surrogate model weights')
    parser.add_argument('--ratio',type=float,default=0.8,help='ratio of training events')
    parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--epochs',type=int,default=500,help='training epochs')
    parser.add_argument('--save_gap',type=int,default=100,help='save model per epochs')
    parser.add_argument('--batch_size',type=int,default=256,help='training batch size')
    parser.add_argument('--balance',action="store_true",help='if use balance not classification loss')

    # network args
    parser.add_argument('--norm',action="store_true",help='if data is normalized with maximum')
    parser.add_argument('--conv',type=str,default='GCNconv',help='convolution type')
    parser.add_argument('--embed_size',type=int,default=128,help='number of channels in each convolution layer')
    parser.add_argument('--n_sp_layer',type=int,default=3,help='number of spatial layers')
    parser.add_argument('--dropout',type=float,default=0.0,help='dropout rate')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')
    parser.add_argument('--recurrent',type=str,default='GRU',help='recurrent type')
    parser.add_argument('--hidden_dim',type=int,default=64,help='number of channels in each recurrent layer')
    parser.add_argument('--kernel_size',type=int,default=3,help='number of channels in each convolution layer')
    parser.add_argument('--n_tp_layer',type=int,default=2,help='number of temporal layers')
    parser.add_argument('--seq_in',type=int,default=6,help='input sequential length')
    parser.add_argument('--seq_out',type=int,default=1,help='out sequential length. seq_out < seq_in ')
    parser.add_argument('--resnet',action='store_true',help='if use resnet')
    parser.add_argument('--if_flood',type=int,default=0,help='if classify flooding with layers or not')
    parser.add_argument('--epsilon',type=float,default=-1.0,help='the depth threshold of flooding')

    # test args
    parser.add_argument('--test',action="store_true",help='if test the emulator')
    parser.add_argument('--result_dir',type=str,default='./results/',help='the test results')
    parser.add_argument('--hotstart',action="store_true",help='if use hotstart to test simulation time')

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
    #            'use_edge':True
    #            }
    # for k,v in simu_de.items():
    #     setattr(args,k,v)

    # train_de = {'train':True,
    #             'env':'astlingen',
    #             'length':1000,
    #             'data_dir':'./envs/data/astlingen/1s_edge_conti_rain50/',
    #             'act':'conti',
    #             'model_dir':'./model/shunqing/30s_20k_conti_1000ledgef_res_norm_flood_gat/',
    #             'batch_size':128,
    #             'epochs':5000,
    #             'n_tp_layer':4,
    #             'resnet':True,
    #             'norm':True,
    #             'use_edge':True,'edge_fusion':True,
    #             'balance':False,
    #             'seq_in':30,'seq_out':30,
    #             'if_flood':True,
    #             'conv':'GAT',
    #             'recurrent':'Conv1D'}
    # for k,v in train_de.items():
    #     setattr(args,k,v)

    # test_de = {'test':True,
    #            'env':'hague',
    #            'act':False,
    #            'model_dir':'./model/hague/12s_20k_res_norm_flood_gcn/',
    #            'resnet':True,
    #            'norm':True,
    #            'seq_in':12,
    #            'seq_out':12,
    #            'if_flood':True,
    #            'use_edge':False,
    #            'balance':False,
    #            'conv':'GCN',
    #            'recurrent':'Conv1D',
    #            'result_dir':'./results/hague/12s_20k_res_norm_flood_gcn/',
    #            'rain_dir':'./envs/config/hg_test_events.csv'}
    # for k,v in test_de.items():
    #     setattr(args,k,v)

    env = get_env(args.env)()
    env_args = env.get_args(args.directed,args.length,args.order)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)
    args.use_edge = args.use_edge or args.edge_fusion

    dG = DataGenerator(env,args.data_dir,args)
    
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
        events = get_inp_files(env.config['swmm_input'],rain_arg)
        dG.generate(events,processes=args.processes,repeats=args.repeats,act=args.act)
        dG.save(args.data_dir)

    if args.train:
        dG.load(args.data_dir)
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        emul = Emulator(args.conv,args.resnet,args.recurrent,args)
        # plot_model(emul.model,os.path.join(args.model_dir,"model.png"),show_shapes=True)
        yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

        seq = max(args.seq_in,args.seq_out) if args.recurrent else 0
        n_events = int(max(dG.event_id))+1

        if args.load_model:
            emul.load(args.model_dir)
            train_ids = np.load(os.path.join(args.model_dir,'train_id.npy'))
        else:
            train_ids = np.random.choice(np.arange(n_events),int(n_events*args.ratio),replace=False)
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        if args.norm:
            emul.set_norm(*dG.get_norm())

        train_idxs = dG.get_data_idxs(seq,train_ids)
        test_idxs = dG.get_data_idxs(seq,test_ids)

        t0 = time.time()
        train_losses,test_losses,secs = [],[],[0]
        for epoch in range(args.epochs):
            train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size)
            x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
            if args.norm:
                x,b,y = [emul.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if args.use_edge:
                ex,ey = [dat for dat in train_dats[-2:]]
                if args.norm:
                    ex,ey = [emul.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            train_loss = emul.fit_eval(x,a,b,y,ex,ey)
            train_loss = train_loss.numpy()
            if epoch >= 500:
                train_losses.append(train_loss)

            test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size)
            x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
            if args.norm:
                x,b,y = [emul.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if args.use_edge:
                ex,ey = [dat for dat in test_dats[-2:]]
                if args.norm:
                    ex,ey = [emul.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            test_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=False)
            test_loss = [los.numpy() for los in test_loss]
            if epoch >= 500:
                test_losses.append(test_loss)

            if train_loss < min([1e6]+train_losses[:-1]):
                emul.save(os.path.join(args.model_dir,'train'))
            if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
                emul.save(os.path.join(args.model_dir,'test'))
            if epoch > 0 and epoch % emul.save_gap == 0:
                emul.save(os.path.join(args.model_dir,'%s'%epoch))
                
            secs.append(time.time()-t0)

            # Log output
            log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],train_loss,sum(test_loss))
            log += " ("
            node_str = "Node bal: " if args.balance else "Node: "
            log += node_str + "{:.4f}".format(test_loss[0])
            i = 1
            if args.if_flood and not args.balance:
                log += " if_flood: {:.4f}".format(test_loss[i])
                i += 1
            if args.use_edge:
                log += " Edge: {:.4f}".format(test_loss[i])
            log += ")"
            print(log)


        # save
        emul.save(args.model_dir)
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
            setattr(args,k,v)
        env_args = env.get_args(args.directed,args.length,args.order)
        for k,v in env_args.items():
            if k == 'act':
                v = v and args.act != 'False' and args.act
            setattr(args,k,v)
        args.use_edge = args.use_edge or args.edge_fusion
        dG = DataGenerator(env,args.data_dir,args)
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
        events = get_inp_files(env.config['swmm_input'],rain_arg)
        for event in events:
            name = os.path.basename(event).strip('.inp')
            if os.path.exists(os.path.join(args.result_dir,name + '_states.npy')):
                states = np.load(os.path.join(args.result_dir,name + '_states.npy'))
                perfs = np.load(os.path.join(args.result_dir,name + '_perfs.npy'))
                if args.act:
                    settings = np.load(os.path.join(args.result_dir,name + '_settings.npy'))
                if args.use_edge:
                    edge_states = np.load(os.path.join(args.result_dir,name + '_edge_states.npy'))
            else:
                t0 = time.time()
                pre_step = rain_arg.get('pre_time',0) // args.interval
                res = dG.simulate(event,act=args.act,hotstart=args.seq_out*args.hotstart)
                states,perfs,settings = [r[pre_step:] if r is not None else None for r in res[:3]]
                print("{} Simulation time: {}".format(name,time.time()-t0))
                np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
                np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)
                if settings is not None:
                    np.save(os.path.join(args.result_dir,name + '_settings.npy'),settings)
                if args.use_edge:
                    edge_states = res[-1][pre_step:]
                    np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),edge_states)

            seq = max(args.seq_in,args.seq_out) if args.recurrent else False
            states,perfs = [dG.expand_seq(dat,seq) for dat in [states,perfs]]
            edge_states = dG.expand_seq(edge_states,seq) if args.use_edge else None

            states[...,1] = states[...,1] - states[...,-1]
            r,true = states[args.seq_out:,...,-1:],states[args.seq_out:,...,:-1]
            if args.tide:
                t = states[args.seq_out:,...,0] * args.is_outfall
                r = np.concatenate([r,np.expand_dims(t,axis=-1)],axis=-1)
                
            # states = states[...,:-1]
            if args.if_flood:
                f = (perfs>0).astype(float)
                # f = np.eye(2)[f].squeeze(-2)
                states = np.concatenate([states[...,:-1],f,states[...,-1:]],axis=-1)
                true = np.concatenate([true,f[args.seq_out:]],axis=-1)
            states = states[:-args.seq_out]

            if args.use_edge:
                edge_true = edge_states[args.seq_out:,...,:-1]
                edge_states = edge_states[:-args.seq_out]
                if args.recurrent:
                    edge_states = edge_states[:,-args.seq_in:,...]
                    edge_true = edge_true[:,:args.seq_out,...]
            else:
                edge_states = None

            if args.act:
                if args.recurrent:
                    a = dG.expand_seq(settings,args.seq_out,zeros=False)
                    a = a[args.seq_out:,:args.seq_out,...]
                else:
                    a = a[1:]
            else:
                a = None
            
            t0 = time.time()
            # lp = LineProfiler()
            # lp_wrapper = lp(emul.simulate)
            # pred = lp_wrapper(states,r,a,edge_states)
            # lp.print_stats()
            pred = emul.simulate(states,r,a,edge_states)
            if args.use_edge:
                pred,edge_pred = pred
            print("{} Emulation time: {}".format(name,time.time()-t0))

            true = np.concatenate([true,perfs[args.seq_out:,...]],axis=-1)  # cumflooding in performance
            if args.recurrent:
                true = true[:,:args.seq_out,...]

            los_str = "{} Testing loss: (".format(name)
            loss = [emul.mse(emul.normalize(pred[...,:3],'y'),emul.normalize(true[...,:3],'y'))]
            los_str += "Node: {:.4f} ".format(loss[-1])
            if args.if_flood:
                loss += [emul.bce(pred[...,-2:-1],true[...,-2:-1])]
                los_str += "if_flood: {:.4f} ".format(loss[-1])
            if args.use_edge:
                loss += [emul.mse(emul.normalize(edge_pred,'e'),emul.normalize(edge_true,'e'))]
                los_str += "Edge: {:.4f}".format(loss[-1])
            print(los_str+')')

            np.save(os.path.join(args.result_dir,name + '_runoff.npy'),r)
            np.save(os.path.join(args.result_dir,name + '_true.npy'),true)
            np.save(os.path.join(args.result_dir,name + '_pred.npy'),pred)
            if args.use_edge:
                edge_true = edge_true[:,:args.seq_out,...] if args.recurrent else edge_true
                np.save(os.path.join(args.result_dir,name + '_edge_true.npy'),edge_true)
                np.save(os.path.join(args.result_dir,name + '_edge_pred.npy'),edge_pred)


