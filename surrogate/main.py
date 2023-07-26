from emulator import Emulator # Emulator should be imported before env
from dataloader import DataGenerator
from utilities import get_inp_files
import argparse,yaml
from envs import get_env
import numpy as np
import os,time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from line_profiler import LineProfiler

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')

    # simulate args
    parser.add_argument('--env',type=str,default='shunqing',help='set drainage scenarios')
    parser.add_argument('--simulate',action="store_true",help='if simulate rainfall events for training data')
    parser.add_argument('--data_dir',type=str,default='./envs/data/',help='the sampling data file')
    parser.add_argument('--act',type=str,default='False',help='if and what control actions')
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
    parser.add_argument('--batch_size',type=int,default=256,help='training batch size')
    parser.add_argument('--roll',action="store_true",help='if rollout simulation')
    parser.add_argument('--balance',action="store_true",help='ratio of balance loss')

    # network args
    parser.add_argument('--norm',action="store_true",help='if data is normalized with maximum')
    parser.add_argument('--conv',type=str,default='GCNconv',help='convolution type')
    parser.add_argument('--embed_size',type=int,default=128,help='number of channels in each convolution layer')
    parser.add_argument('--n_sp_layer',type=int,default=3,help='number of spatial layers')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')
    parser.add_argument('--recurrent',type=str,default='GRU',help='recurrent type')
    parser.add_argument('--hidden_dim',type=int,default=64,help='number of channels in each recurrent layer')
    parser.add_argument('--kernel_size',type=int,default=3,help='number of channels in each convolution layer')
    parser.add_argument('--n_tp_layer',type=int,default=2,help='number of temporal layers')
    parser.add_argument('--seq_in',type=int,default=6,help='input sequential length')
    # TODO: seq out
    parser.add_argument('--seq_out',type=int,default=1,help='out sequential length. if not roll, seq_out < seq_in ')
    parser.add_argument('--resnet',action='store_true',help='if use resnet')
    parser.add_argument('--if_flood',action='store_true',help='if classify flooding or not')

    # test args
    parser.add_argument('--test',action="store_true",help='if test the emulator')
    parser.add_argument('--result_dir',type=str,default='./results/',help='the test results')

    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        parser.set_defaults(**hyps[args.env])
    args = parser.parse_args()

    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k)}
    for k,v in config.items():
        if 'dir' in k:
            setattr(args,k,os.path.join(hyps[args.env][k],v))

    print('Training configs: {}'.format(args))
    return args,config

if __name__ == "__main__":
    args,config = parser('config.yaml')

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
    #             'data_dir':'./envs/data/astlingen/act_edge/',
    #             'act':True,
    #             'model_dir':'./model/astlingen/10s_20k_act_edgef_res_norm_flood_roll/',
    #             'batch_size':32,
    #             'epochs':5000,
    #             'resnet':True,
    #             'norm':True,
    #             'roll':True,
    #             'use_edge':True,'edge_fusion':True,
    #             'balance':False,
    #             'seq_in':10,'seq_out':5,
    #             'if_flood':True,
    #             'conv':'GAT',
    #             'recurrent':'Conv1D'}
    # for k,v in train_de.items():
    #     setattr(args,k,v)

    # test_de = {'test':True,
    #            'env':'astlingen',
    #            'act':'bc',
    #            'model_dir':'./model/astlingen/10s_20k_act_edge_res_norm_flood/',
    #            'resnet':True,
    #            'norm':True,
    #            'seq_in':10,
    #            'seq_out':10,
    #            'if_flood':True,
    #            'use_edge':True,
    #            'balance':False,
    #            'conv':'GAT',
    #            'recurrent':'Conv1D',
    #            'result_dir':'./results/astlingen/10s_20k_bc_edge_res_norm_flood/'}
    # for k,v in test_de.items():
    #     setattr(args,k,v)

    env = get_env(args.env)()
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)
    args.use_edge = args.use_edge or args.edge_fusion

    dG = DataGenerator(env,args.seq_in,args.seq_out,args.recurrent,args.act,args.if_flood,args.use_edge,args.data_dir)


    events = get_inp_files(env.config['swmm_input'],env.config['rainfall'])
    if args.simulate:
        if not os.path.exists(args.data_dir):
            os.mkdir(args.data_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.data_dir,'parser.yaml'),'w'))
        dG.generate(events,processes=args.processes,repeats=args.repeats,act=args.act)
        dG.save(args.data_dir)

    if args.train:
        dG.load(args.data_dir)
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        emul = Emulator(args.conv,args.resnet,args.recurrent,args)
        # plot_model(emul.model,os.path.join(args.model_dir,"model.png"),show_shapes=True)
        yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

        if args.load_model:
            emul.load(args.model_dir)
            train_ids = np.load(os.path.join(args.model_dir,'train_id.npy'))
        else:
            train_ids = None
        if args.norm:
            emul.set_norm(*dG.get_norm())
        train_ids,test_ids,train_losses,test_losses = emul.update_net(dG,args.ratio,args.epochs,args.batch_size,train_ids)

        # save
        emul.save(args.model_dir)
        np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
        np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
        np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
        plt.plot(train_losses,label='train')
        plt.plot(np.array(test_losses).sum(axis=1),label='test')
        plt.legend()
        plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)

    if args.test:
        emul = Emulator(args.conv,args.resnet,args.recurrent,args)
        emul.load(args.model_dir)
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))
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
                res = dG.simulate(event,act=args.act,hotstart=True)
                states,perfs,settings = [r for r in res[:3]]
                print("{} Simulation time: {}".format(name,time.time()-t0))
                np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
                np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)
                if settings is not None:
                    np.save(os.path.join(args.result_dir,name + '_settings.npy'),settings)
                if args.use_edge:
                    edge_states = res[-1]
                    np.save(os.path.join(args.result_dir,name + '_edge_states.npy'),edge_states)

            seq = max(args.seq_in,args.seq_out) if args.recurrent else False
            states,perfs = [dG.expand_seq(dat,seq) for dat in [states,perfs]]
            edge_states = dG.expand_seq(edge_states,seq) if args.use_edge else None

            states[...,1] = states[...,1] - states[...,-1]
            r,true = states[args.seq_out:,...,-1:],states[args.seq_out:,...,:-1]
            # states = states[...,:-1]
            if args.if_flood:
                f = (perfs>0).astype(int)
                f = np.eye(2)[f].squeeze(-2)
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
                loss += [emul.cce(pred[...,-3:-1],true[...,-3:-1])]
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


