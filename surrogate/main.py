from emulator import Emulator
from dataloader import DataGenerator,generate_file
import argparse,yaml,random
from envs import get_env
import numpy as np
import os
import matplotlib.pyplot as plt

def parser(config=None):
    parser = argparse.ArgumentParser(description='surrogate')

    # simulate args
    parser.add_argument('--env',type=str,default='shunqing',help='set drainage scenarios')
    parser.add_argument('--simulate',action="store_true",help='if simulate rainfall events for training data')
    parser.add_argument('--load_data',action="store_true",help='if load simulation data')
    parser.add_argument('--data_dir',type=str,default='./envs/data/',help='the sampling data file')
    parser.add_argument('--act',action="store_true",help='if the environment contains control actions')
    parser.add_argument('--processes',type=int,default=1,help='number of simulation processes')

    # train args
    parser.add_argument('--train',action="store_true",help='if train the emulator')
    parser.add_argument('--load_model',action="store_true",help='if load surrogate model weights')
    parser.add_argument('--model_dir',type=str,default='./model/',help='the surrogate model weights')
    parser.add_argument('--ratio',type=float,default=0.8,help='ratio of training events')
    parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--epochs',type=int,default=100,help='training epochs')
    parser.add_argument('--batch_size',type=int,default=256,help='training batch size')
    parser.add_argument('--roll',action="store_true",help='if rollout simulation')
    parser.add_argument('--balance',type=float,default=0,help='ratio of balance loss')

    # network args
    parser.add_argument('--norm',action="store_true",help='if data is normalized with maximum')
    parser.add_argument('--conv',type=str,default='GCNconv',help='convolution type')
    parser.add_argument('--embed_size',type=int,default=128,help='number of channels in each convolution layer')
    parser.add_argument('--n_layer',type=int,default=3,help='number of convolution layers')
    parser.add_argument('--activation',type=str,default='relu',help='activation function')
    parser.add_argument('--recurrent',type=str,default='GRU',help='recurrent type')
    parser.add_argument('--hidden_dim',type=int,default=64,help='number of channels in each recurrent layer')
    parser.add_argument('--seq_in',type=int,default=6,help='input sequential length')
    # TODO: seq out
    parser.add_argument('--seq_out',type=int,default=1,help='out sequential length. if not roll, seq_out < seq_in ')
    parser.add_argument('--resnet',action='store_true',help='if use resnet')
    parser.add_argument('--if_flood',action='store_true',help='if classify flooding or not')

    # test args
    parser.add_argument('--test',action="store_true",help='if test the emulator')
    parser.add_argument('--result_dir',type=str,default='./results/',help='the test results')

    # https://www.cnblogs.com/zxyfrank/p/15414605.html
    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        parser.set_defaults(**hyps[args.env])
    args = parser.parse_args()
    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k)}
    print('Training configs: {}'.format(args))
    return args,config

# TODO: prediction
def test(env,emul,event=None):
    seq = emul.seq_in

    # Get lateral inflow 
    lat_event = env.get_subc_inp()
    lat_env = get_env(env.name)(swmm_file=lat_event,global_state=True)
    # Use totalinflow instead of lateral_infow_vol
    lat_env.config['global_state'] = [('nodes','totalinflow')]
    state = lat_env.reset(global_state=True,seq=seq)
    runoff,done = [state],False
    while not done:
        done = lat_env.step()
        state = lat_env.state(seq=seq)
        runoff.append(state)
    runoff = np.squeeze(np.array(runoff),axis=-1) # B,T_in,N

    state = env.reset(event,global_state=True,seq=seq)
    true,preds = [],[]
    done,idx = False,0
    while not done:
        # pred_runoff = env.predict(emul.seq_out)
        pred_runoff = np.expand_dims(runoff[idx+emul.seq_out],axis=0)
        pred,qws = emul.simulate(np.expand_dims(state,-1),pred_runoff,roll=False)

        done = env.step()
        state = env.state(seq=seq)
        true.append(state)
        preds.append(np.squeeze(pred))
        idx += 1
    true = np.array(true)[emul.seq_out:,...]
    true = true[...,-emul.seq_out:,...] if emul.recurrent else true
    return true,np.array(preds)

if __name__ == "__main__":
    args,config = parser('config.yaml')

    # simu_de = {'simulate':True,
    #            'data_dir':'./envs/data/shunqing/5s_60s_flood/',
    #            'seq_in':5,
    #            'seq_out':5,
    #            'if_flood':True}
    # for k,v in simu_de.items():
    #     setattr(args,k,v)

    # train_de = {'train':True,'data_dir':'./envs/data/shunqing/5s_60s_flood/','model_dir':'./model/shunqing/5s_5k_res_loop_bal_flood/','load_data':True,'resnet':True,'seq_in':5,'seq_out':5,'if_flood':True,'balance':0.3}
    # for k,v in train_de.items():
    #     setattr(args,k,v)

    env = get_env(args.env)()
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v & args.act
        setattr(args,k,v)
    
    dG = DataGenerator(env,args.seq_in,args.seq_out,args.recurrent,args.act,args.if_flood,args.data_dir)
    events = generate_file(env.config['swmm_input'],env.config['rainfall'])
    if args.simulate:
        dG.generate(events,processes=args.processes,act=args.act)
        dG.save(args.data_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.data_dir,'parser.yaml'),'w'))
    elif args.load_data or args.train:
        dG.load(args.data_dir)
    
    emul = Emulator(args.conv,args.edges,args.resnet,args.recurrent,args)
        
    if args.train:
        if args.norm:
            emul.set_norm(dG.get_norm())
        train_ids,test_ids,train_losses,test_losses = emul.update_net(dG,args.ratio,args.epochs,args.batch_size)

        # save
        emul.save(args.model_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))
        np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
        np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
        np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
        np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
        plt.plot(train_losses,label='train')
        plt.plot(test_losses,label='test')
        plt.legend()
        plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)

    if args.test:
        emul.load(args.model_dir)
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        yaml.dump(data=config,stream=open(os.path.join(args.result_dir,'parser.yaml'),'w'))
        for event in events:
            name = os.path.basename(event).strip('.inp')
            if os.path.exists(os.path.join(args.result_dir,name + '_states.npy')):
                states = np.load(os.path.join(args.result_dir,name + '_states.npy'))
                perfs = np.load(os.path.join(args.result_dir,name + '_perfs.npy'))
            else:
                states,perfs,settings = dG.simulate(event,act=args.act)
                np.save(os.path.join(args.result_dir,name + '_states.npy'),states)
                np.save(os.path.join(args.result_dir,name + '_perfs.npy'),perfs)

            states[...,1] = states[...,1] - states[...,-1]
            r,true = states[args.seq_out:,...,-1:],states[args.seq_out:,...,:-1]

            states = states[...,:-1]
            if args.if_flood:
                f = (perfs>0).astype(int)
                f = np.eye(2)[f].squeeze(-2)
                states = np.concatenate([states,f],axis=-1)
                true = np.concatenate([true,f[args.seq_out:]],axis=-1)
            pred = emul.simulate(states,r)

            true = np.concatenate([true,perfs[args.seq_out:,...]],axis=-1)  # cumflooding in performance
            if args.recurrent:
                true = true[:,:emul.seq_out,...]
            np.save(os.path.join(args.result_dir,name + '_runoff.npy'),r)
            np.save(os.path.join(args.result_dir,name + '_true.npy'),true)
            np.save(os.path.join(args.result_dir,name + '_pred.npy'),pred)
            
