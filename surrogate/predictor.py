from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt,cumsum,tile
from tensorflow.keras.layers import Dense,Input,GRU,LSTM,Conv1D,Flatten
from tensorflow.keras import activations
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError,BinaryCrossentropy
from tensorflow.keras import mixed_precision
import numpy as np
import os
# from line_profiler import LineProfiler
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')

import yaml,time,datetime,matplotlib.pyplot as plt
from main import Argument,HERE
from dataloader import DataGenerator
from envs import get_env

class ArgumentPredictor(Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--full',action='store_true',help='if use full-scale input')
        self.add_argument('--norm',action='store_true',help='if norm targets individually')
        self.add_argument('--vol',action='store_true',help='if only fit volume-based objectives')
        self.add_argument('--cum',action='store_true',help='if fit cumulative values of the horizon')

def parser(config=None):
    parser = ArgumentPredictor(description='prediction')
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

class Predictor:
    def __init__(self,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.moni_nodes = [(args.elements['nodes'].index(idx),args.attrs['nodes'].index(attr))
                           for idx,attr in args.states if attr in args.attrs['nodes']]
        self.moni_links = [(args.elements['links'].index(idx),args.attrs['links'].index(attr))
                           for idx,attr in args.states if attr in args.attrs['links']]
        self.n_moni = len(self.moni_nodes) + len(self.moni_links)
        self.tide = getattr(args,'tide',False)
        self.b_in = 2 if self.tide else 1
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.n_act = len(getattr(args,"act_edges"))
            act_edges = [np.where((args.edges==act_edge).all(1))[0] for act_edge in args.act_edges]
            act_edges = [i for e in act_edges for i in e]
            self.act_edges = sorted(list(set(act_edges)),key=act_edges.index)

        self.full = getattr(args,'full',False)
        self.norm = getattr(args,'norm',False)
        self.vol = getattr(args,'vol',False)
        self.cum = getattr(args,'cum',False)
        self.targets = getattr(args,"performance_targets")
        if self.vol:
            self.n_out = len([i for i,attr,weight in self.targets
                               if attr in ['cumflooding','cuminflow'] and weight>=0.5])
        else:
            self.n_out = len(self.targets)
        self.flood_nodes = [args.elements['nodes'].index(i)
                             for i,attr,_ in self.targets 
                             if attr == 'cumflooding' and i in args.elements['nodes']]
        self.flood_nodes = list(set(self.flood_nodes + np.where(args.area>0)[0].tolist()))
        self.if_flood = getattr(args,"if_flood",0)
        self.n_flood = len(self.flood_nodes)
        if self.if_flood:
            self.n_in += 1

        self.seq_in = getattr(args,'seq_in',6)
        self.seq_out = getattr(args,'seq_out',1)

        self.embed_size = getattr(args,'embed_size',128)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.n_sp_layer = getattr(args,"n_sp_layer",3)
        self.n_tp_layer = getattr(args,"n_tp_layer",3)
        self.dropout = getattr(args,'dropout',0.0)
        self.activation = getattr(args,"activation",'relu')

        self.recurrent = recurrent
        self.model = self.build_network(self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

        self.model_dir = getattr(args,"model_dir")

    def build_network(self,recurrent):
        if self.full:
            x_in = Input(shape=(self.seq_in,self.n_node*self.n_in))
            e_in = Input(shape=(self.seq_in,self.n_edge*self.e_in))
            inp = [x_in,e_in]
            x = Dense(self.embed_size,activation=self.activation)(x_in)
            e = Dense(self.embed_size,activation=self.activation)(e_in)
            x = concat([x,e],axis=-1)
            b_in = Input(shape=(self.seq_out,self.n_node*self.b_in,))
            inp += [b_in]
            b = Dense(self.embed_size,activation=self.activation)(b_in)
        else:
            x_in = Input(shape=(self.seq_in,self.n_moni,))
            b_in = Input(shape=(self.seq_in+self.seq_out,self.n_node*self.b_in,))
            inp = [x_in,b_in]
            x = Dense(self.embed_size,activation=self.activation)(x_in)
            b = Dense(self.embed_size,activation=self.activation)(b_in)
        if self.act:
            a_in = Input(shape=(self.seq_out if self.full else self.seq_in+self.seq_out,self.n_act,))
            inp += [a_in]
            a = Dense(self.embed_size,activation=self.activation)(a_in)
            b = tf.concat([b,a],axis=-1)
        for _ in range(self.n_sp_layer):
            x = Dense(self.embed_size,activation=self.activation)(x)
            b = Dense(self.embed_size,activation=self.activation)(b)
        x = self.get_tem_nets(recurrent)(x if self.full else tf.concat([x,b[:,:self.seq_in,:]],axis=-1))
        x = self.get_tem_nets(recurrent)(tf.concat([x,b if self.full else b[:,-self.seq_out:,:]],axis=-1))
                    
        out = Dense(self.n_out,activation='linear')(x)
        if self.if_flood:
            for _ in range(self.if_flood):
                fl = Dense(self.embed_size,activation=self.activation)(x)
            flood = Dense(self.n_flood,activation='sigmoid')(fl)
            out = [out,flood]
        model = Model(inputs=inp,outputs=out)
        return model
    
    def get_tem_nets(self,recurrent):
        if recurrent == 'Conv1D':
            return Sequential([Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,
                            activation=self.activation) for i in range(self.n_tp_layer)])
        elif recurrent == 'GRU':
            return Sequential([GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)])
        elif recurrent == 'LSTM':
            return Sequential([LSTM(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)])
        else:
            return Sequential()

    @tf.function
    def fit_eval(self,x,e,b,a,objs,fit=True):
        if self.norm:
            objs = self.normalize(objs,'o')
        x,b,e = [self.normalize(dat,item) for dat,item in zip([x,b,e],'xbe')]
        if self.full:
            inp = [tf.reshape(dat,dat.shape[:2]+(np.prod(dat.shape[2:]),)) for dat in [x,e,b]]
            inp += [a] if self.act else []
        else:
            moni = tf.stack([x[...,i,j] for i,j in self.moni_nodes]+[e[...,i,j] for i,j in self.moni_links],axis=-1)
            inp = [moni, tf.concat([x[...,-1],tf.squeeze(b)],axis=1)]
            inp += [tf.concat([tf.gather(e[...,-1],self.act_edges,axis=-1),a],axis=1)] if self.act else []
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model(inp)
            if self.if_flood:
                pred,flood = pred
                loss = self.bce(tf.cast(objs[...,:self.n_flood]>0,tf.float32),flood)
                if not fit:
                    loss = [loss]
                pred = tf.concat([pred[...,:self.n_flood] * tf.cast(flood>0.5,tf.float32),
                                  pred[...,self.n_flood:]],axis=-1)
                if self.cum:
                    pred,objs = [tf.reduce_sum(dat,axis=-2) for dat in [pred,objs]]
                loss += self.mse(objs[...,:self.n_out],pred) if fit else [self.mse(objs[...,:self.n_out],pred)]
            else:
                if self.cum:
                    pred,objs = [tf.reduce_sum(dat,axis=-2) for dat in [pred,objs]]
                loss = self.mse(objs[...,:self.n_out],pred)
            if fit:
                grads = tape.gradient(loss,self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss
    
    def predict(self,state,runoff,settings=None,edge_state=None):
        x,b,e = [self.normalize(dat,item) for dat,item in zip([state,runoff,edge_state],'xbe')]
        if self.full:
            x,b,e = [tf.reshape(dat,dat.shape[:2]+(-1,)) for dat in [x,b,e]]
            inp = [x,e,b,settings] if self.act else [x,e,b]
        else:
            moni = tf.stack([x[...,i,j] for i,j in self.moni_nodes]+[e[...,i,j] for i,j in self.moni_links],axis=-1)
            inp = [moni, tf.squeeze(tf.concat([x[...,-1:],b],axis=1),axis=-1)]
            if self.act:
                inp += [tf.concat([tf.gather(edge_state[...,-1],self.act_edges,axis=-1),settings],axis=1)]
        preds = self.model(inp)
        if self.if_flood:
            preds,flood = preds
            preds = tf.concat([preds[...,:self.n_flood] * tf.cast(flood>0.5,tf.float32),
                               preds[...,self.n_flood:]],axis=-1)
        if self.norm:
            preds = self.normalize(preds,'o',inverse=True)
        return preds

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        setattr(self,'norm_r',norm_r)
        setattr(self,'norm_e',norm_e)

    def normalize(self,dat,item,inverse=False):
        normal = getattr(self,'norm_%s'%item)
        maxi,mini = normal[0,...,:dat.shape[-1]],normal[1,...,:dat.shape[-1]]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)

    def save(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if model_dir.endswith('.h5'):
            self.model.save_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.save_weights(os.path.join(model_dir,'model.h5'))

        for item in 'xbyreo':
            if hasattr(self,'norm_%s'%item):
                np.save(os.path.join(model_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))

    def load(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))

        for item in 'xbyreo':
            if os.path.exists(os.path.join(model_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,np.load(os.path.join(model_dir,'norm_%s.npy'%item)))


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))

    train_de = {
        # 'train':True,
        # 'env':'chaohu',
        # 'data_dir':'./envs/data/chaohu/1s_edge_rand64_rain50/',
        # 'act':'conti',
        # 'model_dir':'./model/chaohu/60s_50k_rand_pred_fitone/',
        # 'load_model':False,
        # 'setting_duration':5,
        # 'batch_size':64,
        # 'epochs':10000,
        # 'n_sp_layer':5,
        # 'n_tp_layer':5,
        # 'seq_in':60,'seq_out':60,
        # # 'if_flood':3,
        # 'recurrent':'LSTM',
        }
    for k,v in train_de.items():
        setattr(args,k,v)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)
    
    dG = DataGenerator(env.config,args.data_dir,args)
    dG.load(args.data_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    seq = max(args.seq_in,args.seq_out)
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

    # Data balance: Only use flooding steps
    # nodes = [args.elements['nodes'].index(node) for node,attr,_ in args.performance_targets if 'flooding' in attr and node != 'T1']
    # iys = np.apply_along_axis(lambda t:np.arange(t,t+seq),axis=1,arr=np.expand_dims(train_idxs,axis=-1))
    # train_idxs = train_idxs[np.take(dG.perfs[:,nodes,-1].sum(axis=-1),iys,axis=0).sum(axis=-1)>0]
    # iys = np.apply_along_axis(lambda t:np.arange(t,t+seq),axis=1,arr=np.expand_dims(test_idxs,axis=-1))
    # test_idxs = test_idxs[np.take(dG.perfs[:,nodes,-1].sum(axis=-1),iys,axis=0).sum(axis=-1)>0]

    emul = Predictor(args.recurrent,args)
    # plot_model(emul.model,os.path.join(args.model_dir,"model.png"),show_shapes=True)
    if args.load_model:
        emul.load(args.model_dir)
        args.model_dir = os.path.join(args.model_dir,'retrain')
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        if 'model_dir' in config:
            config['model_dir'] += '/retrain'
    emul.set_norm(*dG.get_norm())
    emul.norm_o = env.get_obj_norm(emul.norm_y,emul.norm_e,dG.perfs)
    yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

    t0 = time.time()
    train_losses,test_losses,secs = [],[],[0]
    log_dir = "logs/model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    for epoch in range(args.epochs):
        train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
        ex,ey = [dat for dat in train_dats[-2:]]
        objs = env.objective_pred([y,ey],[x,ex],a,keepdim=True)
        if not args.norm:
            objs = env.norm_obj(objs,[x,ex])
        train_loss = emul.fit_eval(x,ex,b,a,objs)
        train_losses.append(train_loss.numpy())

        test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
        ex,ey = [dat for dat in test_dats[-2:]]
        objs = env.objective_pred([y,ey],[x,ex],a,keepdim=True)
        if not args.norm:
            objs = env.norm_obj(objs,[x,ex])
        test_loss = emul.fit_eval(x,ex,b,a,objs,fit=False)
        test_losses.append([loss.numpy() for loss in test_loss] if args.if_flood else test_loss.numpy())

        if train_loss < min([1e6]+train_losses[:-1]):
            emul.save(os.path.join(args.model_dir,'train'))
        if args.if_flood and sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
            emul.save(os.path.join(args.model_dir,'test'))
        if not args.if_flood and test_loss < min([1e6]+ test_losses[:-1]):
            emul.save(os.path.join(args.model_dir,'test'))
        if epoch > 0 and epoch % args.save_gap == 0:
            emul.save(os.path.join(args.model_dir,'%s'%epoch))
            
        secs.append(time.time()-t0)

        # Log output
        log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} ".format(epoch,args.epochs,secs[-1]-secs[-2],train_loss)
        if args.if_flood:
            log += "Test loss: {:.4f} (flood: {:.4f}, mse: {:.4f})".format(sum(test_loss),test_loss[0],test_loss[1])
        else:
            log += "Test loss: {:.4f}".format(test_loss)
        print(log)
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Train loss', train_loss, step=epoch)
            if args.if_flood:
                tf.summary.scalar('Test loss', sum(test_loss), step=epoch)
                tf.summary.scalar('Flood loss', test_loss[0], step=epoch)
                tf.summary.scalar('MSE loss', test_loss[1], step=epoch)
            else:
                tf.summary.scalar('Test loss', test_loss, step=epoch)

    # save
    emul.save(args.model_dir)
    np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
    np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
    np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
    np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
    np.save(os.path.join(args.model_dir,'time.npy'),np.array(secs[1:]))
    plt.plot(train_losses,label='train')
    plt.plot(test_losses,label='test')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)
