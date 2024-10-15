from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt,cumsum,tile
from tensorflow.keras.layers import Dense,Input,GRU,LSTM,Conv1D,Softmax,Add,Subtract,Dropout
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
from main import parser,HERE
from dataloader import DataGenerator
from envs import get_env

class Predictor:
    def __init__(self,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.tide = getattr(args,'tide',False)
        self.b_in = 2 if self.tide else 1
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.n_act = len(getattr(args,"act_edges"))
            act_edges = [np.where((args.edges==act_edge).all(1))[0] for act_edge in args.act_edges]
            act_edges = [i for e in act_edges for i in e]
            self.act_edges = sorted(list(set(act_edges)),key=act_edges.index)

        self.targets = getattr(args,"performance_targets")
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
        self.is_outfall = getattr(args,"is_outfall",np.array([0 for _ in range(self.n_node)]))
        # self.epsilon = getattr(args,"epsilon",-1.0)

        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))
        self.hmin = getattr(args,"hmin",np.array([0.0 for _ in range(self.n_node)]))
        self.area = getattr(args,"area",np.array([0.0 for _ in range(self.n_node)]))
        self.nwei = getattr(args,"nwei",np.array([1.0 for _ in range(self.n_node)]))
        self.pump_in = getattr(args,"pump_in",np.array([0.0 for _ in range(self.n_node)]))
        self.pump_out = getattr(args,"pump_out",np.array([0.0 for _ in range(self.n_node)]))        

        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.model = self.build_network(self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

        self.roll = getattr(args,"roll",0)
        self.model_dir = getattr(args,"model_dir")

    def build_network(self,recurrent):
        # x_in = Input(shape=(self.seq_in,self.n_node*self.n_in) if recurrent else (self.n_node*self.n_in,))
        # x = Dense(self.embed_size,activation=self.activation)(x_in)
        # e_in = Input(shape=(self.seq_in,self.n_edge*self.e_in) if recurrent else (self.n_edge*self.e_in,))
        # inp = [x_in,e_in]
        # e = Dense(self.embed_size,activation=self.activation)(e_in)
        # x = concat([x,e],axis=-1)
        # else:
        #     inp = [x_in]
        # for _ in range(self.n_sp_layer):
        #     x = Dense(self.embed_size,activation=self.activation)(x)
        h_in = Input(shape=(self.seq_in,self.n_flood,) if recurrent else (self.n_flood,))
        h = Dense(self.embed_size,activation=self.activation)(h_in)
        b_in = Input(shape=(self.seq_in+self.seq_out,self.n_node*self.b_in,) if recurrent else (self.n_node*self.b_in,))
        inp = [h_in,b_in]
        if self.act:
            a_in = Input(shape=(self.seq_in+self.seq_out,self.n_act,) if recurrent else (self.n_act,))
            inp += [a_in]
        b = Dense(self.embed_size,activation=self.activation)(b_in)
        # x = concat([x,b],axis=-1)
        if self.act:
            a = Dense(self.embed_size,activation=self.activation)(a_in)
            b = tf.concat([b,a],axis=-1)
        for _ in range(self.n_sp_layer):
            b = Dense(self.embed_size,activation=self.activation)(b)
        if recurrent:
            x = self.get_tem_nets(recurrent)(tf.concat([b[:,:self.seq_in],h],axis=-1))
            x = self.get_tem_nets(recurrent)(tf.concat([x,b[:,-self.seq_out:]],axis=-1))
                    
        out = Dense(self.n_out,activation='hard_sigmoid')(x)
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
    def fit_eval(self,h,b,a,objs,fit=True):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model([h,b,a] if self.act else [h,b])
            if self.if_flood:
                pred,flood = pred
                loss = self.bce(tf.cast(objs[...,:self.n_flood]>0,tf.float32),flood)
                if not fit:
                    loss = [loss]
                pred = tf.concat([pred[...,:self.n_flood] * tf.cast(flood>0.5,tf.float32),
                                  pred[...,self.n_flood:]],axis=-1)
                loss += self.mse(objs,pred) if fit else [self.mse(objs,pred)]
            else:
                loss = self.mse(objs,pred)
            if fit:
                grads = tape.gradient(loss,self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss
    
    def predict(self,state,runoff,settings=None,edge_state=None):
        x,b = [self.normalize(dat,item) for dat,item in zip([state,runoff],'xb')]
        h = tf.gather(x[...,0],self.flood_nodes,axis=-1)
        b = tf.squeeze(tf.concat([x[...,-1:],b],axis=1),axis=-1)
        if self.act:
            settings = tf.concat([tf.gather(edge_state[...,-1],self.act_edges,axis=-1),settings],axis=1)
        preds = self.model([h,b,settings] if self.act else [h,b])
        if self.if_flood:
            preds,flood = preds
            preds = tf.concat([preds[...,:self.n_flood] * tf.cast(flood>0.5,tf.float32),
                               preds[...,self.n_flood:]],axis=-1)
        objs = self.normalize(preds,'o',inverse=True)
            # objs = preds.numpy() * state[...,-1].sum(axis=-1).sum(axis=-1)[:,None]
        return objs

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        setattr(self,'norm_r',norm_r)
        if norm_e is not None:
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
        'train':True,
        'env':'chaohu',
        'data_dir':'./envs/data/chaohu/1s_edge_rand64_rain50/',
        'act':'conti',
        'model_dir':'./model/chaohu/60s_50k_rand_pred/',
        'load_model':False,
        'setting_duration':10,
        'batch_size':64,
        'epochs':10000,
        'n_sp_layer':5,
        'n_tp_layer':5,
        'norm':True,
        'seq_in':60,'seq_out':60,
        # 'if_flood':3,
        'recurrent':'LSTM',
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

    seq = max(args.seq_in,args.seq_out) if args.recurrent else 0
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
        x,b,objs = [emul.normalize(dat,item) for dat,item in zip([x,b,objs],'xbo')]
        b = tf.squeeze(tf.concat([x[...,-1:],b],axis=1),axis=-1)
        a = tf.concat([tf.gather(ex[...,-1],emul.act_edges,axis=-1),a],axis=1)
        h = tf.gather(x[...,0],emul.flood_nodes,axis=-1)
        train_loss = emul.fit_eval(h,b,a,objs)
        train_losses.append(train_loss.numpy())

        test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
        ex,ey = [dat for dat in test_dats[-2:]]
        objs = env.objective_pred([y,ey],[x,ex],a,keepdim=True)
        x,b,objs = [emul.normalize(dat,item) for dat,item in zip([x,b,objs],'xbo')]
        b = tf.squeeze(tf.concat([x[...,-1:],b],axis=1),axis=-1)
        a = tf.concat([tf.gather(ex[...,-1],emul.act_edges,axis=-1),a],axis=1)
        h = tf.gather(x[...,0],emul.flood_nodes,axis=-1)
        test_loss = emul.fit_eval(h,b,a,objs,fit=False)
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
