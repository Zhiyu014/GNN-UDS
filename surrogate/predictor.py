from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt,cumsum,tile
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D,Softmax,Add,Subtract,Dropout
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
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
        self.tide = getattr(args,'tide',False)
        self.b_in = 2 if self.tide else 1
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.n_act = len(getattr(args,"act_edges"))
        self.n_out = len(getattr(args,"perforamce_targets"))

        self.seq_in = getattr(args,'seq_in',6)
        self.seq_out = getattr(args,'seq_out',1)

        self.embed_size = getattr(args,'embed_size',64)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.n_sp_layer = getattr(args,"n_sp_layer",3)
        self.n_tp_layer = getattr(args,"n_tp_layer",2)
        self.dropout = getattr(args,'dropout',0.0)
        self.activation = getattr(args,"activation",'relu')
        self.norm = getattr(args,"norm",False)
        # self.balance = getattr(args,"balance",False)
        # self.if_flood = getattr(args,"if_flood",0)
        # if self.if_flood:
        #     self.n_in += 1
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

        self.roll = getattr(args,"roll",0)
        self.model_dir = getattr(args,"model_dir")

    def build_network(self,recurrent):
        b_in = Input(shape=(self.seq_in+self.seq_out,self.n_node,) if recurrent else (self.n_node,))
        if self.act:
            a_in = Input(shape=(self.seq_in+self.seq_out,self.n_act,) if recurrent else (self.n_act,))
            inp = [b_in,a_in]
        else:
            inp = b_in
        x = Dense(self.embed_size,activation=self.activation)(b_in)
        if self.act:
            a = Dense(self.embed_size,activation=self.activation)(a_in)
            x = concat([x,a],axis=-1)
        
        if recurrent:
            for i in range(self.n_tp_layer):
                if recurrent == 'Conv1D':
                    x = Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,
                            activation=self.activation)(x)
                elif recurrent == 'GRU':
                    x = GRU(self.hidden_dim,return_sequences=True)(x)
                    
        out = Dense(self.n_out,activation='hard_sigmoid' if self.norm else 'linear')(x[...,-1,:])
        model = Model(inputs=inp,outputs=out)
        return model

    @tf.function
    def fit_eval(self,b,a,objs,fit=True):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model([b,a] if self.act else b)
            loss = self.mse(objs,pred)
            if fit:
                grads = tape.gradient(loss,self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        setattr(self,'norm_r',norm_r)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)

    def normalize(self,dat,item,inverse=False):
        normal = getattr(self,'norm_%s'%item)
        maxi,mini = normal[0],normal[1]
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

        if self.norm:
            for item in 'xbyre':
                if hasattr(self,'norm_%s'%item):
                    np.save(os.path.join(model_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))

    def load(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))

        if self.norm:
            for item in 'xbyre':
                if os.path.exists(os.path.join(model_dir,'norm_%s.npy'%item)):
                    setattr(self,'norm_%s'%item,np.load(os.path.join(model_dir,'norm_%s.npy'%item)))


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))

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

    emul = Predictor(args.recurrent,args)
    # plot_model(emul.model,os.path.join(args.model_dir,"model.png"),show_shapes=True)
    if args.load_model:
        emul.load(args.model_dir)
        args.model_dir = os.path.join(args.model_dir,'retrain')
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        if 'model_dir' in config:
            config['model_dir'] += '/retrain'
    if args.norm:
        emul.set_norm(*dG.get_norm())
    yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

    t0 = time.time()
    train_losses,test_losses,secs = [],[],[0]
    log_dir = "logs/model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    for epoch in range(args.epochs):
        train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
        b = np.concatenate([x[...,-1:],b],axis=1)
        if args.norm:
            x,b,y = [emul.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
        if args.use_edge:
            ex,ey = [dat for dat in train_dats[-2:]]
            if args.norm:
                ex,ey = [emul.normalize(dat,'e') for dat in [ex,ey]]
        else:
            ex,ey = None,None
        objs = env.objective_preds([y,ey],[x,ex],a)
        train_loss = emul.fit_eval(np.squeeze(b,axis=-1),a,objs)
        train_losses.append(train_loss.numpy())
        # TODO: get max norm values to inversely normalize the objective

        test_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
        b = np.concatenate([x[...,-1:],b],axis=1)
        if args.norm:
            x,b,y = [emul.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
        if args.use_edge:
            ex,ey = [dat for dat in test_dats[-2:]]
            if args.norm:
                ex,ey = [emul.normalize(dat,'e') for dat in [ex,ey]]
        else:
            ex,ey = None,None
        objs = env.objective_preds([y,ey],[x,ex],a)
        test_loss = emul.fit_eval(np.squeeze(b,axis=-1),a,objs)
        test_losses.append(test_loss.numpy())

        if train_loss < min([1e6]+train_losses[:-1]):
            emul.save(os.path.join(args.model_dir,'train'))
        if test_loss < min([1e6]+ test_losses[:-1]):
            emul.save(os.path.join(args.model_dir,'test'))
        if epoch > 0 and epoch % args.save_gap == 0:
            emul.save(os.path.join(args.model_dir,'%s'%epoch))
            
        secs.append(time.time()-t0)

        # Log output
        log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],train_loss,test_loss)
        print(log)
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Train loss', train_loss, step=epoch)
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
