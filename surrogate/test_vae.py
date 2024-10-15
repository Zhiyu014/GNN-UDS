import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
# from tensorflow_probability.python.distributions import RelaxedOneHotCategorical,Normal
# from tensorflow_probability.python.layers import DistributionLambda
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense,Input,Flatten,GRU,Conv1D,LSTM
from tensorflow import expand_dims,convert_to_tensor,float32,squeeze,GradientTape,transpose,reduce_mean
import numpy as np
from tensorflow.keras.models import Model,Sequential
from spektral.layers import GCNConv,GlobalAttnSumPool,GATConv,DiffusionConv,GeneralConv
import os
from emulator import NodeEdge
from dataloader import DataGenerator
from mbrl import parser
from envs import get_env
import datetime
tf.config.list_physical_devices(device_type='GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress infos and warnings

HERE = os.path.dirname(__file__)

# TODO: No conv and recurrent, directly flatten states
# TODO: Add static features: volume, depth, diameter, slope
class ConvNet:
    def __init__(self,args,conv):
        self.seq_in = getattr(args,"seq_in",None)
        self.seq_out = getattr(args,"seq_out",getattr(args,"setting_duration",None))
        self.recurrent = getattr(args,"recurrent",False)
        self.recurrent = False if self.recurrent in ['None','False','NoneType'] else self.recurrent

        self.hidden_dim = getattr(args,"hidden_dim",128)
        self.n_tp_layer = getattr(args,"n_tp_layer",3)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.conv_dim = getattr(args,"conv_dim",128)
        self.n_sp_layer = getattr(args, "n_sp_layer",3)
        self.lat_dim = getattr(args,"lat_dim",32)

        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        if getattr(args,'if_flood',False):
            self.n_in += 1
        self.b_in = 2 if getattr(args,'tide',False) else 1
        self.graph_base = getattr(args,"graph_base",0)        
        self.adj = getattr(args,"adj",np.eye(self.n_node))
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,3))
        self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
        self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)
        self.activation = getattr(args,"activation",False)
        net = self.get_conv(conv)
        self.encoder = self.build_encoder(net)
        self.decoder = self.build_decoder()
        self.vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.optimizer = Adam(lr=0.001)
        self.mse = MeanSquaredError()

    def get_conv(self,conv):
        if 'GCN' in conv:
            net = GCNConv
            self.filter = GCNConv.preprocess(self.adj)
            self.edge_filter = GCNConv.preprocess(self.edge_adj)
        elif 'Diff' in conv:
            net = DiffusionConv
            self.filter = DiffusionConv.preprocess(self.adj)
            self.edge_filter = DiffusionConv.preprocess(self.edge_adj)
        elif 'GAT' in conv:
            net = GATConv
            self.filter = self.adj.astype(int)
            self.edge_filter = self.edge_adj.astype(int)
        elif 'General' in conv:
            net = GeneralConv
            self.filter = self.adj.astype(int)
            self.edge_filter = self.edge_adj.astype(int)
        else:
            raise AssertionError("Unknown Convolution layer %s"%str(conv))
        return net

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
        
    # input: X,B,Adj,E,Eadj
    # output: X
    def build_encoder(self,net):
        state_shape,bound_shape = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if self.recurrent:
            state_shape = (self.seq_in,) + state_shape
            bound_shape = (self.seq_out,) + bound_shape
        X_in = Input(shape=state_shape)
        B_in = Input(shape=bound_shape)
        Adj_in = Input(shape=(self.n_node,))
        inp = [X_in,B_in,Adj_in]
        
        edge_state_shape = (self.seq_in,self.n_edge,self.e_in,) if self.recurrent else (self.n_edge,self.e_in,)
        E_in = Input(shape=edge_state_shape)
        Eadj_in = Input(shape=(self.n_edge,))
        inp += [E_in,Eadj_in]
        activation = activations.get(self.activation)

        # Embedding
        x = Dense(self.conv_dim,activation=activation)(X_in)
        b = Dense(self.conv_dim//2,activation=activation)(B_in)
        e = Dense(self.conv_dim,activation=activation)(E_in)

        # Spatial block
        x = tf.reshape(x,(-1,) + tuple(x.shape[2:])) if self.recurrent else x
        e = tf.reshape(e,(-1,) + tuple(e.shape[2:])) if self.recurrent else e
        for _ in range(self.n_sp_layer):
            if self.graph_base:
                x = [tf.concat([x,e],axis=-2),Adj_in]
                x = net(self.conv_dim,activation=self.activation)(x)
                x,e = tf.split(x,[self.n_node,self.n_edge],axis=-2)
            else:
                x_e = Dense(self.conv_dim//2,activation=self.activation)(e)
                e_x = Dense(self.conv_dim//2,activation=self.activation)(x)
                x = tf.concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = tf.concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                x = net(self.conv_dim,activation=self.activation)([x,Adj_in])
                e = net(self.conv_dim,activation=self.activation)([e,Eadj_in])

        # (B*T,N,E) (B,N,E)  --> (B,T,N,E) (B,N,E)
        x = tf.reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],))
        # (B*T,N,E)(B,N,E)  --> (B,T,N,E) (B,N,E)
        e = tf.reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],))

        if self.recurrent:
            x = tf.reshape(transpose(x,[0,2,1,3]),(-1,self.seq_in,x.shape[-1]))
            x = self.get_tem_nets(self.recurrent)(x)
            x = x[...,-self.seq_out:,:]
            x = transpose(tf.reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3])
            e = tf.reshape(transpose(e,[0,2,1,3]),(-1,self.seq_in,e.shape[-1]))
            e = self.get_tem_nets(self.recurrent)(e)
            e = e[...,-self.seq_out:,:]
            e = transpose(tf.reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3])
        x = Dense(self.conv_dim,activation=activation)(tf.concat([x,b],axis=-1))
        x = tf.reshape(tf.concat([x,e],axis=-2),(-1,self.n_node+self.n_edge,self.conv_dim))
        x = GlobalAttnSumPool()(x)
        x = tf.reshape(x,(-1,self.seq_out,self.conv_dim))
        x = GRU(self.hidden_dim)(x)
        mu = Dense(self.lat_dim,activation=None)(x)
        logvar = Dense(self.lat_dim,activation=None)(x)
        z = tfp.layers.DistributionLambda(lambda t:tfd.Normal(loc=t[0],scale=tf.exp(0.5*t[1])))([mu,logvar])
        model = Model(inputs=inp, outputs=z)
        return model

    def build_decoder(self):
        h_in = Input(shape=(self.lat_dim,))
        if self.recurrent:
            h = Dense(self.seq_in,activation=self.activation)(h_in)
            h = tf.reshape(h,(-1,self.seq_in,1))
            for _ in range(self.n_tp_layer):
                h = Dense(self.hidden_dim,activation=self.activation)(h)
        # for _ in range(self.n_sp_layer):
        h = Dense(self.n_node*2+self.n_edge,activation=self.activation)(h if self.recurrent else h_in)
        if self.recurrent:
            h = tf.reshape(h,(-1,self.seq_in,self.n_node*2+self.n_edge,1))
        else:
            h = tf.reshape(h,(-1,self.n_node*2+self.n_edge,1))
        for _ in range(self.n_sp_layer):
            h = Dense(self.conv_dim,activation=self.activation)(h)
        x,b,e = tf.split(h,num_or_size_splits=[self.n_node,self.n_node,self.n_edge],axis=-2)
        x = Dense(self.n_in,activation=self.activation)(x)
        b = Dense(self.b_in,activation=self.activation)(b)
        out = [x,b]
        e = Dense(self.e_in,activation=self.activation)(e)
        out += [e]
        model = Model(inputs=h_in, outputs=out)
        return model
    
    def train(self,observ):
        inp = observ[:2] + [self.filter]
        inp += [observ[-1:],self.edge_filter]
        with GradientTape() as tape:
            tape.watch(self.vars)
            distr = self.encoder(inp)
            z = distr.sample()
            recon = self.decoder(z)
            mse_loss = [self.mse(o,r) for o,r in zip(observ,recon)]
            mse_loss = tf.reduce_mean(mse_loss)
            mu,logvar = distr.loc, 2*tf.math.log(distr.scale)
            kl_loss = -0.5 * tf.reduce_sum(logvar + 1 - tf.square(mu) - tf.exp(logvar),axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            loss = mse_loss + kl_loss
        grads = tape.gradient(loss,self.vars)
        self.optimizer.apply_gradients(zip(grads,self.vars))
        return mse_loss.numpy(),kl_loss.numpy()

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        setattr(self,'norm_r',norm_r)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)

    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        if inverse:
            return dat * (normal[0,:,:dim]-normal[1,:,:dim]) + normal[1,:,:dim]
        else:
            return (dat - normal[1,:,:dim])/(normal[0,:,:dim]-normal[1,:,:dim])

if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','policy.yaml'))
    env = get_env(args.env)(initialize=False)

    args.data_dir = './envs/data/astlingen/1s_edge_conti128_rain50'
    args.act = 'conti'
    args.seq_in = 5
    args.seq_out = 5
    args.conv = 'GAT'
    args.recurrent = 'Conv1D'
    args.if_flood = True
    args.episodes = 10000
    args.batch_size = 64

    env_args = env.get_args(act=args.act,mac=args.mac)
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act
        setattr(args,k,v)


    conv = ConvNet(args,args.conv)
    dG = DataGenerator(env.config,args.data_dir,args)
    dG.load(args.data_dir)
    conv.set_norm(*dG.get_norm())

    n_events = int(max(dG.event_id))+1
    train_ids = np.load(os.path.join(args.data_dir,'train_id.npy'))
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]
    events = ['./envs/network/astlingen/astlingen_03_05_2006_01.inp', './envs/network/astlingen/astlingen_07_30_2004_21.inp', './envs/network/astlingen/astlingen_01_13_2002_12.inp', './envs/network/astlingen/astlingen_08_12_2003_08.inp', './envs/network/astlingen/astlingen_10_05_2005_16.inp', './envs/network/astlingen/astlingen_04_12_2003_18.inp', './envs/network/astlingen/astlingen_05_27_2004_06.inp', './envs/network/astlingen/astlingen_12_02_2004_23.inp', './envs/network/astlingen/astlingen_12_28_2006_08.inp', './envs/network/astlingen/astlingen_12_13_2006_23.inp', './envs/network/astlingen/astlingen_03_11_2002_09.inp', './envs/network/astlingen/astlingen_08_11_2003_19.inp', './envs/network/astlingen/astlingen_09_16_2006_05.inp', './envs/network/astlingen/astlingen_03_23_2006_08.inp', './envs/network/astlingen/astlingen_06_13_2000_20.inp', './envs/network/astlingen/astlingen_11_15_2003_17.inp', './envs/network/astlingen/astlingen_02_07_2001_07.inp', './envs/network/astlingen/astlingen_04_17_2005_12.inp', './envs/network/astlingen/astlingen_06_29_2002_07.inp', './envs/network/astlingen/astlingen_05_06_2004_19.inp', './envs/network/astlingen/astlingen_08_21_2001_08.inp', './envs/network/astlingen/astlingen_04_30_2001_09.inp', './envs/network/astlingen/astlingen_03_13_2001_16.inp', './envs/network/astlingen/astlingen_07_27_2000_14.inp', './envs/network/astlingen/astlingen_04_27_2005_00.inp', './envs/network/astlingen/astlingen_08_01_2002_11.inp', './envs/network/astlingen/astlingen_11_28_2006_01.inp', './envs/network/astlingen/astlingen_10_29_2004_11.inp', './envs/network/astlingen/astlingen_07_25_2000_01.inp', './envs/network/astlingen/astlingen_09_11_2006_11.inp', './envs/network/astlingen/astlingen_06_01_2005_10.inp', './envs/network/astlingen/astlingen_02_10_2004_00.inp', './envs/network/astlingen/astlingen_03_07_2003_20.inp', './envs/network/astlingen/astlingen_10_25_2000_13.inp', './envs/network/astlingen/astlingen_12_23_2000_19.inp', './envs/network/astlingen/astlingen_08_08_2005_22.inp', './envs/network/astlingen/astlingen_12_15_2006_17.inp', './envs/network/astlingen/astlingen_04_17_2000_07.inp', './envs/network/astlingen/astlingen_11_12_2005_09.inp', './envs/network/astlingen/astlingen_03_07_2006_18.inp', './envs/network/astlingen/astlingen_10_13_2003_15.inp', './envs/network/astlingen/astlingen_09_26_2002_16.inp', './envs/network/astlingen/astlingen_10_28_2000_08.inp', './envs/network/astlingen/astlingen_10_23_2004_17.inp', './envs/network/astlingen/astlingen_06_11_2006_01.inp', './envs/network/astlingen/astlingen_12_16_2004_17.inp', './envs/network/astlingen/astlingen_03_27_2004_11.inp', './envs/network/astlingen/astlingen_01_04_2004_17.inp', './envs/network/astlingen/astlingen_11_17_2001_18.inp', './envs/network/astlingen/astlingen_04_17_2000_22.inp', './envs/network/astlingen/astlingen_08_22_2006_02.inp']
    train_events,test_events = [events[ix] for ix in train_ids],[events[ix] for ix in test_ids]

    train_losses,eval_losses,train_objss,test_objss,secs = [],[],[],[],[]

    log_dir = "logs/vae/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    for episode in range(args.episodes):
        train_idxs = dG.get_data_idxs(train_ids,args.seq_in)
        train_dats = dG.prepare_batch(train_idxs,args.seq_in,args.batch_size,args.setting_duration)
        x,settings,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
        x_norm,b_norm = [conv.normalize(dat,item) for dat,item in zip([x,b],'xb')]
        s = [x_norm[:,-args.seq_in:,...],b_norm[:,:args.seq_in,...]]
        ex,ey = train_dats[-2:]
        ex_norm,ey_norm = [conv.normalize(dat,'e') for dat in [ex,ey]]
        s += [ex_norm[:,-args.seq_in:,...]]
        mse,kl = conv.train(s)
        print('episode: %d, mse: %.4f, kl: %.4f'%(episode,mse,kl))
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Reconstruction Error', mse, step=episode)
            tf.summary.scalar('KL Divergance', kl, step=episode)
            tf.summary.scalar('Loss', mse+kl, step=episode)