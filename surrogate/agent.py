import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError,SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
# from tensorflow_probability.python.distributions import RelaxedOneHotCategorical,Normal
# from tensorflow_probability.python.layers import DistributionLambda
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense,Input,Lambda,GRU,Conv1D
from tensorflow import expand_dims,convert_to_tensor,float32,squeeze,GradientTape,transpose,reduce_mean
import numpy as np
from numpy import array,save,load,concatenate
from tensorflow.keras.models import Model
from spektral.layers import GCNConv,GlobalAttnSumPool,GATConv,DiffusionConv,GeneralConv
from os.path import join
import os
from emulator import NodeEdge,Emulator
# from envs import get_env

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

        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        if getattr(args,'if_flood',False):
            self.n_in += 1
        self.b_in = 2 if getattr(args,'tide',False) else 1
        self.use_edge = getattr(args,"use_edge",False)
        self.adj = getattr(args,"adj",np.eye(self.n_node))
        if self.use_edge:
            self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,3))
            self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
            self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)
        self.activation = getattr(args,"activation",False)
        net = self.get_conv(conv)
        self.model = self.build_network(net)

    def get_conv(self,conv):
        if 'GCN' in conv:
            net = GCNConv
            self.filter = GCNConv.preprocess(self.adj)
            if self.use_edge:
                self.edge_filter = GCNConv.preprocess(self.edge_adj)
        elif 'Diff' in conv:
            net = DiffusionConv
            self.filter = DiffusionConv.preprocess(self.adj)
            if self.use_edge:
                self.edge_filter = DiffusionConv.preprocess(self.edge_adj)
        elif 'GAT' in conv:
            net = GATConv
            self.filter = self.adj.astype(int)
            if self.use_edge:
                self.edge_filter = self.edge_adj.astype(int)
        elif 'General' in conv:
            net = GeneralConv
            self.filter = self.adj.astype(int)
            if self.use_edge:
                self.edge_filter = self.edge_adj.astype(int)
        else:
            raise AssertionError("Unknown Convolution layer %s"%str(conv))
        return net

    # input: X,B,Adj,E,Eadj
    # output: X
    def build_network(self,net):
        state_shape,bound_shape = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if self.recurrent:
            state_shape = (self.seq_in,) + state_shape
            bound_shape = (self.seq_out,) + bound_shape
        X_in = Input(shape=state_shape)
        B_in = Input(shape=bound_shape)
        Adj_in = Input(shape=(self.n_node,self.n_node,))
        inp = [X_in,B_in,Adj_in]
        
        if self.use_edge:
            edge_state_shape = (self.seq_in,self.n_edge,self.e_in,) if self.recurrent else (self.n_edge,self.e_in,)
            E_in = Input(shape=edge_state_shape)
            Eadj_in = Input(shape=(self.n_edge,self.n_edge,))
            inp += [E_in,Eadj_in]
        activation = activations.get(self.activation)

        # Embedding
        x = Dense(self.conv_dim,activation=activation)(X_in)
        b = Dense(self.conv_dim//2,activation=activation)(B_in)
        if self.use_edge:
            e = Dense(self.conv_dim,activation=activation)(E_in)

        # Spatial block
        x = tf.reshape(x,(-1,) + tuple(x.shape[2:])) if self.recurrent else x
        if self.use_edge:
            e = tf.reshape(e,(-1,) + tuple(e.shape[2:])) if self.recurrent else e
        for _ in range(self.n_sp_layer):
            if self.use_edge:
                x_e = Dense(self.conv_dim//2,activation=self.activation)(e)
                e_x = Dense(self.conv_dim//2,activation=self.activation)(x)
                x = tf.concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = tf.concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
            x = [x,Adj_in]
            x = net(self.conv_dim,activation=self.activation)(x)
            if self.use_edge:
                e = [e,Eadj_in]
                e = net(self.conv_dim,activation=self.activation)(e)

        # (B*T,N,E) (B,N,E)  --> (B,T,N,E) (B,N,E)
        x = tf.reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],))
        if self.use_edge:
            # (B*T,N,E)(B,N,E)  --> (B,T,N,E) (B,N,E)
            e = tf.reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],))

        if self.recurrent:
            if self.recurrent == 'Conv1D':
                x_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:]) for i in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=e.shape[-2:]) for i in range(self.n_tp_layer)]
            elif self.recurrent == 'GRU':
                x_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
            else:
                raise AssertionError("Unknown recurrent layer %s"%str(self.recurrent))
            x = tf.reshape(transpose(x,[0,2,1,3]),(-1,self.seq_in,x.shape[-1]))
            for i in range(self.n_tp_layer):
                x = x_tem_nets[i](x)
            x = x[...,-self.seq_out:,:]
            x = transpose(tf.reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3])
            if self.use_edge:
                e = tf.reshape(transpose(e,[0,2,1,3]),(-1,self.seq_in,e.shape[-1]))
                for i in range(self.n_tp_layer):
                    e = e_tem_nets[i](e)
                e = e[...,-self.seq_out:,:]
                e = transpose(tf.reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3])
        x = Dense(self.conv_dim,activation=activation)(tf.concat([x,b],axis=-1))
        x = tf.reshape(tf.concat([x,e],axis=-2),(-1,self.n_node+self.n_edge,self.conv_dim))
        x = GlobalAttnSumPool()(x)
        x = tf.reshape(x,(-1,self.seq_out,self.conv_dim))
        model = Model(inputs=inp, outputs=x)
        return model

class Actor:
    def __init__(self,
                 action_shape,
                 observ_size,
                 args,
                 conv = None,
                 act_only = True,
                 margs = None):
        self.action_shape = action_shape
        self.observ_size = observ_size

        self.mac = getattr(args, "mac", False)
        self.n_agents = getattr(args, "n_agents", 1)
        self.seq_in = getattr(args,"seq_in",1)
        self.setting_duration = getattr(args,"setting_duration",1)
        self.seq_out = getattr(args,"seq_out",self.setting_duration)
        self.n_step = self.seq_out//self.setting_duration
        self.r_step = self.setting_duration//args.interval
        recurrent = getattr(args,"recurrent",False)
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
        self.action_space = [tf.convert_to_tensor(space,dtype=tf.float32) for space in getattr(args,'action_space',{}).values()]
        # self.env = get_env(args.env)(initialize=False)

        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)
        self.kernel_size = getattr(args,"kernel_size",5)
        self.activation = getattr(args,"activation",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if self.conv in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        elif getattr(args,'if_flood',False):
            self.observ_size += 1
        self.model = self.build_pi_network(self.convnet.model)
        self.target_model = self.build_pi_network(self.convnet.model)
        self.agent_dir = args.agent_dir
        if not act_only:
            # self.gamma = getattr(args, "gamma", 0.98)
            self.gamma = tf.convert_to_tensor([getattr(args, "gamma", 0.98)**i for i in range(self.n_step) for _ in range(self.r_step)])
            self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-4))
            self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            self.emul.load(margs.model_dir)
        if args.load_agent:
            self.load()

    def build_pi_network(self,conv=None):
        if conv is None:
            input_shape = (self.seq_in,self.observ_size) if self.recurrent else (self.observ_size,)
            x_in = Input(shape=input_shape)
            x = x_in
        else:
            x_in = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(x_in)
        for _ in range(self.n_layer):
            x = Dense(self.net_dim, activation=self.activation)(x)
        if self.recurrent:
            x = GRU(self.hidden_dim)(x)
        if self.conti:
            mu = Dense(self.action_shape, activation='sigmoid')(x)
            log_std = Dense(self.action_shape, activation='linear')(x)
            # low = tf.convert_to_tensor([min(sp) for sp in self.action_space])
            # high = tf.convert_to_tensor([max(sp) for sp in self.action_space])
            # output = tfp.layers.DistributionLambda(lambda t: tfd.TruncatedNormal(loc=t[0],scale=tf.exp(t[1]),
            #                                                                      low=tf.ones_like(t[0])*low,high=tf.ones_like(t[0])*high))([mu,log_std])
            output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[0],scale=tf.exp(t[1])))([mu,log_std])
        elif isinstance(self.action_shape,np.ndarray):
            output = [Dense(act_shape, activation='softmax')(x) for act_shape in self.action_shape]
            output = [tfp.layers.DistributionLambda(lambda t: tfd.RelaxedOneHotCategorical(1.0,probs=t))(o) for o in output]
        else:
            output = Dense(self.action_shape, activation='softmax')(x)
            output = tfp.layers.DistributionLambda(lambda t: tfd.RelaxedOneHotCategorical(1.0,probs=t))(output)
        model = Model(inputs=x_in, outputs=output)
        return model

    def get_input(self,observ):
        if self.conv:
            inp = observ[:2] + [self.convnet.filter]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = observ
        return inp
    
    def control(self, observ,train=False):
        observ = [self.normalize(dat,item) if dat is not None else None
                   for dat,item in zip(observ,'xbe')]
        observ = [ob[tf.newaxis,...] for ob in observ]
        action = self.get_action(observ,train)
        settings = self.convert_action_to_setting(action)
        # settings = tf.repeat(settings,self.r_step,axis=-2)
        # inp = self.get_input(observ)
        # pi = self.model(inp)
        # pi = [squeeze(pii).numpy().tolist() for pii in pi] if isinstance(pi,list) else squeeze(pi).numpy().tolist()
        return tf.squeeze(settings).numpy()

    def forward(self, observ, target=False):
        inp = self.get_input(observ)
        probs = self.model(inp) if not target else self.target_model(inp)
        return probs

    def get_action(self, observ, train = True):
        distr = self.forward(observ)
        if self.conti:
            return tf.tanh(distr.sample()) if train else tf.tanh(distr.loc)
        elif isinstance(distr,list):
            return [tf.argmax(distri.sample(),axis=-1) if train else tf.argmax(distri.probs,axis=-1) for distri in distr]
        else:
            return tf.argmax(distr.sample(),axis=-1) if train else tf.argmax(distr.probs,axis=-1)

    def get_action_probs(self, observ, target=False):
        distr = self.forward(observ,target)
        if isinstance(distr,list):
            a = [distri.sample() for distri in distr]
            logp_action = [distri.log_prob(act) for distri,act in zip(distr,a)]
        else:
            a = distr.sample()
            logp_action = distr.log_prob(a)
            if self.conti:
                a = tf.tanh(a)   # Restrict action to be between -1 and 1
                logp_action -= tf.math.log(1.000001-tf.pow(a,2))   # Adjusted Log Probability due to tanh
        return a,logp_action
        
    def convert_action_to_setting(self,action):
        if self.conti:
            return (action+1)/2
        elif isinstance(action,list):
            return tf.stack([tf.gather(space,ai) for space,ai in zip(self.action_space,action)],axis=-1)
        else:
            return tf.gather(self.action_space,action)
        
    def convert_setting_to_action(self,setting):
        if self.conti:
            return tf.multiply(setting,2)-1
        elif isinstance(self.action_space,list):
            return [tf.argmin(tf.abs(setting[...,i]-space),axis=-1) for i,space in enumerate(self.action_space)]
        else:
            return tf.argmin(tf.abs(setting-self.action_space),axis=-1)

    def _hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self,tau):
        target_model_weights = array(self.target_model.get_weights())
        model_weights = array(self.model.get_weights())
        new_weight = (1. - tau) * target_model_weights \
            + tau * model_weights
        self.target_model.set_weights(new_weight)
    
    def save(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        self.model.save_weights(join(agent_dir,'actor%s.h5'%i))
        self.target_model.save_weights(join(agent_dir,'target_actor%s.h5'%i))
        for item in 'xbye':
            if hasattr(self,'norm_%s'%item):
                np.save(join(agent_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))
            
    def load(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.load_weights(join(agent_dir,'actor%s.h5'%i))
        self.target_model.load_weights(join(agent_dir,'target_actor%s.h5'%i))
        for item in 'xbye':
            setattr(self,'norm_%s'%item,np.load(join(agent_dir,'norm_%s.npy'%item)))
            
    def set_norm(self,norm_x,norm_b,norm_y,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)

    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        if inverse:
            return dat * (normal[0,:,:dim]-normal[1,:,:dim]) + normal[1,:,:dim]
        else:
            return (dat - normal[1,:,:dim])/(normal[0,:,:dim]-normal[1,:,:dim])

class QAgent:
    def __init__(self,
                 action_shape,
                 observ_size,
                 args,
                 conv=None):
        self.action_shape = action_shape
        self.observ_size = observ_size

        self.seq_in = getattr(args,"seq_in",1)
        self.setting_duration = getattr(args,"setting_duration",5)
        self.seq_out = getattr(args,"seq_out",self.setting_duration)
        recurrent = getattr(args,"recurrent",False)
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent

        self.hidden_dim = getattr(args,"hidden_dim",128)
        self.n_tp_layer = getattr(args,"n_tp_layer",3)
        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')

        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.activation = getattr(args,"activation",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if self.conv in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        elif getattr(args,'if_flood',False):
            self.observ_size += 1
        self.model = self.build_q_network(self.convnet.model)
        self.target_model = self.build_q_network(self.convnet.model)
        self.target_model.set_weights(self.model.get_weights())
        self.agent_dir = args.agent_dir
        
    def build_q_network(self,conv=None):
        if conv is None:
            input_shape = (self.seq_in,self.observ_size) if self.recurrent else (self.observ_size,)
            inp = Input(shape=input_shape)
            x = inp
        else:
            inp = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(inp)
        if self.conti:
            a_dim = sum(self.action_shape) if isinstance(self.action_shape,(np.ndarray,list)) else self.action_shape
            a_in = Input(shape=(self.seq_out,a_dim) if self.recurrent else (a_dim,))
            a = Dense(self.net_dim, activation=self.activation)(a_in)
            x = tf.concat([x,a],axis=-1)
            inp = inp + [a_in] if isinstance(inp,list) else [inp,a_in]
        for _ in range(self.n_layer):
            x = Dense(self.net_dim, activation=self.activation)(x)
        if self.recurrent:
            x = GRU(self.hidden_dim)(x)
        # out_dim = 1
        if isinstance(self.action_shape,(np.ndarray,list)):
            output = [Dense(dim, activation='linear')(x) for dim in self.action_shape]
        else:
            output = Dense(self.action_shape, activation='linear')(x)
        model = Model(inputs=inp, outputs=output)
        return model
    
    def get_input(self,observ,act):
        if self.conv:
            inp = observ[:2] + [self.convnet.filter]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = [observ]
        if self.conti:
            # if isinstance(act,list):
            #     act = tf.concat(act,axis=-1)
            if self.recurrent:
                act = tf.repeat(act[:,tf.newaxis,:] if len(act.shape) < 3 else act,self.seq_out,axis=1)
            inp += [act]
        return inp

    def forward(self,observ,act=None,target=False):
        inp = self.get_input(observ,act)
        q = self.target_model(inp) if target else self.model(inp)
        if not self.conti and act is not None:
            if isinstance(self.action_shape,(np.ndarray,list)):
                q = tf.stack([tf.gather(qi,ai,axis=-1,batch_dims=1)
                     for qi,ai in zip(q,act)],axis=-1)
            else:
                q = tf.gather(q,act,axis=-1,batch_dims=1)
        return q
    
    def _hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self,tau):
        target_model_weights = array(self.target_model.get_weights())
        model_weights = array(self.model.get_weights())
        new_weight = (1. - tau) * target_model_weights \
            + tau * model_weights
        self.target_model.set_weights(new_weight)
    
    def save(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.save_weights(join(agent_dir,'qnet%s.h5'%i))
        self.target_model.save_weights(join(agent_dir,'qnet%s_target.h5'%i))
            
    def load(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.load_weights(join(agent_dir,'qnet%s.h5'%i))
        self.target_model.load_weights(join(agent_dir,'qnet%s_target.h5'%i))

class Agent:
    def __init__(self,
            action_shape,
            observ_space,
            args = None,
            act_only = False,
            margs = None):
        self.action_shape = action_shape
        self.observ_space = observ_space

        self.mac = getattr(args, "mac", False)
        self.n_agents = getattr(args, "n_agents", 1)
        self.state_shape = getattr(args, "state_shape", 10)
        self.horizon = getattr(args,"horizon",60)
        self.setting_duration = getattr(args,"setting_duration",5)
        self.seq_in = getattr(args,"seq_in",1)
        self.seq_out = self.setting_duration
        self.n_step = self.horizon//self.setting_duration
        self.r_step = self.setting_duration//args.interval
        recurrent = getattr(args,"recurrent",False)
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
        self.action_space = [tf.convert_to_tensor(space,dtype=tf.float32) for space in getattr(args,'action_space',{}).values()]
        # self.env = get_env(args.env)(initialize=False)

        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)
        self.n_tp_layer = getattr(args, "n_tp_layer", self.n_layer)
        self.kernel_size = getattr(args,"kernel_size",5)
        self.activation = getattr(args,"activation",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if self.conv in ['None','False','NoneType'] else self.conv
        if self.conv:
            self.convnet = ConvNet(args,self.conv)
        elif getattr(args,'if_flood',False):
            self.observ_size += 1
            
        self.qnet_0 = QAgent(action_shape,self.state_shape,args,self.convnet if self.conv else False)
        self.qnet_1 = QAgent(action_shape,self.state_shape,args,self.convnet if self.conv else False)
        if self.mac:
            self.actor = [Actor(self.action_shape[i],len(self.observ_space[i]),args) 
            for i in range(self.n_agents)]
        else:
            self.actor = Actor(action_shape,observ_space,args,self.convnet if self.conv else False)

        if not act_only:
            # self.gamma = tf.convert_to_tensor([getattr(args, "gamma", 0.98)**i for i in range(self.n_step) for _ in range(self.r_step)])
            self.gamma = getattr(args, "gamma", 0.98)
            self.batch_size = getattr(args,"batch_size",128)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.target_update_func = self._hard_update_target_model if self.update_interval >1\
                else self._soft_update_target_model
            self.act_optimizer = Adam(learning_rate=getattr(args,"act_lr",1e-4),clipnorm=1.0)
            self.cri_optimizer = Adam(learning_rate=getattr(args,"cri_lr",1e-3),clipnorm=1.0)
            self.mse = MeanSquaredError()

            self.emul = Emulator(margs.conv,margs.resnet,margs.recurrent,margs)
            self.emul.load(margs.model_dir)

            self.alpha_log = tf.Variable(-1,dtype=tf.float32,trainable=True)
            self.alpha_optimizer = Adam(learning_rate=getattr(args,"act_lr",1e-4),clipnorm=1.0)

        self.agent_dir = args.agent_dir
        if args.load_agent:
            self.load()

    def control(self,state,train=True):
        state = [convert_to_tensor(s)[tf.newaxis,...] for s in state]
        state = [self.normalize(s,item) for s,item in zip(state,'xbe')]
        a = self.actor.get_action(state,train)
        sett = self.actor.convert_action_to_setting(a)
        return tf.squeeze(sett).numpy()
        
    
    def _split_observ(self,s):
        # Split as multi-agent & convert to tensor
        if self.recurrent:
            o = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                   for sis in si] for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        else:
            o = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                   for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        return o

    @tf.function
    def rollout(self,dat):
        x,a,b,y,ex,_ = dat
        # Initial perf is not included in the rollout
        xs,exs,settings,perfs = [x],[ex],[a[:,:self.seq_in,:]],[y[:,:self.seq_in,:,-1:]]
        for i in range(self.n_step):
            bi = b[:,i*self.r_step:(i+1)*self.r_step,:]
            x_norm,b_norm,ex_norm = [self.normalize(dat,item)if dat is not None else None
                                      for dat,item in zip([x,bi,ex],'xbe')]
            a = self.actor.get_action([x_norm,b_norm,ex_norm],train=True)
            setting = self.actor.convert_action_to_setting(a)
            setting = tf.repeat(setting[:,tf.newaxis,:],self.r_step,axis=1)
            settings.append(setting)
            preds = self.emul.predict_tf(x,bi,setting,ex)
            if self.emul.if_flood:
                x = tf.concat([preds[0][...,:-2],tf.cast(preds[0][...,-2:-1]>0.5,tf.float32),bi],axis=-1)
            else:
                x = tf.concat([preds[0][...,:-1],bi],axis=-1)
            if self.emul.use_edge:
                ae = self.emul.get_edge_action(setting,True)
                ex = tf.concat([preds[1],ae],axis=-1)
            else:
                ex = None
            xs.append(x)
            exs.append(ex)
            perfs.append(preds[0][...,-1:])
        return [tf.concat(tf.concat(dat,axis=1),axis=0) for dat in [xs,exs,settings,perfs]]

    # TODO: Perhaps multi-agents should be updated individually, not with mean loss
    # TODO: Indepedent q values for each agent? Or only one for all.
    @tf.function
    def update_eval(self,s,a,r,s_,train=True):
        # if self.mac:
        #     o = self._split_observ(s)
        value_loss = self.critic_update(s,a,r,s_,train)
        alpha = self.alpha_update(s) if train else tf.exp(self.alpha_log).numpy()
        policy_loss = self.actor_update(s,train)
        return value_loss,alpha,policy_loss

    @tf.function
    def critic_update(self,s,a,r,s_,train=True):
        if self.conti:
            a_,logprobs_ = self.actor.get_action_probs(s_)
            q_ = tf.minimum(self.qnet_0.forward(s_,a_,target=True),self.qnet_1.forward(s_,a_,target=True))
            if len(logprobs_.shape)>1:
                q_target = tf.expand_dims(r,axis=-1) + self.gamma * (q_ - tf.exp(self.alpha_log) * logprobs_)
            else:
                q_target = r + self.gamma * (tf.squeeze(q_,axis=-1) - tf.exp(self.alpha_log) * logprobs_)
        else:
            distr_ = self.actor.forward(s_)
            if isinstance(self.action_shape,(list,np.ndarray)):
                logprobs_ = [tf.math.log(distri_.probs+1e-5) for distri_ in distr_]
                q_ = self.qnet_0.forward(s_,target=True),self.qnet_1.forward(s_,target=True)
                q_ = [tf.minimum(q0,q1) for q0,q1 in zip(q_[0],q_[1])]
                q_target = tf.stack([r + self.gamma * tf.reduce_sum((qi - tf.exp(self.alpha_log) * lp) * distri_.probs,axis=-1)
                            for qi,lp,distri_ in zip(q_,logprobs_,distr_)],axis=-1)
            else:
                q_ = tf.minimum(self.qnet_0.forward(s_,target=True),self.qnet_1.forward(s_,target=True))
                logprobs_ = tf.math.log(distr_.probs+1e-5)
                q_target = r + self.gamma * tf.reduce_sum((q_ - tf.exp(self.alpha_log) * logprobs_) * distr_.probs,axis=-1)
        train_vars = self.qnet_0.model.trainable_variables+self.qnet_1.model.trainable_variables
        with GradientTape() as tape:
            tape.watch(train_vars)
            q0,q1 = self.qnet_0.forward(s,a),self.qnet_1.forward(s,a)
            value_loss = self.mse(q_target,q0) + self.mse(q_target,q1)
            if train:
                grads = tape.gradient(value_loss, train_vars)
                grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                          for grad in grads]
                grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
                self.cri_optimizer.apply_gradients(zip(grads, train_vars))
        return value_loss
    
    @tf.function
    def alpha_update(self,s):
        _,log_probs = self.actor.get_action_probs(s)
        with GradientTape() as tape:
            tape.watch(self.alpha_log)
            if isinstance(self.action_shape,(list,np.ndarray)):
                alpha_loss = tf.reduce_mean([tf.reduce_mean(self.alpha_log * (shp - log_prob),axis=0) for shp,log_prob in zip(self.action_shape,log_probs)],axis=0)
            else:
                alpha_loss = self.alpha_log * tf.reduce_mean(self.action_shape - log_probs,axis=0)
            grads = tape.gradient(alpha_loss, [self.alpha_log])
            grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                     for grad in grads]
            grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
            self.alpha_optimizer.apply_gradients(zip(grads, [self.alpha_log]))
        return tf.exp(self.alpha_log)

    # Output n steps are infeasible in RL
    # Directly train a multi-step policy net instead (maybe with infinite target) -- Nop
    # Use setting duration as sequential model for auto-regression in RL
    @tf.function
    def actor_update(self,s,train=True):
        variables = self.actor.model.trainable_variables
        with GradientTape() as tape:
            tape.watch(variables)
            if self.conti:
                a_pg,log_probs = self.actor.get_action_probs(s)
                q_pg = tf.minimum(self.qnet_0.forward(s,a_pg), self.qnet_1.forward(s,a_pg))
                if len(log_probs.shape) > 1:
                    policy_loss = tf.reduce_mean(q_pg - log_probs * tf.exp(self.alpha_log),axis=-1)
                else:
                    policy_loss = tf.squeeze(q_pg,axis=-1) - log_probs * tf.exp(self.alpha_log)
            else:
                distr = self.actor.forward(s)
                if isinstance(self.action_shape,(list,np.ndarray)):
                    log_probs = [tf.math.log(distri.probs+1e-5) for distri in distr]
                    q_pg = self.qnet_0.forward(s),self.qnet_1.forward(s)
                    q_pg = [tf.minimum(q0,q1) for q0,q1 in zip(q_pg[0],q_pg[1])]
                    policy_loss = tf.reduce_mean([tf.reduce_sum(distri.probs*(qi - lp * tf.exp(self.alpha_log)),axis=-1)
                                                 for qi,distri,lp in zip(q_pg,distr,log_probs)],axis=0)
                else:
                    q_pg = tf.minimum(self.qnet_0.forward(s),self.qnet_1.forward(s))
                    log_probs = tf.math.log(distr.probs + 1e-5)
                    policy_loss = tf.reduce_sum(distr.probs*(q_pg - log_probs * tf.exp(self.alpha_log)),axis=-1)
            policy_loss = tf.reduce_mean(policy_loss,axis=0)
            if train:
                grads = tape.gradient(-policy_loss, variables)
                grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                         for grad in grads]
                grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
                self.act_optimizer.apply_gradients(zip(grads, variables))
        return policy_loss

    def _hard_update_target_model(self,episode):
        if episode%self.update_interval == 0:
            self.qnet_0._hard_update_target_model()
            self.qnet_1._hard_update_target_model()
            self.actor._hard_update_target_model()

    def _soft_update_target_model(self,episode):
        self.qnet_0._soft_update_target_model(self.update_interval)
        self.qnet_1._soft_update_target_model(self.update_interval)
        self.actor._soft_update_target_model(self.update_interval)


    def save(self,agent_dir=None,agents=True):
        # Save the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbye':
            if hasattr(self,'norm_%s'%item):
                save(join(agent_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))
        # Load the agent paras
        if agents:
            if self.mac:
                for i,actor in enumerate(self.actor):
                    actor.save(agent_dir,i)
            else:
                self.actor.save(agent_dir)
            self.qnet_0.save(agent_dir,0)
            self.qnet_1.save(agent_dir,1)


    def load(self,agent_dir=None,agents=True):
        # Load the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbye':
            if os.path.exists(join(agent_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,load(join(agent_dir,'norm_%s.npy'%item)))
        # Load the agent paras
        if agents:
            if self.mac:
                for i,actor in enumerate(self.actor):
                    actor.load(agent_dir,i)
            else:
                self.actor.load(agent_dir)
            self.qnet_0.load(agent_dir,0)
            self.qnet_1.load(agent_dir,1)

    def set_norm(self,norm_x,norm_b,norm_y,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)

    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        if inverse:
            return dat * (normal[0,:,:dim]-normal[1,:,:dim]) + normal[1,:,:dim]
        else:
            return (dat - normal[1,:,:dim])/(normal[0,:,:dim]-normal[1,:,:dim])

def get_agent(name):
    try:
        return eval(name)
    except:
        raise AssertionError("Unknown agent %s"%str(name))
