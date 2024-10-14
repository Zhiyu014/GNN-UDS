import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
# from tensorflow_probability.python.distributions import RelaxedOneHotCategorical,Normal
# from tensorflow_probability.python.layers import DistributionLambda
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import activations,Sequential
from tensorflow.keras.layers import Dense,Input
import numpy as np
from numpy import array,save,load,concatenate
from tensorflow.keras.models import Model
from spektral.layers import GCNConv,GlobalAttnSumPool,GlobalAvgPool,GlobalSumPool,GATConv,DiffusionConv,GeneralConv
from os.path import join
import os
from emulator import NodeEdge
tf.config.list_physical_devices(device_type='GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress infos and warnings

class ConvNet:
    def __init__(self,args,conv):
        self.conv_dim = getattr(args,"conv_dim",128)
        self.n_sp_layer = getattr(args, "n_sp_layer",3)

        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        if getattr(args,'if_flood',False):
            self.n_in += 1
        self.use_pred = getattr(args,"use_pred",False)
        if self.use_pred:
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
            # self.filter = self.adj.astype(int)
            self.filter = (self.adj>0).astype(int)
            if self.use_edge:
                # self.edge_filter = self.edge_adj.astype(int)
                self.edge_filter = (self.edge_adj>0).astype(int)
        elif 'General' in conv:
            net = GeneralConv
            self.filter = (self.adj>0).astype(int)
            if self.use_edge:
                self.edge_filter = (self.edge_adj>0).astype(int)
        else:
            raise AssertionError("Unknown Convolution layer %s"%str(conv))
        return net

    # input: X,Adj,B,E,Eadj
    # output: X
    def build_network(self,net):
        X_in = Input(shape=(self.n_node,self.n_in,))
        Adj_in = Input(shape=(self.n_node,))
        inp = [X_in,Adj_in]
        if self.use_pred:
            B_in = Input(shape=(self.n_node,self.b_in,))
            inp += [B_in]
        
        if self.use_edge:
            E_in = Input(shape=(self.n_edge,self.e_in,))
            Eadj_in = Input(shape=(self.n_edge,))
            inp += [E_in,Eadj_in]
        activation = activations.get(self.activation)

        # Embedding
        x = Dense(self.conv_dim,activation=activation)(X_in)
        if self.use_pred:
            b = Dense(self.conv_dim//2,activation=activation)(B_in)
            x = Dense(self.conv_dim,activation=activation)(tf.concat([x,b],axis=-1))
        if self.use_edge:
            e = Dense(self.conv_dim,activation=activation)(E_in)

        # Spatial block
        for _ in range(self.n_sp_layer):
            if self.use_edge:
                x_e = Dense(self.conv_dim//2,activation=self.activation)(e)
                e_x = Dense(self.conv_dim//2,activation=self.activation)(x)
                x = tf.concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = tf.concat([e,NodeEdge(tf.transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
            x = [x,Adj_in]
            x = net(self.conv_dim,activation=self.activation)(x)
            if self.use_edge:
                e = [e,Eadj_in]
                e = net(self.conv_dim,activation=self.activation)(e)

        # Global Pooling
        if self.use_edge:
            x = GlobalAttnSumPool()(tf.concat([x,e],axis=-2))
        model = Model(inputs=inp, outputs=x)
        return model

class Actor:
    def __init__(self,
                 action_shape,
                 observ_size,
                 args,
                 conv = None):
        self.action_shape = action_shape
        self.observ_size = observ_size

        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
        self.action_space = [tf.convert_to_tensor(space,dtype=tf.float32) for space in getattr(args,'action_space',{}).values()]
        if not self.conti:
            self.action_table = tf.convert_to_tensor(list(getattr(args,'action_table',{}).values()),dtype=tf.float32)

        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.activation = getattr(args,"activation",False)
        self.mac = getattr(args,"mac",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        self.model = self.build_pi_network(self.convnet.model if self.conv else None)
        self.agent_dir = args.agent_dir
        if args.load_agent:
            self.load()

    def build_pi_network(self,conv=None):
        if conv is None:
            x_in = Input(shape=(self.observ_size,))
            x = x_in
        else:
            x_in = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(x_in)
        for _ in range(self.n_layer):
            x = Dense(self.net_dim, activation=self.activation)(x)
        if self.conti:
            mu = Dense(self.action_shape, activation='linear')(x)
            log_std = Dense(self.action_shape, activation='linear')(x)
            output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[0],scale=tf.exp(t[1])))([mu,log_std])
        elif isinstance(self.action_shape,np.ndarray) and self.mac:
            output = [Dense(act_shape, activation='softmax')(x) for act_shape in self.action_shape]
            output = [tfp.layers.DistributionLambda(lambda t: tfd.RelaxedOneHotCategorical(1.0,probs=t))(o) for o in output]
        else:
            output = Dense(np.product(self.action_shape), activation='softmax')(x)
            output = tfp.layers.DistributionLambda(lambda t: tfd.RelaxedOneHotCategorical(1.0,probs=t))(output)
        model = Model(inputs=x_in, outputs=output)
        return model

    def get_input(self,observ):
        if self.conv:
            inp = observ[:1] + [self.convnet.filter]
            if self.convnet.use_pred:
                inp += [observ[1:2]]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = observ
        return inp
    
    def control(self, observ,train=False,batch=False):
        if not batch:
            if isinstance(observ,list):
                observ = [ob[tf.newaxis,...] for ob in observ]
            else:
                observ = observ[tf.newaxis,...]
        action = self.get_action(observ,train)
        settings = self.convert_action_to_setting(action)
        return tf.squeeze(settings)

    def forward(self, observ):
        inp = self.get_input(observ)
        probs = self.model(inp)
        return probs

    def get_action(self, observ, train = True):
        distr = self.forward(observ)
        if self.conti:
            return tf.tanh(distr.sample()) if train else tf.tanh(distr.loc)
        elif isinstance(distr,list):
            return [tf.argmax(distri.sample(),axis=-1) if train else tf.argmax(distri.probs,axis=-1) for distri in distr]
        else:
            return tf.argmax(distr.sample(),axis=-1) if train else tf.argmax(distr.probs,axis=-1)

    def get_action_probs(self, observ):
        distr = self.forward(observ)
        if self.conti:
            a = distr.sample()
            logp_action = distr.log_prob(a)
            a = tf.tanh(a)   # Restrict action to be between -1 and 1
            logp_action -= tf.math.log(1.000001-tf.pow(a,2))   # Adjusted Log Probability due to tanh
            return a,logp_action
        elif isinstance(distr,list):
            probs = [distri.probs for distri in distr]
            log_probs = [tf.math.log(distri.probs+1e-5) for distri in distr]
            return probs,log_probs
        else:
            return distr.probs,tf.math.log(distr.probs+1e-5)
        
    def convert_action_to_setting(self,action):
        if self.conti:
            return (action+1)/2
        elif isinstance(self.action_shape,(list,np.ndarray)):
            if self.mac:
                return tf.stack([tf.gather(space,ai) for space,ai in zip(self.action_space,action)],axis=-1)
            else:
                return tf.gather(self.action_table,action)
        else:
            return tf.gather(self.action_space,action)
        
    def convert_setting_to_action(self,setting):
        if self.conti:
            return tf.multiply(setting,2)-1
        elif isinstance(self.action_shape,(list,np.ndarray)):
            if self.mac:
                return [tf.argmin([tf.abs(setting[...,i]-sp) for sp in space],axis=0)
                         for i,space in enumerate(self.action_space)]
            else:
                return tf.argmin([tf.reduce_sum(tf.abs(setting-tab),axis=-1) for tab in self.action_table],axis=0)
        else:
            return tf.argmin(tf.abs(setting-self.action_space),axis=-1)

    def save(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        self.model.save_weights(join(agent_dir,'actor%s.h5'%i))
        for item in 'xbyer':
            if hasattr(self,'norm_%s'%item):
                np.save(join(agent_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))
            
    def load(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.load_weights(join(agent_dir,'actor%s.h5'%i))
        for item in 'xbyer':
            setattr(self,'norm_%s'%item,np.load(join(agent_dir,'norm_%s.npy'%item)))
            
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
        maxi,mini = normal[0,...,:dim],normal[1,...,:dim]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)

# TODO: reward shaping
class QAgent:
    def __init__(self,
                 action_shape,
                 observ_size,
                 args,
                 conv=None):
        self.action_shape = action_shape
        self.observ_size = observ_size

        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.activation = getattr(args,"activation",False)

        self.mac = getattr(args,"mac",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        self.model = self.build_q_network(self.convnet.model if self.conv else None)
        self.target_model = self.build_q_network(self.convnet.model if self.conv else None)
        self.target_model.set_weights(self.model.get_weights())
        self.agent_dir = args.agent_dir
        # TODO value normalization
        self.value_tau = getattr(args,"value_tau",0.005)
        self.value_avg,self.value_std = tf.constant(0.0),tf.constant(1.0)
        
    def build_q_network(self,conv=None):
        if conv is None:
            inp = Input(shape=(self.observ_size,))
            x = inp
        else:
            inp = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(inp)
        if self.conti:
            a_dim = sum(self.action_shape) if isinstance(self.action_shape,(np.ndarray,list)) else self.action_shape
            a_in = Input(shape=(a_dim,))
            a = Dense(self.net_dim, activation=self.activation)(a_in)
            x = tf.concat([x,a],axis=-1)
            inp = inp + [a_in] if isinstance(inp,list) else [inp,a_in]
        for _ in range(self.n_layer):
            x = Dense(self.net_dim, activation=self.activation)(x)
        if self.mac and isinstance(self.action_shape,(np.ndarray,list)):
            output = [Dense(shp, activation='linear')(x) for shp in self.action_shape]
        else:
            out_dim = 1 if self.conti else np.product(np.asarray(self.action_shape))
            output = Dense(out_dim, activation='linear')(x)
        model = Model(inputs=inp, outputs=output)
        return model
    
    def get_input(self,observ,act):
        if self.conv:
            inp = observ[:1] + [self.convnet.filter]
            if self.convnet.use_pred:
                inp += [observ[1:2]]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = observ
        if self.conti:
            inp = inp + [act] if isinstance(inp,list) else [inp,act]
        return inp

    def forward(self,observ,act=None,target=False):
        inp = self.get_input(observ,act)
        q = self.target_model(inp) if target else self.model(inp)
        if not self.conti and act is not None:
            if isinstance(self.action_shape,(np.ndarray,list)) and self.mac:
                q = tf.stack([tf.gather(qi,ai,axis=-1,batch_dims=1)
                     for qi,ai in zip(q,act)],axis=-1)
            else:
                q = tf.gather(q,act,axis=-1,batch_dims=1)
        return q
    
    # TODO: value normalization
    def value_re_norm(self,value):
        return value * self.value_std + self.value_avg
    
    def value_update(self,observ,act=None):
        inp = self.get_input(observ,act)
        q = self.model(inp)
        self.value_avg = self.value_tau * tf.reduce_mean(q) + (1-self.value_tau) * self.value_avg 
        self.value_std = self.value_tau * tf.math.reduce_std(q) + (1-self.value_tau) * self.value_std

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

class VAgent:
    def __init__(self,
                 observ_size,
                 args,
                 conv=None):
        self.observ_size = observ_size

        self.net_dim = getattr(args,"net_dim",128)
        self.n_layer = getattr(args, "n_layer", 3)
        self.activation = getattr(args,"activation",False)
        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        self.model = self.build_v_network(self.convnet.model if self.conv else None)
        self.target_model = self.build_v_network(self.convnet.model if self.conv else None)
        self.target_model.set_weights(self.model.get_weights())
        self.agent_dir = args.agent_dir
        # TODO value normalization
        self.value_tau = getattr(args,"value_tau",0.0)
        self.value_avg,self.value_std = tf.constant(0.0),tf.constant(1.0)

    def build_v_network(self,conv=None):
        if conv is None:
            inp = Input(shape=(self.observ_size,))
            x = inp
        else:
            inp = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(inp)
        for _ in range(self.n_layer):
            x = Dense(self.net_dim, activation=self.activation)(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=inp, outputs=output)
        return model
    
    def get_input(self,observ):
        if self.conv:
            inp = observ[:1] + [self.convnet.filter]
            if self.convnet.use_pred:
                inp += [observ[1:2]]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = observ
        return inp

    def forward(self,observ,target=False):
        inp = self.get_input(observ)
        v = self.target_model(inp) if target else self.model(inp)
        return v
    
    # TODO: value normalization
    def value_re_norm(self,value):
        return value * self.value_std + self.value_avg
    
    def value_update(self,observ,act=None):
        inp = self.get_input(observ,act)
        q = self.model(inp)
        self.value_avg = self.value_tau * tf.reduce_mean(q) + (1-self.value_tau) * self.value_avg 
        self.value_std = self.value_tau * tf.math.reduce_std(q) + (1-self.value_tau) * self.value_std
    
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
        self.model.save_weights(join(agent_dir,'vnet%s.h5'%i))
        self.target_model.save_weights(join(agent_dir,'vnet%s_target.h5'%i))
            
    def load(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.load_weights(join(agent_dir,'vnet%s.h5'%i))
        self.target_model.load_weights(join(agent_dir,'vnet%s_target.h5'%i))

class AgentSAC:
    def __init__(self,
            action_shape,
            observ_space,
            args = None,
            act_only = False):
        self.action_shape = action_shape

        self.dec = getattr(args, "dec", False)
        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
<<<<<<< Updated upstream
        self.n_agents = action_shape if self.conti else len(action_shape)
=======
        self.mac = getattr(args,"mac",False)
        if self.mac:
            self.n_agents = action_shape if self.conti else action_shape.shape[0]
>>>>>>> Stashed changes

        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if self.conv:
            self.convnet = ConvNet(args,self.conv)

        if self.dec:
            self.actor = [Actor(action_shape[i],len(observ_space[i]),args) 
            for i in range(self.n_agents)]
            state_shape = len(getattr(args,'states'))
        else:
            state_shape = len(observ_space)
            self.actor = Actor(action_shape,state_shape,args,self.convnet if self.conv else None)

        self.act_only = act_only
        if not self.act_only:
            self.qnet_0 = QAgent(action_shape,state_shape,args,self.convnet if self.conv else None)
            self.qnet_1 = QAgent(action_shape,state_shape,args,self.convnet if self.conv else None)
            if not self.conti and self.mac:
                self.vnet = VAgent(state_shape,args,self.convnet if self.conv else None)
            self.gamma = getattr(args, "gamma", 0.98)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.update_func = self._hard_update_target_model if self.update_interval >1\
                else self._soft_update_target_model
            self.act_optimizer = Adam(learning_rate=getattr(args,"act_lr",1e-4),clipnorm=1.0)
            self.cri_optimizer = Adam(learning_rate=getattr(args,"cri_lr",1e-3),clipnorm=1.0)
            if getattr(self,"vnet",None) is not None:
                self.val_optimizer = Adam(learning_rate=getattr(args,"cri_lr",1e-3),clipnorm=1.0)
            self.mse = MeanSquaredError()

            self.alpha_log = tf.Variable(-1,dtype=tf.float32,trainable=True)
            self.alpha_optimizer = Adam(learning_rate=getattr(args,"act_lr",1e-4),clipnorm=1.0)

        self.agent_dir = args.agent_dir
        if args.load_agent:
            self.load()
        # TODO: reward normalization
        self.value_tau = getattr(args,"value_tau",0.0)
        self.reward_std = tf.constant(1.0)

    # Perhaps multi-agents should be updated individually, not with mean loss
    # Indepedent q values for each agent? Or only one for all.
    @tf.function
    def update_eval(self,s,a,r,s_,train=True):
        # if self.dec:
        #     o = self._split_observ(s)
        # r = tf.no_gradient(self.reward_norm(r,update=True))
        value_loss = self.critic_update(s,a,r,s_,train)
        alpha = self.alpha_update(s) if train else tf.exp(self.alpha_log).numpy()
        policy_loss = self.actor_update(s,train)
        if getattr(self,"vnet",None) is not None:
            vf_loss = self.vnet_update(s,train)
            return value_loss,alpha,policy_loss,vf_loss
        else:
            return value_loss,alpha,policy_loss

    @tf.function
    def critic_update(self,s,a,r,s_,train=True):
        if self.conti:
            a_,logprobs_ = self.actor.get_action_probs(s_)
            q_ = tf.minimum(self.qnet_0.forward(s_,a_,target=True),self.qnet_1.forward(s_,a_,target=True))
            # TODO: sum or mean log_probs over agents?
            if len(logprobs_.shape)>1:
                logprobs_ = tf.reduce_mean(logprobs_,axis=-1)
            q_target = r + self.gamma * (tf.squeeze(q_,axis=-1) - tf.exp(self.alpha_log) * logprobs_)
        elif self.mac:
            vf_target = self.vnet.forward(s_,target=True)
            q_target = r + self.gamma * tf.squeeze(vf_target,axis=-1)
        else:
            # if isinstance(self.action_shape,(list,np.ndarray)):
            #     logprobs_ = [tf.math.log(distri_.probs+1e-5) for distri_ in distr_]
            #     q_ = self.qnet_0.forward(s_,target=True),self.qnet_1.forward(s_,target=True)
            #     q_ = [tf.minimum(q0,q1) for q0,q1 in zip(q_[0],q_[1])]
            #     q_target = tf.stack([r + self.gamma * tf.reduce_sum((qi - tf.exp(self.alpha_log) * lp) * distri_.probs,axis=-1)
            #                 for qi,lp,distri_ in zip(q_,logprobs_,distr_)],axis=-1)
            # else:
            probs_,logprobs_ = self.actor.get_action_probs(s_)
            q_ = tf.minimum(self.qnet_0.forward(s_,target=True),self.qnet_1.forward(s_,target=True))
            q_target = r + self.gamma * tf.reduce_sum((q_ - tf.exp(self.alpha_log) * logprobs_) * probs_,axis=-1)
        train_vars = self.qnet_0.model.trainable_variables+self.qnet_1.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(train_vars)
            q0,q1 = self.qnet_0.forward(s,a),self.qnet_1.forward(s,a)
            # sum/mean over all agents?
            if len(q0.shape) > 1:
                q0,q1 = tf.reduce_mean(q0,axis=-1),tf.reduce_mean(q1,axis=-1)
            value_loss = self.mse(q_target,q0) + self.mse(q_target,q1)
            if train:
                grads = tape.gradient(value_loss, train_vars)
                grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                          for grad in grads]
                # grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
                self.cri_optimizer.apply_gradients(zip(grads, train_vars))
        return value_loss
    
    @tf.function
    def alpha_update(self,s):
        probs,log_probs = self.actor.get_action_probs(s)
        with tf.GradientTape() as tape:
            tape.watch(self.alpha_log)
            if self.conti:
                alpha_loss = self.alpha_log * tf.reduce_mean(self.action_shape-log_probs)
            elif self.mac and isinstance(self.action_shape,(list,np.ndarray)):
                alpha_loss = self.alpha_log * tf.reduce_mean([len(self.action_shape) - tf.reduce_sum(log_prob*prob,axis=-1)
                                              for prob,log_prob in zip(probs,log_probs)],axis=0)
            else:
                alpha_loss = self.alpha_log * tf.reduce_mean(1 - tf.reduce_sum(log_probs*probs,axis=-1),axis=0)
            grads = tape.gradient(alpha_loss, [self.alpha_log])
            grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                     for grad in grads]
            # grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
            self.alpha_optimizer.apply_gradients(zip(grads, [self.alpha_log]))
        return tf.exp(self.alpha_log)

    # Output n steps are infeasible in RL
    # Directly train a multi-step policy net instead (maybe with infinite target) -- Nop
    # Use setting duration as sequential model for auto-regression in RL
    @tf.function
    def actor_update(self,s,train=True):
        variables = self.actor.model.trainable_variables
        if getattr(self,"vnet",None) is not None:
            vpred = self.vnet.forward(s)
        with tf.GradientTape() as tape:
            tape.watch(variables)
            a_pg,log_probs = self.actor.get_action_probs(s)
            if self.conti:
                q_pg = tf.minimum(self.qnet_0.forward(s,a_pg), self.qnet_1.forward(s,a_pg))
                # TODO: sum or mean log_probs over agents?
                if len(log_probs.shape) > 1:
                    log_probs = tf.reduce_mean(log_probs,axis=-1)
                policy_loss = tf.squeeze(q_pg,axis=-1) - log_probs * tf.exp(self.alpha_log)
            elif self.mac:
                q_pg = self.qnet_0.forward(s),self.qnet_1.forward(s)
                q_pg = [tf.minimum(q0,q1) for q0,q1 in zip(q_pg[0],q_pg[1])]
                policy_loss = tf.reduce_mean([tf.reduce_sum(pg*(qi - vpred - lp * tf.exp(self.alpha_log)),axis=-1)
                                                for qi,pg,lp in zip(q_pg,a_pg,log_probs)],axis=0)
            else:
                q_pg = tf.minimum(self.qnet_0.forward(s),self.qnet_1.forward(s))
                policy_loss = tf.reduce_sum(a_pg*(q_pg - log_probs * tf.exp(self.alpha_log)),axis=-1)
            policy_loss = tf.reduce_mean(policy_loss,axis=0)
            # assert tf.abs(policy_loss) < 10, "Policy loss is too large: %.3f"%tf.abs(policy_loss)
            if train:
                grads = tape.gradient(-policy_loss, variables)
                grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                         for grad in grads]
                # grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
                self.act_optimizer.apply_gradients(zip(grads, variables))
        return policy_loss
    
    @tf.function
    def vnet_update(self,s,train=True):
        a_pg,log_probs = self.actor.get_action_probs(s)
        if self.conti:
            q_pg = tf.minimum(self.qnet_0.forward(s,a_pg),self.qnet_1.forward(s,a_pg))
            # TODO: sum or mean log_probs over agents?
            if len(log_probs.shape)>1:
                log_probs = tf.reduce_mean(log_probs,axis=-1)
            v_target = tf.squeeze(q_pg,axis=-1) - tf.exp(self.alpha_log) * log_probs
        elif self.mac and isinstance(self.action_shape,(list,np.ndarray)):
                q_pg = self.qnet_0.forward(s),self.qnet_1.forward(s)
                q_pg = [tf.minimum(q0,q1) for q0,q1 in zip(q_pg[0],q_pg[1])]
                v_target = tf.reduce_mean([tf.reduce_sum(pg*(qi - lp * tf.exp(self.alpha_log)),axis=-1)
                                        for qi,pg,lp in zip(q_pg,a_pg,log_probs)],axis=0)
        else:
            q_pg = tf.minimum(self.qnet_0.forward(s),self.qnet_1.forward(s))
            v_target = tf.reduce_sum(a_pg*(q_pg - log_probs * tf.exp(self.alpha_log)),axis=-1)
        with tf.GradientTape() as tape:
            tape.watch(self.vnet.model.trainable_variables)
            vf_pred = self.vnet.forward(s)
            vf_loss = self.mse(v_target,vf_pred)
            if train:
                grads = tape.gradient(vf_loss, self.vnet.model.trainable_variables)
                grads = [tf.zeros_like(grad) if tf.reduce_any(tf.math.is_inf(grad)) or tf.reduce_any(tf.math.is_nan(grad)) else grad
                         for grad in grads]
                # grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
                self.val_optimizer.apply_gradients(zip(grads, self.vnet.model.trainable_variables))
        return vf_loss

    def control(self,observ,train=False,batch=False):
        return self.actor.control(observ,train,batch)
    
    def convert_setting_to_action(self,setting):
        return self.actor.convert_setting_to_action(setting)

    def reward_norm(self,r,update=False):
        if update:
            self.reward_std = (1-self.value_tau) * self.reward_std + self.value_tau * tf.math.reduce_std(r)
        return r / (self.reward_std + 1e-6)

    def _hard_update_target_model(self,episode):
        if episode%self.update_interval == 0:
            self.qnet_0._hard_update_target_model()
            self.qnet_1._hard_update_target_model()
            if getattr(self,"vnet",None) is not None:
                self.vnet._hard_update_target_model()

    def _soft_update_target_model(self,episode):
        self.qnet_0._soft_update_target_model(self.update_interval)
        self.qnet_1._soft_update_target_model(self.update_interval)
        if getattr(self,"vnet",None) is not None:
            self.vnet._soft_update_target_model(self.update_interval)

    def save(self,agent_dir=None,agents=True):
        # Save the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbyer':
            if hasattr(self,'norm_%s'%item):
                save(join(agent_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))
        # Load the agent paras
        if agents:
            if self.dec:
                for i,actor in enumerate(self.actor):
                    actor.save(agent_dir,i)
            else:
                self.actor.save(agent_dir)
            self.qnet_0.save(agent_dir,0)
            self.qnet_1.save(agent_dir,1)
            if getattr(self,"vnet",None) is not None:
                self.vnet.save(agent_dir)

    def load(self,agent_dir=None,agents=True):
        # Load the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbyer':
            if os.path.exists(join(agent_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,load(join(agent_dir,'norm_%s.npy'%item)))
        # Load the agent paras
        if agents:
            if self.dec:
                for i,actor in enumerate(self.actor):
                    actor.load(agent_dir,i)
            else:
                self.actor.load(agent_dir)
            if not self.act_only:
                self.qnet_0.load(agent_dir,0)
                self.qnet_1.load(agent_dir,1)
                if getattr(self,"vnet",None) is not None:
                    self.vnet.load(agent_dir)

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        setattr(self,'norm_r',norm_r)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)
        self.actor.set_norm(norm_x,norm_b,norm_y,norm_r,norm_e)

    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        maxi,mini = normal[0,...,:dim],normal[1,...,:dim]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)

#TODO: AgentPPO

#TODO: debug qmix
class AgentQMIX:
    def __init__(self,
            action_shape,
            observ_space,
            args = None,
            act_only = False):
        self.action_shape = action_shape
        self.observ_space = observ_space

        self.net_dim = getattr(args,"net_dim",128)
        self.activation = getattr(args,"activation",False)

        self.mac = getattr(args,"mac",False)
        if self.mac:
            self.n_agents = self.action_shape.shape[0]
        self.dec = getattr(args, "dec", False)
        self.act = getattr(args,"act","")
        self.action_space = [tf.convert_to_tensor(space,dtype=tf.float32)
                              for space in getattr(args,'action_space',{}).values()]
        self.action_table = tf.convert_to_tensor(list(getattr(args,'action_table',{}).values()),dtype=tf.float32)
        
        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if self.conv:
            self.convnet = ConvNet(args,self.conv)
            
        if self.dec:
            self.qnet = [QAgent(self.action_shape[i],len(self.observ_space[i]),args) 
            for i in range(self.n_agents)]
            state_shape = len(getattr(args,"states"))
        else:
            state_shape = len(self.observ_space)
            self.qnet = QAgent(self.action_shape,state_shape,args,self.convnet if self.conv else None)
        self.epsilon_decay,self.epsilon_min = getattr(args,"epsilon_decay",0.9996),0.1
        self._epsilon_decay(getattr(args,"episode",0))
        self.act_only = act_only
        if not self.act_only:
            if self.mac:
                self.mix = MixNet(self.action_shape,state_shape,args,self.convnet if self.conv else None)
            self.double = getattr(args,"double",True) # Not defined in args
            self.gamma = getattr(args, "gamma", 0.98)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.optimizer = Adam(learning_rate=getattr(args,"cri_lr",1e-3),clipnorm=1.0)
            self.mse = MeanSquaredError()

        self.agent_dir = args.agent_dir
        if args.load_agent:
            self.load(self.agent_dir)
        # TODO: reward normalization
        self.value_tau = getattr(args,"value_tau",0.0)
        self.reward_std = tf.constant(1.0)

    def control(self, observ,train=False,batch=False):
        if train and np.random.uniform() < self.epsilon:
            batch_size = 1 if not batch else observ[0].shape[0] if isinstance(observ,list) else observ.shape[0]
            if self.mac:
                action = [tf.random.uniform((batch_size,),maxval=self.action_shape[i],dtype=tf.int32)
                          for i in range(self.n_agents)]
            else:
                action = tf.random.uniform((batch_size,),maxval=self.action_shape,dtype=tf.int32)
        else:  
            if not batch:
                observ = [ob[tf.newaxis,...] for ob in observ] if isinstance(observ,list) else observ[tf.newaxis,...]
            q = self.qnet.forward(observ)
            action = [tf.argmax(qi,axis=-1) for qi in q] if self.mac else tf.argmax(q,axis=-1)
        settings = self.convert_action_to_setting(action)
        return tf.squeeze(settings)

    def convert_action_to_setting(self,action):
        if isinstance(self.action_shape,(list,np.ndarray)):
            if self.mac:
                return tf.stack([tf.gather(space,ai) for space,ai in zip(self.action_space,action)],axis=-1)
            else:
                return tf.gather(self.action_table,action)
        else:
            return tf.gather(self.action_space,action)
        
    def convert_setting_to_action(self,setting):
        if isinstance(self.action_shape,(list,np.ndarray)):
            if self.mac:
                return [tf.argmin([tf.abs(setting[...,i]-sp) for sp in space],axis=0)
                         for i,space in enumerate(self.action_space)]
            else:
                return tf.argmin([tf.reduce_sum(tf.abs(setting-tab),axis=-1)
                                   for tab in self.action_table],axis=0)
        else:
            return tf.argmin(tf.abs(setting-self.action_space),axis=-1)
    
    @tf.function
    def update_eval(self,s,a,r,s_,train=True):
        # if self.dec:
        #     o = self._split_observ(s)
        # r = tf.no_gradient(self.reward_norm(r,update=True))
        target = self._calculate_target(r,s_)
        variables = self.qnet.model.trainable_variables
        variables += self.mix.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)
            q_values = self.qnet.forward(s)
            if self.mac:
                q_values = [tf.reduce_sum(q * tf.one_hot(a[idx],self.action_shape[idx]),axis=-1)
                        for idx,q in enumerate(q_values)]
                q_values = tf.transpose(tf.convert_to_tensor(q_values))
                q_tot = self.mix.forward(s,q_values)
                # q_tot = tf.reduce_sum(q_values,axis=-1)
            else:
                q_tot = tf.reduce_sum(q_values * tf.one_hot(a,self.action_shape),axis=-1)
            loss = self.mse(target, q_tot)
        if train:
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def _calculate_target(self,r,s_):
        tqs_ = self.qnet.forward(s_,target=True)
        if self.mac:
            if self.double:
                qs_ = self.qnet.forward(s_)
                target_q_values = [tf.reduce_sum(tq_*\
                    tf.one_hot(tf.argmax(q_,axis=-1),self.action_shape[idx]),axis=-1)
                    for idx,(q_,tq_) in enumerate(zip(qs_,tqs_))]
            else:
                target_q_values = [tf.reduce_max(tq_,axis=-1) for tq_ in tqs_]
            target_q_values = tf.transpose(tf.convert_to_tensor(target_q_values))
            target_q_tot = self.mix.forward(s_,target_q_values,target=True)
            # target_q_tot = tf.reduce_sum(target_q_values,axis=-1)
        else:
            if self.double:
                qs_ = self.qnet.forward(s_)
                target_q_tot = tf.reduce_sum(tqs_*tf.one_hot(tf.argmax(qs_,axis=-1),self.action_shape),axis=-1)
            else:
                target_q_tot = tf.reduce_max(tqs_,axis=-1)
        return r + self.gamma * target_q_tot

    def reward_norm(self,r,update=False):
        if update:
            self.reward_std = (1-self.value_tau) * self.reward_std + self.value_tau * tf.math.reduce_std(r)
        return r / (self.reward_std + 1e-6)

    def update_func(self,episode):
        if self.update_interval > 1:
            self._hard_update_target_model(episode)
        else:
            self._soft_update_target_model(episode)
        self._epsilon_decay(episode)

    def _hard_update_target_model(self,episode):
        if episode%self.update_interval == 0:
            self.qnet._hard_update_target_model()
            if self.mac:
                self.mix._hard_update_target_model()

    def _soft_update_target_model(self,episode):
        self.qnet._soft_update_target_model(self.update_interval)
        if self.mac:
            self.mix._soft_update_target_model(self.update_interval)

    def _epsilon_decay(self,episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay**episode)

    def save(self,agent_dir=None,agents=True):
        # Save the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbyer':
            if hasattr(self,'norm_%s'%item):
                save(join(agent_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))
        # Load the agent paras
        if agents:
            self.qnet.save(agent_dir)
            if self.mac:
                self.mix.save(agent_dir)

    def load(self,agent_dir=None,agents=True):
        # Load the normalization paras
        agent_dir = self.agent_dir if agent_dir is None else agent_dir
        for item in 'xbyer':
            if os.path.exists(join(agent_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,load(join(agent_dir,'norm_%s.npy'%item)))
        # Load the agent paras
        if agents:
            self.qnet.load(agent_dir)
            if self.mac and not self.act_only:
                self.mix.load(agent_dir)

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
        maxi,mini = normal[0,...,:dim],normal[1,...,:dim]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)
        
class MixNet:
    def __init__(self,action_shape,state_shape,args,conv=None):
        self.state_shape = state_shape

        self.net_dim = getattr(args,"net_dim",128)
        self.activation = getattr(args,"activation",False)

        self.n_agents = len(action_shape)
        self.conv = getattr(args,"conv",False)
        self.conv = False if str(self.conv) in ['None','False','NoneType'] else self.conv
        if conv is not None:
            self.convnet = conv
        elif self.conv:
            self.convnet = ConvNet(args,self.conv)
        self.model = self.build_mixing_network(self.convnet.model if self.conv else None)
        self.target_model = self.build_mixing_network(self.convnet.model if self.conv else None)
        self.target_model.set_weights(self.model.get_weights())
        self.agent_dir = args.agent_dir
        # TODO value normalization
        self.value_tau = getattr(args,"value_tau",0.005)
        self.value_avg,self.value_std = tf.constant(0.0),tf.constant(1.0)

    def build_mixing_network(self,conv=None):
        if conv is None:
            x = Input(shape=(self.state_shape,))
            inp = [x]
        else:
            inp = [Input(shape=ip.shape[1:]) for ip in conv.input]
            x = conv(inp)
        q_in = Input(shape=(self.n_agents,))
        inp += [q_in]
        q = tf.reshape(q_in, [-1, 1, self.n_agents])
        w1 = tf.abs(Dense(self.net_dim*self.n_agents, activation=None)(x))
        w1 = tf.reshape(w1, [-1, self.n_agents, self.net_dim])
        b1 = Dense(self.net_dim, activation=None)(x)
        b1 = tf.reshape(b1, [-1, 1, self.net_dim])
        hidden = activations.elu(tf.matmul(q, w1) + b1)
        w2 = tf.abs(Dense(self.net_dim, activation=None)(x))
        w2 = tf.reshape(w2, [-1, self.net_dim, 1])
        b2 = Dense(self.net_dim, activation=self.activation)(x)
        b2 = Dense(1, activation=None)(b2)
        b2 = tf.reshape(b2, [-1, 1, 1])
        y = tf.matmul(hidden, w2) + b2
        q_tot = tf.reshape(y, [-1])
        model = Model(inputs=inp, outputs=q_tot)
        return model

    def get_input(self,observ,q):
        if self.conv:
            inp = observ[:1] + [self.convnet.filter]
            if self.convnet.use_pred:
                inp += [observ[1:2]]
            if self.convnet.use_edge:
                inp += observ[-1:] + [self.convnet.edge_filter]
        else:
            inp = observ
        inp = inp + [q] if isinstance(inp,list) else [inp,q]
        return inp

    def forward(self,observ,q,target=False):
        inp = self.get_input(observ,q)
        q_tot = self.target_model(inp) if target else self.model(inp)
        return q_tot
    
    # TODO: value normalization
    def value_re_norm(self,value):
        return value * self.value_std + self.value_avg
    
    def value_update(self,observ,act=None):
        inp = self.get_input(observ,act)
        q = self.model(inp)
        self.value_avg = self.value_tau * tf.reduce_mean(q) + (1-self.value_tau) * self.value_avg 
        self.value_std = self.value_tau * tf.math.reduce_std(q) + (1-self.value_tau) * self.value_std

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
        self.model.save_weights(join(agent_dir,'mixnet%s.h5'%i))
        self.target_model.save_weights(join(agent_dir,'mixnet%s_target.h5'%i))
            
    def load(self,agent_dir=None,i=None):
        i = '' if i is None else str(i)
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        self.model.load_weights(join(agent_dir,'mixnet%s.h5'%i))
        self.target_model.load_weights(join(agent_dir,'mixnet%s_target.h5'%i))

def get_agent(name):
    try:
        return eval("Agent"+name)
    except:
        raise AssertionError("Unknown agent %s"%str(name))
