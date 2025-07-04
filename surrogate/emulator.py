from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt,cumsum,tile
from keras.layers import Dense,Input,GRU,Conv1D,LSTM,Softmax,Add,Subtract,Dropout
from keras import activations
from keras.models import Model,Sequential
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.losses import MeanSquaredError,BinaryCrossentropy,BinaryFocalCrossentropy,MeanAbsoluteError
import numpy as np
import os
# from line_profiler import LineProfiler
from spektral.layers import GCNConv,GATConv,ECCConv,GeneralConv,DiffusionConv
import tensorflow as tf
from keras import mixed_precision
# tf.config.list_physical_devices(device_type='GPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MixedGAT(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.dropout = Dropout(self.dropout_rate, 
                               dtype=mixed_precision.global_policy())
        self.built = True

class NodeEdge(tf.keras.layers.Layer):
    def __init__(self, inci, **kwargs):
        super(NodeEdge,self).__init__(**kwargs)
        self.inci = inci

    def build(self,input_shape):
        assert input_shape[-2] == self.inci.shape[1]
        self.w = self.add_weight(
            name='weight',shape=self.inci.shape,initializer='random_normal',trainable=True,
        )
        self.b = self.add_weight(
            name='bias',shape=self.inci.shape,initializer='zeros',trainable=True,
        )
        super(NodeEdge,self).build(input_shape)

    def call(self,inputs):
        mat = self.w * tf.cast(self.inci,mixed_precision.global_policy().compute_dtype) + self.b
        # mat = self.w * self.inci + self.b
        return tf.matmul(mat, inputs)
    
class Emulator:
    def __init__(self,conv=None,resnet=False,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.tide = getattr(args,'tide',False)
        # Runoff (tide) is boundary
        self.b_in = 2 if self.tide else 1
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'

        self.n_out = self.n_in - 1
        self.seq_in = getattr(args,'seq_in',6)
        self.seq_out = getattr(args,'seq_out',1)

        self.embed_size = getattr(args,'embed_size',64)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.n_sp_layer = getattr(args,"n_sp_layer",3)
        self.n_tp_layer = getattr(args,"n_tp_layer",2)
        self.dropout = getattr(args,'dropout',0.0)
        self.activation = getattr(args,"activation",'relu')
        self.balance = getattr(args,"balance",False)
        self.if_flood = getattr(args,"if_flood",0)
        if self.if_flood:
            self.n_in += 1
        self.is_outfall = getattr(args,"is_outfall",np.array([0 for _ in range(self.n_node)]))
        self.epsilon = getattr(args,"epsilon",-1.0)

        self.graph_base = getattr(args,"graph_base",0)
        self.edge_fusion = getattr(args,"edge_fusion",False)
        self.edges = getattr(args,"edges")
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.e_out = self.e_in - 1 # exclude setting
        self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
        self.ehmax = getattr(args,"ehmax",np.array([0.5 for _ in range(self.n_edge)]))
        self.pump = getattr(args,"pump",np.array([0.0 for _ in range(self.n_edge)]))
        self.ewei = getattr(args,"ewei",np.array([1.0 for _ in range(self.n_edge)]))
        self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)
        if self.edge_fusion:
            self.n_out -= 2 # exclude q_in, q_out
            # self.node_index = tf.convert_to_tensor(getattr(args,"node_index"),dtype=tf.int32)
            # self.edge_index = tf.convert_to_tensor(getattr(args,"edge_index"),dtype=tf.int32)
        self.adj = getattr(args,"adj",np.eye(self.n_node))

        if self.act:
            self.act_edges = getattr(args,"act_edges")
            self.use_adj = getattr(args,"use_adj",False)
        self.area = getattr(args,"area",np.array([0.0 for _ in range(self.n_node)]))
        self.pump_in = getattr(args,"pump_in",np.array([0.0 for _ in range(self.n_node)]))
        self.pump_out = getattr(args,"pump_out",np.array([0.0 for _ in range(self.n_node)]))        
        self.offset = getattr(args,"offset",np.array([0.0 for _ in range(self.n_edge)]))
        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))
        self.hmin = getattr(args,"hmin",np.array([0.0 for _ in range(self.n_node)]))
        self.nwei = getattr(args,"nwei",np.array([1.0 for _ in range(self.n_node)]))
        self.nwei = np.repeat(self.nwei[:,np.newaxis],3+int(self.balance),axis=-1).astype(np.float32)
        # narrow down norm range of water head
        if self.hmin.max() > 0:
            wei = (self.hmax-self.hmin)*(1-self.is_outfall) + (self.hmax-self.hmin).mean()*self.is_outfall
            wei = (self.hmax.max()-self.hmin.min())/wei
            wei = np.stack([wei] + (2+int(self.balance)) * [np.ones_like(wei)],axis=-1)
            self.nwei *= wei

        self.conv = False if conv in ['None','False','NoneType'] else conv
        self.recurrent = recurrent
        self.model = self.build_network(self.conv,resnet,self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        if mixed_precision.global_policy().compute_dtype != 'float32':
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
        self.mse = MeanSquaredError()
        if self.if_flood:
            self.poswei = getattr(args,"poswei",np.array([1.0 for _ in range(self.n_node)])).astype(np.float32)
            self.bce = BinaryCrossentropy()
        # GradNorm for multi-task learning
        self.gradnorm = getattr(args,"gradnorm",False)
        if self.gradnorm:
            self.alpha_reg = tf.Variable(1.0,dtype=tf.float32,trainable=True)
            self.alpha_cls = tf.Variable(1.0,dtype=tf.float32,trainable=True)
            self.alpha_optimizer = Adam(learning_rate=1e-4)
            self.mae = MeanAbsoluteError()
            self.alpha = 0.5
        self.roll = getattr(args,"roll",0)
        self.model_dir = getattr(args,"model_dir")

    def get_conv(self,conv):
        # maybe problematic for directed graph, use GeneralConv instead
        if 'GCN' in conv:
            net = GCNConv
            self.filter = GCNConv.preprocess(self.adj)
            self.edge_filter = GCNConv.preprocess(self.edge_adj)
        elif 'Diff' in conv:
            net = DiffusionConv
            self.filter = DiffusionConv.preprocess(self.adj)
            self.edge_filter = DiffusionConv.preprocess(self.edge_adj)
        elif 'GAT' in conv:
            net = MixedGAT
            # filter needs changed <1 -->0 for models (2024.8.9)
            # self.filter = self.adj.astype(int)
            self.filter = (self.adj>0).astype(int)
            # self.edge_filter = self.edge_adj.astype(int)
            self.edge_filter = (self.edge_adj>0).astype(int)
        elif 'General' in conv:
            net = GeneralConv
            self.filter = (self.adj>0).astype(int)
            self.edge_filter = (self.edge_adj>0).astype(int)
        else:
            raise AssertionError("Unknown Convolution layer %s"%str(conv))
        return net
        
    def get_tem_nets(self,recurrent):
        if recurrent == 'Conv1D':
            return [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,
                           activation=self.activation) for i in range(self.n_tp_layer)]
        elif recurrent == 'GRU':
            return [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
        elif recurrent == 'LSTM':
            return [LSTM(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
        else:
            return []
        
    # TODO: GCNConv cause NodeEdge bug
    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        state_shape,bound_shape = (self.seq_in,self.n_node,self.n_in),(self.seq_out,self.n_node,self.b_in)
        X_in = Input(shape=state_shape)
        B_in = Input(shape=bound_shape)
        inp = [X_in,B_in]
        if conv:
            n_ele = self.n_node + self.n_edge if self.graph_base else self.n_node
            Adj_in = Input(shape=(n_ele,))
            inp += [Adj_in]
            net = self.get_conv(conv)
            # TODO: dynamic adj input for past and future
            if self.act and self.use_adj:
                A_in = Input(shape=(self.seq_out,n_ele,n_ele))
                inp += [A_in]
        else:
            net = Dense

        edge_state_shape = (self.seq_in,self.n_edge,self.e_in,)
        E_in = Input(shape=edge_state_shape)
        inp += [E_in]
        if conv and not self.graph_base:
            Eadj_in = Input(shape=(self.n_edge,))
            inp += [Eadj_in]
        if self.act:
            AE_in = Input(shape=edge_state_shape[:-1]+(1,))
            inp += [AE_in]
        activation = activations.get(self.activation)

        # Embedding block
        # (B,T,N,in) (B,N,in)--> (B,T,N*in) (B,N*in)
        x = reshape(X_in,(-1,)+tuple(state_shape[:-2])+(self.n_node*self.n_in,)) if not conv else X_in
        x = Dense(self.embed_size,activation='linear')(x) # Embedding
        x = Dropout(0.2)(x) if self.dropout else x
        res = x[:,-1:,...]  # Keep the identity original with no activation
        x = activation(x)
        b = reshape(B_in,(-1,)+tuple(state_shape[:-2])+(self.n_node*self.b_in,)) if not conv else B_in
        b = Dense(self.embed_size//2,activation=self.activation)(b)  # Boundary Embedding (B,T_out,N,E)
        b = Dropout(0.2)(b) if self.dropout else b
        e = reshape(E_in,(-1,)+tuple(edge_state_shape[:-2])+(self.n_edge*self.e_in,)) if not conv else E_in
        e = Dense(self.embed_size,activation='linear')(e)  # Edge attr Embedding (B,T_in,N,E)
        e = Dropout(0.2)(e) if self.dropout else e
        res_e = e[:,-1:,...]
        e = activation(e)
        if self.act:
            ae = reshape(AE_in,(-1,)+tuple(edge_state_shape[:-2])+(self.n_edge*1,)) if not conv else AE_in
            ae = Dense(self.embed_size//2,activation=self.activation)(ae)  # Control Embedding (B,T_out,N,E)
            ae = Dropout(0.2)(ae) if self.dropout else ae

        # Spatial block: Does spatial and temporal nets need combination one-by-one?
        # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
        x = reshape(x,(-1,) + tuple(x.shape[2:]))
        e = reshape(e,(-1,) + tuple(e.shape[2:]))
        for _ in range(self.n_sp_layer):
            if conv and self.graph_base:
                x = [concat([x,e],axis=-2),Adj_in]
                x = net(self.embed_size,activation=self.activation)(x)
                x,e = tf.split(x,[self.n_node,self.n_edge],axis=-2)
            elif conv:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                x = net(self.embed_size,activation=self.activation)([x,Adj_in])
                e = net(self.embed_size,activation=self.activation)([e,Eadj_in])
            else:
                x = Dense(self.embed_size*2,activation=self.activation)(concat([x,e],axis=-1))
                x,e = tf.split(x,2,axis=-1)
            x = Dropout(self.dropout)(x) if self.dropout else x
            e = Dropout(self.dropout)(e) if self.dropout else e

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x = reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+state_shape[:-2]+(x.shape[-1],)) 
        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        e = reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_state_shape[:-2]+(e.shape[-1],)) 

        # Recurrent block: No need for b and ae temporal nets
        # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
        x = reshape(transpose(x,[0,2,1,3]),(-1,self.seq_in,x.shape[-1])) if conv else x
        # (B*N,T,E) (B,T,E) --> (B*N,T_out,H) (B,T_out,H)
        # x = self.get_tem_nets(recurrent)(x)
        for ly in self.get_tem_nets(recurrent):
            x = ly(x)
        x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll
        # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
        x = transpose(reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else x
        e = reshape(transpose(e,[0,2,1,3]),(-1,self.seq_in,e.shape[-1])) if conv else e
        # e = self.get_tem_nets(recurrent)(e)
        for ly in self.get_tem_nets(recurrent):
            e = ly(e)
        e = e[...,-self.seq_out:,:] # seq_in >= seq_out if roll
        e = transpose(reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3]) if conv else e

        # Boundary in: Add or concat? maybe add makes more sense
        x = concat([x,b],axis=-1)
        if self.act:
            e = concat([e,ae],axis=-1)

        # Spatial block 2
        x = reshape(x,(-1,) + tuple(x.shape[2:]))
        e = reshape(e,(-1,) + tuple(e.shape[2:]))
        if conv:
            if self.act and self.use_adj:
                A = reshape(A_in,(-1,) + tuple(A_in.shape[-2:]))
            else:
                A = Adj_in
        for _ in range(self.n_sp_layer):
            if conv and self.graph_base:
                x = [concat([x,e],axis=-2),A]
                x = net(self.embed_size,activation=self.activation)(x)
                x,e = tf.split(x,[self.n_node,self.n_edge],axis=-2)
            elif conv:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                x = net(self.embed_size,activation=self.activation)([x,A])
                e = net(self.embed_size,activation=self.activation)([e,Eadj_in])
            else:
                x = Dense(self.embed_size*2,activation=self.activation)(concat([x,e],axis=-1))
                x,e = tf.split(x,2,axis=-1)
            x = Dropout(self.dropout)(x) if self.dropout else x
            e = Dropout(self.dropout)(e) if self.dropout else e

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        state_shape = (self.seq_out,) + state_shape[1:]
        x = reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+state_shape[:-2]+(x.shape[-1],)) 
        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        edge_state_shape = (self.seq_out,) + edge_state_shape[1:]
        e = reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_state_shape[:-2]+(e.shape[-1],)) 

        # Recurrent block 2: T_out --> T_out
        # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
        x = reshape(transpose(x,[0,2,1,3]),(-1,self.seq_out,x.shape[-1])) if conv else x
        # (B*N,T,E) (B,T,E) --> (B*N,T_out,H) (B,T_out,H)
        # x = self.get_tem_nets(recurrent)(x)
        for ly in self.get_tem_nets(recurrent):
            x = ly(x)
        # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
        x = transpose(reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else x
        e = reshape(transpose(e,[0,2,1,3]),(-1,self.seq_out,e.shape[-1])) if conv else e
        # e = self.get_tem_nets(recurrent)(e)
        for ly in self.get_tem_nets(recurrent):
            e = ly(e)
        e = transpose(reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3]) if conv else e
        
        # Resnet
        x_out = Dense(self.embed_size,activation='linear',name='dense_resx')(x) # TODO: rename current models
        x_out = Dropout(self.dropout)(x_out) if self.dropout else x_out
        x = Add()([cumsum(x_out,axis=1),tile(res,(1,self.seq_out,)+(1,)*len(res.shape[2:]))]) if resnet else x_out
        x = activation(x)   # if activation is needed here?
        e_out = Dense(self.embed_size,activation='linear')(e)
        e_out = Dropout(self.dropout)(e_out) if self.dropout else e_out
        e = Add()([cumsum(e_out,axis=1),tile(res_e,(1,self.seq_out,)+(1,)*len(res_e.shape[2:]))]) if resnet else e_out
        e = activation(e)

        out_shape = self.n_out if conv else self.n_out * self.n_node
        # (B,T_out,N,H) (B,T_out,H) --> (B,T_out,N,n_out)
        out = Dense(out_shape,activation='hard_sigmoid')(x)   # if tanh is better than linear here when norm==True?
        out = reshape(out,(-1,self.seq_out,self.n_node,self.n_out))
        if self.if_flood:
            out_shape = 1 if conv else 1 * self.n_node
            for _ in range(self.if_flood):
                x = Dense(self.embed_size//2,activation=self.activation)(x)
            flood = Dense(out_shape,activation='sigmoid')(x)
                        #   kernel_regularizer=l2(0.01),bias_regularizer=l1(0.01)
            flood = reshape(flood,(-1,self.seq_out,self.n_node,1))
            out = concat([out,flood],axis=-1)

        out_shape = self.e_out if conv else self.e_out * self.n_edge
        e_out = Dense(out_shape,activation='tanh')(e)
        e_out = reshape(e_out,(-1,self.seq_out,self.n_edge,self.e_out))
        out = [out,e_out]

        model = Model(inputs=inp, outputs=out)
        return model

    def get_adj_action(self,a,g=False):
        # (B,T,N_act) --> (B,T,N,N)
        if g:
            out = np.zeros_like(self.adj)
            out[tuple(self.act_edges.T)] = range(1,a.shape[-1]+1)
            adj = self.adj * tf.gather(tf.concat([tf.ones_like(a[...,:1]),a],axis=-1),tf.cast(out,tf.int32),axis=-1)
        else:
            def get_act(s):
                adj = self.adj.copy()
                adj[tuple(self.act_edges.T)] = s
                return adj
            adj = np.apply_along_axis(get_act,-1,a)

        if 'GCN' in self.conv:
            adj = GCNConv.preprocess(adj)
        elif 'Diff' in self.conv:
            adj = DiffusionConv.preprocess(adj)
        else:
            adj = tf.cast(adj,tf.int32) if g else adj.astype(int)
        return adj

    def get_action(self,a,g=False):
        if g:
            out_o = np.zeros(self.n_node)
            out_i = np.zeros(self.n_node)
            out_o[self.act_edges[:,0]] = range(1,a.shape[-1]+1)
            out_i[self.act_edges[:,1]] = range(1,a.shape[-1]+1)
            a_out = tf.gather(tf.concat([tf.ones_like(a[...,:1]),a],axis=-1),tf.cast(out_o,tf.int32),axis=-1)
            a_in = tf.gather(tf.concat([tf.ones_like(a[...,:1]),a],axis=-1),tf.cast(out_i,tf.int32),axis=-1)
        else:
            def set_outflow(s):
                out = np.ones(self.n_node)
                out[self.act_edges[:,0]] = s
                return out
            def set_inflow(s):
                out = np.ones(self.n_node)
                out[self.act_edges[:,1]] = s
                return out
            a_out = np.apply_along_axis(set_outflow,-1,a)
            a_in = np.apply_along_axis(set_inflow,-1,a)
        return a_out,a_in
    
    def get_edge_action(self,a,g=False):
        act_edges = [np.where((self.edges==act_edge).all(1))[0] for act_edge in self.act_edges]
        act_edges = [i for e in act_edges for i in e]
        act_edges = sorted(list(set(act_edges)),key=act_edges.index)
        if g:
            out = np.zeros(self.n_edge)
            out[act_edges] = range(1,a.shape[-1]+1)
            return tf.expand_dims(tf.gather(tf.concat([tf.ones_like(a[...,:1]),a],axis=-1),tf.cast(out,tf.int32),axis=-1),axis=-1)
        else:
            def set_edge_action(s):
                out = np.ones(self.n_edge)
                out[act_edges] = s
                return out
            return np.expand_dims(np.apply_along_axis(set_edge_action,-1,a),-1)

    def _model(self,x,a,b,ex,ae,adj=None,fit=True):
        if self.roll:       # Curriculum learning (long-term)
            predss,edge_predss = [],[]
            for i in range(self.roll):
                inp = [x[:,-self.seq_in:,...],b[:,i*self.seq_out:(i+1)*self.seq_out,:]]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj[:,i*self.seq_out:(i+1)*self.seq_out,...]] if self.act and self.use_adj else []
                inp += [ex[:,-self.seq_in:,...]]
                inp += [self.edge_filter] if self.conv and not self.graph_base else []
                inp += [ae[:,i*self.seq_out:(i+1)*self.seq_out,...]] if self.act else []
                preds = self.model(inp,training=fit) if self.dropout else self.model(inp)
                preds = [tf.cast(pred,tf.float32) for pred in preds]
                preds,edge_preds = self.post_proc_tf(preds,a[:,i*self.seq_out:(i+1)*self.seq_out,...],b[:,i*self.seq_out:(i+1)*self.seq_out,...])
                predss.append(preds)
                edge_predss.append(edge_preds)
                if self.if_flood:
                    x_new = tf.concat([preds[...,:-1],tf.cast(preds[...,-1:]>0.5,tf.float32),b[:,i*self.seq_out:(i+1)*self.seq_out,:]],axis=-1)
                else:
                    x_new = tf.concat([preds,b[:,i*self.seq_out:(i+1)*self.seq_out,:]],axis=-1)
                x = tf.concat([x[:,-(self.seq_in-self.seq_out):,...],x_new],axis=1) if self.seq_in > self.seq_out else x_new
                ae_new = self.get_edge_action(a[:,i*self.seq_out:(i+1)*self.seq_out,...],True)
                ex_new = tf.concat([edge_preds,ae_new],axis=-1)
                ex = tf.concat([ex[:,-(self.seq_in-self.seq_out):,...],ex_new],axis=1) if self.seq_in > self.seq_out else ex_new
            preds = concat(predss,axis=1)
            edge_preds = concat(edge_predss,axis=1)
        else:
            inp = [x,b]
            if self.conv:
                inp += [self.filter]
                inp += [adj] if self.act and self.use_adj else []
            inp += [ex]
            inp += [self.edge_filter] if self.conv and not self.graph_base else []
            inp += [ae] if self.act else []
            preds = self.model(inp,training=fit) if self.dropout else self.model(inp)
            preds = [tf.cast(pred,tf.float32) for pred in preds]
            preds,edge_preds = self.post_proc_tf(preds,a,b)
        preds = tf.clip_by_value(preds,0,1) # avoid large loss value
        return preds,edge_preds
    
    def get_node_loss(self,y,b,preds):
        if self.balance:
            q_w,preds = self.constrain_tf(self.normalize(preds,'y',inverse=True),
                                          self.normalize(b,'b',inverse=True)[...,:1])
            q_w,preds = tf.expand_dims(q_w/self.norm_y[0,:,-1],axis=-1),self.normalize(preds,'y')
            preds = tf.clip_by_value(preds,0,1) # avoid large loss value
            node_loss = self.mse(tf.concat([y[...,:3],y[...,-1:]],axis=-1)*self.nwei,
                                    tf.concat([preds[...,:3],q_w],axis=-1)*self.nwei)
        else:
            node_loss = self.mse(y[...,:3]*self.nwei,preds[...,:3]*self.nwei)
        return node_loss

    def get_flood_loss(self,y,preds):
        weight = self.poswei * y[...,-2] + self.nwei[:,-1] * (1-y[...,-2])
        fl_loss = self.bce(y[...,-2:-1],preds[...,-1:],sample_weight=weight)
        return fl_loss
    
    @tf.function
    def fit_eval(self,x,a,b,y,ex,ey,fit=True):
        ae = self.get_edge_action(a,True) if self.act else None
        adj = self.get_adj_action(a,True) if self.act and self.use_adj else None
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            preds,edge_preds = self._model(x,a,b,ex,ae,adj,fit)
            # Loss funtion
            node_loss = self.get_node_loss(y,b,preds)
            if self.if_flood and not self.balance:
                fl_loss = self.get_flood_loss(y,preds)
            edge_loss = self.mse(ey,edge_preds,sample_weight=self.ewei)
            if fit:
                reg_loss = node_loss + edge_loss
                loss = (self.alpha_reg if self.gradnorm else 1) * reg_loss
                if self.if_flood:
                    loss += (self.alpha_cls if self.gradnorm else 1) * fl_loss
                tf.debugging.assert_all_finite(loss, 'Loss contains NaN or Inf values.')
                if mixed_precision.global_policy().name != 'float32':
                    scaled_loss = self.optimizer.get_scaled_loss(loss)
                    scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                    grads = self.optimizer.get_unscaled_gradients(scaled_gradients)
                else:
                    grads = tape.gradient(loss, self.model.trainable_variables)
                for grad in grads:
                    tf.debugging.assert_all_finite(grad, "grads contain NaN/Inf!")
                self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return [node_loss,fl_loss,edge_loss] if self.if_flood else [node_loss,edge_loss]

    @tf.function
    def fit_grad_norm(self,x,a,b,y,ex,ey,ini_loss):
        ae = self.get_edge_action(a,True) if self.act else None
        adj = self.get_adj_action(a,True) if self.act and self.use_adj else None
        inps,outs = [x,a,b,ex,ae,adj],[y,ey]
        with tf.GradientTape() as tape:
            tape.watch([self.alpha_reg,self.alpha_cls])
            alpha_loss = self._get_grad_norm(inps,outs,ini_loss)
            alpha_grads = tape.gradient(alpha_loss, [self.alpha_reg,self.alpha_cls])
        # update alpha
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.alpha_reg,self.alpha_cls]))
        
        # alpha normalized with the sum as 2
        alpha_sum = self.alpha_reg + self.alpha_cls
        self.alpha_reg.assign(2.0 * self.alpha_reg / alpha_sum)
        self.alpha_cls.assign(2.0 * self.alpha_cls / alpha_sum)

    def _get_grad_norm(self,inps,outs,ini_loss):
        W = self.model.get_layer('dense_resx').kernel
        y,ey = outs
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(W)
            preds,edge_preds = self._model(*inps)
            node_loss = self.get_node_loss(y,inps[2],preds)
            edge_loss = self.mse(ey,edge_preds,sample_weight=self.ewei)
            reg_loss = node_loss + edge_loss
            fl_loss = self.get_flood_loss(y,preds)
        grad_reg_norm = tf.norm(self.alpha_reg*tape.gradient(reg_loss,W))
        grad_cls_norm = tf.norm(self.alpha_cls*tape.gradient(fl_loss,W))
        r_reg,r_cls = reg_loss / (ini_loss[0]+ini_loss[-1]), fl_loss / ini_loss[1]
        r_avg = (r_reg + r_cls) / 2
        grad_norm = tf.stack([grad_reg_norm,grad_cls_norm])
        target_grad = tf.stop_gradient(tf.reduce_mean(grad_norm)) * (tf.stack([r_reg,r_cls])/ r_avg) ** self.alpha
        return self.mae(target_grad,grad_norm)

    def simulate(self,states,runoff,a=None,edge_states=None):
        # runoff shape: T_out, T_in, N
        if self.act:
            ae = self.get_edge_action(a)
            if self.use_adj:
                adj = self.get_adj_action(a)
        preds,edge_preds = [],[]
        for idx,bi in enumerate(runoff):
            x = states[idx,-self.seq_in:,...]
            ex = edge_states[idx,-self.seq_in:,...]
                
            bi = bi[:self.seq_out]

            inp = [self.normalize(x,'x'),self.normalize(bi,'b')]
            inp = [expand_dims(dat,0) for dat in inp]
            if self.conv:
                inp += [self.filter]
                inp += [adj[idx:idx+1]] if self.act and self.use_adj else []
            inp += [expand_dims(self.normalize(ex,'e'),0)]
            inp += [self.edge_filter] if self.conv and not self.graph_base else []
            inp += [ae[idx:idx+1]] if self.act else []
            y = self.model(inp,training=False) if self.dropout else self.model(inp)
            y,ey = [tf.cast(pred,tf.float32) for pred in y]

            y,ey = self.post_proc(y.numpy(),ey.numpy(),a[idx:idx+1],self.normalize(bi,'b'))
            ey = self.normalize(np.squeeze(ey,0),'e',True)
            ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
            y = self.normalize(np.squeeze(y,0),'y',True)

            # Pumped storage depth calculation: boundary condition differs from orifice
            if sum(getattr(self,'pump_in',[0])) + sum(getattr(self,'pump_out',[0])) + sum(getattr(self,'pump',[0])) > 0:
                ps = (self.area * np.matmul(np.clip(self.node_edge,0,1),np.expand_dims(self.pump,axis=-1)).squeeze())>0
                h,qin,qout = [y[...,i] for i in [0,1,2]]
                de = []
                for t in range(self.seq_out):
                    de += [np.clip(x[-1,:,0] if t==0 else de[-1] +\
                                    (qin-qout)[t,:]/(self.area+1e-6),self.hmin,self.hmax)]
                y[...,0] = h*(1-ps) + np.stack(de,axis=0) * ps

            q_w,y = self.constrain(y,bi[...,:1],x[-1:,:,0])
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            edge_preds.append(ey)
        return np.array(preds),np.array(edge_preds)

    def predict(self,states,b,a=None,edge_state=None):
        x = states[:,-self.seq_in:,...]
        if edge_state is not None:
            ex = edge_state[:,-self.seq_in:,...]
        assert b.shape[1] == self.seq_out
        if self.act:
            ae = self.get_edge_action(a)
            if self.use_adj:
                adj = self.get_adj_action(a)
        inp = [self.normalize(x,'x'),self.normalize(b,'b')]
        if self.conv:
            inp += [self.filter]
            inp += [adj] if self.act and self.use_adj else []
        inp += [self.normalize(ex,'e')]
        inp += [self.edge_filter] if self.conv and not self.graph_base else []
        inp += [ae] if self.act else []
        y,ey = self.model(inp,training=False) if self.dropout else self.model(inp)
        y,ey = [tf.cast(pred,tf.float32) for pred in (y,ey)]

        y,ey = self.post_proc(y.numpy(),ey.numpy(),a,self.normalize(b,'b'))
        y = self.normalize(y,'y',True)
        ey = self.normalize(ey,'e',True)
        ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)

        # Pumped storage depth calculation: boundary condition differs from orifice
        if sum(getattr(self,'pump_in',[0])) + sum(getattr(self,'pump_out',[0])) + sum(getattr(self,'pump',[0])) > 0:
            ps = (self.area * np.matmul(np.clip(self.node_edge,0,1),self.pump))>0
            h,qin,qout = [y[...,i] for i in [0,1,2]]
            de = np.zeros_like(h)
            for t in range(de.shape[1]):
                de[:,t,:] = np.clip(x[:,-1,:,0] if t==0 else de[:,t-1,:] +\
                                    (qin-qout)[:,t,:]/(self.area+1e-6),self.hmin,self.hmax)
            y[...,0] = h*(1-ps) + de * ps

        q_w,y = self.constrain(y,b[...,:1],x[:,-1:,:,0])
        y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
        return y,ey

    @tf.function
    def predict_tf(self,states,b,a=None,edge_state=None):
        x = states[:,-self.seq_in:,...]
        if edge_state is not None:
            ex = edge_state[:,-self.seq_in:,...]
        assert b.shape[1] == self.seq_out
        if self.act:
            ae = self.get_edge_action(a,True)
            if self.use_adj:
                adj = self.get_adj_action(a,True)
        inp = [self.normalize(x,'x'),self.normalize(b,'b')]
        if self.conv:
            inp += [self.filter]
            inp += [adj] if self.act and self.use_adj else []
        inp += [self.normalize(ex,'e')]
        inp += [self.edge_filter] if self.conv and not self.graph_base else []
        inp += [ae] if self.act else []
        y,ey = self.model(inp,training=False) if self.dropout else self.model(inp)
        y,ey = [tf.cast(pred,tf.float32) for pred in (y,ey)]

        y,ey = self.post_proc_tf((y,ey),a,self.normalize(b,'b'))
        ey = self.normalize(ey,'e',True)
        ey = tf.concat([tf.expand_dims(tf.clip_by_value(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
        y = self.normalize(y,'y',True)

        # Pumped storage depth calculation: boundary condition differs from orifice
        if sum(getattr(self,'pump_in',[0])) + sum(getattr(self,'pump_out',[0])) + sum(getattr(self,'pump',[0])) > 0:
            ps = tf.cast((self.area * tf.squeeze(tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.expand_dims(tf.cast(self.pump,tf.float32),axis=-1))))>0,tf.float32)
            h,qin,qout = [y[...,i] for i in [0,1,2]]
            de = []
            for t in range(self.seq_out):
                de += [tf.clip_by_value(x[:,-1,:,0] if t==0 else de[-1] +\
                                        (qin-qout)[:,t,:]/(self.area+1e-6),self.hmin,self.hmax)]
            de = tf.stack(de,axis=1)
            y = tf.concat([tf.expand_dims(h*(1-ps) + de * ps,axis=-1),y[...,1:]],axis=-1)
        q_w,y = self.constrain_tf(y,b[...,:1],x[:,-1:,:,0])
        y = tf.concat([y,tf.expand_dims(q_w,axis=-1)],axis=-1)
        return y,ey

    def post_proc(self,y,ey,a,b):
        # tide boundary
        if self.tide:
            h = y[...,0] * (1 - self.is_outfall) + b[...,-1]
            y = np.concatenate([np.expand_dims(h,axis=-1),y[...,1:]],axis=-1)
        # TODO: if need to regulate pipe inflow offset
        inoff = np.matmul(self.normalize(y,'y',True)[...,0] - self.hmin,np.clip(self.node_edge,0,1))
        flow = np.expand_dims(ey[...,-1]*(ey[...,-1]>0)*(self.offset>0)*(inoff > self.offset)+\
                                ey[...,-1]*(ey[...,-1]<=0)*(self.offset>0)+\
                                    ey[...,-1]*(self.offset==0),axis=-1)
        ey = np.concatenate([ey[...,:-1],flow],axis=-1)
        if self.act:
            ae = self.get_edge_action(a)
            # regulate pumping flow (rated value if there is volume in inlet tank)
            fl = self.pump*np.matmul(y[...,0]>0.01,np.clip(self.node_edge,0,1))
            fl *= (self.norm_e[0,:,2]>1e-3)/self.norm_e[0,:,2]
            ey[...,-1] = ey[...,-1] * (fl==0) + fl
            ey[...,-1:] *= ae
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...])
                # regulate pumping flow (rated value if there is volume in inlet tank)
                fli = self.pump_in * (y[...,0]>0.01)/self.norm_y[0,:,1]
                flo = self.pump_out * (y[...,0]>0.01)/self.norm_y[0,:,2]
                y[...,1] = y[...,1]*(fli==0) + fli
                y[...,2] = y[...,2]*(flo==0) + flo
                # regulate flow with setting
                y[...,2] *= a_out
                y[...,1] *= a_in
        if self.edge_fusion:
            efl = self.normalize(ey,'e',True)[...,-1:]
            node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(efl,0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(efl,-np.inf,0))
            node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(efl,0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(efl,-np.inf,0))
            node_outflow *= (self.norm_y[0,:,2:3]>1e-3)/self.norm_y[0,:,2:3]
            node_inflow *= (self.norm_y[0,:,1:2]>1e-3)/self.norm_y[0,:,1:2]
            y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)
        return y,ey

    @tf.function
    def post_proc_tf(self,preds,a,b):
        preds,edge_preds = preds
        # tide boundary
        if self.tide:
            h = preds[...,0] * (1 - self.is_outfall) + b[...,-1]
            preds = tf.concat([tf.expand_dims(h,axis=-1),preds[...,1:]],axis=-1)
        # TODO: if need to regulate pipe inflow offset
        if self.offset.max() > 0:
            inoff = tf.matmul(self.normalize(preds,'y',True)[...,0]-self.hmin,tf.clip_by_value(self.node_edge,0,1))
            flow = edge_preds[...,-1]
            flow = tf.expand_dims(flow * tf.cast(flow>0,tf.float32) * tf.cast(self.offset>0,tf.float32) * tf.cast(inoff > self.offset,tf.float32) +\
                                  flow * tf.cast(flow<=0,tf.float32) * tf.cast(self.offset>0,tf.float32) +\
                                  flow * tf.cast(self.offset==0,tf.float32),axis=-1)
            edge_preds = concat([edge_preds[...,:-1],flow],axis=-1)
        # Control action regulation
        if self.act:
            # regulate pumping flow (rated value if there is volume in inlet tank)
            if self.pump.min()>0:
                fl = self.pump*tf.matmul(tf.cast(preds[...,0]>0.01,tf.float32),tf.clip_by_value(self.node_edge,0,1))
                fl *= tf.cast(self.norm_e[0,:,2]>1e-3,tf.float32)/self.norm_e[0,:,2]
                flow = tf.expand_dims(edge_preds[...,-1] * tf.cast(fl==0,tf.float32) + fl,axis=-1)
            else:
                flow = edge_preds[...,-1:]
            # regulate flow with setting
            ae = self.get_edge_action(a,True)
            edge_preds = concat([edge_preds[...,:-1],tf.multiply(flow,ae)],axis=-1)
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...],True)
                # regulate pumping flow (rated value if there is volume in inlet tank)
                fli = self.pump_in * tf.cast(preds[...,0]>0,tf.float32)/self.norm_y[0,:,1]
                flo = self.pump_out * tf.cast(preds[...,0]>0,tf.float32)/self.norm_y[0,:,2]
                inflow = preds[...,1] * tf.cast(fli==0,tf.float32) + fli
                outflow = preds[...,2] * tf.cast(flo==0,tf.float32) + flo
                # regulate flow with setting
                preds = concat([tf.stack([preds[...,0],tf.multiply(inflow,a_in),tf.multiply(outflow,a_out)],axis=-1),preds[...,3:]],axis=-1)

        # Node edge flow balance
        if self.edge_fusion:
            edge_flow = self.normalize(edge_preds,'e',True)[...,-1:]
            node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(edge_flow,-np.inf,0))
            node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(edge_flow,-np.inf,0))
            node_outflow *= tf.cast(self.norm_y[0,:,2:3]>1e-3,tf.float32)/self.norm_y[0,:,2:3]
            node_inflow *= tf.cast(self.norm_y[0,:,1:2]>1e-3,tf.float32)/self.norm_y[0,:,1:2]
            preds = concat([preds[...,:1],node_inflow,node_outflow,preds[...,1:]],axis=-1)
        return preds,edge_preds
     
    def constrain(self,y,r,h0=None):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = np.clip(h,self.hmin,self.hmax)
        # dv = self.area * np.diff(np.concatenate([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = np.clip(q_us + r - q_ds - dv,0,np.inf) * (1-self.is_outfall)
        if self.if_flood:
            # f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            f = y[...,-1] > 0.5
            h = self.hmax * f + h * ~f
            # y = np.stack([h,q_us,q_ds,y[...,-2],y[...,-1]],axis=-1)
            y = np.stack([h,q_us,q_ds,y[...,-1]],axis=-1)
        else:
            y = np.stack([h,q_us,q_ds],axis=-1)
        if self.epsilon > 0:
            q_w *= ((self.hmax - h) < self.epsilon)
        elif self.epsilon == 0:
            pass
        elif self.if_flood:
            q_w *= f
        return q_w,y
    
    def constrain_tf(self,y,r,h0=None):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = tf.squeeze(r,axis=-1)
        h = tf.clip_by_value(h,self.hmin,self.hmax)
        # dv = self.area * tf.experimental.numpy.diff(tf.concat([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = tf.clip_by_value(q_us + r - q_ds - dv,0,np.inf) * tf.cast(1-self.is_outfall,tf.float32)
        if self.if_flood:
            # f = tf.cast(tf.argmax(y[...,-2:],axis=-1),tf.float32)
            f = tf.cast(y[...,-1] > 0.5,tf.float32)
            h = self.hmax * f + h * (1-f)
            y = tf.stack([h,q_us,q_ds,y[...,-1]],axis=-1)
        else:
            y = tf.stack([h,q_us,q_ds],axis=-1)
        if self.epsilon > 0:
            q_w = q_w * tf.cast((self.hmax - h) < self.epsilon, tf.float32)
        elif self.epsilon == 0:
            pass
        elif self.if_flood:
            q_w *= f
        return q_w,y
    
    def get_flood(self,y,r,h0):
        # B,T,N
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = np.clip(h,self.hmin,self.hmax)
        # dv = self.area * np.diff(np.concatenate([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = np.clip(q_us + r - q_ds - dv,0,np.inf) * (1-self.is_outfall)
        if self.epsilon > 0:
            q_w *= ((self.hmax - h) < self.epsilon)
        elif self.epsilon == 0:
            pass
        elif self.if_flood:
            # f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            f = y[...,-1] > 0.5
            q_w *= f
        return q_w
    
    def head_to_depth(self,h):
        h = tf.clip_by_value(h,self.hmin,self.hmax)
        return (h-self.hmin)/(self.hmax-self.hmin)

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
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



    def save(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if model_dir.endswith('.h5'):
            self.model.save_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.save_weights(os.path.join(model_dir,'model.h5'))

        for item in 'xbyre':
            if hasattr(self,'norm_%s'%item):
                np.save(os.path.join(model_dir,'norm_%s.npy'%item),getattr(self,'norm_%s'%item))

        np.save(os.path.join(model_dir,'optim.npy'), self.optimizer.get_weights())
        if self.gradnorm:
            checkpoint = tf.train.Checkpoint(cls=self.alpha_cls,reg=self.alpha_reg,optimizer=self.alpha_optimizer)
            checkpoint.write(os.path.join(model_dir,"gradnorm.ckpt"))

    def load(self,model_dir=None,retrain=False):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))

        for item in 'xbyre':
            if os.path.exists(os.path.join(model_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,np.load(os.path.join(model_dir,'norm_%s.npy'%item)))
        
        if retrain and os.path.exists(os.path.join(model_dir,'optim.npy')):
            if len(self.optimizer.get_weights()) == 0:
                self.optimizer.apply_gradients(zip([tf.zeros_like(va) for va in self.model.trainable_variables],
                                                   self.model.trainable_variables))
            self.optimizer.set_weights(np.load(os.path.join(model_dir,'optim.npy'),allow_pickle=True))
        if retrain and self.gradnorm:
            checkpoint = tf.train.Checkpoint(cls=self.alpha_cls,reg=self.alpha_reg,optimizer=self.alpha_optimizer)
            checkpoint.restore(os.path.join(model_dir,"gradnorm.ckpt"))
