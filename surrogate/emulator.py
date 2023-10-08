from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt,cumsum,tile
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D,Softmax,Add,Subtract,Dropout
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError,CategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras import mixed_precision
import numpy as np
import os
# from line_profiler import LineProfiler
from spektral.layers import GCNConv,GATConv,ECCConv,GeneralConv,DiffusionConv
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf=2.3.0
# policy = mixed_precision.experimental.Policy('mixed_float16')
# mixed_precision.experimental.set_policy(policy)
# tf=2.6.0
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# - **Model**: STGCN may be a possible method to handle with spatial-temporal prediction. Why such structure is needed?  TO reduce model
#     - [pytorch implementation](https://github.com/LMissher/STGNN)
#     - [original](https://github.com/VeritasYin/STGCN_IJCAI-18)
# - **Predict**: *T*-times 1-step prediction OR T-step prediction?

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
        # mat = self.w * tf.cast(self.inci,policy.compute_dtype) + self.b
        mat = self.w * self.inci + self.b
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
        self.roll = getattr(args,"roll",False)
        if self.roll:
            self.seq_out = 1

        self.embed_size = getattr(args,'embed_size',64)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.n_sp_layer = getattr(args,"n_sp_layer",3)
        self.n_tp_layer = getattr(args,"n_tp_layer",2)
        self.dropout = getattr(args,'dropout',0.0)
        self.activation = getattr(args,"activation",'relu')
        self.norm = getattr(args,"norm",False)
        self.balance = getattr(args,"balance",0)
        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 2
        self.is_outfall = getattr(args,"is_outfall",np.array([0 for _ in range(self.n_node)]))
        self.epsilon = getattr(args,"epsilon",0.1)

        self.use_edge = getattr(args,"use_edge",False)
        self.edge_fusion = getattr(args,"edge_fusion",False)
        self.use_edge = self.edge_fusion or self.use_edge
        if self.use_edge:
            self.edges = getattr(args,"edges")
            self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
            self.e_out = self.e_in - 1 # exclude setting
            self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
            self.ehmax = getattr(args,"ehmax",np.array([0.5 for _ in range(self.n_edge)]))
            if self.edge_fusion:
                self.n_out -= 2 # exclude q_in, q_out
                self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)
                self.node_index = tf.convert_to_tensor(getattr(args,"node_index"),dtype=tf.int32)
                self.edge_index = tf.convert_to_tensor(getattr(args,"edge_index"),dtype=tf.int32)
        self.adj = getattr(args,"adj",np.eye(self.n_node))
        if self.act:
            self.act_edges = getattr(args,"act_edges")
            self.use_adj = getattr(args,"use_adj",False)
        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))
        self.hmin = getattr(args,"hmin",np.array([0.0 for _ in range(self.n_node)]))
        self.area = getattr(args,"area",np.array([0.0 for _ in range(self.n_node)]))

        self.conv = False if conv in ['None','False','NoneType'] else conv
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.model = self.build_network(self.conv,resnet,self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        self.mse = MeanSquaredError()
        if self.if_flood:
            self.bce = BinaryCrossentropy()
            # self.cce = CategoricalCrossentropy()

        self.ratio = getattr(args,"ratio",0.8)
        self.batch_size = getattr(args,"batch_size",256)
        self.epochs = getattr(args,"epochs",100)
        self.model_dir = getattr(args,"model_dir")
        
    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        state_shape,bound_shape = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if recurrent:
            state_shape = (self.seq_in,) + state_shape
            bound_shape = (self.seq_out,) + bound_shape
        X_in = Input(shape=state_shape)
        B_in = Input(shape=bound_shape)
        inp = [X_in,B_in]
        if conv:
            Adj_in = Input(shape=(self.n_node,self.n_node,))
            inp += [Adj_in]
            # maybe problematic for directed graph, use GeneralConv instead
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
            if self.act and self.use_adj:
                A_in = Input(shape=(self.n_node,self.n_node,))
                inp += [A_in]
        else:
            net = Dense

        if self.use_edge:
            edge_state_shape = (self.seq_in,self.n_edge,self.e_in,) if recurrent else (self.n_edge,self.e_in,)
            E_in = Input(shape=edge_state_shape)
            inp += [E_in]
            if conv:
                Eadj_in = Input(shape=(self.n_edge,self.n_edge,))
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
        res = x[:,-1:,...] if recurrent else x  # Keep the identity original with no activation
        x = activation(x)
        b = reshape(B_in,(-1,)+tuple(state_shape[:-2])+(self.n_node*self.b_in,)) if not conv else B_in
        b = Dense(self.embed_size//2,activation=self.activation)(b)  # Boundary Embedding (B,T_out,N,E)
        b = Dropout(0.2)(b) if self.dropout else b
        if self.use_edge:
            e = reshape(E_in,(-1,)+tuple(edge_state_shape[:-2])+(self.n_edge*self.e_in,)) if not conv else E_in
            e = Dense(self.embed_size,activation='linear')(e)  # Edge attr Embedding (B,T_in,N,E)
            e = Dropout(0.2)(e) if self.dropout else e
            res_e = e[:,-1:,...] if recurrent else e
            e = activation(e)
            # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
            e = reshape(e,(-1,) + tuple(e.shape[2:])) if recurrent else e
            if self.act:
                ae = reshape(AE_in,(-1,)+tuple(edge_state_shape[:-2])+(self.n_edge*1,)) if not conv else AE_in
                ae = Dense(self.embed_size//2,activation=self.activation)(ae)  # Control Embedding (B,T_out,N,E)
                ae = Dropout(0.2)(ae) if self.dropout else ae

        # Spatial block: Does spatial and temporal nets need combination one-by-one?
        # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        for _ in range(self.n_sp_layer):
            if conv and self.use_edge and self.edge_fusion:
                # n,n,e   B,E,H  how to convert e from beh to bnnh?
                # x_e = tf.gather(e,self.edge_index,axis=1)
                # e_x = tf.gather(x,self.node_index,axis=1)
                # x = ECCConv(self.embed_size)([x,Adj_in,x_e])
                # e = ECCConv(self.embed_size)([e,Eadj_in,e_x])
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
            x = [x,Adj_in] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            x = Dropout(self.dropout)(x) if self.dropout else x
            if self.use_edge:
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)
                e = Dropout(self.dropout)(e) if self.dropout else e

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x = reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+state_shape[:-2]+(x.shape[-1],)) 
        if self.use_edge:
            # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
            e = reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_state_shape[:-2]+(e.shape[-1],)) 

        # Recurrent block: No need for b and ae temporal nets
        if recurrent:
            # Model the spatio-temporal differences
            if recurrent == 'Conv1D':
                x_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:]) for i in range(self.n_tp_layer)]
                # b_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=b.shape[-2:]) for i in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=e.shape[-2:]) for i in range(self.n_tp_layer)]
                    # if self.act:
                    #     ae_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=ae.shape[-2:]) for i in range(self.n_tp_layer)]
            elif recurrent == 'GRU':
                x_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                # b_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                    # if self.act:
                    #     ae_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
            else:
                raise AssertionError("Unknown recurrent layer %s"%str(recurrent))
            
            # x = Subtract()([x,concat([tf.zeros_like(x[:,0:1,...]),x[:,1:,...]],axis=1)])
            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
            x = reshape(transpose(x,[0,2,1,3]),(-1,self.seq_in,x.shape[-1])) if conv else x
            # b = reshape(transpose(b,[0,2,1,3]),(-1,self.seq_out,b.shape[-1])) if conv else b
            # (B*N,T,E) (B,T,E) --> (B*N,T_out,H) (B,T_out,H)
            for i in range(self.n_tp_layer):
                x = x_tem_nets[i](x)
                # b = b_tem_nets[i](b)
            x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll
            # b = b[...,-self.seq_out:,:] # seq_in >= seq_out if roll
            # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
            x = transpose(reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else x
            # b = transpose(reshape(b,(-1,self.n_node,self.seq_out,b.shape[-1])),[0,2,1,3]) if conv else b
            if self.use_edge:
                # e = Subtract()([e,concat([tf.zeros_like(e[:,0:1,...]),e[:,1:,...]],axis=1)])
                e = reshape(transpose(e,[0,2,1,3]),(-1,self.seq_in,e.shape[-1])) if conv else e
                for i in range(self.n_tp_layer):
                    e = e_tem_nets[i](e)
                e = e[...,-self.seq_out:,:] # seq_in >= seq_out if roll
                e = transpose(reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3]) if conv else e
                # if self.act:
                #     ae = reshape(transpose(ae,[0,2,1,3]),(-1,self.seq_out,ae.shape[-1])) if conv else ae
                #     for i in range(self.n_tp_layer):
                #         ae = ae_tem_nets[i](ae)
                #     ae = ae[...,-self.seq_out:,:] # seq_in >= seq_out if roll
                #     ae = transpose(reshape(ae,(-1,self.n_edge,self.seq_out,ae.shape[-1])),[0,2,1,3]) if conv else ae


        # Boundary in: Add or concat? maybe add makes more sense
        # b = Dense(self.hidden_dim,activation=self.activation)(B_in)  # Boundary Embedding (B,T_out,N,H)
        x = concat([x,b],axis=-1)
        # x = Add()([x,b])
        if self.use_edge and self.act:
            # ae = Dense(self.hidden_dim//2,activation=self.activation)(ae)  # Control Embedding (B,T_out,N,H)
            e = concat([e,ae],axis=-1)
            # e = Add()([e,ae])


        # Spatial block 2
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        if self.use_edge:
            e = reshape(e,(-1,) + tuple(e.shape[2:])) if recurrent else e
        if conv:
            if self.act and self.use_adj:
                A = reshape(A_in,(-1,) + (self.n_node,self.n_node)) if recurrent else A_in
            else:
                A = Adj_in
        for _ in range(self.n_sp_layer):
            if conv and self.use_edge and self.edge_fusion:
                # n,n,e   B,E,H  how to convert e from beh to bnnh?
                # x_e = tf.gather(e,self.edge_index,axis=1)
                # e_x = tf.gather(x,self.node_index,axis=1)
                # x = ECCConv(self.embed_size)([x,Adj_in,x_e])
                # e = ECCConv(self.embed_size)([e,Eadj_in,e_x])
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
            x = [x,A] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            x = Dropout(self.dropout)(x) if self.dropout else x
            if self.use_edge:
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)
                e = Dropout(self.dropout)(e) if self.dropout else e

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        state_shape = (self.seq_out,) + state_shape[1:] if recurrent else state_shape
        x = reshape(x,(-1,)+state_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+state_shape[:-2]+(x.shape[-1],)) 
        if self.use_edge:
            # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
            edge_state_shape = (self.seq_out,) + edge_state_shape[1:] if recurrent else edge_state_shape
            e = reshape(e,(-1,)+edge_state_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_state_shape[:-2]+(e.shape[-1],)) 

        # Recurrent block 2: T_out --> T_out
        if recurrent:
            # Model the spatio-temporal differences
            if recurrent == 'Conv1D':
                x_tem_nets2 = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:]) for i in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets2 = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=e.shape[-2:]) for i in range(self.n_tp_layer)]
            elif recurrent == 'GRU':
                x_tem_nets2 = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets2 = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]

            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
            x = reshape(transpose(x,[0,2,1,3]),(-1,self.seq_out,x.shape[-1])) if conv else x
            # (B*N,T,E) (B,T,E) --> (B*N,T_out,H) (B,T_out,H)
            for i in range(self.n_tp_layer):
                x = x_tem_nets2[i](x)
            # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
            x = transpose(reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else x
            if self.use_edge:
                # e = Subtract()([e,concat([tf.zeros_like(e[:,0:1,...]),e[:,1:,...]],axis=1)])
                e = reshape(transpose(e,[0,2,1,3]),(-1,self.seq_out,e.shape[-1])) if conv else e
                for i in range(self.n_tp_layer):
                    e = e_tem_nets2[i](e)
                e = transpose(reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])),[0,2,1,3]) if conv else e
        
        # Resnet
        x_out = Dense(self.embed_size,activation='linear')(x)
        x_out = Dropout(self.dropout)(x_out) if self.dropout else x_out
        x = Add()([cumsum(x_out,axis=1),tile(res,(1,self.seq_out,)+(1,)*len(res.shape[2:]))]) if recurrent else Add()([res,x_out]) if resnet else x_out
        x = activation(x)   # if activation is needed here?
        if self.use_edge:
            e_out = Dense(self.embed_size,activation='linear')(e)
            e_out = Dropout(self.dropout)(e_out) if self.dropout else e_out
            e = Add()([cumsum(e_out,axis=1),tile(res_e,(1,self.seq_out,)+(1,)*len(res_e.shape[2:]))]) if recurrent else Add()([res_e,e_out]) if resnet else e_out
            e = activation(e)

        out_shape = self.n_out if conv else self.n_out * self.n_node
        # (B,T_out,N,H) (B,T_out,H) --> (B,T_out,N,n_out)
        out = Dense(out_shape,activation='hard_sigmoid' if self.norm else 'linear')(x)   # if tanh is better than linear here when norm==True?
        out = reshape(out,(-1,self.seq_out,self.n_node,self.n_out))
        if self.if_flood:
            # flood = Dense(self.embed_size//2,activation=self.activation,
            #               kernel_regularizer=l2(0.01),bias_regularizer=l1(0.01),activity_regularizer=l1(0.01))(x)
            # flood = Dropout(self.dropout)(flood) if self.dropout else flood
            out_shape = 1 if conv else 1 * self.n_node
            flood = Dense(out_shape,activation='sigmoid')(x)
                        #   kernel_regularizer=l2(0.01),bias_regularizer=l1(0.01)
            # flood = Dense(out_shape,activation='linear')(x)
            flood = reshape(flood,(-1,self.seq_out,self.n_node,1))
            # flood = Softmax()(flood)
            out = concat([out,flood],axis=-1)

        if self.use_edge:
            out_shape = self.e_out if conv else self.e_out * self.n_edge
            e_out = Dense(out_shape,activation='tanh' if self.norm else 'linear')(e)
            e_out = reshape(e_out,(-1,self.seq_out,self.n_edge,self.e_out))
            out = [out,e_out]

        model = Model(inputs=inp, outputs=out)
        return model

    def get_adj_action(self,a,g=False):
        # (B,T,N_act) --> (B,T,N,N)
        if g:
            out = np.zeros((self.n_node,self.n_node))
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
        act_edges = np.squeeze([np.where((self.edges==act_edge).all(1))[0] for act_edge in self.act_edges])
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

    @tf.function
    def fit_eval(self,x,a,b,y,ex=None,ey=None,fit=True):
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a,True)
            if self.use_edge:
                ae = self.get_edge_action(a,True)
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a,True)
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            if self.roll:
                preds = []
                if self.use_edge:
                    edge_preds = []
                # TODO: what if not recurrent
                for i in range(b.shape[1]):
                    # ai = a[:,i:i+self.seq_out,...] if self.conv and self.act else a
                    inp = [x,b[:,i:i+self.seq_out,...]]
                    if self.conv:
                        inp += [self.filter]
                        inp += [adj[:,i:i+self.seq_out,...] if self.recurrent else adj] if self.act and self.use_adj else []
                    if self.use_edge:
                        inp += [ex]
                        inp += [self.edge_filter] if self.conv else []
                        inp += [ae[:,i:i+self.seq_out,...] if self.recurrent else ae] if self.act else []
                    pred = self.model(inp,training=fit) if self.dropout else self.model(inp)

                    if self.use_edge:
                        pred,edge_pred = pred

                    if self.act:
                        if self.use_edge:
                            edge_pred = concat([edge_pred[...,:-1],tf.multiply(edge_pred[...,-1:],ae[:,i:i+self.seq_out,...] if self.recurrent else ae)],axis=-1)                                
                        if not self.edge_fusion:
                            pred = concat([tf.stack([pred[...,0],
                            tf.multiply(pred[...,1],a_in[:,i:i+self.seq_out,...] if self.recurrent else a_in),
                            tf.multiply(pred[...,2],a_out[:,i:i+self.seq_out,...] if self.recurrent else a_out)],axis=-1),
                            pred[...,3:]],axis=-1)

                    if self.use_edge and self.edge_fusion:
                        edge_flow = self.normalize(edge_pred,'e',True)[...,-1:] if self.norm else edge_pred[...,-1:]
                        node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(edge_flow,-np.inf,0))
                        node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(edge_flow,-np.inf,0))
                        if self.norm:
                            node_outflow *= tf.cast(self.norm_y[0,:,2:3]>1e-3,tf.float32)
                            node_inflow *= tf.cast(self.norm_y[0,:,1:2]>1e-3,tf.float32)
                            node_outflow /= self.norm_y[0,:,2:3]
                            node_inflow /= self.norm_y[0,:,1:2]
                            # node_outflow = tf.clip_by_value(node_outflow,-10,10)
                            # node_inflow = tf.clip_by_value(node_inflow,-10,10)
                        pred = concat([pred[...,:1],node_inflow,node_outflow,pred[...,1:]],axis=-1)

                    preds.append(pred)
                    x = concat([x[:,1:,...],concat([pred[:,:1,...],b[:,i:i+self.seq_out,:,:1]],axis=-1)],axis=1) if self.recurrent else concat([pred,b[...,:1]],axis=-1)
                    if self.use_edge:
                        edge_preds.append(edge_pred)
                        if self.act:
                            ex = concat([ex[:,1:,...],concat([edge_pred[:,:1,...],ae[:,i:i+self.seq_out,...]],axis=-1)],axis=1) if self.recurrent else concat([edge_pred,ae],axis=-1)
                        else:
                            ex = concat([ex[:,1:,...],concat([edge_pred[:,:1,...],ex[:-1:,:,-1:]],axis=-1)],axis=1) if self.recurrent else concat([edge_pred,ex[...,-1:]],axis=-1)
                preds = concat(preds,axis=1)
                if self.use_edge:
                    edge_preds = concat(edge_preds,axis=1)
            else:
                inp = [x,b]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj] if self.act and self.use_adj else []
                if self.use_edge:
                    inp += [ex]
                    inp += [self.edge_filter] if self.conv else []
                    inp += [ae] if self.act else []
                preds = self.model(inp,training=fit) if self.dropout else self.model(inp)

                if self.use_edge:
                    preds,edge_preds = preds

                if self.act:
                    if self.use_edge:
                        edge_preds = concat([edge_preds[...,:-1],tf.multiply(edge_preds[...,-1:],ae)],axis=-1)
                    if not self.edge_fusion:
                        preds = concat([tf.stack([preds[...,0],tf.multiply(preds[...,1],a_in),tf.multiply(preds[...,2],a_out)],axis=-1),preds[...,3:]],axis=-1)

                if self.use_edge and self.edge_fusion:
                    edge_flow = self.normalize(edge_preds,'e',True)[...,-1:] if self.norm else edge_preds[...,-1:]
                    node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(edge_flow,-np.inf,0))
                    node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(edge_flow,-np.inf,0))
                    if self.norm:
                        node_outflow *= tf.cast(self.norm_y[0,:,2:3]>1e-3,tf.float32)
                        node_inflow *= tf.cast(self.norm_y[0,:,1:2]>1e-3,tf.float32)
                        node_outflow /= self.norm_y[0,:,2:3]
                        node_inflow /= self.norm_y[0,:,1:2]
                        # node_outflow = tf.clip_by_value(node_outflow,-10,10)
                        # node_inflow = tf.clip_by_value(node_inflow,-10,10)
                    preds = concat([preds[...,:1],node_inflow,node_outflow,preds[...,1:]],axis=-1)
            
            # Loss funtion
            if self.balance:
                if self.norm:
                    preds_re_norm = self.normalize(preds,'y',inverse=True)
                    b = self.normalize(b,'b',inverse=True)
                    q_w,preds_re_norm = self.constrain_tf(preds_re_norm,b[...,:1],self.normalize(x[:,-1:,:,0],'x',True))
                    # q_w = self.get_flood(preds_re_norm,b[...,:1])
                    q_w = q_w/self.norm_y[0,:,-1]
                    preds = self.normalize(preds_re_norm,'y')
                else:
                    q_w,preds = self.constrain_tf(preds,b[...,:1],x[:,-1:,:,0])
                    # q_w = self.get_flood(preds,b[...,:1])
                q_w = expand_dims(q_w,axis=-1)
            # narrow down norm range of water head
            if self.norm and self.hmin.max() > 0:
                # preds = concat([expand_dims(self.head_to_depth(self.normalize(preds,'y',True)[...,0]),axis=-1),
                #                 preds[...,1:]],axis=-1)
                # y = concat([expand_dims(self.head_to_depth(self.normalize(y,'y',True)[...,0]),axis=-1),
                #             y[...,1:]],axis=-1)
                # or use sample weights
                wei = (self.norm_y[0,:,0].max()-self.norm_y[1,:,0].min())/(self.hmax-self.hmin).mean()
                preds = concat([preds[...,:1] * wei,preds[...,1:]],axis=-1)
                y = concat([y[...,:1] * wei,y[...,1:]],axis=-1)
            preds = tf.clip_by_value(preds,0,1) # avoid large loss value
            if self.balance:
                loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],q_w],axis=-1))
            else:
                loss = self.mse(y[...,:3],preds[...,:3])
            if not fit:
                loss = [loss]
            if self.if_flood:
                loss += self.bce(y[...,-2:-1],preds[...,-1:]) if fit else [self.bce(y[...,-2:-1],preds[...,-1:])]
                # loss += self.cce(y[...,-3:-1],preds[...,-2:]) if fit else [self.cce(y[...,-3:-1],preds[...,-2:])]
            if self.use_edge:
                edge_preds = tf.clip_by_value(edge_preds,0,1) # avoid large loss value
                loss += self.mse(edge_preds,ey) if fit else [self.mse(edge_preds,ey)]
        if fit:
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        #     return loss.numpy()
        # else:
        #     return [los.numpy() for los in loss]
        return loss


    def update_net(self,dG,ratio=None,epochs=None,batch_size=None,train_ids=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs

        seq = max(self.seq_in,self.seq_out) if self.recurrent else 0
        n_events = int(max(dG.event_id))+1

        if train_ids is None:
            ratio = self.ratio if ratio is None else ratio
            train_ids = np.random.choice(np.arange(n_events),int(n_events*ratio),replace=False)
            test_ids = [ev for ev in range(n_events) if ev not in train_ids]
        else:
            test_ids = [ev for ev in range(n_events) if ev not in train_ids]

        # try:
        #     train_datas = dG.prepare(seq,train_ids)
        #     test_datas = dG.prepare(seq,test_ids)
        #     _dataloaded = True
        # except:
        print('Full data load failed, prepare per batch')
        _dataloaded = False
        train_idxs = dG.get_data_idxs(seq,train_ids)
        test_idxs = dG.get_data_idxs(seq,test_ids)

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            # Training
            # if _dataloaded:
            #     idxs = np.random.choice(range(train_datas[0].shape[0]),batch_size)
            #     train_dats = [dat[idxs] if dat is not None else dat for dat in train_datas]
            # else:
            train_dats = dG.prepare_batch(train_idxs,seq,batch_size)
            x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if self.use_edge:
                ex,ey = [dat for dat in train_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            train_loss = self.fit_eval(x,a,b,y,ex,ey)
            train_loss = train_loss.numpy()
            if epoch >= 500:
                train_losses.append(train_loss)

            # Validation
            # if _dataloaded:
            #     idxs = np.random.choice(range(test_datas[0].shape[0]),batch_size)
            #     test_dats = [dat[idxs] if dat is not None else dat for dat in test_datas]
            # else:
            test_dats = dG.prepare_batch(test_idxs,seq,batch_size)
            x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if self.use_edge:
                ex,ey = [dat for dat in test_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            test_loss = self.fit_eval(x,a,b,y,ex,ey,fit=False)
            test_loss = [los.numpy() for los in test_loss]
            if epoch >= 500:
                test_losses.append(test_loss)

            # Log output
            log = "Epoch {}/{} Train loss: {:.4f} Test loss: {:.4f}".format(epoch,epochs,train_loss,sum(test_loss))
            log += " ("
            node_str = "Node bal: " if self.balance else "Node: "
            log += node_str + "{:.4f}".format(test_loss[0])
            i = 1
            if self.if_flood:
                log += " if_flood: {:.4f}".format(test_loss[i])
                i += 1
            if self.use_edge:
                log += " Edge: {:.4f}".format(test_loss[i])
            log += ")"
            print(log)

            if train_loss < min([1e6]+train_losses[:-1]):
                self.save(os.path.join(self.model_dir,'train'))
            if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
                self.save(os.path.join(self.model_dir,'test'))
        return train_ids,test_ids,train_losses,test_losses

    def simulate(self,states,runoff,a=None,edge_states=None):
        # runoff shape: T_out, T_in, N
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            if self.use_edge:
                ae = self.get_edge_action(a)
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a)
        preds = []
        if self.use_edge:
            edge_preds = []
        for idx,bi in enumerate(runoff):
            x = states[idx,-self.seq_in:,...] if self.recurrent else states[idx]
            if self.use_edge:
                ex = edge_states[idx,-self.seq_in:,...] if self.recurrent else edge_states[idx]
            if self.roll:
                # TODO: What if not recurrent
                qws,ys = [],[]
                if self.use_edge:
                    eys = []
                for i in range(bi.shape[0]):
                    b_i = bi[i:i+self.seq_out]
                    inp = [self.normalize(x,'x'),self.normalize(b_i,'r')] if self.norm else [x,b_i]
                    inp = [expand_dims(dat,0) for dat in inp]
                    if self.conv:
                        inp += [self.filter]
                        inp += [adj[idx:idx+1,i:i+self.seq_out,...] if self.recurrent else adj[idx]] if self.act and self.use_adj else []
                    if self.use_edge:
                        inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
                        inp += [self.edge_filter] if self.conv else []
                        inp += [ae[idx:idx+1,i:i+self.seq_out,...] if self.recurrent else ae[idx]] if self.act else []
                    y = self.model(inp,training=False) if self.dropout else self.model(inp)

                    if self.use_edge:
                        y,ey = y
                        ey = squeeze(ey,0).numpy()
                        if self.norm:
                            ey = self.normalize(ey,'e',True)
                        ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
                    y = squeeze(y,0).numpy()
                    if self.act:
                        if self.use_edge:
                            ey[...,-1:] *= ae[idx,i:i+self.seq_out,...] if self.recurrent else ae[idx]
                        if not self.edge_fusion:
                            y[...,2] *= a_out[idx,i:i+self.seq_out,...] if self.recurrent else a_out[idx,...]
                            y[...,1] *= a_in[idx,i:i+self.seq_out,...] if self.recurrent else a_in[idx,...]

                    if self.use_edge and self.edge_fusion:
                        node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
                        node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
                        
                        if self.norm:
                            node_outflow = node_outflow*(self.norm_y[0,:,2:3]>1e-3)/self.norm_y[0,:,2:3]
                            node_inflow = node_inflow*(self.norm_y[0,:,1:2]>1e-3)/self.norm_y[0,:,1:2]
                        y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)

                    if self.norm:
                        y = self.normalize(y,'y',True)

                    q_w,y = self.constrain(y,b_i[...,:1],x[-1:,:,0])
                    x = np.concatenate([x[1:],np.concatenate([y[:1],b_i[...,:1]],axis=-1)],axis=0) if self.recurrent else np.concatenate([y,b_i[...,:1]],axis=-1)
                    qws.append(q_w)
                    ys.append(y)
                    if self.use_edge:
                        ex = np.concatenate([ex[1:],np.concatenate([ey[:1],ae[idx,i:i+self.seq_out,...] if self.recurrent else ae[idx]],axis=-1) if self.act else np.concatenate([ey[:1],ex[-1:,:,-1:]],axis=-1)],axis=0) if self.recurrent else np.concatenate([ey,ex[...,-1:]],axis=-1)
                        eys.append(ey)

                q_w,y = np.concatenate(qws,axis=0),np.concatenate(ys,axis=0)
                if self.use_edge:
                    ey = np.concatenate(eys,axis=0)
            else:
                bi = bi[:self.seq_out] if self.recurrent else bi

                inp = [self.normalize(x,'x'),self.normalize(bi,'b')] if self.norm else [x,bi]
                inp = [expand_dims(dat,0) for dat in inp]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj[idx:idx+1]] if self.act and self.use_adj else []
                if self.use_edge:
                    inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
                    inp += [self.edge_filter] if self.conv else []
                    inp += [ae[idx:idx+1]] if self.act else []
                y = self.model(inp,training=False) if self.dropout else self.model(inp)

                if self.use_edge:
                    y,ey = y
                    ey = squeeze(ey,0).numpy()
                    if self.norm:
                        ey = self.normalize(ey,'e',True)
                    ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
                y = squeeze(y,0).numpy()
                if self.act:
                    if self.use_edge:
                        ey[...,-1:] *= ae[idx]
                    if not self.edge_fusion:
                        y[...,2] *= a_out[idx,...]
                        y[...,1] *= a_in[idx,...]

                if self.use_edge and self.edge_fusion:
                    node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
                    node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
                    if self.norm:
                        node_outflow = node_outflow*(self.norm_y[0,:,2:3]>1e-3)/self.norm_y[0,:,2:3]
                        node_inflow = node_inflow*(self.norm_y[0,:,1:2]>1e-3)/self.norm_y[0,:,1:2]
                    y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)

                if self.norm:
                    y = self.normalize(y,'y',True)
                q_w,y = self.constrain(y,bi[...,:1],x[-1:,:,0])
                # q_w = self.get_flood(y,bi)
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            if self.use_edge:
                edge_preds.append(ey)
        if self.use_edge:
            return np.array(preds),np.array(edge_preds)
        else:
            return np.array(preds)

    # No roll
    # TODO
    def predict(self,states,b,a=None,edge_state=None):
        x = states[:,-self.seq_in:,...] if self.recurrent else states
        if edge_state is not None:
            ex = edge_state[:,-self.seq_in:,...] if self.recurrent else states
        assert b.shape[1] == self.seq_out
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            if self.use_edge:
                ae = self.get_edge_action(a)
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a)
        inp = [self.normalize(x,'x'),self.normalize(b,'b')] if self.norm else [x,b]
        # inp = [expand_dims(dat,0) for dat in inp]
        if self.conv:
            inp += [self.filter]
            inp += [adj] if self.act and self.use_adj else []
        if self.use_edge:
            # inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
            inp += [self.normalize(ex,'e') if self.norm else ex]
            inp += [self.edge_filter] if self.conv else []
            inp += [ae] if self.act else []
        y = self.model(inp,training=False) if self.dropout else self.model(inp)
        if self.use_edge:
            y,ey = y
            ey = ey.numpy()
            if self.norm:
                ey = self.normalize(ey,'e',True)
            ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
        y = y.numpy()
        if self.act:
            if self.use_edge:
                ey[...,-1:] *= ae
            if not self.edge_fusion:
                y[...,2] *= a_out
                y[...,1] *= a_in
        if self.use_edge and self.edge_fusion:
            node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
            node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
            if self.norm:
                node_outflow = node_outflow*(self.norm_y[0,:,2:3]>1e-3)/self.norm_y[0,:,2:3]
                node_inflow = node_inflow*(self.norm_y[0,:,1:2]>1e-3)/self.norm_y[0,:,1:2]
            y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)
        if self.norm:
            y = self.normalize(y,'y',True)
        q_w,y = self.constrain(y,b[...,:1],x[:,-1:,:,0])
        y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
        return y,ey if self.use_edge else y

    @tf.function
    def predict_tf(self,states,b,a=None,edge_state=None):
        x = states[:,-self.seq_in:,...] if self.recurrent else states
        if edge_state is not None:
            ex = edge_state[:,-self.seq_in:,...] if self.recurrent else states
        assert b.shape[1] == self.seq_out
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a,True)
            if self.use_edge:
                ae = self.get_edge_action(a,True)
            if not self.edge_fusion:
                a_out,a_in = self.get_action(a,True)
        inp = [self.normalize(x,'x'),self.normalize(b,'b')] if self.norm else [x,b]
        # inp = [expand_dims(dat,0) for dat in inp]
        if self.conv:
            inp += [self.filter]
            inp += [adj] if self.act and self.use_adj else []
        if self.use_edge:
            # inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
            inp += [self.normalize(ex,'e') if self.norm else ex]
            inp += [self.edge_filter] if self.conv else []
            inp += [ae] if self.act else []
        y = self.model(inp,training=False) if self.dropout else self.model(inp)
        if self.use_edge:
            y,ey = y
            if self.norm:
                ey = self.normalize(ey,'e',True)
            ey = tf.concat([tf.expand_dims(tf.clip_by_value(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
        if self.act:
            if self.use_edge:
                ey = concat([ey[...,:-1],tf.multiply(ey[...,-1:],ae)],axis=-1)
            if not self.edge_fusion:
                y = concat([tf.stack([y[...,0],tf.multiply(y[...,1],a_in),tf.multiply(y[...,2],a_out)],axis=-1),y[...,3:]],axis=-1)

        if self.use_edge and self.edge_fusion:
            node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(ey[...,-1:],0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(ey[...,-1:],-np.inf,0))
            node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(ey[...,-1:],0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(ey[...,-1:],-np.inf,0))
            if self.norm:
                node_outflow *= tf.cast(self.norm_y[0,:,2:3]>1e-3,tf.float32)/self.norm_y[0,:,2:3]
                node_inflow *= tf.cast(self.norm_y[0,:,1:2]>1e-3,tf.float32)/self.norm_y[0,:,1:2]
            y = tf.concat([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)
        if self.norm:
            y = self.normalize(y,'y',True)
        q_w,y = self.constrain_tf(y,b[...,:1],x[:,-1:,:,0])
        y = tf.concat([y,tf.expand_dims(q_w,axis=-1)],axis=-1)
        return y,ey if self.use_edge else y
        
    def constrain(self,y,r,h0):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = np.clip(h,self.hmin,self.hmax)
        # dv = self.area * np.diff(np.concatenate([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = np.clip(q_us + r - q_ds - dv,0,np.inf) * (1-self.is_outfall)
        if self.if_flood:
            # f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            f = y[...,-1] > 0.5
            q_w *= f
            h = self.hmax * f + h * ~f
            # y = np.stack([h,q_us,q_ds,y[...,-2],y[...,-1]],axis=-1)
            y = np.stack([h,q_us,q_ds,y[...,-1]],axis=-1)
        else:
            if self.epsilon > 0:
                q_w *= ((self.hmax - h) < self.epsilon)
            y = np.stack([h,q_us,q_ds],axis=-1)
        return q_w,y
    
    def constrain_tf(self,y,r,h0):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = tf.squeeze(r,axis=-1)
        h = tf.clip_by_value(h,self.hmin,self.hmax)
        # dv = self.area * tf.experimental.numpy.diff(tf.concat([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = tf.clip_by_value(q_us + r - q_ds - dv,0,np.inf) * tf.cast(1-self.is_outfall,tf.float32)
        if self.if_flood:
            # f = tf.cast(tf.argmax(y[...,-2:],axis=-1),tf.float32)
            f = tf.cast(y[...,-1] > 0.5,tf.float32)
            q_w *= f
            h = self.hmax * f + h * (1-f)
            y = tf.stack([h,q_us,q_ds,y[...,-1]],axis=-1)
        else:
            if self.epsilon > 0:
                q_w = q_w * tf.cast((self.hmax - h) < self.epsilon, tf.float32)
            y = tf.stack([h,q_us,q_ds],axis=-1)
        return q_w,y
    
    def get_flood(self,y,r,h0):
        # B,T,N
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = np.clip(h,self.hmin,self.hmax)
        # dv = self.area * np.diff(np.concatenate([h0,h],axis=1),axis=1) if self.area.max() > 0 else 0.0
        dv = 0.0
        q_w = np.clip(q_us + r - q_ds - dv,0,np.inf) * (1-self.is_outfall)
        if self.if_flood:
            # f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            f = y[...,-1] > 0.5
            q_w *= f
        elif self.epsilon > 0:
            q_w *= ((self.hmax - h) < self.epsilon)
        return q_w
    
    def head_to_depth(self,h):
        h = tf.clip_by_value(h,self.hmin,self.hmax)
        return (h-self.hmin)/(self.hmax-self.hmin)

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
            for item in 'xbye':
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
            for item in 'xbye':
                if os.path.exists(os.path.join(model_dir,'norm_%s.npy'%item)):
                    setattr(self,'norm_%s'%item,np.load(os.path.join(model_dir,'norm_%s.npy'%item)))