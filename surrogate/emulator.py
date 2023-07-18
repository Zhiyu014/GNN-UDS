from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat,sqrt
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D,Softmax,Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError,CategoricalCrossentropy
from tensorflow.keras import mixed_precision
import numpy as np
import os
from spektral.layers import GCNConv,GATConv
from spektral.utils.convolution import gcn_filter
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# mixed_precision.set_global_policy('mixed_float16')

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
        return tf.matmul(self.w * self.inci + self.b, inputs)
    

class Emulator:
    def __init__(self,conv=None,resnet=False,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        # Runoff is boundary
        self.b_in = 1
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
        self.activation = getattr(args,"activation",'relu')
        self.norm = getattr(args,"norm",False)
        self.balance = getattr(args,"balance",0)
        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 2

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

        self.adj = getattr(args,"adj",np.eye(self.n_node))
        if self.act:
            self.act_edges = getattr(args,"act_edges")
            self.use_adj = getattr(args,"use_adj",False)
        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))

        self.conv = False if conv in ['None','False','NoneType'] else conv
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.model = self.build_network(self.conv,resnet,self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        self.mse = MeanSquaredError()
        self.cce = CategoricalCrossentropy()

        self.ratio = getattr(args,"ratio",0.8)
        self.batch_size = getattr(args,"batch_size",256)
        self.epochs = getattr(args,"epochs",100)
        self.model_dir = getattr(args,"model_dir")
        
    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        input_shape,bound_input = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if recurrent:
            input_shape = (self.seq_in,) + input_shape
            bound_input = (self.seq_out,) + bound_input
        X_in = Input(shape=input_shape)
        B_in = Input(shape=bound_input)
        inp = [X_in,B_in]
        if conv:
            Adj_in = Input(self.n_node,)
            inp += [Adj_in]
            if 'GCN' in conv:
                net = GCNConv
                self.filter = gcn_filter(self.adj)
                if self.use_edge:
                    self.edge_filter = gcn_filter(self.edge_adj)
            elif 'GAT' in conv:
                net = GATConv
                self.filter = self.adj.copy()
                if self.use_edge:
                    self.edge_filter = self.edge_adj.copy()
            else:
                raise AssertionError("Unknown Convolution layer %s"%str(conv))
            if self.act and self.use_adj:
                A_in = Input(self.n_node,)
                inp += [A_in]
        else:
            net = Dense


        if self.use_edge:
            edge_input_shape = (self.seq_in,self.n_edge,self.e_in,) if recurrent else (self.n_edge,self.e_in,)
            E_in = Input(shape=edge_input_shape)
            inp += [E_in]
            if conv:
                Eadj_in = Input(self.n_edge,)
                inp += [Eadj_in]
            if self.act:
                AE_in = Input(shape=edge_input_shape[:-1]+(1,))
                inp += [AE_in]

        # Embedding block
        # (B,T,N,in) (B,N,in)--> (B,T,N*in) (B,N*in)
        x = reshape(X_in,(-1,)+tuple(input_shape[:-2])+(self.n_node*self.n_in,)) if not conv else X_in
        x = Dense(self.embed_size,activation=self.activation)(x) # Embedding
        res = [x]
        b = Dense(self.embed_size//2,activation=self.activation)(B_in)  # Boundary Embedding (B,T_out,N,E)
        if self.use_edge:
            e = Dense(self.embed_size,activation=self.activation)(E_in)  # Edge attr Embedding (B,T_in,N,E)
            res_e = [e]
            # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
            e = reshape(e,(-1,) + tuple(e.shape[2:])) if recurrent else e
            if self.act:
                ae = Dense(self.embed_size//2,activation=self.activation)(AE_in)  # Control Embedding (B,T_out,N,E)

        # Spatial block
        # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        for _ in range(self.n_sp_layer):
            if self.use_edge and self.edge_fusion:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                # x = concat([x,tf.matmul(tf.abs(self.node_edge),x_e)],axis=-1)
                # e = concat([e,tf.matmul(transpose(tf.abs(self.node_edge)),e_x)],axis=-1)
            x = [x,Adj_in] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            # b = Dense(self.embed_size//2,activation=self.activation)(b)
            if self.use_edge:
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)


        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x_out = reshape(x,(-1,)+input_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+input_shape[:-2]+(x.shape[-1],)) 
        res += [x_out]
        #  (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> （B,T,N,E)
        x = Add()(res) if resnet else x_out

        if self.use_edge:
            # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
            e_out = reshape(e,(-1,)+edge_input_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_input_shape[:-2]+(e.shape[-1],)) 
            res_e += [e_out]
            #  (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B,T,N,E)
            e = Add()(res_e) if resnet else e_out


        # Recurrent block
        if recurrent:
            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E)
            x = transpose(x,[0,2,1,3]) if conv else x
            b = transpose(b,[0,2,1,3])
            if self.use_edge:
                e = transpose(e,[0,2,1,3]) if conv else e
                if self.act:
                    ae = transpose(ae,[0,2,1,3])
            if recurrent == 'Conv1D':
                x_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:]) for i in range(self.n_tp_layer)]
                b_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=b.shape[-2:]) for i in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=e.shape[-2:]) for i in range(self.n_tp_layer)]
                    if self.act:
                        ae_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=ae.shape[-2:]) for i in range(self.n_tp_layer)]
            elif recurrent == 'GRU':
                x_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                b_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
                if self.use_edge:
                    e_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
                    if self.act:
                        ae_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
            else:
                raise AssertionError("Unknown recurrent layer %s"%str(recurrent))
            
            # (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
            x = reshape(x,(-1,self.seq_in,x.shape[-1])) if conv else x
            b = reshape(b,(-1,self.seq_out,b.shape[-1]))
            # (B*N,T,E) (B,T,E) --> (B*N,T_out,H) (B,T_out,H)
            for i in range(self.n_tp_layer):
                x = x_tem_nets[i](x)
                b = b_tem_nets[i](b)
            x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll
            # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H)
            x = reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)) if conv else x
            b = reshape(b,(-1,self.n_node,self.seq_out,b.shape[-1]))
            # (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
            x = transpose(x,[0,2,1,3]) if conv else x
            b = transpose(b,[0,2,1,3])
            if self.use_edge:
                e = reshape(e,(-1,self.seq_in,e.shape[-1])) if conv else e
                for i in range(self.n_tp_layer):
                    e = e_tem_nets[i](e)
                e = e[...,-self.seq_out:,:] # seq_in >= seq_out if roll
                e = reshape(e,(-1,self.n_edge,self.seq_out,e.shape[-1])) if conv else e
                e = transpose(e,[0,2,1,3]) if conv else e
                if self.act:
                    ae = reshape(ae,(-1,self.seq_out,ae.shape[-1]))
                    for i in range(self.n_tp_layer):
                        ae = ae_tem_nets[i](ae)
                    ae = reshape(ae,(-1,self.n_edge,self.seq_out,ae.shape[-1]))
                    ae = transpose(ae,[0,2,1,3])

        # boundary in
        x = concat([x,b],axis=-1)
        if self.use_edge and self.act:
            e = concat([e,ae],axis=-1)

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
            if self.use_edge and self.edge_fusion:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = concat([e,NodeEdge(transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                # x = concat([x,tf.matmul(tf.abs(self.node_edge),x_e)],axis=-1)
                # e = concat([e,tf.matmul(transpose(tf.abs(self.node_edge)),e_x)],axis=-1)
            x = [x,A] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            if self.use_edge:
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x = reshape(x,(-1,)+input_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+input_shape[:-2]+(x.shape[-1],)) 

        if self.use_edge:
            # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
            e = reshape(e,(-1,)+edge_input_shape[:-1]+(e.shape[-1],)) if conv else reshape(e,(-1,)+edge_input_shape[:-2]+(e.shape[-1],)) 


        out_shape = self.n_out if conv else self.n_out * self.n_node
        # (B,T_out,N,H) (B,T_out,H) --> (B,T_out,N,n_out)
        out = Dense(out_shape,activation='linear')(x)
        out = reshape(out,(-1,self.seq_out,self.n_node,self.n_out))
        if self.if_flood:
            out_shape = 2 if conv else 2 * self.n_node
            flood = Dense(out_shape,activation='linear')(x)
            flood = reshape(flood,(-1,self.seq_out,self.n_node,2))
            flood = Softmax()(flood)
            out = concat([out,flood],axis=-1)

        if self.use_edge:
            out_shape = self.e_out if conv else self.e_out * self.n_edge
            e_out = Dense(out_shape,activation='linear')(e)
            e_out = reshape(e_out,(-1,self.seq_out,self.n_edge,self.e_out))
            out = [out,e_out]

        model = Model(inputs=inp, outputs=out)
        return model

    def get_adj_action(self,a):
        # (B,T,N_act) --> (B,T,N,N)
        def get_act(s):
            adj = self.adj.copy()
            adj[tuple(self.act_edges.T)] = s
            return adj
        adj = np.apply_along_axis(get_act,-1,a)

        if 'GCN' in self.conv:
            adj = gcn_filter(adj)
        return adj

    def get_action(self,a):
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
    
    def get_edge_action(self,a):
        act_edges = np.squeeze([np.where((self.edges==act_edge).all(1))[0] for act_edge in self.act_edges])
        def set_edge_action(s):
            out = np.ones(self.n_edge)
            out[act_edges] = s
            return out
        return np.expand_dims(np.apply_along_axis(set_edge_action,-1,a),-1)

    def fit_eval(self,x,a,b,y,ex=None,ey=None,fit=True):
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            if self.use_edge:
                ae = self.get_edge_action(a)
            else:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a)
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
                    pred = self.model(inp)

                    if self.use_edge:
                        pred,edge_pred = pred

                    if self.act:
                        if self.use_edge:
                            edge_pred = concat([edge_pred[...,:-1],tf.multiply(edge_pred[...,-1:],ae[:,i:i+self.seq_out,...] if self.recurrent else ae)],axis=-1)                                
                        if not (self.use_edge and self.edge_fusion):
                            pred = concat([tf.stack([pred[...,0],
                            tf.multiply(pred[...,1],a_in[:,i:i+self.seq_out,...] if self.recurrent else a_in),
                            tf.multiply(pred[...,2],a_out[:,i:i+self.seq_out,...] if self.recurrent else a_out)],axis=-1),
                            pred[...,3:]],axis=-1)

                    if self.use_edge and self.edge_fusion:
                        edge_flow = self.normalize(edge_pred,'e',True)[...,-1:] if self.norm else edge_pred[...,-1:]
                        node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(edge_flow,-np.inf,0))
                        node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(edge_flow,-np.inf,0))
                        if self.norm:
                            node_outflow *= tf.cast(self.norm_y[:,2:3]>1e-3,tf.float32)
                            node_inflow *= tf.cast(self.norm_y[:,1:2]>1e-3,tf.float32)
                            node_outflow /= self.norm_y[:,2:3]
                            node_inflow /= self.norm_y[:,1:2]
                            # node_outflow = tf.clip_by_value(node_outflow,-10,10)
                            # node_inflow = tf.clip_by_value(node_inflow,-10,10)
                        pred = concat([pred[...,:1],node_inflow,node_outflow,pred[...,1:]],axis=-1)

                    preds.append(pred)
                    x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
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
                preds = self.model(inp)

                if self.use_edge:
                    preds,edge_preds = preds

                if self.act:
                    if self.use_edge:
                        edge_preds = concat([edge_preds[...,:-1],tf.multiply(edge_preds[...,-1:],ae)],axis=-1)
                    if not (self.use_edge and self.edge_fusion):
                        preds = concat([tf.stack([preds[...,0],tf.multiply(preds[...,1],a_in),tf.multiply(preds[...,2],a_out)],axis=-1),preds[...,3:]],axis=-1)

                if self.use_edge and self.edge_fusion:
                    edge_flow = self.normalize(edge_preds,'e',True)[...,-1:] if self.norm else edge_preds[...,-1:]
                    node_outflow = tf.matmul(tf.clip_by_value(self.node_edge,0,1),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),-tf.clip_by_value(edge_flow,-np.inf,0))
                    node_inflow = tf.matmul(tf.abs(tf.clip_by_value(self.node_edge,-1,0)),tf.clip_by_value(edge_flow,0,np.inf)) + tf.matmul(tf.clip_by_value(self.node_edge,0,1),-tf.clip_by_value(edge_flow,-np.inf,0))
                    if self.norm:
                        node_outflow *= tf.cast(self.norm_y[:,2:3]>1e-3,tf.float32)
                        node_inflow *= tf.cast(self.norm_y[:,1:2]>1e-3,tf.float32)
                        node_outflow /= self.norm_y[:,2:3]
                        node_inflow /= self.norm_y[:,1:2]
                        # node_outflow = tf.clip_by_value(node_outflow,-10,10)
                        # node_inflow = tf.clip_by_value(node_inflow,-10,10)
                    preds = concat([preds[...,:1],node_inflow,node_outflow,preds[...,1:]],axis=-1)
            
            # Loss funtion
            if self.balance:
                if self.norm:
                    preds_re_norm = self.normalize(preds,'y',inverse=True)
                    b = self.normalize(b,'b',inverse=True)
                    q_w = self.get_flood(preds_re_norm,b)
                    q_w = q_w/self.norm_y[...,-1]
                else:
                    q_w = self.get_flood(preds,b)
                # loss += self.balance_alpha * self.mse(y[...,-1],q_w)
                loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],expand_dims(q_w,axis=-1)],axis=-1))
            else:
                loss = self.mse(y[...,:3],preds[...,:3])
            if not fit:
                loss = [loss]
            if self.if_flood:
                loss += self.cce(y[...,-3:-1],preds[...,-2:]) if fit else [self.cce(y[...,-3:-1],preds[...,-2:])]
            if self.use_edge:
                loss += self.mse(edge_preds,ey) if fit else [self.mse(edge_preds,ey)]
        if fit:
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
            return loss.numpy()
        else:
            return [los.numpy() for los in loss]


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

        train_dats = dG.prepare(seq,train_ids)
        test_dats = dG.prepare(seq,test_ids)

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            # Training
            idxs = np.random.choice(range(train_dats[0].shape[0]),batch_size)
            x,a,b,y = [dat[idxs] if dat is not None else dat for dat in train_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if self.use_edge:
                ex,ey = [dat[idxs] for dat in train_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            train_loss = self.fit_eval(x,a,b,y,ex,ey)
            train_losses.append(train_loss)

            # Validation
            idxs = np.random.choice(range(test_dats[0].shape[0]),batch_size)
            x,a,b,y = [dat[idxs] if dat is not None else dat for dat in test_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]
            if self.use_edge:
                ex,ey = [dat[idxs] for dat in test_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            test_loss = self.fit_eval(x,a,b,y,ex,ey,fit=False)
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
            else:
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
                    y = self.model(inp)

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
                        if not (self.use_edge and self.edge_fusion):
                            y[...,2] *= a_out[idx,i:i+self.seq_out,...] if self.recurrent else a_out[idx,...]
                            y[...,1] *= a_in[idx,i:i+self.seq_out,...] if self.recurrent else a_in[idx,...]

                    if self.use_edge and self.edge_fusion:
                        node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
                        node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
                        
                        if self.norm:
                            node_outflow = node_outflow*(self.norm_y[:,2:3]>1e-3)/self.norm_y[:,2:3]
                            node_inflow = node_inflow*(self.norm_y[:,1:2]>1e-3)/self.norm_y[:,1:2]
                        y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)

                    if self.norm:
                        y = self.normalize(y,'y',True)

                    q_w,y = self.constrain(y,b_i)
                    x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
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
                y = self.model(inp,training=False)

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
                    if not (self.use_edge and self.edge_fusion):
                        y[...,2] *= a_out[idx,...]
                        y[...,1] *= a_in[idx,...]

                if self.use_edge and self.edge_fusion:
                    node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
                    node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
                    if self.norm:
                        node_outflow = node_outflow*(self.norm_y[:,2:3]>1e-3)/self.norm_y[:,2:3]
                        node_inflow = node_inflow*(self.norm_y[:,1:2]>1e-3)/self.norm_y[:,1:2]
                    y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)

                if self.norm:
                    y = self.normalize(y,'y',True)
                q_w,y = self.constrain(y,bi)
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
    def predict(self,states,runoff,a=None,edge_state=None):
        x = states[:,-self.seq_in:,...] if self.recurrent else states
        if edge_state is not None:
            ex = edge_state[:,-self.seq_in:,...] if self.recurrent else states
        n_roll = runoff.shape[1] // self.seq_out
        preds = []
        if self.use_edge:
            edge_preds = []
        for i in range(n_roll):
            sc = slice(i*self.seq_out,(i+1)*self.seq_out)
            b = runoff[:,sc,...]
            if self.act:
                if self.use_adj:
                    adj = self.get_adj_action(a[:,sc,...])
                if self.use_edge:
                    ae = self.get_edge_action(a[:,sc,...])
                else:
                    a_out,a_in = self.get_action(a[:,sc,...])
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
            y = self.model(inp)
            if self.use_edge:
                y,ey = y
                # ey = squeeze(ey,0).numpy()
                ey = ey.numpy()
                if self.norm:
                    ey = self.normalize(ey,'e',True)
                ey = np.concatenate([np.expand_dims(np.clip(ey[...,0],0,self.ehmax),axis=-1),ey[...,1:]],axis=-1)
            # y = squeeze(y,0).numpy()
            y = y.numpy()
            if self.act:
                if self.use_edge:
                    ey[...,-1:] *= ae
                if not (self.use_edge and self.edge_fusion):
                    y[...,2] *= a_out
                    y[...,1] *= a_in
            if self.use_edge and self.edge_fusion:
                node_outflow = np.matmul(np.clip(self.node_edge,0,1),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.abs(np.clip(self.node_edge,-1,0)),-np.clip(ey[...,-1:],-np.inf,0))
                node_inflow = np.matmul(np.abs(np.clip(self.node_edge,-1,0)),np.clip(ey[...,-1:],0,np.inf)) + np.matmul(np.clip(self.node_edge,0,1),-np.clip(ey[...,-1:],-np.inf,0))
                if self.norm:
                    node_outflow = node_outflow*(self.norm_y[:,2:3]>1e-3)/self.norm_y[:,2:3]
                    node_inflow = node_inflow*(self.norm_y[:,1:2]>1e-3)/self.norm_y[:,1:2]
                y = np.concatenate([y[...,:1],node_inflow,node_outflow,y[...,1:]],axis=-1)
            if self.norm:
                y = self.normalize(y,'y',True)
            q_w,y = self.constrain(y,b)

            if self.recurrent and self.seq_in > self.seq_out:    
                x = np.concatenate([x[:,self.seq_out-self.seq_in:,...],
                                    np.concatenate([y[...,:-2],b,y[...,-2:]] if self.if_flood else [y,b],axis=-1)],axis=1)
                if self.use_edge:
                    ex = np.concatenate([ex[:,self.seq_out-self.seq_in:,...],np.concatenate([ey,ae],axis=-1) if self.act else ey],axis=1)
            else:
                x = np.concatenate([y[...,:-2],b,y[...,-2:]] if self.if_flood else [y,b],axis=-1)
                if self.use_edge:
                    ex = np.concatenate([ey,ae],axis=-1)

            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            if self.use_edge:
                edge_preds.append(ey)
        if self.use_edge:
            return np.concatenate(preds,axis=1),np.concatenate(edge_preds,axis=1)
        else:
            return np.concatenate(preds,axis=1)

    def constrain(self,y,r):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = np.clip(h,0,self.hmax)
        if self.if_flood:
            f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            q_w = np.clip(q_us + r - q_ds,0,np.inf) * f
            h = self.hmax * f + h * ~f
            y = np.stack([h,q_us,q_ds,y[...,-2],y[...,-1]],axis=-1)
        else:
            # q_w = np.clip(q_us + r - q_ds,0,np.inf) * ((self.hmax - h) < 0.1) * (self.hmax > 0)
            q_w = np.clip(q_us + r - q_ds,0,np.inf)
            y = np.stack([h,q_us,q_ds],axis=-1)
        return q_w,y
    
    def get_flood(self,y,r):
        # B,T,N
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        if self.if_flood:
            f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            q_w = np.clip(q_us + r - q_ds,0,np.inf) * f
        else:
            h = np.clip(h,0,self.hmax)
            # q_w = np.clip(q_us + r - q_ds,0,np.inf) * ((self.hmax - h) < 0.1) * (self.hmax > 0)
            q_w = np.clip(q_us + r - q_ds,0,np.inf)
        # err = q_us + r - q_ds - q_w
        # return tf.sqrt(self.mse(q_us + r, q_ds + q_w))/reduce_mean(q_us + r)
        return q_w
    
    def set_norm(self,norm_x,norm_b,norm_y,norm_e=None):
        setattr(self,'norm_x',norm_x)
        setattr(self,'norm_b',norm_b)
        setattr(self,'norm_y',norm_y)
        if norm_e is not None:
            setattr(self,'norm_e',norm_e)


    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        return dat * normal[:,:dim] if inverse else dat/normal[:,:dim]



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