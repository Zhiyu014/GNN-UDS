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
# mixed_precision.set_global_policy('mixed_float16')

# - **Model**: STGCN may be a possible method to handle with spatial-temporal prediction. Why such structure is needed?  TO reduce model
#     - [pytorch implementation](https://github.com/LMissher/STGNN)
#     - [original](https://github.com/VeritasYin/STGCN_IJCAI-18)
# - **Predict**: *T*-times 1-step prediction OR T-step prediction?

class Emulator:
    def __init__(self,conv=None,resnet=False,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        # Runoff is boundary
        self.b_in = 1
        self.act = getattr(args,"act",False)

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

        # TODO: edges fusion model
        self.edge_fusion = getattr(args,"edge_fusion",False)
        if self.edge_fusion:
            self.edges = getattr(args,"edges")
            self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
            self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
            self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)

        self.adj = getattr(args,"adj",np.eye(self.n_node))
        if self.act:
            self.act_edges = getattr(args,"act_edges")
            self.use_adj = getattr(args,"use_adj",False)
        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))


        self.conv = False if conv in ['None','False','NoneType'] else conv
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.model = self.build_network(self.conv,resnet,self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3))
        self.mse = MeanSquaredError()
        self.cce = CategoricalCrossentropy()

        self.ratio = getattr(args,"ratio",0.8)
        self.batch_size = getattr(args,"batch_size",256)
        self.epochs = getattr(args,"epochs",100)
        self.model_dir = getattr(args,"model_dir","./model/shunqing/model.h5")



    # TODO: node-edge fusion model
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
                if self.edge_fusion:
                    self.edge_filter = gcn_filter(self.edge_adj)
            elif 'GAT' in conv:
                net = GATConv
                self.filter = self.adj.copy()
                if self.edge_fusion:
                    self.edge_filter = self.edge_adj.copy()
            else:
                raise AssertionError("Unknown Convolution layer %s"%str(conv))
            if self.act and self.use_adj:
                A_in = Input(self.n_node,)
                inp += [A_in]
        else:
            net = Dense


        if self.edge_fusion:
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
        if self.edge_fusion:
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
            if self.edge_fusion:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = concat([x,tf.matmul(self.node_edge,x_e)],axis=-1)
                e = concat([e,tf.matmul(transpose(self.node_edge),e_x)],axis=-1)
            x = [x,Adj_in] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            # b = Dense(self.embed_size//2,activation=self.activation)(b)
            if self.edge_fusion:
                # TODO: edge fusion model
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)


        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x_out = reshape(x,(-1,)+input_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+input_shape[:-2]+(x.shape[-1],)) 
        res += [x_out]
        #  (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> （B,T,N,E)
        x = Add()(res) if resnet else x_out

        if self.edge_fusion:
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
            if self.edge_fusion:
                e = transpose(e,[0,2,1,3]) if conv else e
                if self.act:
                    ae = transpose(ae,[0,2,1,3])
            if recurrent == 'Conv1D':
                x_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:]) for i in range(self.n_tp_layer)]
                b_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=b.shape[-2:]) for i in range(self.n_tp_layer)]
                if self.edge_fusion:
                    e_tem_nets = [Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=e.shape[-2:]) for i in range(self.n_tp_layer)]
                    if self.act:
                        ae_tem_nets = [Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=ae.shape[-2:]) for i in range(self.n_tp_layer)]
            elif recurrent == 'GRU':
                x_tem_nets = [GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)]
                b_tem_nets = [GRU(self.hidden_dim//2,return_sequences=True) for _ in range(self.n_tp_layer)]
                if self.edge_fusion:
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
            if self.edge_fusion:
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
        if self.edge_fusion:
            e = concat([e,ae],axis=-1)

        # Spatial block 2
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        if self.edge_fusion:
            e = reshape(e,(-1,) + tuple(e.shape[2:])) if recurrent else e
        if conv:
            if self.act and self.use_adj:
                A = reshape(A_in,(-1,) + (self.n_node,self.n_node)) if recurrent else A_in
            else:
                A = Adj_in
        for _ in range(self.n_sp_layer):
            x = [x,A] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)
            if self.edge_fusion:
                e = [e,Eadj_in] if conv else e
                e = net(self.embed_size,activation=self.activation)(e)

                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                
                x = concat([x,tf.matmul(self.node_edge,x_e)],axis=-1)
                e = concat([e,tf.matmul(transpose(self.node_edge),e_x)],axis=-1)

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x = reshape(x,(-1,)+input_shape[:-1]+(x.shape[-1],)) if conv else reshape(x,(-1,)+input_shape[:-2]+(x.shape[-1],)) 

        if self.edge_fusion:
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

        if self.edge_fusion:
            out_shape = self.e_in-1 if conv else (self.e_in-1) * self.n_edge
            e_out = Dense(out_shape,activation='linear')(e)
            e_out = reshape(e_out,(-1,self.seq_out,self.n_edge,self.e_in-1))
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
        return np.apply_along_axis(set_edge_action,-1,a)


    # TODO: network input and output debugging
    def fit(self,x,a,b,y,ex=None,ey=None):
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            elif self.edge_fusion:
                ae = self.get_edge_action(a)
            else:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a)
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            if self.roll:
                preds = []
                if self.edge_fusion:
                    edge_preds = []
                # TODO: what if not recurrent
                for i in range(b.shape[1]):
                    # ai = a[:,i:i+self.seq_out,...] if self.conv and self.act else a
                    inp = [x,b[:,i:i+self.seq_out,...]]
                    if self.conv:
                        inp += [self.filter]
                        inp += [adj[:,i:i+self.seq_out,...]] if self.act and self.use_adj else []
                    if self.edge_fusion:
                        inp += [ex]
                        inp += [self.edge_filter] if self.conv else []
                        inp += [ae[:,i:i+self.seq_out,...]] if self.act else []
                    pred = self.model(inp)
                    if self.edge_fusion:
                        pred,edge_pred = pred
                        edge_preds.append(edge_pred)
                        ex = concat([ex[:,1:,...],edge_pred[:,:1,...]],axis=1) if self.recurrent else edge_pred
                    if self.act and not self.use_adj and not self.edge_fusion:
                        pred = concat([tf.stack([tf.clip_by_value(pred[...,0],0,1),tf.multiply(pred[...,1],a_in),tf.multiply(pred[...,2],a_out)],axis=-1),pred[...,3:]],axis=-1)
                    else:
                        pred = concat([tf.clip_by_value(pred[...,:1],0,1),pred[...,1:]],axis=-1)
                    preds.append(pred)
                    x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
                preds = concat(preds,axis=1)
                if self.edge_fusion:
                    edge_preds = concat(edge_preds,axis=1)
            else:
                inp = [x,b]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj] if self.act and self.use_adj else []
                if self.edge_fusion:
                    inp += [ex]
                    inp += [self.edge_filter] if self.conv else []
                    inp += [ae] if self.act else []
                preds = self.model(inp)
                if self.edge_fusion:
                    preds,edge_preds = preds
                if self.act and not self.use_adj and not self.edge_fusion:
                    preds = concat([tf.stack([tf.clip_by_value(preds[...,0],0,1),tf.multiply(preds[...,1],a_in),tf.multiply(preds[...,2],a_out)],axis=-1),preds[...,3:]],axis=-1)

            if self.balance:
                if self.norm:
                    preds_re_norm = self.normalize(preds,'y',inverse=True)
                    b = self.normalize(b,'b',inverse=True)
                q_w = self.get_flood(preds_re_norm,b)
                if self.norm:
                    q_w = q_w/self.normal[...,-2]
                # loss += self.balance_alpha * self.mse(y[...,-1],q_w)
                loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],expand_dims(q_w,axis=-1)],axis=-1))
            else:
                loss = self.mse(y[...,:3],preds[...,:3])

            if self.if_flood:
                loss += self.cce(y[...,-3:-1],preds[...,-2:])

            if self.edge_fusion:
                loss += self.mse(edge_preds,ey)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss.numpy()
    
    def evaluate(self,x,a,b,y,ex=None,ey=None):
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            elif self.edge_fusion:
                ae = self.get_edge_action(a)
            else:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a)
        if self.roll:
            preds = []
            if self.edge_fusion:
                edge_preds = []
            # TODO: what if not recurrent
            for i in range(b.shape[1]):
                inp = [x,b[:,i:i+self.seq_out,...]]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj[:,i:i+self.seq_out,...]] if self.act and self.use_adj else []
                if self.edge_fusion:
                    inp += [ex]
                    inp += [self.edge_filter] if self.conv else []
                    inp += [ae[:,i:i+self.seq_out,...]] if self.act else []
                pred = self.model(inp)
                if self.edge_fusion:
                    pred,edge_pred = pred
                    edge_preds.append(edge_pred)
                    ex = concat([ex[:,1:,...],edge_pred[:,:1,...]],axis=1) if self.recurrent else edge_pred
                if self.act and not self.use_adj and not self.edge_fusion:
                    pred = concat([tf.stack([tf.clip_by_value(pred[...,0],0,1),tf.multiply(pred[...,1],a_in),tf.multiply(pred[...,2],a_out)],axis=-1),pred[...,3:]],axis=-1)
                else:
                    pred = concat([tf.clip_by_value(pred[...,:1],0,1),pred[...,1:]],axis=-1)
                preds.append(pred)
                x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
            preds = concat(preds,axis=1)
            if self.edge_fusion:
                edge_preds = concat(edge_preds,axis=1)
        else:
            inp = [x,b]
            if self.conv:
                inp += [self.filter]
                inp += [adj] if self.act and self.use_adj else []
            if self.edge_fusion:
                inp += [ex]
                inp += [self.edge_filter] if self.conv else []
                inp += [ae] if self.act else []
            preds = self.model(inp)
            if self.edge_fusion:
                preds,edge_preds = preds
            if self.act and not self.use_adj and not self.edge_fusion:
                preds = concat([tf.stack([tf.clip_by_value(preds[...,0],0,1),tf.multiply(preds[...,1],a_in),tf.multiply(preds[...,2],a_out)],axis=-1),preds[...,3:]],axis=-1)
                
        if self.balance:
            if self.norm:
                preds_re_norm = self.normalize(preds,'y',inverse=True)
                b = self.normalize(b,'b',inverse=True)
            q_w = self.get_flood(preds_re_norm,b)
            if self.norm:
                q_w = q_w/self.normal[...,-2]
            # loss += self.balance_alpha * self.mse(y[...,-1],q_w)
            loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],expand_dims(q_w,axis=-1)],axis=-1))
        else:
            loss = self.mse(y[...,:3],preds[...,:3])
        if self.if_flood:
            loss += self.cce(y[...,-3:-1],preds[...,-2:])
        if self.edge_fusion:
            loss += self.mse(edge_preds,ey)
        return loss.numpy()


    def update_net(self,dG,ratio=None,epochs=None,batch_size=None):
        ratio = self.ratio if ratio is None else ratio
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs

        seq = max(self.seq_in,self.seq_out) if self.recurrent else 0

        n_events = int(max(dG.event_id))+1
        train_ids = np.random.choice(np.arange(n_events),int(n_events*ratio),replace=False)
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]

        train_dats = dG.prepare(seq,train_ids)
        test_dats = dG.prepare(seq,test_ids)

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            idxs = np.random.choice(range(train_dats[0].shape[0]),batch_size)
            x,a,b,y = [dat[idxs] if dat is not None else dat for dat in train_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]

            if self.edge_fusion:
                ex,ey = [dat[idxs] for dat in train_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            train_loss = self.fit(x,a,b,y,ex,ey)
            train_losses.append(train_loss)

            idxs = np.random.choice(range(test_dats[0].shape[0]),batch_size)
            x,a,b,y = [dat[idxs] if dat is not None else dat for dat in test_dats[:4]]
            if self.norm:
                x,b,y = [self.normalize(dat,item) for dat,item in zip([x,b,y],'xby')]

            if self.edge_fusion:
                ex,ey = [dat[idxs] for dat in train_dats[-2:]]
                if self.norm:
                    ex,ey = [self.normalize(dat,'e') for dat in [ex,ey]]
            else:
                ex,ey = None,None
            test_loss = self.evaluate(x,a,b,y,ex,ey)
            test_losses.append(test_loss)
            print("Epoch {}/{} Train loss: {} Test loss: {}".format(epoch,epochs,train_loss,test_loss))
        return train_ids,test_ids,train_losses,test_losses

    # TODO: settings
    def simulate(self,states,runoff,a=None,edge_states=None):
        # runoff shape: T_out, T_in, N
        if self.act:
            if self.use_adj:
                adj = self.get_adj_action(a)
            elif self.edge_fusion:
                ae = self.get_edge_action(a)
            else:
                a_out,a_in = self.get_action(a[:,:self.seq_out,...] if self.recurrent else a)
        preds = []
        if self.edge_fusion:
            edge_preds = []
        for idx,bi in enumerate(runoff):
            x = states[idx,-self.seq_in:,...] if self.recurrent else states[idx]
            ex = edge_states[idx,-self.seq_in:,...] if self.recurrent else edge_states[idx]
            if self.roll:
                # TODO: What if not recurrent
                qws,ys = [],[]
                if self.edge_fusion:
                    eys = []
                for i in range(bi.shape[0]):
                    b_i = bi[i:i+self.seq_out]
                    inp = [self.normalize(x,'x'),self.normalize(b_i,'r')] if self.norm else [x,b_i]
                    inp = [expand_dims(dat,0) for dat in inp]
                    if self.conv:
                        inp += [self.filter]
                        inp += [adj[idx:idx+1,i:i+self.seq_out,...]] if self.act and self.use_adj else []
                    if self.edge_fusion:
                        inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
                        inp += [self.edge_filter] if self.conv else []
                        inp += [ae[idx:idx+1,i:i+self.seq_out,...]] if self.act else []
                    y = self.model(inp)
                    if self.edge_fusion:
                        y,ey = y
                        if self.norm:
                            ey = self.normalize(ey,'e',True)
                        ex = np.concatenate([ex[1:],ey[:1]],axis=0) if self.recurrent else ey
                        eys.append(ey)
                    y = squeeze(y,0).numpy()
                    if self.norm:
                        y = self.normalize(y,'y',True)
                    if self.act and not self.use_adj and not self.edge_fusion:
                        y[...,2] *= a_out[idx]
                        y[...,1] *= a_in[idx]
                    q_w,y = self.constrain(y,b_i)
                    x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
                    qws.append(q_w)
                    ys.append(y)
                q_w,y = np.concatenate(qws,axis=0),np.concatenate(ys,axis=0)
                if self.edge_fusion:
                    ey = np.concatenate(eys,axis=0)
            else:
                bi = bi[:self.seq_out] if self.recurrent else bi

                inp = [self.normalize(x,'x'),self.normalize(bi,'b')] if self.norm else [x,bi]
                inp = [expand_dims(dat,0) for dat in inp]
                if self.conv:
                    inp += [self.filter]
                    inp += [adj[idx:idx+1]] if self.act and self.use_adj else []
                if self.edge_fusion:
                    inp += [expand_dims(self.normalize(ex,'e') if self.norm else ex,0)]
                    inp += [self.edge_filter] if self.conv else []
                    inp += [ae[idx:idx+1]] if self.act else []
                y = self.model(inp)
                if self.edge_fusion:
                    y,ey = y
                    ey = squeeze(ey,0).numpy()
                    if self.norm:
                        ey = self.normalize(ey,'e',True)
                y = squeeze(y,0).numpy()
                if self.norm:
                    y = self.normalize(y,'y',True)
                if self.act and not self.use_adj and not self.edge_fusion:
                    y[...,2] *= a_out[idx]
                    y[...,1] *= a_in[idx]
                q_w,y = self.constrain(y,bi)
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            if self.edge_fusion:
                edge_preds.append(ey)
        if self.edge_fusion:
            return np.array(preds),np.array(edge_preds)
        else:
            return np.array(preds)
    
    # # Parallel emulation (GPU out of memory)
    # def simulate(self,states,runoff):
    #     # runoff shape: T_out, T_in, N
    #     if self.roll:
    #         x = states[0,-self.seq_in:,...] if self.recurrent else states[0]
    #         preds = []
    #         for idx,ri in enumerate(runoff):
    #             # x = x if self.roll else state
    #             # TODO: What if not recurrent
    #             qws,ys = []
    #             for i in range(ri.shape[0]):
    #                 r_i = ri[i:i+self.seq_out]
    #                 y = self.predict(x,r_i)
    #                 q_w,y = self.constrain(y,r_i)
    #                 x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
    #                 qws.append(q_w)
    #                 ys.append(y)
    #             q_w,y = np.concatenate(qws,axis=0),np.concatenate(ys,axis=0)
    #             y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
    #             preds.append(y)
    #     else:
    #         states = states[:,-self.seq_in:,...] if self.recurrent else states
    #         runoff = runoff[:,:self.seq_out,...] if self.recurrent else runoff
    #         if self.norm:
    #             x = self.normalize(states)
    #             b = self.normalize(runoff)
    #         y = self.model([x,self.filter,b]).numpy()
    #         if self.norm:
    #             y = self.normalize(y,inverse=True).clip(0)
    #         q_w,y = self.constrain(y,runoff)
    #         preds = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
    #     return np.array(preds)
    


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
            q_w = np.clip(q_us + r - q_ds,0,np.inf) * ((self.hmax - h) < 0.1)
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
            q_w = np.clip(q_us + r - q_ds,0,np.inf) * ((self.hmax - h) < 0.1)
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
        return dat * normal[...,:dim] if inverse else dat/normal[...,:dim]



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