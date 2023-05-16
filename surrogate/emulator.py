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
        self.n_in -= 1
        self.act = getattr(args,"act",False)

        self.n_out = self.n_in
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

        self.adj = getattr(args,"adj",np.eye(self.n_node))
        if self.act:
            self.act_edges = getattr(args,"act_edges")
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
        if args.load_model:
            self.load()



    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        input_shape,bound_input = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if recurrent:
            input_shape = (self.seq_in,) + input_shape
            bound_input = (self.seq_out,) + bound_input
        X_in = Input(shape=input_shape)
        B_in = Input(shape=bound_input)

        if conv:
            A_in = Input(self.n_node,)
            inp = [X_in,A_in,B_in]
            if 'GCN' in conv:
                net = GCNConv
            elif 'GAT' in conv:
                net = GATConv
            else:
                raise AssertionError("Unknown Convolution layer %s"%str(conv))
        else:
            inp,net = [X_in,B_in],Dense
        
        # (B,T,N,in) (B,N,in)--> (B,T,N*in) (B,N*in)
        x = reshape(X_in,(-1,)+tuple(input_shape[:-2])+(self.n_node*self.n_in,)) if not conv else X_in
        x = Dense(self.embed_size,activation=self.activation)(x) # Embedding
        res = [x]

        b = Dense(self.embed_size//2,activation=self.activation)(B_in)  # Boundary Embedding (B,T_out,N,E)

        # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        if conv:
            A = reshape(A_in,(-1,) + (self.n_node,self.n_node)) if recurrent and self.act else A_in
        for _ in range(self.n_sp_layer):
            x = [x,A] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)

            b = Dense(self.embed_size//2,activation=self.activation)(b)

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x_out = reshape(x,(-1,)+input_shape[:-1]+(self.embed_size,)) if conv else reshape(x,(-1,)+input_shape[:-2]+(self.embed_size,)) 
        # res.append(x_out)

        res += [x_out]
        #  (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> （B*R,T,N,E)
        # x = concat(res,axis=0) if resnet else x_out
        x = Add()(res) if resnet else x_out

        if recurrent:
            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E)
            x = transpose(x,[0,2,1,3]) if conv else x
            b = transpose(b,[0,2,1,3])
            if recurrent == 'Conv1D':
                # (B,N,T,E) (B,T,E) --> (B,N,T_out,H) (B,T_out,H)
                for i in range(self.n_tp_layer):
                    x = Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=x.shape[-2:])(x)

                x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll

                if self.seq_out > 1:
                    for i in range(self.n_tp_layer):
                        b = Conv1D(self.hidden_dim//2,self.kernel_size,padding='causal',dilation_rate=2**i,activation=self.activation,input_shape=b.shape[-2:])(b)

            elif recurrent == 'GRU':
                # (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
                x = reshape(x,(-1,self.seq_in,self.embed_size)) if conv else x
                for i in range(self.n_tp_layer):
                    x = GRU(self.hidden_dim,return_sequences=True)(x)
                x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll
                # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H)
                x = reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)) if conv else x

                if self.seq_out > 1:
                    # (B,N,T_out,E) --> (B*N,T_out,E) --> (B,N,T_out,E)
                    b = reshape(b,(-1,self.seq_out,self.embed_size//2))
                    for i in range(self.n_tp_layer):
                        b = GRU(self.hidden_dim//2,return_sequences=True)(b)
                    b = reshape(b,(-1,self.n_node,self.seq_out,self.hidden_dim//2))

            else:
                raise AssertionError("Unknown recurrent layer %s"%str(recurrent))
            
            # (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
            x = transpose(x,[0,2,1,3]) if conv else x
            b = transpose(b,[0,2,1,3])

        # （B*R,T_out,N,H) --> (B,T_out,N,H)
        # x = reduce_sum(reshape(x,(len(res),-1,)+tuple(x.shape[1:])),axis=0) if resnet else x
        # boundary in
        x = concat([x,b],axis=-1)

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
        model = Model(inputs=inp, outputs=out)
        return model
            
    # TODO: setting loss
    def fit(self,x,a,b,y):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            if self.roll:
                preds = []
                # TODO: what if not recurrent
                for i in range(b.shape[1]):
                    # ai = a[:,i:i+self.seq_out,...] if self.conv and self.act else a
                    pred = self.model([x,a,b[:,i:i+self.seq_out,...]]) if self.conv else self.model([x,b[:,i:i+self.seq_out,...]])
                    pred = concat([tf.clip_by_value(pred[...,:1],0,1),pred[...,1:]],axis=-1)
                    preds.append(pred)
                    x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
                preds = concat(preds,axis=1)
            else:
                preds = self.model([x,a,b]) if self.conv else self.model([x,b])

            # loss = self.loss_fn(y[...,:-1],preds)

            if self.balance:
                if self.norm:
                    preds_re_norm = self.normalize(preds,inverse=True)
                    b = self.normalize(b,inverse=True)
                q_w = self.get_flood(preds_re_norm,b)
                if self.norm:
                    q_w = q_w/self.normal[...,-2]
                # loss += self.balance_alpha * self.mse(y[...,-1],q_w)
                loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],expand_dims(q_w,axis=-1)],axis=-1))
            else:
                loss = self.mse(y[...,:3],preds[...,:3])

            if self.if_flood:
                loss += self.cce(y[...,-3:-1],preds[...,-2:])

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss.numpy()
    
    def evaluate(self,x,a,b,y):
        if self.roll:
            preds = []
            # TODO: what if not recurrent
            for i in range(b.shape[1]):
                pred = self.model([x,a,b[:,i:i+self.seq_out,...]]) if self.conv else self.model([x,b[:,i:i+self.seq_out,...]])
                pred = concat([tf.clip_by_value(pred[...,:1],0,1),pred[...,1:]],axis=-1)
                preds.append(pred)
                x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
            preds = concat(preds,axis=1)
        else:
            preds = self.model([x,a,b]) if self.conv else self.model([x,b])

        if self.balance:
            if self.norm:
                preds_re_norm = self.normalize(preds,inverse=True)
                b = self.normalize(b,inverse=True)
            q_w = self.get_flood(preds_re_norm,b)
            if self.norm:
                q_w = q_w/self.normal[...,-2]
            # loss += self.balance_alpha * self.mse(y[...,-1],q_w)
            loss = self.mse(concat([y[...,:3],y[...,-1:]],axis=-1),concat([preds[...,:3],expand_dims(q_w,axis=-1)],axis=-1))
        else:
            loss = self.mse(y[...,:3],preds[...,:3])
        if self.if_flood:
            loss += self.cce(y[...,-3:-1],preds[...,-2:])
        return loss.numpy()

    def predict(self,x,a,b):
        if self.norm:
            x = self.normalize(x)
            b = self.normalize(b)
        if 'GCN' in self.conv:
            a = gcn_filter(a)
        if len(a.shape) > 2:
            a = expand_dims(a,0)
        x,b = expand_dims(x,0),expand_dims(b,0)
        y = squeeze(self.model([x,a,b]) if self.conv else self.model([x,b]),0).numpy()
        if self.norm:
            y = self.normalize(y,inverse=True)
        return y

    def set_norm(self,normal):
        setattr(self,'normal',normal)

    def normalize(self,dat,inverse=False):
        dim = dat.shape[-1]
        if dim >= 3:
            return dat * self.normal[...,:dim] if inverse else dat/self.normal[...,:dim]
        else:
            return dat * self.normal[...,-dim:] if inverse else dat/self.normal[...,-dim:]


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
    
    def update_net(self,dG,ratio=None,epochs=None,batch_size=None):
        ratio = self.ratio if ratio is None else ratio
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs

        seq = max(self.seq_in,self.seq_out) if self.recurrent else 0

        n_events = int(max(dG.event_id))+1
        train_ids = np.random.choice(np.arange(n_events),int(n_events*ratio),replace=False)
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]

        X_train,set_train,B_train,Y_train = dG.prepare(seq,train_ids)
        X_test,set_test,B_test,Y_test = dG.prepare(seq,test_ids)

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            idxs = np.random.choice(range(X_train.shape[0]),batch_size)
            x,b,y = [dat[idxs] for dat in [X_train,B_train,Y_train]]
            if self.norm:
                x,b,y = [self.normalize(dat) for dat in [x,b,y]]
            if self.act:
                # (B,T,N_act) --> (B,T,N,N)
                def get_act(s):
                    adj = self.adj.copy()
                    adj[tuple(self.act_edges.T)] = s
                    return adj
                a = np.apply_along_axis(get_act,-1,set_train[idxs])
            else:
                a = self.adj.copy()
            if 'GCN' in self.conv:
                a = gcn_filter(a)
            train_loss = self.fit(x,a,b,y)
            train_losses.append(train_loss)

            idxs = np.random.choice(range(X_test.shape[0]),batch_size)
            x,b,y = [dat[idxs] for dat in [X_test,B_test,Y_test]]
            if self.norm:
                x,b,y = [self.normalize(dat) for dat in[x,b,y]]
            if self.act:
                # (B,T,N_act) --> (B,T,N,N)
                def get_act(s):
                    adj = self.adj.copy()
                    adj[tuple(self.act_edges.T)] = s
                    return adj
                a = np.apply_along_axis(get_act,-1,set_test[idxs])
            else:
                a = self.adj.copy()
            if 'GCN' in self.conv:
                a = gcn_filter(a)
            test_loss = self.evaluate(x,a,b,y)
            test_losses.append(test_loss)
            print("Epoch {}/{} Train loss: {} Test loss: {}".format(epoch,epochs,train_loss,test_loss))
        return train_ids,test_ids,train_losses,test_losses

    # TODO: settings
    def simulate(self,states,runoff,a):
        # runoff shape: T_out, T_in, N
        preds = []
        for idx,ri in enumerate(runoff):
            x = states[idx,-self.seq_in:,...] if self.recurrent else states[idx]
            ai = a[idx] if len(a.shape)>2 else a
            # x = x if self.roll else state
            if self.roll:
                # TODO: What if not recurrent
                qws,ys = [],[]
                for i in range(ri.shape[0]):
                    r_i = ri[i:i+self.seq_out]
                    a_i = ai[i:i+self.seq_out,...] if len(ai.shape)>2 else ai
                    y = self.predict(x,a_i,r_i)
                    q_w,y = self.constrain(y,r_i)
                    x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
                    qws.append(q_w)
                    ys.append(y)
                q_w,y = np.concatenate(qws,axis=0),np.concatenate(ys,axis=0)
            else:
                ri = ri[:self.seq_out] if self.recurrent else ri
                y = self.predict(x,ai,ri)
                q_w,y = self.constrain(y,ri)
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
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
    
    def save(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if model_dir.endswith('.h5'):
            self.model.save_weights(model_dir)
            if self.norm:
                np.save(os.path.join(os.path.dirname(model_dir),'normal.npy'),self.normal)
        else:
            self.model.save_weights(os.path.join(model_dir,'model.h5'))
            if self.norm:
                np.save(os.path.join(model_dir,'normal.npy'),self.normal)


    def load(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
            if self.norm:
                self.normal = np.load(os.path.join(os.path.dirname(model_dir),'normal.npy'))
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))
            if self.norm:
                self.normal = np.load(os.path.join(model_dir,'normal.npy'))
