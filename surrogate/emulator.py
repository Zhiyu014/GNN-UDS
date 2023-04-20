from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean,reduce_sum,concat
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError,CategoricalCrossentropy
import numpy as np
import os
from spektral.layers import GCNConv,GATConv
from spektral.utils.convolution import gcn_filter
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')
# - **Model**: STGCN may be a possible method to handle with spatial-temporal prediction. Why such structure is needed?
#     - [pytorch implementation](https://github.com/LMissher/STGNN)
#     - [original](https://github.com/VeritasYin/STGCN_IJCAI-18)
# - **Predict**: *T*-times 1-step prediction OR T-step prediction?

class Emulator:
    def __init__(self,conv=None,edges=None,resnet=False,recurrent=None,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        # Runoff and actions are boundary
        self.b_in = 2 if getattr(args,"act",False) else 1
        self.n_in -= 1

        self.n_out = self.n_in
        self.seq_in = getattr(args,'seq_in',6)
        self.seq_out = getattr(args,'seq_out',1)
        self.roll = getattr(args,"roll",False)
        if self.roll:
            self.seq_out = 1

        self.embed_size = getattr(args,'embed_size',64)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.n_layer = getattr(args,"n_layer",3)
        self.activation = getattr(args,"activation",'relu')
        self.norm = getattr(args,"norm",False)
        self.balance_alpha = getattr(args,"balance",0)
        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 2

        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))
        if edges is not None:
            self.edges = edges
            self.filter = self.get_adj(edges)
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


    def get_adj(self,edges):
        # self-loop
        # TODO: if directed graph
        A = np.eye(edges.max()+1) # adjacency matrix
        for u,v in edges:
            A[u,v] += 1
            A[v,u] += 1
        return A

    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        input_shape,bound_input = (self.n_node,self.n_in),(self.n_node,self.b_in)
        if recurrent:
            input_shape = (self.seq_in,) + input_shape
            bound_input = (self.seq_out,) + bound_input
        X_in = Input(shape=input_shape)
        B_in = Input(shape=bound_input)

        if conv:
            A_in = Input(self.filter.shape[0],)
            inp = [X_in,A_in,B_in]
            if 'GCN' in conv:
                self.filter,net = gcn_filter(self.filter),GCNConv
            elif 'GAT' in conv:
                net = GATConv
            # elif 'CNN' in conv:
            #     # TODO: CNN
            #     net = Conv2D
            else:
                raise AssertionError("Unknown Convolution layer %s"%str(conv))
        else:
            inp,net = [X_in,B_in],Dense
        
        # (B,T,N,in) (B,N,in)--> (B,T,N*in) (B,N*in)
        x = reshape(X_in,tuple(x.shape[:-2])+(-1,)) if not conv else X_in
        x = Dense(self.embed_size,activation=self.activation)(x) # Embedding
        res = [x]

        b = Dense(self.embed_size//2,activation=self.activation)(B_in)  # Boundary Embedding (B,T_out,N,E)

        # (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> (B*T,N,E) (B*T,E)
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        for _ in range(self.n_layer):
            x = [x,A_in] if conv else x
            x = net(self.embed_size,activation=self.activation)(x)

            b = Dense(self.embed_size//2,activation=self.activation)(b)

        # (B*T,N,E) (B*T,E) (B,N,E) (B,E) --> (B,T,N,E) (B,T,E) (B,N,E) (B,E)
        x_out = reshape(x,(-1,)+input_shape[:-1]+(self.embed_size,)) if conv else reshape(x,(-1,)+input_shape[:-2]+(self.embed_size,)) 
        # res.append(x_out)
        #  (B,T,N,E) (B,T,E) (B,N,E) (B,E) --> （B*R,T,N,E)
        res += [x_out]
        x = concat(res,axis=0) if resnet else x_out

        if recurrent:
            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E)
            x = transpose(x,[0,2,1,3]) if conv else x
            b = transpose(b,[0,2,1,3])
            if recurrent == 'Conv1D':
                # (B,N,T,E) (B,T,E) --> (B,N,T_out,H) (B,T_out,H)
                x = Conv1D(self.hidden_dim,self.seq_in-self.seq_out+1,activation=self.activation,input_shape=x.shape[-2:])(x)

                b = Conv1D(self.hidden_dim//2,self.seq_in-self.seq_out+1,activation=self.activation,input_shape=b.shape[-2:])(b)

            elif recurrent == 'GRU':
                # (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
                x = reshape(x,(-1,self.seq_in,self.embed_size)) if conv else x
                x = GRU(self.hidden_dim,return_sequences=True)(x)
                x = x[...,-self.seq_out:,:] # seq_in >= seq_out if roll
                # (B*N,T_out,H) (B,T_out,H) --> (B,N,T_out,H) (B,T_out,H)
                x = reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)) if conv else x

                # (B,N,T_out,E) --> (B*N,T_out,E) --> (B,N,T_out,E)
                b = reshape(b,(-1,self.seq_out,self.embed_size//2))
                b = GRU(self.hidden_dim//2,return_sequences=True)(b)
                b = reshape(b,(-1,self.n_node,self.seq_out,self.hidden_dim//2))

                # (B,N,T_out,H) (B,T_out,H) --> (B,T_out,N,H) (B,T_out,H)
                x = transpose(x,[0,2,1,3]) if conv else x
                b = transpose(b,[0,2,1,3])

            else:
                raise AssertionError("Unknown recurrent layer %s"%str(recurrent))
        
        # （B*R,T_out,N,H) --> (B,T_out,N,H)
        x = reduce_sum(reshape(x,(len(res),-1,)+tuple(x.shape[1:])),axis=0) if resnet else x
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
    
    def loss_fn(self,true,pred):
        if self.if_flood:
            loss = self.mse(pred[...,:-2],true[...,:-2]) + self.cce(pred[...,-2:],true[...,-2:])
        else:
            loss = self.mse(pred,true)
        return loss
            

    # TODO: classify if flooding
    # TODO: setting loss
    def fit(self,x,b,y):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            if self.roll:
                preds = []
                # TODO: what if not recurrent
                for i in range(b.shape[1]):
                    pred = self.model([x,self.filter,b[:,i:i+self.seq_out,...]])
                    preds.append(pred)
                    x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
                preds = concat(preds,axis=1)
                loss = self.loss_fn(y,preds)
            else:
                preds = self.model([x,self.filter,b])
                loss = self.loss_fn(y,preds)
            if self.norm:
                preds = self.normalize(preds,inverse=True)
                b = self.normalize(b,inverse=True)
            if self.balance_alpha:
                loss += self.balance_alpha * self.balance(preds,b)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss.numpy()
    
    def evaluate(self,x,b,y):
        if self.roll:
            preds = []
            # TODO: what if not recurrent
            for i in range(b.shape[1]):
                pred = self.model([x,self.filter,b[:,i:i+1,...]])
                preds.append(pred)
                x = concat([x[:,1:,...],pred[:,:1,...]],axis=1) if self.recurrent else pred
            preds = concat(preds,axis=1)
            loss = self.loss_fn(y,preds)
        else:
            preds = self.model([x,self.filter,b])
            loss = self.loss_fn(y,preds)
        if self.balance_alpha:
            loss += self.balance_alpha * self.balance(preds,b)
        return loss.numpy()

    def predict(self,x,b):
        if self.norm:
            x = self.normalize(x)
            b = self.normalize(b)
        x,b = expand_dims(x,0),expand_dims(b,0)
        y = squeeze(self.model([x,self.filter,b]),0).numpy()
        if self.norm:
            y = self.normalize(y,inverse=True).clip(0)
        return y

    def set_norm(self,normal):
        setattr(self,'normal',normal)

    def normalize(self,dat,inverse=False):
        dim = dat.shape[-1]
        if dim >= 3:
            return dat * self.normal[...,:dim] if inverse else dat/self.normal[...,:dim]
        else:
            return dat * self.normal[...,-dim:] if inverse else dat/self.normal[...,-dim:]


    # Problems: use flooding volume at each node as label?
    def constrain(self,y,r):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = h.clip(0,self.hmax)
        if self.if_flood:
            f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            q_w = (q_us + r - q_ds).clip(0) * f
            h = self.hmax * f + h * ~f
            y = np.stack([h,q_us,q_ds,y[...,-2],y[...,-1]],axis=-1)
        else:
            q_w = (q_us + r - q_ds).clip(0) * ((self.hmax - h) < 0.1)
            y = np.stack([h,q_us,q_ds],axis=-1)
        return q_w,y
    
    def balance(self,y,r):
        h,q_us,q_ds = [y[...,i].numpy() for i in range(3)]
        r = np.squeeze(r,axis=-1)
        h = h.clip(0,self.hmax)
        if self.if_flood:
            f = np.argmax(y[...,-2:],axis=-1).astype(bool)
            q_w = (q_us + r - q_ds).clip(0) * f
        else:
            q_w = (q_us + r - q_ds).clip(0) * ((self.hmax - h) < 0.1)
        # err = q_us + r - q_ds - q_w
        return self.mse(q_us + r, q_ds + q_w)

    # TODO: settings
    def simulate(self,states,runoff):
        # runoff shape: T_out, T_in, N
        x = states[0]
        if self.recurrent:
            x = x[-self.seq_in:,...]
        pred = []
        for idx,ri in enumerate(runoff):
            state = states[idx,-self.seq_in:,...] if self.recurrent else states[idx]
            x = x if self.roll else state
            ri = ri[:self.seq_out] if self.recurrent else ri
            y = self.predict(x,ri)
            q_w,y = self.constrain(y,ri)
            if self.roll:
                x = np.concatenate([x[1:],y[:1]],axis=0) if self.recurrent else y
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            pred.append(y)
        return np.array(pred)
    
    def update_net(self,dG,ratio=None,epochs=None,batch_size=None):
        ratio = self.ratio if ratio is None else ratio
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs

        n_events = int(max(dG.event_id))+1
        train_ids = np.random.choice(np.arange(n_events),int(n_events*ratio))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            x,b,y = dG.sample(batch_size,train_ids,self.norm)
            train_loss = self.fit(x,b,y)
            train_losses.append(train_loss)
            x,b,y = dG.sample(batch_size,test_ids,self.norm)
            test_loss = self.evaluate(x,b,y)
            test_losses.append(test_loss)
            print("Epoch {}/{} Train loss: {} Test loss: {}".format(epoch,epochs,train_loss,test_loss))
        return train_ids,test_ids,train_losses,test_losses

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
