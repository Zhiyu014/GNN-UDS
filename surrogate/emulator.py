from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims,reduce_mean
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import losses,optimizers
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
        self.n_out = getattr(args,'n_out',3)
        self.seq_len = getattr(args,'seq_len',6)
        self.embed_size = getattr(args,'embed_size',64)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.n_layer = getattr(args,"n_layer",3)
        self.activation = getattr(args,"activation",'relu')
        self.norm = getattr(args,"norm",'False')

        self.hmax = getattr(args,"hmax",np.array([1.5 for _ in range(self.n_node)]))
        if edges is not None:
            self.edges = edges
            self.filter = self.get_adj(edges)
        self.conv = conv
        self.recurrent = recurrent
        self.model = self.build_network(conv,resnet,recurrent)
        self.loss_fn = losses.get(getattr(args,"loss_function","MeanSquaredError"))
        self.optimizer = optimizers.get(getattr(args,"optimizer","Adam"))
        self.optimizer.learning_rate = getattr(args,"learning_rate",1e-3)

        self.ratio = getattr(args,"ratio",0.8)
        self.batch_size = getattr(args,"batch_size",256)
        self.epochs = getattr(args,"epochs",100)
        self.model_dir = getattr(args,"model_dir","./model/shunqing/model.h5")
        if args.load_model:
            self.load()


    def get_adj(self,edges):
        A = np.zeros((edges.max()+1,edges.max()+1)) # adjacency matrix
        for u,v in edges:
            A[u,v] += 1
        return A

    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (N,in)
        input_shape = (self.n_node,self.n_in)
        if recurrent:
            input_shape = (self.seq_len,) + input_shape
        X_in = Input(shape=input_shape)
        # x = X_in.copy()
        
        if conv:
            A_in = Input(self.filter.shape[0],)
            inp = [X_in,A_in]
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
            inp,net = X_in,Dense
        
        # (B,T,N,in) (B,N,in)--> (B,T,N*in) (B,N*in)
        x = reshape(X_in,input_shape[:-2]+(-1,)) if not conv else X_in
        # (B,T,N,in) (B,T,N*in) (B,N,in) (B,N*in) --> (B*T,N,in)
        x = reshape(x,(-1,) + tuple(x.shape[2:])) if recurrent else x
        x = Dense(self.embed_size,activation=self.activation)(x) # Embedding
        for i in range(self.n_layer):
            x = [x,A_in] if conv else x
            x_out = net(self.embed_size,activation=self.activation)(x)
            x = x[0] if conv else x
            x = x_out + x if resnet and i>0 else x_out

        if recurrent:
            # (B*T,N,E) (B*T,E) --> (B,T,N,E) (B,T,E)
            x = reshape(x,(-1,)+input_shape[:-1]+(self.embed_size,))
            # (B,T,N,E) (B,T,E) --> (B,N,T,E) (B,T,E)
            x = transpose(x,[0,2,1,3]) if conv else x
            if recurrent == 'Conv1D':
                # (B,N,T,E) (B,T,E) --> (B,N,H) (B,H)
                x = Conv1D(self.hidden_dim,self.seq_len,activation=self.activation,input_shape=x.shape[-2:])(x)
                x = squeeze(x)
            elif recurrent == 'GRU':
                # (B,N,T,E) (B,T,E) --> (B*N,T,E) (B,T,E)
                x = reshape(x,(-1,self.seq_len,self.embed_size)) if conv else x
                x = GRU(self.hidden_dim)(x)
                # (B*N,H) (B,H) --> (B,N,H) (B,H)
                x = reshape(x,(-1,self.n_node,self.hidden_dim)) if conv else x
            else:
                raise AssertionError("Unknown recurrent layer %s"%str(recurrent))

        out_shape = self.n_out if conv else self.n_out * self.n_node
        # (B,N,H) (B,H) --> (B,N,n_out)
        out = Dense(out_shape,activation='linear')(x)
        out = reshape(out,(-1,self.n_node,self.n_out)) 
        model = Model(inputs=inp, outputs=out)
        return model
    

    # TODO: setting loss
    def fit(self,x,y):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model([x,self.filter])
            loss = self.loss_fn(y,pred)
            # if self.norm:
            #     x = self.normalize(x,inverse=True)
            #     pred = self.normalize(pred,inverse=True)
            # r = x[:,-1,...,-1] if self.recurrent else x[...,-1]
            # loss += self.balance(pred,r)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss.numpy()
    
    def evaluate(self,x,y):
        loss = self.loss_fn(self.model([x,self.filter]),y)
        return loss.numpy()

    def predict(self,x):
        if self.norm:
            x = self.normalize(x)
        x = expand_dims(x,0)
        y = squeeze(self.model([x,self.filter]),0).numpy()
        if self.norm:
            y = self.normalize(y,inverse=True)
        return y

    def set_norm(self,normal):
        setattr(self,'normal',normal)

    def normalize(self,dat,inverse = False):
        if inverse:
            return dat * self.normal[...,:dat.shape[-1]]
        else:
            return dat/self.normal[...,:dat.shape[-1]]

    # Problems: h is static at the end of the interval -- use 30s step
    # Problems: use flooding volume at each node as label?
    def constrain(self,y,r):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        q_w = (q_us + r - q_ds).clip(0) * ((self.hmax - h) < 0.01)
        h = h.clip(0,self.hmax)
        y = np.stack([h,q_us,q_ds],axis=-1)
        return q_w,y
    
    def balance(self,y,r):
        _,q_us,q_ds = [y[...,i] for i in range(3)]
        q_w,_ = self.constrain(y,r)
        err = q_us + r - q_ds - q_w
        return reduce_mean(err ** 2)

    # TODO: settings
    def simulate(self,ini_state,r,settings=None):
        x = ini_state[...,:-1] # exclude runoff
        states,qws = [x[-1,...]],[] # exclude recurrent info
        for ri in r:
            x = np.concatenate([x,np.expand_dims(ri,-1)],axis=-1) # T,N,n_in
            y = self.predict(x) # N,n_out
            ri = ri[-1,...] if self.recurrent else ri
            q_w,y = self.constrain(y,ri)
            x = np.concatenate([x[1:,:,:-1],np.expand_dims(y,0)],axis=0) if self.recurrent else y
            states.append(y)
            qws.append(q_w)
        return np.array(states),np.array(qws)
    
    def update_net(self,dG,ratio=None,epochs=None,batch_size=None):
        ratio = self.ratio if ratio is None else ratio
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs

        n_events = int(max(dG.event_id))+1
        train_ids = np.random.choice(np.arange(n_events),int(n_events*ratio))
        test_ids = [ev for ev in range(n_events) if ev not in train_ids]

        train_losses,test_losses = [],[]
        for epoch in range(epochs):
            x,y = dG.sample(batch_size,train_ids,self.norm)
            train_loss = self.fit(x,y)
            train_losses.append(train_loss)
            x,y = dG.sample(batch_size,test_ids,self.norm)
            test_loss = self.evaluate(x,y)
            test_losses.append(test_loss)
            print("Epoch {}/{} Train loss: {} Test loss: {}".format(epoch,epochs,train_loss,test_loss))
        return train_losses,test_losses

    def save(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.save_weights(model_dir)
        else:
            self.model.save_weights(os.path.join(model_dir,'model.h5'))


    def load(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))