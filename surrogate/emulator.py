from tensorflow import reshape,transpose,squeeze,GradientTape,expand_dims
from tensorflow.keras.layers import Dense,Input,GRU,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import losses,optimizers
import numpy as np
from spektral.layers import GCNConv,GATConv
from spektral.utils.convolution import gcn_filter

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

    def get_adj(self,edges):
        A = np.zeros((edges.max()+1,edges.max()+1)) # adjacency matrix
        for u,v in edges:
            A[u,v] += 1
        return A

    def build_network(self,conv=None,resnet=False,recurrent=None):
        # (T,N,in) (T,in*N) (N,in) (in*N)
        input_shape = (self.n_node,self.n_in) if conv else (self.n_node * self.n_in,)
        if recurrent:
            input_shape = (self.seq_len,) + input_shape
        X_in = Input(shape=input_shape)
        x = X_in.copy()
        
        if conv:
            A_in = Input(self.filter.shape[0],)
            inp = [X_in,A_in]
            if 'GCN' in conv:
                self.filter,net = gcn_filter(self.filter),GCNConv
            elif 'GAT' in conv:
                net = GATConv
            elif 'CNN' in conv:
                # TODO: CNN
                net 
            else:
                raise AssertionError("Unknown Convolution layer %s"%str(conv))
        else:
            inp,net = X_in,Dense
        
        # (B,T,N,in) (B,T,in*N) --> (B*T,N,in) (B*T,in*N)
        x = reshape(x,(-1,)+input_shape[1:]) if recurrent else x
        for _ in range(self.n_layer):
            x = [x,A_in] if conv else x
            x_out = net(self.embed_size,activation=self.activation)(x)
            x = x_out + x if resnet else x_out

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
        out = Dense(out_shape,activation='linear')(x)
        model = Model(inputs=inp, outputs=out)
        return model
    
    def fit(self,x,y):
        with GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model(x)
            loss = self.loss_fn(y,pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss.numpy()
    
    def predict(self,x):
        x = expand_dims(x,0)
        return squeeze(self.model(x),0).numpy()

    def constrain(self,y,r):
        # y,r are 2-d
        # r should be at the same step with q_ds
        if self.conv:
            h,q_us,q_ds = [y[:,:,i] for i in range(3)]
        else:
            h,q_us,q_ds = y[:,:self.n_node],y[:,self.n_node:self.n_node*2],y[:,self.n_node*2:]
        # q_us = [np.zeros((y.shape[0],)) for _ in self.n_node]
        # for u,v in self.edges:
        #     q_us[v] += q_ds[u]
        # q_us = np.array(q_us).T
        q_w = (q_us + r - q_ds).clip(0) * (h > self.hmax)
        return (h,q_us,q_ds,q_w)
        
        