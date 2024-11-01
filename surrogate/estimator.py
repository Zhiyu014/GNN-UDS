import numpy as np,tensorflow as tf,os
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, GRU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from spektral.layers import GCNConv, GATConv, DiffusionConv, GeneralConv
from emulator import NodeEdge

import yaml,time,datetime,matplotlib.pyplot as plt
from main import parser,HERE
from dataloader import DataGenerator
from envs import get_env

class Estimator:
    def __init__(self, conv, recurrent, args):
        self.n_node,self.n_out = getattr(args,'state_shape',(40,4))
        self.n_edge,self.e_out = getattr(args,'edge_state_shape',(40,4))
        self.n_out -= 1
        self.e_out -= 1
        self.moni_nodes = [(args.elements['nodes'].index(idx),args.attrs['nodes'].index(attr))
                           for idx,attr in args.states if attr in args.attrs['nodes'][:-1]]
        self.moni_links = [(args.elements['links'].index(idx),args.attrs['links'].index(attr))
                           for idx,attr in args.states if attr in args.attrs['links'][:-1]]
        self.n_in,self.e_in = len(self.moni_nodes),len(self.moni_links)
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.n_act = len(getattr(args,"act_edges"))
            act_edges = [np.where((args.edges==act_edge).all(1))[0] for act_edge in args.act_edges]
            act_edges = [i for e in act_edges for i in e]
            self.act_edges = sorted(list(set(act_edges)),key=act_edges.index)

        self.seq_in = getattr(args,'seq_in',6)
        self.seq_out = getattr(args,'seq_out',1)

        self.graph_base = getattr(args,"graph_base",0)
        self.edge_fusion = getattr(args,"edge_fusion",False)
        self.edges = getattr(args,"edges")
        self.edge_adj = getattr(args,"edge_adj",np.eye(self.n_edge))
        self.ehmax = getattr(args,"ehmax",np.array([0.5 for _ in range(self.n_edge)]))
        self.pump = getattr(args,"pump",np.array([0.0 for _ in range(self.n_edge)]))
        self.node_edge = tf.convert_to_tensor(getattr(args,"node_edge"),dtype=tf.float32)
        self.adj = getattr(args,"adj",np.eye(self.n_node))

        self.embed_size = getattr(args,'embed_size',128)
        self.hidden_dim = getattr(args,"hidden_dim",64)
        self.kernel_size = getattr(args,"kernel_size",3)
        self.n_sp_layer = getattr(args,"n_sp_layer",3)
        self.n_tp_layer = getattr(args,"n_tp_layer",3)
        self.dropout = getattr(args,'dropout',0.0)
        self.activation = getattr(args,"activation",'relu')
        self.is_outfall = getattr(args,"is_outfall",np.array([0 for _ in range(self.n_node)]))
        self.if_flood = getattr(args,"if_flood",0)
        if self.if_flood:
            self.n_out += 1

        self.conv = False if conv in ['None','False','NoneType'] else conv
        self.recurrent = recurrent
        self.model = self.build_network(self.conv,self.recurrent)
        self.optimizer = Adam(learning_rate=getattr(args,"learning_rate",1e-3),clipnorm=1.0)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.nwei = getattr(args,"nwei",np.array([1.0 for _ in range(self.n_node)]))
        self.ewei = getattr(args,"ewei",np.array([1.0 for _ in range(self.n_edge)]))

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
            net = GATConv
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
        
    def build_network(self,conv,recurrent):
        seq = self.seq_in + self.seq_out
        X_in = Input((seq,self.n_in))
        B_in = Input((seq,self.n_node,1))
        inp = [X_in,B_in]
        if conv:
            n_ele = self.n_node + self.n_edge if self.graph_base else self.n_node
            Adj_in = Input(shape=(n_ele,n_ele,))
            inp += [Adj_in]
            net = self.get_conv(conv)
        else:
            net = Dense

        if conv:
            x = self._ungather_node(X_in,[n for n,_ in self.moni_nodes],g=True)
        else:
            b = tf.reshape(B_in,(-1,seq,self.n_node,))            
        x = Dense(self.embed_size,activation=self.activation)(x if conv else X_in)
        b = Dense(self.embed_size,activation=self.activation)(B_in if conv else b)
        x = tf.concat([x,b],axis=-1)
        if self.e_in>0:
            E_in = Input((seq,self.e_in))
            inp += [E_in]
            if conv:
                e = self._ungather_edge(E_in,[v for v,_ in self.moni_links],0,g=True)
            e = Dense(self.embed_size,activation=self.activation)(e if conv else E_in)
        else:
            e = tf.zeros_like(x)[...,:self.n_edge,:self.embed_size]
        if conv and not self.graph_base:
            Eadj_in = Input(shape=(self.n_edge,self.n_edge,))
            inp += [Eadj_in]
        if self.act:
            A_in = Input((seq,self.n_act,))
            inp += [A_in]
            if conv:
                a = self._ungather_edge(A_in,self.act_edges,g=True)
            a = Dense(self.embed_size,activation=self.activation)(a if conv else A_in)
            e = tf.concat([e,a],axis=-1)

        x = tf.reshape(x,(-1,) + tuple(x.shape[2:]))
        e = tf.reshape(e,(-1,) + tuple(e.shape[2:]))
        for _ in range(self.n_sp_layer):
            if conv and self.graph_base:
                x = [tf.concat([x,e],axis=-2),Adj_in]
                x = net(self.embed_size,activation=self.activation)(x)
                x,e = tf.split(x,[self.n_node,self.n_edge],axis=-2)
            elif conv:
                x_e = Dense(self.embed_size//2,activation=self.activation)(e)
                e_x = Dense(self.embed_size//2,activation=self.activation)(x)
                x = tf.concat([x,NodeEdge(tf.abs(self.node_edge))(x_e)],axis=-1)
                e = tf.concat([e,NodeEdge(tf.transpose(tf.abs(self.node_edge)))(e_x)],axis=-1)
                x = net(self.embed_size,activation=self.activation)([x,Adj_in])
                e = net(self.embed_size,activation=self.activation)([e,Eadj_in])
            else:
                x = Dense(self.embed_size*2,activation=self.activation)(tf.concat([x,e],axis=-1))
                x,e = tf.split(x,2,axis=-1)

        # (B*T,N,E) (B*T,E) --> (B,T,N,E) (B,T,E)
        x = tf.reshape(x,(-1,seq,self.n_node,self.embed_size,) if conv else (-1,seq,self.embed_size,)) 
        e = tf.reshape(e,(-1,seq,self.n_edge,self.embed_size,) if conv else (-1,seq,self.embed_size,)) 

        x = tf.reshape(tf.transpose(x,[0,2,1,3]),(-1,seq,self.embed_size)) if conv else x
        x = self.get_tem_nets(recurrent)(x)[...,-self.seq_out:,:]
        x = tf.transpose(tf.reshape(x,(-1,self.n_node,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else x
        e = tf.reshape(tf.transpose(e,[0,2,1,3]),(-1,seq,self.embed_size)) if conv else e
        e = self.get_tem_nets(recurrent)(e)[...,-self.seq_out:,:]
        e = tf.transpose(tf.reshape(e,(-1,self.n_edge,self.seq_out,self.hidden_dim)),[0,2,1,3]) if conv else e

        out_shape = self.n_out if conv else self.n_out * self.n_node
        # (B,T_out,N,H) (B,T_out,H) --> (B,T_out,N,n_out)
        out = Dense(out_shape,activation='hard_sigmoid')(x)   # if tanh is better than linear here when norm==True?
        out = tf.reshape(out,(-1,self.seq_out,self.n_node,self.n_out))
        if self.if_flood:
            out_shape = 1 if conv else 1 * self.n_node
            for _ in range(self.if_flood):
                x = Dense(self.embed_size//2,activation=self.activation)(x)
            flood = Dense(out_shape,activation='sigmoid')(x)
                        #   kernel_regularizer=l2(0.01),bias_regularizer=l1(0.01)
            flood = tf.reshape(flood,(-1,self.seq_out,self.n_node,1))
            out = tf.concat([out,flood],axis=-1)

        out_shape = self.e_out if conv else self.e_out * self.n_edge
        e_out = Dense(out_shape,activation='tanh')(e)
        e_out = tf.reshape(e_out,(-1,self.seq_out,self.n_edge,self.e_out))
        out = [out,e_out]
        model = Model(inputs=inp, outputs=out)
        return model
    
    def _ungather_node(self,a,nodes,default = 0,g=False):
        if g:
            out = np.zeros(self.n_node)
            out[nodes] = range(1,a.shape[-1]+1)
            return tf.expand_dims(tf.gather(tf.concat([tf.ones_like(a[...,:1]) * default,a],axis=-1),tf.cast(out,tf.int32),axis=-1),axis=-1)
        else:
            def set_node(s):
                out = np.ones(self.n_node) * default
                out[nodes] = s
                return out
            return np.expand_dims(np.apply_along_axis(set_node,-1,a),-1)
        
    def _ungather_edge(self,a,edges,default = 1,g=False):
        if g:
            out = np.zeros(self.n_edge)
            out[edges] = range(1,a.shape[-1]+1)
            return tf.expand_dims(tf.gather(tf.concat([tf.ones_like(a[...,:1]) * default,a],axis=-1),tf.cast(out,tf.int32),axis=-1),axis=-1)
        else:
            def set_edge(s):
                out = np.ones(self.n_edge) * default
                out[edges] = s
                return out
            return np.expand_dims(np.apply_along_axis(set_edge,-1,a),-1)
        
    def get_tem_nets(self,recurrent):
        if recurrent == 'Conv1D':
            return Sequential([Conv1D(self.hidden_dim,self.kernel_size,padding='causal',dilation_rate=2**i,
                            activation=self.activation) for i in range(self.n_tp_layer)])
        elif recurrent == 'GRU':
            return Sequential([GRU(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)])
        elif recurrent == 'LSTM':
            return Sequential([LSTM(self.hidden_dim,return_sequences=True) for _ in range(self.n_tp_layer)])
        else:
            return Sequential()

    @tf.function
    def fit_eval(self,x,a,b,y,ex,ey,fit=True):
        x,b,y,ex,ey = [self.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            xm = tf.concat([tf.stack([x[...,i,j] for i,j in self.moni_nodes],axis=-1),
                            tf.stack([y[...,i,j] for i,j in self.moni_nodes],axis=-1)],axis=1)
            inp = [xm,tf.concat([x[...,-1:],b],axis=1)]
            if self.conv:
                inp += [self.filter]
            if self.e_in>0:
                em = tf.concat([tf.stack([ex[...,i,j] for i,j in self.moni_links],axis=-1),
                                tf.stack([ey[...,i,j] for i,j in self.moni_links],axis=-1)],axis=1)
                inp += [em]
            inp += [self.edge_filter] if self.conv and not self.graph_base else []
            inp += [tf.concat([tf.gather(ex[...,-1],self.act_edges,axis=-1),a],axis=1)] if self.act else []
            preds,edge_preds = self.model(inp,training=fit) if self.dropout else self.model(inp)

            loss = self.mse(y[...,:3],preds[...,:3],sample_weight=self.nwei)
            if not fit:
                loss = [loss]
            if self.if_flood:
                loss += self.bce(y[...,-2:-1],preds[...,-1:],sample_weight=self.nwei) if fit else [self.bce(y[...,-2:-1],preds[...,-1:],sample_weight=self.nwei)]
            loss += self.mse(edge_preds,ey,sample_weight=self.ewei) if fit else [self.mse(edge_preds,ey,sample_weight=self.ewei)]

        if fit:
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss

    def predict(self,states,b,a=None,edge_state=None):
        x,b,ex = [self.normalize(dat,item) for dat,item in zip([states,b,edge_state],'xbe')]
        xm = tf.stack([x[...,i,j] for i,j in self.moni_nodes],axis=-1)
        inp = [xm, tf.concat([x[...,-1:],b],axis=1)]
        if self.conv:
            inp += [self.filter]
        if self.e_in>0:
            em = tf.stack([ex[...,i,j] for i,j in self.moni_links],axis=-1)
            inp += [em]
        inp += [self.edge_filter] if self.conv and not self.graph_base else []
        if self.act:
            inp += [tf.concat([tf.gather(ex[...,-1],self.act_edges,axis=-1),a],axis=1)]
        preds,edge_preds = self.model(inp,training=False) if self.dropout else self.model(inp)
        preds,edge_preds = tf.concat([preds,b],axis=-1),tf.concat([edge_preds,a],axis=-1)
        preds,edge_preds = [self.normalize(pred,item,inverse=True) for pred,item in zip([preds,edge_preds],'ye')]
        return preds,edge_preds
    
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

    def load(self,model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        if model_dir.endswith('.h5'):
            self.model.load_weights(model_dir)
            model_dir = os.path.dirname(model_dir)
        else:
            self.model.load_weights(os.path.join(model_dir,'model.h5'))

        for item in 'xbyre':
            if os.path.exists(os.path.join(model_dir,'norm_%s.npy'%item)):
                setattr(self,'norm_%s'%item,np.load(os.path.join(model_dir,'norm_%s.npy'%item)))


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))

    train_de = {
        'train':True,
        'env':'astlingen',
        'order':1,
        'data_dir':'./envs/data/astlingen/1s_edge_conti128_rain50/',
        'act':'conti',
        'model_dir':'./model/astlingen/60s_50k_conti_esti/',
        'load_model':False,
        'setting_duration':5,
        'batch_size':16,
        'embed_size':64,
        'epochs':50000,
        'n_sp_layer':5,
        'n_tp_layer':5,
        'seq_in':60,'seq_out':60,
        # 'if_flood':5,
        'conv': 'Diff',
        'recurrent':'Conv1D',
        }
    for k,v in train_de.items():
        setattr(args,k,v)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)
    
    dG = DataGenerator(env.config,args.data_dir,args)
    dG.load(args.data_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    seq = max(args.seq_in,args.seq_out)
    n_events = int(max(dG.event_id))+1
    if os.path.isfile(os.path.join(args.data_dir,args.train_event_id)):
        train_ids = np.load(os.path.join(args.data_dir,args.train_event_id))
    elif args.load_model:
        train_ids = np.load(os.path.join(args.model_dir,'train_id.npy'))
    else:
        train_ids = np.random.choice(np.arange(n_events),int(n_events*args.ratio),replace=False)
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]
    train_idxs = dG.get_data_idxs(train_ids,seq)
    test_idxs = dG.get_data_idxs(test_ids,seq)

    emul = Estimator(args.conv,args.recurrent,args)
    if args.load_model:
        emul.load(args.model_dir)
        args.model_dir = os.path.join(args.model_dir,'retrain')
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        if 'model_dir' in config:
            config['model_dir'] += '/retrain'
    emul.set_norm(*dG.get_norm())
    yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

    t0 = time.time()
    train_losses,test_losses,secs = [],[],[0]
    log_dir = "logs/model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    for epoch in range(args.epochs):
        train_dats = dG.prepare_batch(train_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
        ex,ey = [dat for dat in train_dats[-2:]]
        train_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=True)
        train_losses.append(train_loss.numpy())

        test_dats = dG.prepare_batch(test_idxs,seq,args.batch_size,interval=args.setting_duration,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
        ex,ey = [dat for dat in test_dats[-2:]]
        test_loss = emul.fit_eval(x,a,b,y,ex,ey,fit=False)
        test_losses.append(test_loss)

        if train_loss < min([1e6]+train_losses[:-1]):
            emul.save(os.path.join(args.model_dir,'train'))
        if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
            emul.save(os.path.join(args.model_dir,'test'))
        if epoch > 0 and epoch % args.save_gap == 0:
            emul.save(os.path.join(args.model_dir,'%s'%epoch))
            
        secs.append(time.time()-t0)
        
        # Log output
        log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],train_loss,sum(test_loss))
        log += "(Node: {:.4f}".format(test_loss[0])
        i = 1
        if args.if_flood:
            log += " if_flood: {:.4f}".format(test_loss[i])
            i += 1
        log += " Edge: {:.4f})".format(test_loss[i])
        print(log)
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('Node loss', test_loss[0], step=epoch)
            i = 1
            if args.if_flood:
                tf.summary.scalar('Flood classification', test_loss[i], step=epoch)
                i += 1
            tf.summary.scalar('Edge loss', test_loss[i], step=epoch)

    # save
    emul.save(args.model_dir)
    np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
    np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
    np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
    np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
    np.save(os.path.join(args.model_dir,'time.npy'),np.array(secs[1:]))
    plt.plot(train_losses,label='train')
    plt.plot(np.array(test_losses).sum(axis=1),label='test')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)
