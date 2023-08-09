import numpy as np
import multiprocessing as mp
import os
from swmm_api import swmm5_run

class DataGenerator:
    def __init__(self,env,seq_in = 4,seq_out=1,recurrent=True,act = False,if_flood=False,use_edge=False,data_dir=None):
        self.env = env
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.if_flood = if_flood
        self.use_edge = use_edge
        self.data_dir = data_dir if data_dir is not None else './envs/data/{}/'.format(env.config['env_name'])
        self.act = act
        if act:
            self.action_table = env.config['action_space']
            # self.adj = env.get_adj()
            # self.act_edges = env.get_edge_list(list(self.action_table.keys()))

    
    def simulate(self, event, seq = False, act = False, hotstart = False):
        state = self.env.reset(event,global_state=True,seq=seq)
        perf = self.env.flood(seq=seq)
        states,perfs = [state],[perf]
        if self.use_edge:
            edge_state = self.env.state_full(seq,'links')
            edge_states = [edge_state]
        setting = [1 for _ in self.action_table] if act else None
        settings = [setting]
        done,i = False,0
        while not done:
            if hotstart:
                eval_file = self.env.get_eval_file()
                _ = swmm5_run(eval_file)
            setting = self.env.controller(state,act) if act and i % (self.env.config['control_interval']//self.env.config['interval']) == 0 else setting
            done = self.env.step(setting)
            state = self.env.state(seq=seq)
            perf = self.env.flood(seq=seq)
            states.append(state)
            perfs.append(perf)
            settings.append(setting)
            if self.use_edge:
                edge_state = self.env.state_full(seq,'links')
                edge_states.append(edge_state)
            i += 1
        if self.use_edge:
            return np.array(states),np.array(perfs),np.array(settings) if act else None,np.array(edge_states)
        else:
            return np.array(states),np.array(perfs),np.array(settings) if act else None
        
    def generate(self,events,processes=1,repeats=1,seq=False,act=False):
        if processes > 1:
            pool = mp.Pool(processes)
            res = [pool.apply_async(func=self.simulate,args=(event,seq,act,))
                    for event in events for _ in range(repeats)]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [self.simulate(event,seq,act)
                    for event in events for _ in range(repeats)]
        self.states,self.perfs = [np.concatenate([r[i] for r in res],axis=0) for i in range(2)]
        self.settings = np.concatenate([r[2] for r in res],axis=0) if act else None
        if self.use_edge:
            self.edge_states = np.concatenate([r[-1] for r in res],axis=0) if self.use_edge else None
        self.event_id = np.concatenate([np.repeat(i,sum([r[0].shape[0] for r in res[i*repeats:(i+1)*repeats]]))
                                         for i,_ in enumerate(events)])

    def state_split(self,states,perfs):
        h,q_totin,q_ds,r = [states[...,i] for i in range(4)]
        # h,q_totin,q_ds,r,q_w = [states[...,i] for i in range(5)]
        q_us = q_totin - r
        # B,T,N,in
        n_spl = self.seq_out if self.recurrent else 1
        X = np.stack([h[:-n_spl],q_us[:-n_spl],q_ds[:-n_spl],r[:-n_spl]],axis=-1)
        Y = np.stack([h[n_spl:],q_us[n_spl:],q_ds[n_spl:]],axis=-1)

        if self.if_flood:
            f = (perfs>0).astype(int)
            f = np.eye(2)[f].squeeze(-2)
            X,Y = np.concatenate([X[...,:-1],f[:-n_spl],X[...,-1:]],axis=-1),np.concatenate([Y,f[n_spl:]],axis=-1)
        Y = np.concatenate([Y,perfs[n_spl:]],axis=-1)
        B = np.expand_dims(r[n_spl:],axis=-1)
        if self.recurrent:
            X = X[:,-self.seq_in:,...]
            B = B[:,:self.seq_out,...]
            Y = Y[:,:self.seq_out,...]
        return X,B,Y

    def edge_state_split(self,edge_states):
        n_spl = self.seq_out if self.recurrent else 1
        ex = edge_states[:-n_spl]
        ey = edge_states[n_spl:,...,:-1]
        if self.recurrent:
            ex,ey = ex[:,-self.seq_in:,...],ey[:,:self.seq_out,...]
        return ex,ey

    def expand_seq(self,dats,seq,zeros=True):
        dats = np.stack([np.concatenate([np.tile(np.zeros_like(s) if zeros else np.ones_like(s),(max(seq-idx,0),)+tuple(1 for _ in s.shape)),dats[max(idx-seq,0):idx]],axis=0) for idx,s in enumerate(dats)])
        return dats

    def prepare(self,seq=0,event=None):
        res = []
        event = np.arange(int(max(self.event_id))+1) if event is None else event
        for idx in event:
            num = np.where(self.event_id==idx)[0]
            if seq > 0:
                states,perfs = [self.expand_seq(dat[num],seq) for dat in [self.states,self.perfs]]
                edge_states = self.expand_seq(self.edge_states[num],seq) if self.use_edge else None
                settings = self.expand_seq(self.settings[num],seq,False) if self.settings is not None else None
            else:
                states,perfs = self.states[num],self.perfs[num]
                edge_states = self.edge_states[num] if self.use_edge else None
                settings = self.settings[num] if self.settings is not None else None
            x,b,y = self.state_split(states,perfs)
            ex,ey = self.edge_state_split(edge_states) if self.use_edge else (None,None)
            if self.settings is not None:
                settings = settings[self.seq_out:,:self.seq_out,...] if self.recurrent else settings[1:]            
            r = [dat.astype(np.float32) if dat is not None else dat for dat in [x,settings,b,y,ex,ey]]
            res.append(r)
        return [np.concatenate([r[i] for r in res],axis=0) if r[i] is not None else None for i in range(6)]

    def prepare_batch(self,seq=0,event=None,batch_size=32):
        res = []
        event = np.arange(int(max(self.event_id))+1) if event is None else event
        event_idxs = [np.argwhere(self.event_id == idx).flatten() for idx in event]
        numbs = sum([idx.shape[0] for idx in event_idxs])
        event_batch = np.random.choice(range(len(event_idxs)),batch_size,p=[idx.shape[0]/numbs for idx in event_idxs])
        for idx in event_batch:
            lm,um = event_idxs[idx].min(),event_idxs[idx].max()+1
            ix = np.random.randint(lm+seq,um-seq-1)
            # sli = slice(max(ix-seq+1,lm),min(ix+seq+1,um))
            if seq > 0:
                states = (self.states[ix-seq+1:ix+1],self.states[ix+1:ix+seq+1])
                perfs = (self.perfs[ix-seq+1:ix+1],self.perfs[ix+1:ix+seq+1])
                edge_states = (self.edge_states[ix-seq+1:ix+1],self.edge_states[ix+1:ix+seq+1]) if self.use_edge else None
                settings = self.settings[ix+1:ix+seq+1] if self.settings is not None else None
            else:
                states,perfs = (self.states[ix],self.states[ix+1]),(self.perfs[ix],self.perfs[ix+1])
                edge_states = (self.edge_states[ix],self.edge_states[ix+1]) if self.use_edge else None
                settings = self.settings[ix+1] if self.settings is not None else None
            x,b,y = self.state_split_batch(states,perfs)
            ex,ey = self.edge_state_split_batch(edge_states) if self.use_edge else (None,None)
            if self.settings is not None:
                settings = settings[:self.seq_out,...] if self.recurrent else settings
            r = [dat.astype(np.float32) if dat is not None else dat for dat in [x,settings,b,y,ex,ey]]
            res.append(r)
        return [np.stack([r[i] for r in res],axis=0) if r[i] is not None else None for i in range(6)]

    def state_split_batch(self,states,perfs):
        h,q_totin,q_ds,r = [states[0][...,i] for i in range(4)]
        q_us = q_totin - r
        X = np.stack([h,q_us,q_ds,r],axis=-1)

        h,q_totin,q_ds,r = [states[1][...,i] for i in range(4)]
        q_us = q_totin - r
        Y = np.stack([h,q_us,q_ds],axis=-1)
        B = np.expand_dims(r,axis=-1)

        if self.if_flood:
            f1 = (perfs[0]>0).astype(int)
            f1 = np.eye(2)[f1].squeeze(-2)
            f2 = (perfs[1]>0).astype(int)
            f2 = np.eye(2)[f2].squeeze(-2)
            X,Y = np.concatenate([X[...,:-1],f1,X[...,-1:]],axis=-1),np.concatenate([Y,f2],axis=-1)
        Y = np.concatenate([Y,perfs[1]],axis=-1)
        if self.recurrent:
            X = X[-self.seq_in:,...]
            B = B[:self.seq_out,...]
            Y = Y[:self.seq_out,...]
        return X,B,Y

    def edge_state_split_batch(self,edge_states):
        ex = edge_states[0]
        ey = edge_states[1][...,:-1]
        if self.recurrent:
            ex,ey = ex[-self.seq_in:,...],ey[:self.seq_out,...]
        return ex,ey

    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save(os.path.join(data_dir,'states.npy'),self.states)
        np.save(os.path.join(data_dir,'perfs.npy'),self.perfs)
        if self.use_edge:
            np.save(os.path.join(data_dir,'edge_states.npy'),self.edge_states)
        if self.settings is not None:
            np.save(os.path.join(data_dir,'settings.npy'),self.settings)
        np.save(os.path.join(data_dir,'event_id.npy'),self.event_id)


    def load(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in ['states','perfs','settings','event_id']:
            if os.path.isfile(os.path.join(data_dir,name+'.npy')):
                dat = np.load(os.path.join(data_dir,name+'.npy')).astype(np.float32)
            else:
                dat = None
            setattr(self,name,dat)
        if self.use_edge:
            self.edge_states = np.load(os.path.join(data_dir,'edge_states.npy')).astype(np.float32)
        self.get_norm()

    def get_norm(self):
        norm = np.concatenate([self.states,self.perfs],axis=-1)
        norm[...,1] = norm[...,1] - norm[...,3]
        while len(norm.shape) > 2:
            norm = norm.max(axis=0)
        norm_b = (norm[...,-2:-1] + 1e-6).astype(np.float32)
        
        if self.if_flood:
            norm_x = np.concatenate([norm[...,:-2] + 1e-6,np.ones(norm.shape[:-1]+(2,),dtype=np.float32),norm[...,-2:-1] + 1e-6],axis=-1)
            norm_y = np.concatenate([norm[...,:-2] + 1e-6,np.ones(norm.shape[:-1]+(2,),dtype=np.float32),np.tile(np.float32(norm[...,-1].max())+1e-6,(norm.shape[0],1))],axis=-1)
        else:
            norm_x = norm[...,:-1]+1e-6
            norm_y = np.concatenate([norm[...,:-2] + 1e-6,np.tile(np.float32(norm[...,-1].max())+1e-6,(norm.shape[0],1))],axis=-1)
        norm_x = norm_x.astype(np.float32)
        norm_y = norm_y.astype(np.float32)
        if self.use_edge:
            norm_e = np.abs(self.edge_states.copy())
            while len(norm_e.shape) > 2:
                norm_e = norm_e.max(axis=0)
            norm_e = np.concatenate([norm_e[:,:-1]+1e-6,norm_e[:,-1:]],axis=-1) if self.act else norm_e+1e-6
            
            # norm_e_min = self.edge_states.copy()
            # while len(norm_e_min.shape) > 2:
            #     norm_e_min = norm_e_min.min(axis=0)
            # norm_e = np.stack([norm_e,norm_e_min],axis=-1)
            norm_e = norm_e.astype(np.float32)
            return norm_x,norm_b,norm_y,norm_e
        else:
            return norm_x,norm_b,norm_y
            

    # def normalize(self,dat,inverse=False):
    #     norm = self.get_norm()
    #     dim = dat.shape[-1]
    #     return dat * norm[...,:dim] if inverse else dat/norm[...,:dim]

