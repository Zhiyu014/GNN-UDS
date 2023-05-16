import numpy as np
import multiprocessing as mp
import os

class DataGenerator:
    def __init__(self,env,seq_in = 4,seq_out=1,recurrent=True,act = False,if_flood=False,data_dir=None):
        self.env = env
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.recurrent = False if recurrent in ['None','False','NoneType'] else recurrent
        self.if_flood = if_flood
        self.data_dir = data_dir if data_dir is not None else './envs/data/{}/'.format(env.config['env_name'])
        if act:
            self.action_table = env.config['action_space']
            # self.adj = env.get_adj()
            # self.act_edges = env.get_edge_list(list(self.action_table.keys()))
    
    def simulate(self, event, seq = False, act = False):
        state = self.env.reset(event,global_state=True,seq=seq)
        perf = self.env.performance(seq=seq)
        states,perfs = [state],[perf]
        setting = [1 for _ in self.action_table] if act else None
        settings = [setting]
        done,i = False,0
        while not done:
            setting = [table[np.random.randint(0,len(table))] for table in self.action_table.values()] if act and i % (self.env.config['control_interval']//self.env.config['interval']) == 0 else setting

            done = self.env.step(setting)
            state = self.env.state(seq=seq)
            perf = self.env.performance(seq=seq)
            states.append(state)
            perfs.append(perf)
            settings.append(setting)
            i += 1
        return np.array(states),np.array(perfs),np.array(settings) if act else None

    def generate(self,events,processes=1,seq=False,act=False):
        if processes > 1:
            pool = mp.Pool(processes)
            res = [pool.apply_async(func=self.simulate,args=(event,seq,act,)) for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [self.simulate(event,seq,act) for event in events]
        self.states,self.perfs = [np.concatenate([r[i] for r in res],axis=0) for i in range(2)]
        self.settings = np.concatenate([r[2] for r in res],axis=0) if act else None
        self.event_id = np.concatenate([np.repeat(i,r[0].shape[0]) for i,r in enumerate(res)])

    def state_split(self,states,perfs):
        h,q_totin,q_ds,r = [states[...,i] for i in range(4)]
        # h,q_totin,q_ds,r,q_w = [states[...,i] for i in range(5)]
        q_us = q_totin - r
        # B,T,N,in
        n_spl = self.seq_out if self.recurrent else 1
        X = np.stack([h[:-n_spl],q_us[:-n_spl],q_ds[:-n_spl]],axis=-1)
        Y = np.stack([h[n_spl:],q_us[n_spl:],q_ds[n_spl:]],axis=-1)

        if self.if_flood:
            f = (perfs>0).astype(int)
            f = np.eye(2)[f].squeeze(-2)
            X,Y = np.concatenate([X,f[:-n_spl]],axis=-1),np.concatenate([Y,f[n_spl:]],axis=-1)
        Y = np.concatenate([Y,perfs[n_spl:]],axis=-1)
        B = np.expand_dims(r[n_spl:],axis=-1)
        if self.recurrent:
            X = X[:,-self.seq_in:,...]
            B = B[:,:self.seq_out,...]
            Y = Y[:,:self.seq_out,...]
        return X,B,Y

    def expand_seq(self,dats,seq,zeros=True):
        dats = np.stack([np.concatenate([np.tile(np.zeros_like(s) if zeros else np.ones_like(s),(max(seq-idx,0),)+tuple(1 for _ in s.shape)),dats[max(idx-seq,0):idx]],axis=0) for idx,s in enumerate(dats)])
        return dats

    def prepare(self,seq=0,event=None):
        res = []
        event = np.arange(int(max(self.event_id))+1) if event is None else event
        for idx in event:
            num = self.event_id==idx
            if seq > 0:
                states,perfs = [self.expand_seq(dat[num],seq) for dat in [self.states,self.perfs]]
                settings = self.expand_seq(self.settings[num],seq,False) if self.settings is not None else None
            x,b,y = self.state_split(states,perfs)
            if self.settings is not None:
                settings = settings[self.seq_out:,:self.seq_out,...] if self.recurrent else settings[1:]
            r = [dat.astype(np.float32) if dat is not None else dat for dat in [x,settings,b,y]]
            res.append(r)
        return [np.concatenate([r[i] for r in res],axis=0) if r[i] is not None else None for i in range(4)]



    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save(os.path.join(data_dir,'states.npy'),self.states)
        np.save(os.path.join(data_dir,'perfs.npy'),self.perfs)
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
        self.get_norm()

    # def get_norm(self):
    #     if not hasattr(self,'normal'):
    #         norm = self.X.copy() + 1e-6
    #         while len(norm.shape) > 2:
    #             norm = norm.max(axis=0)
    #         if self.if_flood:
    #             norm[:,-2:] = 1
    #         norm_b = self.B.copy() + 1e-6
    #         while len(norm_b.shape) > 2:
    #             norm_b = norm_b.max(axis=0)
    #         self.normal = np.concatenate([norm,norm_b],axis=-1)
    #     return self.normal
    
    def get_norm(self):
        if not hasattr(self,'normal'):
            norm = np.concatenate([self.states[...,:-1],self.perfs,self.states[...,-1:]],axis=-1)
            norm[...,1] = norm[...,1] - norm[...,-1]
            while len(norm.shape) > 2:
                norm = norm.max(axis=0)
            if self.if_flood:
                norm = np.concatenate([norm[...,:-2] + 1e-6,np.ones(norm.shape[:-1]+(2,),dtype=np.float32),np.tile(np.float32(norm[...,-2].max())+1e-6,(norm.shape[0],1)),norm[...,-1:] + 1e-6],axis=-1)
                # norm = np.concatenate([norm[...,:-2] + 1e-6,np.ones(norm.shape[:-1]+(2,),dtype=np.float32),norm[...,-2:] + 1e-6],axis=-1)
            else:
                norm += 1e-6

            # if self.if_flood:
            #     norm = np.concatenate([norm[...,:-1] + 1e-6,np.ones((norm.shape[0],1),dtype=np.float32),norm[...,-1:] + 1e-6],axis=-1)
            # else:
            self.normal = norm.astype(np.float32)
        return self.normal
            

    def normalize(self,dat,inverse=False):
        norm = self.get_norm()
        dim = dat.shape[-1]
        if dim >= 3:
            return dat * norm[...,:dim] if inverse else dat/norm[...,:dim]
        else:
            return dat * norm[...,-dim:] if inverse else dat/norm[...,-dim:]

