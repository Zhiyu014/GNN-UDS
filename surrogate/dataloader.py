import numpy as np
import multiprocessing as mp
from datetime import datetime
from swmm_api import read_inp_file
import os

class DataGenerator:
    def __init__(self,env,seq_in = 4,seq_out=1,recurrent=True,act = False,data_dir=None):
        self.env = env
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.recurrent = False if recurrent in ['False','None','NoneType'] else True
        self.data_dir = data_dir if data_dir is not None else './envs/data/{}/'.format(env.config['env_name'])
        if act:
            self.action_table = list(env.config['action_space'].values())
    
    def simulate(self, event, act = False):
        state = self.env.reset(event,global_state=True,seq=self.seq_in & self.recurrent)
        states,settings = [state],[]
        done = False
        while not done:
            setting = [table[np.random.randint(0,len(table))] for table in self.action_table] if act else None
            done = self.env.step(setting)
            state = self.env.state(seq=self.seq_in & self.recurrent)
            states.append(state)
            settings.append(setting)
        return np.array(states),np.array(settings) if act else None
    
    def state_split(self,states,settings=None):
        if settings is not None:
            # B,T,N,S
            states = states[:settings.shape[0]+1]
            # B,T,n_act
            a = np.tile(np.expand_dims(settings,axis=1),[1,self.seq_in,1])
        h,q_totin,q_ds,r = [states[...,i] for i in range(3)]
        # h,q_totin,q_ds,r,q_w = [states[...,i] for i in range(4)]
        q_us = q_totin - r
        # B,T,N,in
        # TODO: seq_out
        n_spl = self.seq_out if self.recurrent else 1
        X = np.stack([h[:-n_spl],q_us[:-n_spl],q_ds[:-n_spl],r[n_spl:]],axis=-1)
        Y = np.stack([h[n_spl:],q_us[n_spl:],q_ds[n_spl:]],axis=-1)
        # Y = np.stack([h[1:],q_us[1:],q_ds[1:],q_w[1:]],axis=-1)
        if self.recurrent:
            Y = Y[:,-self.seq_out:,...]
        if settings is not None:
            X = np.concatenate([X,a],axis=-1)
        return X,Y

    def generate(self,events,processes=1,act=False):
        pool = mp.Pool(processes)
        if processes > 1:
            res = [pool.apply_async(func=self.simulate,args=(event,act,)) for event in events]
            pool.close()
            pool.join()
            res = [self.state_split(*r.get()) for r in res]
        else:
            res = [self.state_split(*self.simulate(event,act)) for event in events]
        self.X,self.Y = [np.concatenate([r[i] for r in res],axis=0) for i in range(2)]
        self.event_id = np.concatenate([np.repeat(i,r[0].shape[0]) for i,r in enumerate(res)])
    
    def sample(self,size,event=None,norm=False):
        if event is not None:
            n_rain = np.in1d(self.event_id,event)
            X,Y = self.X[n_rain],self.Y[n_rain]
        else:
            X,Y = self.X,self.Y
        idx = np.random.choice(range(X.shape[0]),size)
        return (self.normalize(X[idx]),self.normalize(Y[idx])) if norm else (X[idx],Y[idx])


    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        np.save(os.path.join(data_dir,'X.npy'),self.X)
        np.save(os.path.join(data_dir,'Y.npy'),self.Y)
        np.save(os.path.join(data_dir,'event_id.npy'),self.event_id)


    def load(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in ['X','Y','event_id']:
            dat = np.load(os.path.join(data_dir,name+'.npy')).astype(np.float32)
            setattr(self,name,dat)
        self.get_norm()

    def get_norm(self):
        if not hasattr(self,'normal'):
            norm = self.X.copy()
            while len(norm.shape) > 2:
                norm = norm.max(axis=0)
            self.normal = norm + 1e-6
        return self.normal
    
    def normalize(self,dat,inverse=False):
        norm = self.get_norm()
        if inverse:
            return dat * norm[...,:dat.shape[-1]]
        else:
            return dat/norm[...,:dat.shape[-1]]

def generate_file(file,arg):
    inp = read_inp_file(file)
    for k,v in inp.TIMESERIES.items():
        if k.startswith(arg['suffix']):
            dura = v.data[-1][0] - v.data[0][0]
            st = (inp.OPTIONS['START_DATE'],inp.OPTIONS['START_TIME'])
            st = datetime(st[0].year,st[0].month,st[0].day,st[1].hour,st[1].minute,st[1].second)
            et = (st + dura)
            inp.OPTIONS['END_DATE'],inp.OPTIONS['END_TIME'] = et.date(),et.time()
            inp.RAINGAGES['RainGage'].Timeseries = k
            if not os.path.exists(arg['filedir']+k+'.inp'):
                inp.write_file(arg['filedir']+k+'.inp')
    events = [arg['filedir']+k+'.inp' for k in inp.TIMESERIES if k.startswith(arg['suffix'])]
    return events