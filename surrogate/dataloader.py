import numpy as np
import multiprocessing as mp
from datetime import datetime
from swmm_api import read_inp_file
import os

class DataGenerator:
    def __init__(self,env,seq_len = 4,act = False,data_dir=None):
        self.env = env
        self.seq_len = seq_len
        self.data_dir = data_dir if data_dir is not None else './envs/data/{}/'.format(env.config['env_name'])
        if act:
            self.action_table = list(env.config['action_space'].values())
    
    def simulate(self, event, act = False):
        state = self.env.reset(event,global_state=True,seq=self.seq_len)
        states,settings = [state],[]
        done = False
        while not done:
            setting = [table[np.random.randint(0,len(table))] for table in self.action_table] if act else None
            done = self.env.step(setting)
            state = self.env.state(seq=self.seq_len)
            states.append(state)
            settings.append(setting)
        return np.array(states),np.array(settings) if act else None
    
    def state_split(self,states,settings=None,recurrent=True):
        if settings is not None:
            # B,T,N,S
            states = states[:settings.shape[0]+1]
            # B,T,n_act
            a = np.tile(np.expand_dims(settings,axis=1),[1,self.seq_len,1])
        h,q_totin,q_ds,r = [states[...,i] for i in range(4)]
        q_us = q_totin - r
        # B,T,N,in
        X = np.stack([h[:-1],q_us[:-1],q_ds[:-1],r[1:]],axis=-1)
        Y = np.stack([h[1:],q_us[1:],q_ds[1:]],axis=-1)
        if recurrent:
            Y = Y[:,-1,...]
        if settings is not None:
            X = np.concatenate([X,a],axis=-1)
        return X,Y

    def generate(self,events,processes=1,act=False,recurrent=True):
        pool = mp.Pool(processes)
        if processes > 1:
            res = [pool.apply_async(func=self.simulate,args=(event,act,)) for event in events]
            pool.close()
            pool.join()
            res = [self.state_split(*r.get(),recurrent) for r in res]
        else:
            res = [self.state_split(*self.simulate(event,act),recurrent) for event in events]
        self.X,self.Y = [np.concatenate([r[i] for r in res],axis=0) for i in range(2)]
        self.event_id = np.concatenate([np.repeat(i,r[0].shape[0]) for i,r in enumerate(res)])
    
    def sample(self,size,event=None):
        if event is not None:
            n_rain = np.in1d(self.event_id,event)
            X,Y = self.X[n_rain],self.Y[n_rain]
        else:
            X,Y = self.X,self.Y
        idx = np.random.choice(range(X.shape[0]),size)
        return X[idx],Y[idx]

    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        np.save(os.path.join(data_dir,'X.npy'),self.X)
        np.save(os.path.join(data_dir,'Y.npy'),self.Y)
        np.save(os.path.join(data_dir,'event_id.npy'),self.event_id)


    def load(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in ['X','Y','event_id']:
            dat = np.load(os.path.join(data_dir,name+'.npy'))
            setattr(self,name,dat)


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