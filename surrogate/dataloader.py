import numpy as np
import multiprocessing as mp
import os
from swmm_api import swmm5_run,read_inp_file
from datetime import timedelta
from envs import get_env

class DataGenerator:
    def __init__(self,env_config,data_dir=None,args=None):
        self.config = env_config
        self.data_dir = data_dir if data_dir is not None else './envs/data/{}/'.format(self.config['env_name'])
        self.items = ['states','perfs','settings','rains','edge_states','event_id','dones']
        self.pre_step = args.rainfall.get('pre_time',0) // self.config['interval']
        self.seq_in = self.seq_out = getattr(args,"seq",5)
        self.if_flood = getattr(args,"if_flood",False)
        self.is_outfall = getattr(args,"is_outfall",np.array([0]))
        self.act = getattr(args,"act",False)
        self.setting_duration = getattr(args,"setting_duration",5)
        if self.act:
            self.action_space = self.config['action_space']
        self.limit = 2**getattr(args,"limit",22)
        self.cur_capa = 0
        self.update_num = 0

    def simulate(self, env, event, seq = False, act = False, hotstart = False):
        state = env.reset(event,global_state=True,seq=seq)
        perf = env.flood(seq=seq)
        states,perfs,edge_states = [],[],[]
        edge_state = env.state_full(seq,'links')
        setting = [1 for _ in self.action_space] if act else None
        rains,settings = [],[]
        done,i = False,0
        while not done:
            if hotstart:
                eval_file = env.get_eval_file()
                ct = env.env.methods['simulation_time']()
                inp = read_inp_file(eval_file)
                inp['OPTIONS']['END_DATE'] = (ct + timedelta(minutes=hotstart)).date()
                inp['OPTIONS']['END_TIME'] = (ct + timedelta(minutes=hotstart)).time()
                inp.write_file(eval_file)
                _ = swmm5_run(eval_file)
            setting = env.controller(act,state,setting) if act and i % (self.setting_duration//self.config['interval']) == 0 else setting
            done = env.step(setting)
            state = env.state_full(seq=seq)
            rain = env.rainfall(seq=seq)
            perf = env.flood(seq=seq)
            states.append(state)
            perfs.append(perf)
            settings.append(setting)
            rains.append(rain)
            edge_state = env.state_full(seq,'links')
            edge_states.append(edge_state)
            i += 1
        return np.array(states),np.array(perfs),np.array(settings) if act else None,np.array(rains),np.array(edge_states)
        
    def generate(self,events,processes=1,repeats=1,seq=False,act=False):
        env = get_env(self.config['env_name'])(initialize=False)
        if processes > 1:
            pool = mp.Pool(processes)
            res = [pool.apply_async(func=self.simulate,args=(env,event,seq,act,))
                    for _ in range(repeats) for event in events]
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [self.simulate(env,event,seq,act)
                    for _ in range(repeats) for event in events]
        self.states,self.perfs = [np.concatenate([r[i][self.pre_step:] for r in res],axis=0) for i in range(2)]
        self.settings = np.concatenate([r[2][self.pre_step:] for r in res],axis=0) if act else None
        self.rains = np.concatenate([r[3][self.pre_step:] for r in res],axis=0)
        self.edge_states = np.concatenate([r[-1][self.pre_step:] for r in res],axis=0)
        self.event_id = np.concatenate([np.repeat(i,res[idx][0][self.pre_step:].shape[0])
                                         for idx,i in enumerate([i for _ in range(repeats) for i,_ in enumerate(events)])],axis=0)
        self.dones = np.concatenate([np.eye(r[0][self.pre_step:].shape[0],dtype=np.int32)[-1] for r in res],axis=0)
        self.cur_capa = self.states.shape[0]

    def get_flood_weight(self,seq=0):
        if not hasattr(self,f"flood_weight_{seq}"):
            wei = self.perfs.sum(axis=-1).sum(axis=-1)
            wei = np.pad(np.convolve(wei,np.ones(max(seq,1)),'valid'),(0,max(seq,1)-1),'constant')
            n_flood,n_dry = wei[wei>0].shape[0],wei[wei==0].shape[0]
            if n_flood > 0 and n_flood / wei.shape[0] < 0.5:
                ratio = n_dry / n_flood
                wei = np.ones_like(wei) * (wei == 0) + np.ones_like(wei) * ratio * (wei > 0)
            else:
                wei = np.ones_like(wei)
            setattr(self,f"flood_weight_{seq}",wei)
        return getattr(self,f"flood_weight_{seq}")

    def expand_seq(self,dats,seq,zeros=True):
        dats = np.stack([np.concatenate([np.tile(np.zeros_like(s) if zeros else np.ones_like(s),(max(seq-idx,0),)+tuple(1 for _ in s.shape)),dats[max(idx-seq,0):idx]],axis=0) for idx,s in enumerate(dats)])
        return dats

    def get_data_idxs(self,event=None,seq=0,seq_out=None):
        event = np.arange(int(max(self.event_id))+1) if event is None else event
        event_idxs = [np.argwhere(self.event_id == idx).flatten() for idx in event]
        event_idxs = [np.split(event_idx,np.where(self.dones[event_idx]==1)[0]+1) for event_idx in event_idxs]
        seq_out = seq_out if seq_out is not None else seq
        event_idxs = np.concatenate([np.concatenate([dat[seq:-seq_out] for dat in data],axis=0) for data in event_idxs],axis=0)
        return event_idxs
        

    def prepare_batch(self,event_idxs,seq=0,batch_size=32,interval=1,continuous=False,trim=True,return_idx=False):
        if continuous:
            idxs = np.random.randint(event_idxs.shape[0]//interval-batch_size)
            idxs = interval*np.arange(idxs,idxs+batch_size)
        else:
            wei = self.get_flood_weight(seq)[event_idxs][np.arange(0,event_idxs.shape[0],interval)]
            idxs = interval*np.random.choice(event_idxs.shape[0]//interval,batch_size,replace=False,
                                             p=wei/wei.sum(),
                                             )
            # idxs = interval*np.random.choice(event_idxs.shape[0]//interval,batch_size,replace=False)
        idxs = event_idxs[idxs]
        if seq > 0:
            ixs = np.apply_along_axis(lambda t:np.arange(t-seq,t),axis=1,arr=np.expand_dims(idxs,axis=-1))
            iys = np.apply_along_axis(lambda t:np.arange(t,t+seq),axis=1,arr=np.expand_dims(idxs,axis=-1))
            states = (np.take(self.states,ixs,axis=0),np.take(self.states,iys,axis=0))
            perfs = (np.take(self.perfs,ixs,axis=0),np.take(self.perfs,iys,axis=0))
            rx,ry = np.take(self.rains,ixs,axis=0),np.take(self.rains,iys,axis=0)
            edge_states = (np.take(self.edge_states,ixs,axis=0),np.take(self.edge_states,iys,axis=0))
            settings = np.take(self.settings,iys,axis=0) if self.settings is not None else None
        else:
            states,perfs = (self.states[idxs-1],self.states[idxs]),(self.perfs[idxs-1],self.perfs[idxs])
            rx,ry = self.rains[idxs-1],self.rains[idxs]
            edge_states = (self.edge_states[idxs-1],self.edge_states[idxs])
            settings = self.settings[idxs] if self.settings is not None else None
        x,b,y = self.state_split_batch(states,perfs,trim)
        ex,ey = self.edge_state_split_batch(edge_states,trim)
        if trim:
            rx,ry = rx[:,-self.seq_in:,...],ry[:,:self.seq_out,...]
        if self.settings is not None and trim:
            settings = settings[:,:self.seq_out,...]
        dats = [x,settings,b,y,rx,ry,ex,ey]
        if continuous:
            done = np.eye(batch_size)[np.where(np.diff(idxs) != interval)[0]].sum(axis=0)
        else:
            done = np.take(self.dones,iys).sum(axis=-1) if seq > 0 else self.dones[idxs].sum(axis=-1)
        dats.append(done)
        if return_idx:
            dats.append(self.event_id[idxs])
        return [dat.astype(np.float32) if dat is not None else dat for dat in dats]
    
    def state_split_batch(self,states,perfs,trim=True):
        h,q_totin,q_ds,r = [states[0][...,i] for i in range(4)]
        q_us = q_totin - r
        X = np.stack([h,q_us,q_ds,r],axis=-1)

        h,q_totin,q_ds,r = [states[1][...,i] for i in range(4)]
        q_us = q_totin - r
        Y = np.stack([h,q_us,q_ds],axis=-1)
        B = np.expand_dims(r,axis=-1)
        if self.config['tide']:
            t = h * self.is_outfall
            B = np.concatenate([B,np.expand_dims(t,axis=-1)],axis=-1)

        if self.if_flood:
            f1 = (perfs[0]>0).astype(float)
            # f1 = np.eye(2)[f1].squeeze(-2)
            f2 = (perfs[1]>0).astype(float)
            # f2 = np.eye(2)[f2].squeeze(-2)
            X,Y = np.concatenate([X[...,:-1],f1,X[...,-1:]],axis=-1),np.concatenate([Y,f2],axis=-1)
        Y = np.concatenate([Y,perfs[1]],axis=-1)
        if trim:
            X = X[:,-self.seq_in:,...]
            B = B[:,:self.seq_out,...]
            Y = Y[:,:self.seq_out,...]
        return X,B,Y

    def edge_state_split_batch(self,edge_states,trim=True):
        ex = edge_states[0]
        ey = edge_states[1][...,:-1]
        if trim:
            ex,ey = ex[:,-self.seq_in:,...],ey[:,:self.seq_out,...]
        return ex,ey
    
    def get_flood_poswei(self):
        if self.if_flood:
            n_flood = np.bincount(np.where(self.perfs>0)[1])
            flood_wei = self.perfs.shape[0]/n_flood - 1
            flood_wei[np.isinf(flood_wei)] = 1
            return flood_wei
        else:
            return np.ones(self.perfs.shape[1])

    def update(self,trajs,test_id=None):
        for traj,item in zip(trajs,self.items):
            if getattr(self,item,None) is None:
                setattr(self,item,np.zeros((0,)+traj.shape[1:],np.float32))
            if test_id is not None:
                test_idxs = np.concatenate([np.argwhere(self.event_id == idx).flatten() for idx in test_id],axis=0)
                train_idxs = np.setdiff1d(np.arange(self.event_id.shape[0]),test_idxs)
                setattr(self,item,np.concatenate([np.take(getattr(self,item),train_idxs,axis=0),np.take(getattr(self,item),test_idxs,axis=0),traj],axis=0)[-self.limit:])
            else:
                setattr(self,item,np.concatenate([getattr(self,item),traj],axis=0)[-self.limit:])
        self.cur_capa = min(self.cur_capa + traj.shape[0],self.limit)
        self.update_num = min(self.update_num + traj.shape[0],self.limit)

    def save(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for name in self.items:
            if getattr(self,name,None) is not None:
                np.save(os.path.join(data_dir,name+'.npy'),getattr(self,name))

    def load(self,data_dir=None):
        data_dir = data_dir if data_dir is not None else self.data_dir
        for name in self.items:
            if os.path.isfile(os.path.join(data_dir,name+'.npy')):
                dat = np.load(os.path.join(data_dir,name+'.npy'),mmap_mode='r').astype(np.float32)
            else:
                dat = None
            setattr(self,name,dat)
        self.cur_capa = self.states.shape[0] if self.states is not None else 0

    def clear(self):
        for name in self.items:
            setattr(self,name,None)
        self.cur_capa = 0
        self.update_num = 0

    def get_norm(self):
        norm = np.concatenate([self.states,self.perfs],axis=-1)
        norm[...,1] = norm[...,1] - norm[...,3]
        while len(norm.shape) > 2:
            norm = norm.max(axis=0)
        if self.config['global_state'][0][-1] == 'head':
            norm_h = np.tile(np.float32(norm[...,0].max()+1e-6),(norm.shape[0],1))
        else:
            norm_h = norm[...,0:1]+1e-6
        norm_b = (norm[...,-2:-1] + 1e-6)
        if self.config['tide']:
            norm_b = np.concatenate([norm_b,norm_h*np.expand_dims(self.is_outfall,axis=-1)+1e-6],axis=-1)
        if self.if_flood:
            norm_x = np.concatenate([norm_h,norm[...,1:-2] + 1e-6,np.ones(norm.shape[:-1]+(1,)),norm[...,-2:-1] + 1e-6],axis=-1)
            norm_y = np.concatenate([norm_h,norm[...,1:-2] + 1e-6,np.ones(norm.shape[:-1]+(1,)),np.tile(np.float32(norm[...,-1].max())+1e-6,(norm.shape[0],1))],axis=-1)
        else:
            norm_x = np.concatenate([norm_h,norm[...,1:-1]+1e-6],axis=-1)
            norm_y = np.concatenate([norm_h,norm[...,1:-2] + 1e-6,np.tile(np.float32(norm[...,-1].max())+1e-6,(norm.shape[0],1))],axis=-1)
        if self.config['global_state'][0][-1] == 'head':
            norm_hmin = np.tile(np.float32(self.states[...,0].min()),(norm.shape[0],1))
            if self.config['tide']:
                norm_b = np.stack([norm_b,np.concatenate([np.zeros_like(norm_b[...,:1]),norm_hmin*np.expand_dims(self.is_outfall,axis=-1)],axis=-1)])
            else:
                norm_b = np.stack([norm_b,np.zeros_like(norm_b)])
            norm_x = np.stack([norm_x,np.concatenate([norm_hmin,np.zeros_like(norm_x[...,1:])],axis=-1)])
            norm_y = np.stack([norm_y,np.concatenate([norm_hmin,np.zeros_like(norm_y[...,1:])],axis=-1)])
        else:
            norm_b = np.stack([norm_b,np.zeros_like(norm_b)])
            norm_x = np.stack([norm_x,np.zeros_like(norm_x)])
            norm_y = np.stack([norm_y,np.zeros_like(norm_y)])
        norm_r = self.rains.max(axis=0)
        norm_r = np.stack([norm_r,np.zeros_like(norm_r)])

        norm_e = np.abs(self.edge_states.copy())
        while len(norm_e.shape) > 2:
            norm_e = norm_e.max(axis=0)
        norm_e = np.concatenate([norm_e[:,:-1]+1e-6,norm_e[:,-1:]],axis=-1) if self.act else norm_e+1e-6
        norm_e = np.stack([norm_e,np.zeros_like(norm_e)])
        # norm_e_min = self.edge_states.copy()
        # while len(norm_e_min.shape) > 2:
        #     norm_e_min = norm_e_min.min(axis=0)
        # norm_e = np.stack([norm_e,norm_e_min],axis=-1)
        return [norm.astype(np.float32) for norm in [norm_x,norm_b,norm_y,norm_r,norm_e]]