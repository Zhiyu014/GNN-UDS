import numpy as np
import torch as th
import os
device = th.device("cuda" if th.cuda.is_available() else "cpu")
class Memory:
    def __init__(self,limit,conv=False,cwd=None):     
        self.items = ['state','action','reward','next_state','done']
        self.limit = 2**limit
        self.conv = conv
        self.cur_capa = 0
        self.cwd = cwd

    def __len__(self):
        return self.cur_capa
    
    def sample(self, batch_size, continuous = False):
        if continuous:
            ind = np.random.randint(self.cur_capa-batch_size)
            ind = th.arange(ind,ind+batch_size,device=device)
        else:
            ind = th.randperm(self.cur_capa, device=device)[:batch_size]
        batch = []
        for item in self.items:
            if self.conv and 'state' in item and not hasattr(self,item):
                batch.append([getattr(self, f"{item}_{it}")[ind] for it in 'xe'])
            else:
                batch.append(getattr(self,item)[ind])
        return batch
        
    def update(self, trajs):
        assert len(trajs) == len(self.items), "s a r s' d"
        for item,traj in zip(self.items,trajs):
            if self.conv and 'state' in item and isinstance(traj, list):
                for it,tra in zip('xe',traj):
                    name = f"{item}_{it}"
                    setattr(self,name,
                            th.concat([getattr(self,name,tra[:0,...]),tra],dim=0)[-self.limit:])
            else:
                setattr(self,item,th.concat([getattr(self,item,traj[:0,...]),traj],dim=0)[-self.limit:])
        self.cur_capa = min(self.limit,self.cur_capa + len(trajs[1]))

    def clear(self):
        for item in self.items:
            if self.conv and 'state' in item and not hasattr(self,item):
                for it in 'xe':
                    name = f"{item}_{it}"
                    if hasattr(self,name):
                        setattr(self,name,getattr(self,name)[:0,...])
            else:
                if hasattr(self,item):
                    setattr(self,item,getattr(self,item)[:0,...])
        self.cur_capa = 0

    def save(self,cwd=None):
        cwd = self.cwd if cwd is None else cwd
        for item in self.items:
            if self.conv and 'state' in item and not hasattr(self,item):
                for it in 'xe':
                    name = f"{item}_{it}"
                    if hasattr(self,name):
                        np.save(os.path.join(cwd,f'experience_{name}.npy'),getattr(self,name).cpu().numpy())
            else:
                if hasattr(self,item):
                    np.save(os.path.join(cwd,f'experience_{item}.npy'),getattr(self,item).cpu().numpy())
            print('Save experience %s'%item)

    def load(self,cwd=None):
        cwd = self.cwd if cwd is None else cwd
        for item in self.items:
            if self.conv and 'state' in item and not hasattr(self,item):
                for it in 'xe':
                    name = f"{item}_{it}"
                    if os.path.exists(os.path.join(cwd,f'experience_{name}.npy')):
                        setattr(self,name,th.Tensor(np.load(os.path.join(cwd,f'experience_{name}.npy'))).to(device))
                        print('Load experience %s'%name)
            else:
                if os.path.exists(os.path.join(cwd,f'experience_{item}.npy')):
                    setattr(self,item,th.Tensor(np.load(os.path.join(cwd,f'experience_{item}.npy'))).to(device))
                    print('Load experience %s'%item)
        self.cur_capa = len(getattr(self,item))

    # def get_state_norm(self):
    #     state = np.asarray(self.state)
    #     mean = state.mean(axis=0)
    #     std = state.std(axis=0)
    #     return np.array([mean,std])


    # def get_reward_norm(self):
    #     reward = np.asarray(self.reward)
    #     mean = reward.mean()
    #     std = reward.std()
    #     return (mean,std)
