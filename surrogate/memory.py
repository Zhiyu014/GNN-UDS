import numpy as np
class Memory:
    def __init__(self,limit,conv=False):     
        self.items = ['state','action','reward','next_state','done']
        self.limit = 2**limit
        self.conv = conv
        self.cur_capa = 0

    def __len__(self):
        return self.cur_capa
    
    def sample(self, batch_size, continuous = False):
        if continuous:
            ind = np.random.randint(self.cur_capa-batch_size)
            ind = np.arange(ind,ind+batch_size)
        else:
            ind = np.random.choice(range(self.cur_capa),batch_size,replace=False)
        batch = []
        for item in self.items:
            if self.conv and 'state' in item:
                batch.append([getattr(self,item+'_%s'%it)[ind] for it in 'xe'])
            else:
                batch.append(getattr(self,item)[ind])
        return batch
        
    def update(self, trajs):
        assert len(trajs) == len(self.items), "s a r s' d"
        for item,traj in zip(self.items,trajs):
            if self.conv and 'state' in item:
                for it,tra in zip('xe',traj):
                    setattr(self,item+'_%s'%it,
                            np.concatenate([getattr(self,item+'_%s'%it,tra[:0,...]),tra],axis=0)[-self.limit:])
            else:
                setattr(self,item,np.concatenate([getattr(self,item,traj[:0,...]),traj],axis=0)[-self.limit:])
        self.cur_capa = min(self.limit,self.cur_capa + len(trajs[1]))

    def clear(self):
        for item in self.items:
            if self.conv and 'state' in item:
                for it in 'xe':
                    if hasattr(self,item+'_%s'%it):
                        setattr(self,item+'_%s'%it,getattr(self,item+'_%s'%it)[:0,...])
            else:
                if hasattr(self,item):
                    setattr(self,item,getattr(self,item)[:0,...])
        self.cur_capa = 0

    # def save(self,cwd=None):
    #     cwd = self.cwd if cwd is None else cwd
    #     for item in self.items:
    #         np.save(os.path.join(cwd,'experience_%s.npy'%item),np.array(getattr(self,item)))
    #         print('Save experience %s'%item)

    # def load(self,cwd=None):
    #     cwd = self.cwd if cwd is None else cwd
    #     for item in self.items:
    #         data = np.load(os.path.join(cwd,'experience_%s.npy'%item)).tolist()
    #         setattr(self,item,deque(data,maxlen=self.limit))
    #         print('Load experience %s'%item)
    #     self.cur_capa = len(self.reward)

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
