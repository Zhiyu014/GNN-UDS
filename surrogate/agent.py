import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.nn import GraphConv, SAGEConv, GATConv, global_mean_pool,SAGPooling
from torch.distributions import Normal, Categorical, RelaxedOneHotCategorical
from typing import List, Tuple, Optional, Union
from emulator import get_batch_index
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.dim = getattr(args, "dim", 128)
        self.nly = getattr(args, "nly", 3)
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.n_in += 1 if getattr(args,'if_flood',False) else 0
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,3))
        self.n_in += 2 if getattr(args, 'tide', False) else 1

        self.encs = nn.ModuleDict({
            'x': nn.Sequential(nn.Linear(self.n_in, self.dim), nn.ReLU()),
            'e': nn.Sequential(nn.Linear(self.e_in, self.dim), nn.ReLU())
        })
        # self.conv = GATConv(self.dim, self.dim, add_self_loops=False)
        self.conv = GraphConv(self.dim, self.dim)
        # self.conv = SAGPooling(self.dim,)
        self.out = nn.Linear(self.dim, self.dim//2)

        node_edge_index = th.LongTensor(getattr(args,"node_edge_index",None)).T
        indice = get_batch_index(node_edge_index, getattr(args,"batch_size",128))
        batch = th.concat([th.zeros(self.n_node,dtype=th.long),th.ones(self.n_edge,dtype=th.long)])
        self.register_buffer('node_edge_index', node_edge_index)
        self.register_buffer('indice', indice)
        self.register_buffer('batch', batch)

    def forward(self, inputs, batch = True):
        x,e = inputs
        x,e = self.encs['x'](x),self.encs['e'](e)
        h = th.concat([x,e],dim=-2)
        # h = F.relu(self.conv(h.view(-1,self.dim), self.indice if batch else self.node_edge_index))
        # h = h.view(-1,self.n_node+self.n_edge,self.dim)
        # h = F.relu(self.conv(h, self.node_edge_index))
        return global_mean_pool(self.out(h), self.batch, 2).view(-1,self.dim)

class Actor(nn.Module):
    def __init__(self, 
                 action_shape: Union[int, List[int]], 
                 observ_size: int, 
                 args):
        super(Actor, self).__init__()
        self.conti = args.act.startswith('conti')
        self.mac = getattr(args, "mac", False)
        self.action_shape = action_shape
        self.nly = getattr(args, "nly", 3)
        self.dim = getattr(args, "dim", 128)
        
        self.conv = getattr(args, "conv", False)
        if self.conv:
            self.convnet = ConvNet(args)
            self.observ_size = self.convnet.dim
        else:
            self.observ_size = observ_size
        self.agent_dir = getattr(args, "agent_dir")

    def save(self,agent_dir=None,name=None):
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        th.save(self.state_dict(),
                os.path.join(agent_dir,f'actor{name if name is not None else ''}.pt'))

    def load(self,agent_dir=None,name=None):
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        path = os.path.join(agent_dir,f'actor{name if name is not None else ''}.pt')
        self.load_state_dict(th.load(path, weights_only=True))
 
class Qnet(nn.Module):
    def __init__(self, 
                 observ_size: int, 
                 args,
                 target: bool = False):
        super(Qnet, self).__init__()
        self.conti = args.act.startswith('conti')
        self.dueling = getattr(args, "dueling", False)
        self.mac = getattr(args,"mac",False)

        self.nly = getattr(args, "nly", 3)
        self.dim = getattr(args, "dim", 128)
        
        self.conv = getattr(args, "conv", False)
        if self.conv:
            self.convnet = ConvNet(args)
            self.observ_size = self.convnet.dim
        else:
            self.observ_size = observ_size
        self.agent_dir = getattr(args, "agent_dir")
        self.target = target

    def save(self,agent_dir=None,name=None):
        agent_dir = agent_dir if agent_dir is not None else self.agent_dir
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        th.save(self.state_dict(),
                os.path.join(agent_dir,f'qnet{name if name is not None else ''}{'_target' if self.target else ''}.pt'))
            
    def load(self,agent_dir=None,name=None):
        path = os.path.join(agent_dir,f'qnet{name if name is not None else ''}{'_target' if self.target else ''}.pt')
        self.load_state_dict(th.load(path, weights_only=True))

class Agent:
    def __init__(self, action_shape: int, args, act_only: bool = False):
        self.action_shape = action_shape
        self.dec = getattr(args, "dec", False)
        self.act = getattr(args,"act","")
        self.conti = self.act.startswith('conti')
        self.mac = getattr(args,"mac",False)
        self.device = getattr(args,"device",device)
        self.action_space = [th.tensor(space,dtype=th.float32).to(self.device)
                             for space in getattr(args,'action_space',{}).values()]
        if not self.conti:
            self.action_table = th.tensor(list(getattr(args,'action_table',{}).values()),dtype=th.float32).to(self.device)
        self.act_only = act_only
        self.agent_dir = getattr(args, "agent_dir")
        if not act_only:
            self.gamma = getattr(args, "gamma", 0.98)
            self.tau = getattr(args,"tau",0.005)

    def convert_action_to_setting(self,action):
        if self.conti:
            return (action+1)/2
        elif not self.conti:
            if self.mac:
                sett = [th.gather(space,-1,ai) for space,ai in zip(self.action_space,action)]
                return th.stack(sett,dim=-1)
            else:
                return th.gather(self.action_table,-1,action)
        else:
            return th.gather(self.action_space,-1,action)
        
    def convert_setting_to_action(self,setting):
        if self.conti:
            return th.multiply(setting,2)-1
        elif isinstance(self.action_shape,(list,np.ndarray)):
            if self.mac:
                if len(self.action_space[0].shape) > 1:
                    spdim = [space.shape[-1] for space in self.action_space]
                    return th.stack([th.argmin([th.abs(setting[...,sum(spdim[:i]):sum(spdim[:i+1])]-sp).sum(dim=-1) for sp in space],dim=0)
                            for i,space in enumerate(self.action_space)],dim=-1)
                else:
                    return th.stack([th.argmin(th.abs(setting[...,i:i+1]-space.repeat(setting.shape[0],1)),dim=-1)
                            for i,space in enumerate(self.action_space)],dim=-1)
            else:
                return th.argmin([th.abs(setting-tab).sum(dim=-1) for tab in self.action_table],dim=0)
        else:
            return th.argmin(th.abs(setting-self.action_space),dim=0)
        
    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
        for item in 'xbyre':
            setattr(self,'norm_%s'%item, th.Tensor(eval('norm_%s'%item)).to(self.device))

    def normalize(self,dat,item,inverse=False):
        normal = getattr(self,'norm_%s'%item[0])
        maxi,mini = normal[0,...],normal[1,...]
        if len(item) > 1:
            maxi,mini = maxi[...,:-1],mini[...,:-1]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)

    def to_tensor(self,dat):
        return th.Tensor(dat).to(self.device)

class ActorSAC(Actor):
    def __init__(self, 
                 action_shape: Union[int, List[int]], 
                 observ_size: int, 
                 args):
        super(ActorSAC, self).__init__(action_shape, observ_size, args)
        # Build network
        self.net = nn.Sequential(*[nn.Linear(self.observ_size, self.dim),nn.ReLU()]+\
                                 [nn.Linear(self.dim, self.dim),nn.ReLU()]*(self.nly-1))
        
        # distributional parameters
        if self.conti:
            self.out = nn.ModuleDict({'mu':nn.Linear(self.dim, action_shape),
                                      'logstd':nn.Linear(self.dim, action_shape)})
        elif isinstance(action_shape, (list,np.ndarray)) and self.mac:
            self.out = nn.ModuleList([nn.Linear(self.dim, act_shape) for act_shape in action_shape])
        else:
            self.out = nn.Linear(self.dim, np.prod(action_shape))
    
    def forward(self, x: Union[th.Tensor, List[th.Tensor]], batch: bool = True) -> Union[th.Tensor, List[th.Tensor]]:
        if self.conv:
            x = self.convnet(x, batch)
        x = self.net(x)
        if self.conti:
            return self.out['mu'](x), self.out['logstd'](x).clamp(-20,2).exp()
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            return [F.softmax(out(x),dim=-1) for out in self.out]
        else:
            return F.softmax(self.out(x),dim=-1)
    
    def get_action(self, 
                   obs: Union[th.Tensor, List[th.Tensor]], 
                   stochastic: bool = False,
                   batch: bool = False,) -> Union[th.Tensor, List[th.Tensor]]:
        params = self(obs, batch)
        if self.conti:
            return Normal(*params).sample().tanh() if stochastic else params[0].tanh()
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            return [Categorical(p).sample() if stochastic else p.argmax(dim=-1) for p in params]
        else:
            return Categorical(params).sample() if stochastic else params.argmax(dim=-1)

    def get_action_probs(self, obs: th.Tensor) -> Tuple[Union[th.Tensor, List[th.Tensor]], Union[th.Tensor, List[th.Tensor]]]:
        params = self(obs)
        if self.conti:
            dist = Normal(params[0],params[1])
            a = dist.rsample()
            logp_action = dist.log_prob(a)
            # logp_action -= tf.math.log(1.000001-tf.pow(a.tanh(),2))   # Adjusted Log Probability due to tanh
            logp_action -= (np.log(2.0) - a - F.softplus(-2. * a)) * 2.   # Adjusted Log Probability due to tanh
            return a.tanh(), logp_action.sum(dim=-1)
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            return params,[th.log(p+1e-6) for p in params]
        else:
            return params,th.log(params+1e-6)
   
class CriSAC(Qnet):
    def __init__(self, action_shape: Union[int, List[int]], 
                 observ_size: int, 
                 args,
                 target: bool = False):
        super(CriSAC, self).__init__(observ_size,args,target)
        # Build network
        adim = action_shape if self.conti else 0
        self.net = nn.Sequential(*[nn.Linear(self.observ_size + adim, self.dim),nn.ReLU()]+\
                                 [nn.Linear(self.dim, self.dim),nn.ReLU()]*(self.nly-1))
        if self.conti:
            self.out = nn.Linear(self.dim, 1)
        elif self.mac and not self.conti:
            self.out = nn.ModuleList([nn.Linear(self.dim,shp+1 if self.dueling else shp)
                         for shp in action_shape])
        else:
            self.out = nn.Linear(self.dim, np.prod(action_shape)+1 if self.dueling else np.prod(action_shape))
    
    def forward(self, x: th.Tensor, a: Optional[th.Tensor] = None) -> th.Tensor:
        if self.conv:
            x = self.convnet(x)
        params = self.net(th.concat([x,a],dim=-1) if self.conti else x)
        if self.conti:
            return self.out(params).squeeze(-1)
        elif self.mac and not self.conti:
            out = [out(params) for out in self.out]
            if self.dueling:
                out = [o[...,:1] + o[...,1:] - o[...,1:].mean(dim=-1,keepdim=True) for o in out]
            return th.stack([(o*ai).sum(dim=-1) for o,ai in zip(out,a)],dim=-1) if a is not None else out
        else:
            out = self.out(params)
            if self.dueling:
                out = out[...,:1] + out[...,1:] - out[...,1:].mean(dim=-1,keepdim=True)
            return (out * a).sum(dim=-1) if a is not None else out

class AgentSAC(Agent):
    def __init__(self, action_shape: int, observ_size: int, args, act_only: bool = False):
        super(AgentSAC, self).__init__(action_shape, args, act_only)
        self.on_policy = False
        self.actor = ActorSAC(action_shape, observ_size, args).to(self.device)

        if not act_only:
            self.q1 = CriSAC(action_shape, observ_size, args).to(self.device)
            self.q2 = CriSAC(action_shape, observ_size, args).to(self.device)
            self.act_optim = th.optim.Adam(self.actor.parameters(),lr=getattr(args,"act_lr",1e-4))
            cri_params = [{'params': self.q1.parameters()}, {'params': self.q2.parameters()}]
            self.cri_optim = th.optim.Adam(cri_params,lr=getattr(args,"cri_lr",1e-3))
            self.q1_target = CriSAC(action_shape, observ_size, args, target=True).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target = CriSAC(action_shape, observ_size, args, target=True).to(self.device)
            self.q2_target.load_state_dict(self.q2.state_dict())

            target_entropy = - action_shape if self.conti else np.log(action_shape) if self.mac else np.log(np.prod(action_shape))
            self.target_entropy = th.tensor(target_entropy*getattr(args,"en_disc",1),requires_grad=False).to(self.device)
            log_alpha = getattr(args,'log_alpha',-1.0)
            self.auto_alpha = log_alpha <= 0
            if self.auto_alpha:
                self.log_alpha = nn.Parameter(th.tensor(log_alpha,dtype=th.float32).to(self.device), requires_grad=True)
                self.alpha_optim = th.optim.Adam([self.log_alpha], lr=getattr(args,"act_lr",1e-4))
            else:
                self.log_alpha = th.tensor(np.log(log_alpha),dtype=th.float32).to(self.device)
        if getattr(args,"load_agent",False):
            self.load()

    def update(self, batch: Tuple, pretrain: bool = False,) -> list:
        return self._update_conti(batch,pretrain) if self.conti else self._update_disc(batch,pretrain)

    def _update_conti(self, batch: Tuple, pretrain: bool = False) -> list:
        states, actions, rewards, next_states, *_ = batch
        
        # Compute Q-values
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        
        # Compute target values
        with th.no_grad():
            next_actions, next_log_probs = self.actor.get_action_probs(next_states)
            next_q = th.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )
            target_q = rewards + self.gamma * (next_q - self.alpha * next_log_probs)
        
        # Critic loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        # Conservative Q learning (CQL): need to calculate logsumexp of concated q values of random and current qs
        # if pretrain:
        #     rand_actions = th.rand_like(actions)
        #     q1_loss += th.logsumexp(self.q1(states, th.rand_like(actions)), dim=1).mean() - q1.mean()
        #     q2_loss += th.logsumexp(self.q2(states, th.rand_like(actions)), dim=1).mean() - q2.mean()
        
        # Actor loss
        actions_pred, log_probs = self.actor.get_action_probs(states)
        if pretrain:
            dist = Normal(self.actor(states))
            likelihood = dist.log_prob(th.atanh(actions)).sum(dim=-1)
            actor_loss = (self.alpha * log_probs - likelihood).mean()
        else:
            q_pred = th.min(self.q1(states, actions_pred), self.q2(states, actions_pred))
            actor_loss = (self.alpha * log_probs - q_pred).mean()
        
        # Alpha loss
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Optimization
        self.act_optim.zero_grad()
        actor_loss.backward()
        self.act_optim.step()

        self.cri_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        self.cri_optim.step()
        
        if self.auto_alpha:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
                
        # Update target networks
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)
        
        return [
            (q1_loss + q2_loss).item(),
            actor_loss.item(),
            -log_probs.mean().item(),
            self.alpha.item(),
            ]

    def _update_disc(self, batch: Tuple, pretrain: bool = False) -> list:
        states, actions, rewards, next_states, *_ = batch
        actions = actions.long()
        
        # Compute Q-values
        q1,q2 = self.q1(states),self.q2(states)
        if self.mac:
            qa1 = th.stack([th.gather(qi,-1,a.unsqueeze(-1)).squeeze(-1) for qi,a in zip(q1,actions.T)],dim=-1).mean(dim=-1)
            qa2 = th.stack([th.gather(qi,-1,a.unsqueeze(-1)).squeeze(-1) for qi,a in zip(q2,actions.T)],dim=-1).mean(dim=-1)
        else:
            qa1,qa2 = th.gather(q1,-1,actions.unsqueeze(-1)),th.gather(q2,-1,actions.unsqueeze(-1))

        # Compute target values
        with th.no_grad():
            next_actions, next_log_probs = self.actor.get_action_probs(next_states)
            if self.mac:
                next_q = [th.min(q1_,q2_) for q1_,q2_ in zip(self.q1_target(next_states),self.q2_target(next_states))]
                next_v = th.stack([((qi - self.alpha * lpi) * ai).sum(-1)
                          for qi, ai, lpi in zip(next_q, next_actions, next_log_probs)],dim=-1).mean(dim=-1)
            else:
                next_q = th.min(self.q1_target(next_states),self.q2_target(next_states))
                next_v = ((next_q - self.alpha * next_log_probs) * next_actions).sum(-1)
            target_q = rewards + self.gamma * next_v
        
        # Critic loss
        q1_loss = F.mse_loss(qa1, target_q)
        q2_loss = F.mse_loss(qa2, target_q)
        # Conservative Q learning (CQL): need to calculate logsumexp of concated q values of random and current qs
        # if pretrain:
        #     q1_loss += th.stack([th.logsumexp(qi, dim=1).mean() - qi.mean() for qi in q1]).mean()
        #     q2_loss += th.stack([th.logsumexp(qi, dim=1).mean() - qi.mean() for qi in q2]).mean()

        # Actor loss
        actions_pred, log_probs = self.actor.get_action_probs(states)
        if pretrain:
            if self.mac:
                actor_loss = th.stack([(self.alpha * lpi * p).sum(dim=-1) + F.cross_entropy(p,a,reduction='none')
                                    for p,lpi,a in zip(actions_pred,log_probs,actions.T)],dim=-1).mean()
            else:
                actor_loss = ((self.alpha * log_probs * actions_pred).sum(dim=-1) +\
                              F.cross_entropy(actions_pred, actions, reduction='none')).mean()
        else:
            if self.mac:
                q_pred = [th.min(q1_, q2_) for q1_, q2_ in zip(q1, q2)]
                actor_loss = th.stack([((self.alpha * lpi - qi.detach()) * ai).sum(dim=-1)
                                    for qi, ai, lpi in zip(q_pred, actions_pred, log_probs)],dim=-1).mean()
            else:
                actor_loss = ((self.alpha * log_probs - th.min(q1, q2).detach()) * actions_pred).sum(dim=-1).mean()

        # Alpha loss
        if self.auto_alpha:
            if self.mac:
                entropy = - th.stack([(ai * lpi).sum(dim=-1) for ai, lpi in zip(actions_pred, log_probs)],dim=-1)
            else: 
                entropy = - (log_probs * actions_pred).sum(dim=-1)
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        # Optimization
        self.cri_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        self.cri_optim.step()
        
        if self.auto_alpha:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        self.act_optim.zero_grad()
        actor_loss.backward()
        self.act_optim.step()

        # Update target networks
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)
        
        return [
            (q1_loss + q2_loss).item(),
            actor_loss.item(),
            entropy.mean().item(),
            self.alpha.item(),
        ]

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @th.no_grad()
    def control(self,observ,train=False,batch=False):
        if not batch:
            if isinstance(observ,list):
                observ = [ob[th.newaxis,...] for ob in observ]
            else:
                observ = observ[th.newaxis,...]
        return self.actor.get_action(observ,train,batch)
    
    def save(self,epoch=None):
        agent_dir = self.agent_dir if epoch is None else os.path.join(self.agent_dir, f'{epoch}')
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        self.actor.save(agent_dir)
        for item in 'xbyre':
            norm_path = os.path.join(agent_dir,'norm_%s.npy'%item)
            if hasattr(self,'norm_%s'%item) and not os.path.exists(norm_path):
                np.save(norm_path,getattr(self,'norm_%s'%item).cpu())
        if not self.act_only:
            self.q1.save(agent_dir,'1')
            self.q2.save(agent_dir,'2')
            self.q1_target.save(agent_dir,'1')
            self.q2_target.save(agent_dir,'2')
            th.save({
                'log_alpha': self.log_alpha,
                'act_optim': self.act_optim.state_dict(),
                'cri_optim': self.cri_optim.state_dict(),
                'alpha_optim': self.alpha_optim.state_dict()
                }, os.path.join(agent_dir, 'optim.pth'))
               
    def load(self,epoch=None):
        agent_dir = self.agent_dir if epoch is None else os.path.join(self.agent_dir, f'{epoch}')
        self.actor.load(agent_dir)
        for item in 'xbyre':
            norm_path = os.path.join(agent_dir,'norm_%s.npy'%item)
            if os.path.exists(norm_path):
                setattr(self,'norm_%s'%item,th.Tensor(np.load(norm_path)).to(self.device))
        if not self.act_only:
            self.q1.load(agent_dir,'1')
            self.q2.load(agent_dir,'2')
            # else:
            self.q1_target.load(agent_dir,'1')
            self.q2_target.load(agent_dir,'2')
            checkpoint = th.load(os.path.join(agent_dir, 'optim.pth'), weights_only=True)
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
            self.log_alpha.requires_grad = True
            self.act_optim.load_state_dict(checkpoint['act_optim'])
            self.cri_optim.load_state_dict(checkpoint['cri_optim'])
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
    
class ActorPPO(Actor):
    def __init__(self, 
                 action_shape: Union[int, List[int]], 
                 observ_size: int, 
                 args):
        super(ActorPPO, self).__init__(action_shape, observ_size, args)
        # Build network
        self.net = nn.Sequential(*[nn.Linear(self.observ_size, self.dim),nn.ReLU()]+\
                                 [nn.Linear(self.dim, self.dim),nn.ReLU()]*(self.nly-1))
        if self.conti:
            self.std_log = nn.Parameter(th.zeros((1,action_shape)), requires_grad=True)
        
        # distributional parameters
        if self.conti:
            self.out = nn.Linear(self.dim, action_shape)
        elif isinstance(action_shape, (list,np.ndarray)) and self.mac:
            self.out = nn.ModuleList([nn.Linear(self.dim, act_shape) for act_shape in action_shape])
        else:
            self.out = nn.Linear(self.dim, np.prod(action_shape))

    def forward(self, x: Union[th.Tensor, List[th.Tensor]], batch: bool = True) -> Union[th.Tensor, List[th.Tensor]]:
        if self.conv:
            x = self.convnet(x, batch)
        x = self.net(x)
        if self.conti:
            return self.out(x)
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            return [F.softmax(out(x),dim=-1) for out in self.out]
        else:
            return F.softmax(self.out(x),dim=-1)
        
    def get_action(self, 
                   obs: Union[th.Tensor, List[th.Tensor]], 
                   stochastic: bool = False,
                   batch: bool = False,) -> Union[th.Tensor, List[th.Tensor]]:
        params = self(obs, batch)
        if self.conti:
            return Normal(params, self.std_log.exp()).sample().tanh() if stochastic else params.tanh()
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            return [Categorical(p).sample() if stochastic else p.argmax(dim=-1) for p in params]
        else:
            return Categorical(params).sample() if stochastic else params.argmax(dim=-1)
        
    def get_probs_entropy(self, obs: Union[th.Tensor, List[th.Tensor]],
                          action: Union[th.Tensor, List[th.Tensor]]) -> Union[th.Tensor, List[th.Tensor]]:
        params = self(obs)
        if self.conti:
            dist = Normal(params, self.std_log.exp())
            log_probs = dist.log_prob(action.atanh()).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        elif isinstance(self.action_shape, (list,np.ndarray)) and self.mac:
            log_probs = [th.log(p+1e-6) for p in params]
            entropy = th.stack([-(p*lp).sum(dim=-1) for p,lp in zip(params, log_probs)],dim=-1).sum(-1)
            log_probs = th.stack([lp.gather(dim=-1, index=ai.unsqueeze(-1)).squeeze(-1)
                         for ai,lp in zip(action.T,log_probs)],dim=-1).sum(-1)
        else:
            log_probs = th.log(params+1e-6)
            entropy = -(params*log_probs).sum(dim=-1)
            log_probs = log_probs.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        return log_probs, entropy

class CriPPO(Qnet):
    def __init__(self,
                 observ_size: int, 
                 args,
                 target: bool = False):
        super(CriPPO, self).__init__(observ_size,args,target)
        # Build network
        self.net = nn.Sequential(*[nn.Linear(self.observ_size, self.dim),nn.ReLU()]+\
                                 [nn.Linear(self.dim, self.dim),nn.ReLU()]*(self.nly-1)+\
                                    [nn.Linear(self.dim, 1)])

    def forward(self, x: Union[th.Tensor,List[th.Tensor]]) -> th.Tensor:
        return self.net(x).squeeze(-1) if not self.conv else self.net(self.convnet(x)).squeeze(-1)

class AgentPPO(Agent):
    def __init__(self, action_shape: int, observ_size: int, args, act_only: bool = False):
        super(AgentPPO, self).__init__(action_shape, args, act_only)
        self.on_policy = True
        self.actor = ActorPPO(action_shape, observ_size, args).to(self.device)

        if not act_only:
            self.cri = CriPPO(observ_size, args).to(self.device)
            self.act_optim = th.optim.Adam(self.actor.parameters(), lr=getattr(args,"act_lr",1e-4))
            self.cri_optim = th.optim.Adam(self.cri.parameters(), lr=getattr(args,"cri_lr",1e-3))
            self.lambda_gae = getattr(args, "lambda_gae", 0.95)
            self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)
            self.clip_ratio = getattr(args, "clip_ratio", 0.2)
        if getattr(args,"load_agent",False):
            self.load()
    
    def get_advantages(self,r,d,value,next_value):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        # from ElegantRL
        # update rewards when truncated
        advs = th.zeros_like(r)
        bs = r.shape[0]
        last_ = th.where(d==1)[0].cpu().numpy()
        first_ = [0] + (last_ + 1).tolist()
        last_ = last_.tolist() + [bs-1]
        for idx0,idx1 in zip(first_,last_):
            next_, adv = next_value[idx1], 0
            for t in range(idx1, idx0-1, -1):
                next_ = r[t] + self.gamma * ((1-d[t]) * next_ + d[t] * next_value[t])
                advs[t] = adv = next_ - value[t] + self.gamma * self.lambda_gae * (1-d[t]) * adv
                next_ = value[t]
        return advs
    
    def get_adv_traj(self,r,d,value,next_value):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        # from ElegantRL
        # update rewards when truncated
        bs,advs = r.shape[0],th.zeros_like(r)
        next_, adv = next_value[-1], 0
        for t in range(bs-1, -1, -1):
            next_ = r[t] + self.gamma * ((1-d[t]) * next_ + d[t] * next_value[t])
            advs[t] = adv = next_ - value[t] + self.gamma * self.lambda_gae * (1-d[t]) * adv
            next_ = value[t]
        return advs
        
    @th.no_grad()
    def update_trajs(self, batch: Tuple) -> Tuple[th.Tensor, th.Tensor]:
        states, actions, rewards, next_states, dones = batch
        if not self.conti:
            actions = actions.long()
        values, next_values = self.cri(states), self.cri(next_states)
        advs = self.get_adv_traj(rewards, dones, values, next_values)
        returns = advs + values
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        log_probs, _ = self.actor.get_probs_entropy(states, actions)
        return states, actions, advs, returns, log_probs

    def update(self, batch: Tuple, pretrain: bool = False) -> list:
        states, actions, advs, returns, log_probs = batch
        if not self.conti:
            actions = actions.long()
        
        # TODO: calculate advs norm, log_probs for the whole batch
        # Compute advantages
        # with th.no_grad():
        #     values, next_values = self.cri(states), self.cri(next_states)
        #     advs = self.get_advantages(rewards, dones, values, next_values)
        #     returns = advs + values
        #     # Normalize advantages
        #     advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        #     log_probs, _ = self.actor.get_probs_entropy(states, actions)
        
        # Update critic
        value_loss = F.mse_loss(self.cri(states), returns)
        self.cri_optim.zero_grad()
        value_loss.backward()
        self.cri_optim.step()
        
        # Update actor
        new_log_probs, entropy = self.actor.get_probs_entropy(states, actions)
        if pretrain:
            actor_loss = - new_log_probs.mean()
        else:
            ratio = th.exp(new_log_probs - log_probs)
            min_adv = th.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advs
            actor_loss = - th.min(ratio * advs, min_adv).mean()
        actor_loss -= self.lambda_entropy * entropy.mean()
        self.act_optim.zero_grad()
        actor_loss.backward()
        self.act_optim.step()
        
        return [
            value_loss.item(),
            actor_loss.item(),
            entropy.mean().item(),
                 ]

    @th.no_grad()
    def control(self,observ,train=False,batch=False):
        if not batch:
            if isinstance(observ,list):
                observ = [ob[th.newaxis,...] for ob in observ]
            else:
                observ = observ[th.newaxis,...]
        return self.actor.get_action(observ,train,batch)
    
    def save(self,epoch=None):
        agent_dir = self.agent_dir if epoch is None else os.path.join(self.agent_dir, f'{epoch}')
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
        self.actor.save(agent_dir)
        for item in 'xbyre':
            norm_path = os.path.join(agent_dir,'norm_%s.npy'%item)
            if hasattr(self,'norm_%s'%item) and not os.path.exists(norm_path):
                np.save(norm_path,getattr(self,'norm_%s'%item).cpu())
        if not self.act_only:
            self.cri.save(agent_dir)
            th.save({
                'act_optim': self.act_optim.state_dict(),
                'cri_optim': self.cri_optim.state_dict(),
                }, os.path.join(agent_dir, 'optim.pth'))
               
    def load(self,epoch=None):
        agent_dir = self.agent_dir if epoch is None else os.path.join(self.agent_dir, f'{epoch}')
        self.actor.load(agent_dir)
        for item in 'xbyre':
            norm_path = os.path.join(agent_dir,'norm_%s.npy'%item)
            if os.path.exists(norm_path):
                setattr(self,'norm_%s'%item,th.Tensor(np.load(norm_path)).to(self.device))
        if not self.act_only:
            self.cri.load(agent_dir)
            checkpoint = th.load(os.path.join(agent_dir, 'optim.pth'), weights_only=True)
            self.act_optim.load_state_dict(checkpoint['act_optim'])
            self.cri_optim.load_state_dict(checkpoint['cri_optim'])
    
# Utility functions
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def get_agent(name: str):
    agents = {
        'SAC': AgentSAC,
        'PPO': AgentPPO,
        # 'TD3': AgentTD3,
        # 'QMIX': AgentQMIX
    }
    return agents.get(name, AgentSAC)