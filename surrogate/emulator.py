import torch as th
from torch import nn
from torch_geometric.nn import SAGEConv,GraphConv,GATConv,HeteroConv
import torch_geometric.transforms as T
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import numpy as np,os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class BatchGATConv(GATConv):
    def __init__(self,*args,**kwargs):
        super(BatchGATConv,self).__init__(*args,**kwargs)
    
    def forward(self,x,edge_index,edge_attr=None,*args,**kwargs):
        batch,nodes = x.shape[:-2],x.shape[-2]
        edge_index = repeat(edge_index, 'p e -> p e b', b=np.prod(batch))
        edge_index = edge_index + th.Tensor(th.arange(np.prod(batch))*nodes).to(device)
        edge_index = rearrange(edge_index,'p e b -> p (e b)')
        if edge_attr is not None:
            edge_attr = repeat(edge_attr,'e d -> (tile e) d',tile=np.prod(batch))
        out = super().forward(rearrange(x, 'b t n d -> (b t n) d'),edge_index,edge_attr,*args,**kwargs)
        return rearrange(out,'(b t n) d -> b t n d',b=batch[0],t=batch[1])
    

class NodeEdge(nn.Module):
    def __init__(self, dim, inci, **kwargs):
        super(NodeEdge,self).__init__(**kwargs)
        self.xes = nn.Linear(dim,dim//2)
        self.inci = nn.Parameter(inci,requires_grad=False)
        self.w = nn.Parameter(th.randn(inci.shape),requires_grad=True)
        self.b = nn.Parameter(th.zeros_like(inci),requires_grad=True)

    def forward(self,inputs):
        xe = th.relu(self.xes(inputs))
        return th.matmul(self.w * self.inci + self.b, xe)

class STBlock(nn.Module):
    def __init__(self,nly,dim,kernel,node_edge):
        super(STBlock,self).__init__()
        self.sps_x = nn.ModuleList([BatchGATConv(dim+dim//2,dim) for _ in range(nly)])
        self.sps_e = nn.ModuleList([BatchGATConv(dim+dim//2,dim) for _ in range(nly)])
        self.nes = nn.ModuleList([NodeEdge(dim,th.abs(node_edge)) for _ in range(nly)])
        self.ens = nn.ModuleList([NodeEdge(dim,th.abs(node_edge).T) for _ in range(nly)])
        self.tps = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        self.padding = [(kernel-1)*(2**i) for i in range(nly)]

    def forward(self,x,e,edge_ind,node_ind):
        for spx,ne,spe,en in zip(self.sps_x,self.nes,self.sps_e,self.ens):
            x = th.relu(spx(th.concat([x,ne(e)],dim=-1),edge_ind))
            e = th.relu(spe(th.concat([e,en(x)],dim=-1),node_ind))
        x,e = rearrange(x,'b t n d -> (b n) d t'),rearrange(e,'b t n d -> (b n) d t')
        h = th.concat([x,e],dim=0)
        for tp,pad in zip(self.tps,self.padding):
            h = F.pad(h,(pad,0))
            h = th.relu(tp(h))
        x,e = h[:x.shape[0]],h[x.shape[0]:]
        x = rearrange(x,'(b n) d t -> b t n d',n=edge_ind.max()+1)
        e = rearrange(e,'(b n) d t -> b t n d',n=edge_ind.shape[1])
        return x,e
    
#TODO: Heterogeneous Graphs for node/edge based graph
class STBlock2(nn.Module):
    def __init__(self,nly,dim,kernel,*args,**kwargs):
        super(STBlock2,self).__init__()
        self.sps = nn.ModuleList([BatchGATConv(dim,dim) for _ in range(nly)])
        self.tps = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        self.padding = [(kernel-1)*(2**i) for i in range(nly)]

    def forward(self,x,e,edge_ind):
        h = th.concat([x,e],dim=-2)
        for sp in self.sps:
            h = th.relu(sp(h,edge_ind))
        h = rearrange(h,'b t n d -> (b n) d t')
        for tp,pad in zip(self.tps,self.padding):
            h = F.pad(h,(pad,0))
            h = th.relu(tp(h))
        h = rearrange(h,'(b n) d t -> b t n d', n=x.shape[-2]+e.shape[-2])
        x,e = th.split(h, [x.shape[-2],e.shape[-2]], dim=-2)
        return x,e
    

class STM(nn.Module):
    def __init__(self,in_dims,dim,nly,node_edge,kernel,graph_base=False,if_flood=False):
        super(STM,self).__init__()
        self.encs = nn.ModuleList([nn.Linear(in_dim,dim if i < 2 else dim//2)
                      for i,in_dim in enumerate(in_dims)])
        stblock = STBlock2 if graph_base else STBlock
        self.st1 = stblock(nly, dim, kernel, node_edge)
        self.concs = nn.ModuleList([nn.Linear(dim+dim//2, dim)] * 2)
        self.st2 = stblock(nly, dim, kernel, node_edge)
        self.res = nn.ModuleList([nn.Linear(dim, dim)] * 2)
        self.outs = nn.ModuleList([nn.Linear(dim, 1)] * 2)
        self.if_flood = if_flood
        if self.if_flood:
            self.flood = nn.Sequential(*[nn.Linear(dim,dim),nn.ReLU()]*nly+\
                                       [nn.Linear(dim,1),nn.Sigmoid()])

    def forward(self,inputs,indices):
        x,e = self.pred(inputs,indices)
        out,e = F.hardsigmoid(self.outs[0](x)),th.tanh(self.outs[1](e))
        # flood classification
        if self.if_flood:
            out = th.concat([out,self.flood(x)],dim=-1)
        return out,e

    def pred(self,inputs,indices):
        # Encoding
        x,e,b,a = [enc(inp) for enc,inp in zip(self.encs, inputs)]
        res = [repeat(x[:,-1:,...],'b t n d -> b (tile t) n d',tile=b.shape[1]),
               repeat(e[:,-1:,...],'b t n d -> b (tile t) n d',tile=b.shape[1])]
        x,e,b,a = [th.relu(inp) for inp in [x,e,b,a]]
        # Spatio-temporal block
        x,e = self.st1(x,e,*indices)
        # link st1 and st2
        x = th.relu(self.concs[0](th.concat([x,b],dim=-1)))
        e = th.relu(self.concs[1](th.concat([e,a],dim=-1)))
        x,e = self.st2(x,e,*indices)
        # resnet
        x = th.relu(th.cumsum(self.res[0](x),axis=1) + res[0])
        e = th.relu(th.cumsum(self.res[1](e),axis=1) + res[1])
        return x,e
        
    def output(self,x,e):
        return F.hardsigmoid(self.outs[0](x)),th.tanh(self.outs[1](e))
        
class Emulator:
    def __init__(self,args=None):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.tide = getattr(args,'tide',False)
        # Runoff (tide) is boundary
        self.b_in = 2 if self.tide else 1
        self.seq = getattr(args,'seq',5)

        self.dim = getattr(args,'dim',64)
        self.kernel = getattr(args,"kernel",3)
        self.nly = getattr(args,"nly",3)
        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 1
        self.epsilon = getattr(args,"epsilon",-1.0)
        in_dims = [self.n_in,self.e_in,self.b_in,1]

        self.graph_base = getattr(args,"graph_base",False)
        self.edges = th.LongTensor(getattr(args,"edges").T).to(device)
        self.node_index = th.LongTensor(getattr(args,"node_index").T).to(device)
        self.node_edge_index = th.LongTensor(getattr(args,"node_edge_index").T).to(device)
        self.indices = [self.node_edge_index] if self.graph_base else [self.edges,self.node_index]
        self.node_edge = th.Tensor(getattr(args,"node_edge")).to(device)

        self.model = STM(in_dims,self.dim,self.nly,self.node_edge,self.kernel,self.graph_base,self.if_flood).to(device)
        self.optimizer = th.optim.Adam(self.model.parameters(),lr=getattr(args,"learning_rate",1e-3))
        self.mixed_precision = getattr(args,"mixed_precision",False)
        self.scaler = th.amp.GradScaler()
        self.mse = F.mse_loss
        if self.if_flood:
            self.bce = F.binary_cross_entropy
            self.poswei = th.Tensor(getattr(args,"poswei",[1.0 for _ in range(self.n_node)])).to(device)
        # GradNorm for multi-task learning
        self.gradnorm = getattr(args,"gradnorm",False)
        if self.gradnorm:
            self.alpha_reg = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
            self.alpha_cls = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
            self.alpha_optimizer = th.optim.Adam([self.alpha_reg,self.alpha_cls],lr=1e-4)
            self.alpha_temperature = 0.5
            self.gradnorm_lambda = 0.1
        self.roll = getattr(args,"roll",0)
        self.model_dir = getattr(args,"model_dir")

        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.act_edges = act_edges = getattr(args,"act_edges",[])
            sett = np.zeros(self.n_edge)
            sett[act_edges] = range(1,len(act_edges)+1)
            self.sett = th.LongTensor(sett).to(device)

        self.is_outfall = th.Tensor(getattr(args,"is_outfall",[0 for _ in range(self.n_node)])).to(device)
        self.pump = th.Tensor(getattr(args,"pump",[0.0 for _ in range(self.n_edge)])).to(device)
        self.hmax = th.Tensor(getattr(args,"hmax",[1.5 for _ in range(self.n_node)])).to(device)
        self.hmin = th.Tensor(getattr(args,"hmin",[0.0 for _ in range(self.n_node)])).to(device)
        self.area = th.Tensor(getattr(args,"area",[0.0 for _ in range(self.n_node)])).to(device)
        self.nwei = th.Tensor(getattr(args,"nwei",[1.0 for _ in range(self.n_node)])).to(device)
        self.ewei = th.Tensor(getattr(args,"ewei",[1.0 for _ in range(self.n_edge)])).to(device)
        self.pump_in = th.Tensor(getattr(args,"pump_in",[0.0 for _ in range(self.n_node)])).to(device)
        self.pump_out = th.Tensor(getattr(args,"pump_out",[0.0 for _ in range(self.n_node)])).to(device)      
        self.offset = th.Tensor(getattr(args,"offset",[0.0 for _ in range(self.n_edge)])).to(device)
    
    def get_edge_action(self,a,g=False):
        if g:
            a = th.concat([th.ones_like(a[...,:1]),a],dim=-1)
            return th.gather(a, -1, repeat(self.sett,'n -> b t n',b=a.shape[0],t=a.shape[1])).unsqueeze(dim=-1)
        else:
            def set_edge_action(s):
                out = np.ones(self.n_edge)
                out[self.act_edges] = s
                return out
            return np.expand_dims(np.apply_along_axis(set_edge_action,-1,a),-1)

    def fit_eval(self,x,a,b,y,ex,ey,fit=True):
        ae = self.get_edge_action(a,g=True)
        with th.autocast(device_type='cuda',dtype=th.float16 if self.mixed_precision else th.float32):
            if self.roll:       # Curriculum learning (long-term)
                predss,edge_predss = [],[]
                for i in range(self.roll):
                    bi,aei = b[:,i*self.seq:(i+1)*self.seq,:],ae[:,i*self.seq:(i+1)*self.seq,...]
                    inp = [x[:,-self.seq:,...],ex[:,-self.seq:,...]]
                    inp += [bi,aei]
                    preds = self.model(inp,self.indices)
                    preds,edge_preds = self.post_proc(*preds,bi,aei)
                    predss.append(preds)
                    edge_predss.append(edge_preds)
                    if self.if_flood:
                        x_new = th.concat([preds[...,:-1],(preds[...,-1:]>0.5).to(th.float32),bi],dim=-1)
                    else:
                        x_new = th.concat([preds,bi],dim=-1)
                    ex_new = th.concat([edge_preds,aei],dim=-1)
                    x,ex = x_new,ex_new
                preds = th.concat(predss,dim=1)
                edge_preds = th.concat(edge_predss,dim=1)
            else:
                inp = [x,ex,b,ae]
                preds,edge_preds = self.model(inp,self.indices)
                preds,edge_preds = self.post_proc(preds,edge_preds,b,ae)

            # Loss funtion
            # narrow down norm range of water head
            preds = preds.clip(0,1) # avoid large loss value
            if self.hmin.max() > 0:
                wei = (self.hmax-self.hmin)*(1-self.is_outfall) + (self.hmax-self.hmin).mean()*self.is_outfall
                wei = (self.norm_y[0,:,0].max()-self.norm_y[1,:,0].min())/wei
                preds = th.concat([preds[...,:1] * wei,preds[...,1:]],dim=-1)
                y = th.concat([y[...,:1] * wei,y[...,1:]],dim=-1)
            node_loss = self.mse(th.einsum('btnd,n->btnd',[y[...,:3],self.nwei]),
                                th.einsum('btnd,n->btnd',[preds[...,:3],self.nwei]))
            if self.if_flood:
                pos_mask = (y[...,-2:-1]>0.5).to(th.float32)
                weight = repeat(self.nwei*self.poswei,'n -> n d', d = 1) * pos_mask +\
                    repeat(self.nwei,'n -> n d', d = 1) * (1-pos_mask)
                flood_loss = self.bce(preds[...,-1:], y[...,-2:-1], weight = weight)
            edge_loss = self.mse(th.einsum('btnd,n->btnd',[edge_preds,self.ewei]), th.einsum('btnd,n->btnd',[ey,self.ewei]))
        if fit:
            # th.autograd.set_detect_anomaly(True)
            reg_loss = node_loss + edge_loss
            # --- GradNorm ---
            if self.if_flood and self.gradnorm:
                self._fit_grad_norm(reg_loss,flood_loss)
            loss = (self.alpha_reg if self.gradnorm else 1) * reg_loss
            if self.if_flood:
                loss += (self.alpha_cls if self.gradnorm else 1) * flood_loss
            self.optimizer.zero_grad()
            self.scaler(loss).backward() if self.mixed_precision else loss.backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler(self.optimizer).step() if self.mixed_precision else self.optimizer.step()
        else:
            loss = [node_loss,flood_loss,edge_loss] if self.if_flood else [node_loss,edge_loss]
        return loss

    def _fit_grad_norm(self,reg_loss,flood_loss):
        self.optimizer.zero_grad()
        W = self.model.res[0].weight
        grad_reg = th.autograd.grad(reg_loss,W,retain_graph=True,create_graph=True)
        # reg_loss.backward(retain_graph=True,create_graph=True)
        grad_reg_norm = th.norm(grad_reg[0].clone().detach())

        self.optimizer.zero_grad()
        grad_cls = th.autograd.grad(flood_loss,W,retain_graph=True,create_graph=True)
        # flood_loss.backward(retain_graph=True,create_graph=True)
        grad_cls_norm = th.norm(grad_cls[0].clone().detach())

        grad_ratio = grad_cls_norm / (grad_reg_norm + 1e-6)
        target_reg = (grad_ratio ** self.alpha_temperature).detach()
        target_cls = (1.0 / (grad_ratio ** self.alpha_temperature + 1e-8)).detach()

        gradnorm_loss = (
            th.abs(grad_reg_norm - target_reg) +
            th.abs(grad_cls_norm - target_cls)
        )
        alpha_loss = gradnorm_loss + self.gradnorm_lambda * (self.alpha_reg**2 + self.alpha_cls**2)
        
        # update alpha
        self.alpha_optimizer.zero_grad()
        self.scaler(alpha_loss).backward() if self.mixed_precision else alpha_loss.backward()
        self.scaler(self.alpha_optimizer).step() if self.mixed_precision else self.alpha_optimizer.step()
        
        # alpha normalized with the sum as 2
        alpha_sum = self.alpha_reg + self.alpha_cls
        self.alpha_reg.data = 2.0 * self.alpha_reg / alpha_sum
        self.alpha_cls.data = 2.0 * self.alpha_cls / alpha_sum

    def simulate(self,states,runoff,a,edge_states):
        # runoff shape: T_out, T_in, N
        if self.act:
            ae = self.get_edge_action(a)
        preds,edge_preds = [],[]
        for idx,bi in enumerate(runoff):
            x = states[idx,-self.seq:,...]
            ex = edge_states[idx,-self.seq:,...]
                
            bi = bi[:self.seq]

            inp = [self.normalize(x,'x'),self.normalize(bi,'b')]
            inp = [np.expand_dims(dat,0) for dat in inp]
            inp += [np.expand_dims(self.normalize(ex,'e'),0)]
            inp += [ae[idx:idx+1]] if self.act else []
            y = self.model(inp,self.indices)

            y,ey = self.post_proc(y,ey,self.normalize(bi,'b'),ae[idx:idx+1])
            ey = self.normalize(np.squeeze(ey,0),'e',True)
            y = self.normalize(np.squeeze(y,0),'y',True)

            # Pumped storage depth calculation: boundary condition differs from orifice
            if sum(getattr(self,'pump_in',[0])) + sum(getattr(self,'pump_out',[0])) + sum(getattr(self,'pump',[0])) > 0:
                ps = (self.area * np.matmul(np.clip(self.node_edge,0,1),np.expand_dims(self.pump,axis=-1)).squeeze())>0
                h,qin,qout = [y[...,i] for i in [0,1,2]]
                de = []
                for t in range(self.seq):
                    de += [np.clip(x[-1,:,0] if t==0 else de[-1] +\
                                    (qin-qout)[t,:]/(self.area+1e-6),self.hmin,self.hmax)]
                y[...,0] = h*(1-ps) + np.stack(de,axis=0) * ps

            q_w,y = self.constrain(y,bi[...,:1],x[-1:,:,0])
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            edge_preds.append(ey)
        return np.array(preds),np.array(edge_preds)

    def predict(self,states,b,a,edge_state):
        x = states[:,-self.seq:,...]
        ex = edge_state[:,-self.seq:,...]
        assert b.shape[1] == self.seq
        if self.act:
            ae = self.get_edge_action(a)
        inp = [self.normalize(x,'x'),self.normalize(ex,'e'),self.normalize(b,'b'),ae]
        y,ey = self.model(inp,self.indices)

        y,ey = self.post_proc(y,ey,self.normalize(b,'b'),ae)
        y = self.normalize(y,'y',True)
        ey = self.normalize(ey,'e',True)

        # Pumped storage depth calculation: boundary condition differs from orifice
        if sum(getattr(self,'pump_in',[0])) + sum(getattr(self,'pump_out',[0])) + sum(getattr(self,'pump',[0])) > 0:
            ps = (self.area * th.einsum('ne,e->n',th.clip(self.node_edge,0,1),self.pump))>0
            h,qin,qout = [y[...,i] for i in [0,1,2]]
            de = th.zeros_like(h)
            for t in range(de.shape[1]):
                de[:,t,:] = th.clip(x[:,-1,:,0] if t==0 else de[:,t-1,:] +\
                                    (qin-qout)[:,t,:]/(self.area+1e-6),self.hmin,self.hmax)
            y[...,0] = h*(1-ps) + de * ps

        q_w,y = self.constrain(y,b[...,:1])
        y = th.concat([y,repeat(q_w,'b t n -> b t n d',d=1)],dim=-1)
        return y,ey

    def post_proc(self,y,ey,b,ae):
        # tide boundary
        if self.tide:
            h = y[...,0] * (1 - self.is_outfall) + b[...,-1]
            y = th.concat([repeat(h,'b t n -> b t n d',d=1), y[...,1:]],dim=-1)
        inoff = th.einsum("btn,ne->bte",[self.normalize(y,'y',True)[...,0] - self.hmin,
                                         th.clip(self.node_edge,0,1)])
        ey = repeat(ey[...,0] * (inoff > self.offset) * (self.offset) + ey[...,0] * (self.offset==0),'b t n -> b t n d',d=1)
        if self.act:
            # regulate pumping flow (rated value if there is volume in inlet tank)
            fl = self.pump * th.einsum("btn,ne->bte",[(y[...,0]>0.01).type(th.float32),th.clip(self.node_edge,0,1)])
            fl *= (self.norm_e[0,:,0]>1e-3).type(th.float32)/self.norm_e[0,:,0]
            ey = repeat((ey[...,0] * (fl==0).type(th.float32) + fl)*ae[...,0],'b t e -> b t e d',d=1)
        efl = self.normalize(ey,'e',True)[...,:1]
        node_outflow = th.einsum('ne,bted->btnd',[th.clip(self.node_edge,0,1),th.clip(efl,0,th.inf)]) +\
              th.einsum('ne,bted->btnd',[th.abs(th.clip(self.node_edge,-1,0)),-th.clip(efl,-th.inf,0)])
        node_inflow = th.einsum('ne,bted->btnd',[th.abs(th.clip(self.node_edge,-1,0)),th.clip(efl,0,th.inf)]) +\
              th.einsum('ne,bted->btnd',[th.clip(self.node_edge,0,1),-th.clip(efl,-th.inf,0)])
        node_outflow *= (self.norm_y[0,:,2:3]>1e-3).type(th.float32)/self.norm_y[0,:,2:3]
        node_inflow *= (self.norm_y[0,:,1:2]>1e-3).type(th.float32)/self.norm_y[0,:,1:2]
        y = th.concat([y[...,:1],node_inflow,node_outflow,y[...,1:]],dim=-1)
        return y,ey

    def constrain(self,y,r):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        h = th.clip(h,self.hmin,self.hmax)
        q_w = th.clip(q_us + r.squeeze(dim=-1) - q_ds,0,np.inf) * (1-self.is_outfall)
        if self.if_flood:
            f = (y[...,-1] > 0.5).type(th.float32)
            h = self.hmax * f + h * (1-f)
            y = th.stack([h,q_us,q_ds,y[...,-1]],dim=-1)
        else:
            y = th.stack([h,q_us,q_ds],dim=-1)
        if self.epsilon > 0:
            q_w *= ((self.hmax - h) < self.epsilon).type(th.float32)
        elif self.epsilon == 0:
            pass
        elif self.if_flood:
            q_w *= f
        return q_w,y
    
    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
        for item in 'xbyre':
            setattr(self,'norm_%s'%item, th.Tensor(eval('norm_%s'%item)).to(device))

    def normalize(self,dat,item,inverse=False):
        dim = dat.shape[-1]
        normal = getattr(self,'norm_%s'%item)
        maxi,mini = normal[0,...,:dim],normal[1,...,:dim]
        if inverse:
            return dat * (maxi-mini) + mini
        else:
            return (dat - mini)/(maxi-mini)

    def save(self,epoch=None):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        th.save({
            'optimizer_state': self.optimizer.state_dict(),
            'alpha_optimizer_state': self.alpha_optimizer.state_dict() if hasattr(self,'alpha_optimizer') else None,
            'model_state': self.model.state_dict(),
            'epoch': epoch,
            }, os.path.join(self.model_dir,f'{epoch if epoch is not None else 'model'}.pth'))
        for item in 'xbyre':
            norm_path = os.path.join(self.model_dir,'norm_%s.npy'%item)
            if hasattr(self,'norm_%s'%item) and not os.path.exists(norm_path):
                np.save(norm_path,getattr(self,'norm_%s'%item).cpu())

    def load(self,epoch=None,retrain=False):
        path = os.path.join(self.model_dir,f'{epoch if epoch is not None else 'model'}.pth')
        checkpoint = th.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        if retrain:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if hasattr(self,'alpha_optimizer') and checkpoint.get('alpha_optimizer_state') is not None:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state'])
            print(f'Load model at {checkpoint['epoch']}')
        for item in 'xbyre':
            norm_path = os.path.join(self.model_dir,'norm_%s.npy'%item)
            if os.path.exists(norm_path):
                setattr(self,'norm_%s'%item,th.Tensor(np.load(norm_path).to(device)))