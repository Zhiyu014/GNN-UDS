import torch as th
from torch import nn
from torch_geometric.nn import SAGEConv,GraphConv,GATConv,HeteroConv
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import numpy as np,os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class BatchGATConv(GATConv):
    def __init__(self,*args,**kwargs):
        super(BatchGATConv,self).__init__(*args,**kwargs)
    
    def forward(self,x,edge_index,edge_attr=None,*args,**kwargs):
        # TODO: remove tensor.shape
        bat,seq,nodes = x.shape[:-1]
        edge_index = repeat(edge_index, 'p e -> p e b', b=bat*seq)
        edge_index = edge_index + th.Tensor(th.arange(bat*seq)*nodes).to(device)
        edge_index = rearrange(edge_index,'p e b -> p (e b)')
        if edge_attr is not None:
            edge_attr = repeat(edge_attr,'e d -> (tile e) d',tile=bat*seq)
        out = super().forward(rearrange(x, 'b t n d -> (b t n) d'),edge_index,edge_attr,*args,**kwargs)
        return rearrange(out,'(b t n) d -> b t n d',b=bat,t=seq)
    
def get_batch_index(edge_index: th.Tensor,batch: th.Tensor):
    edge_index = repeat(edge_index, 'p e -> p e b', b=batch)
    edge_index = edge_index + th.Tensor(th.arange(batch)*(edge_index.max().item()+1)).to(device)
    edge_index = rearrange(edge_index,'p e b -> p (e b)')
    return edge_index

class NodeEdge(nn.Module):
    def __init__(self, dim, activation, inci, **kwargs):
        super(NodeEdge,self).__init__(**kwargs)
        self.xes = nn.Linear(dim,dim//2)
        self.activation = getattr(F,activation,F.relu)
        self.inci = nn.Parameter(inci,requires_grad=False)
        self.w = nn.Parameter(th.randn(inci.shape),requires_grad=True)
        self.b = nn.Parameter(th.zeros_like(inci),requires_grad=True)

    def forward(self,inputs):
        xe = self.activation(self.xes(inputs))
        return th.matmul(self.w * self.inci + self.b, xe)

class STBlock(nn.Module):
    def __init__(self,nly,dim,seq,kernel,activation,node_edge):
        super(STBlock,self).__init__()
        self.dim = dim
        self.conv_dim = dim+dim//2
        self.seq = seq
        # self.sps_x = nn.ModuleList([BatchGATConv(dim+dim//2,dim) for _ in range(nly)])
        # self.sps_e = nn.ModuleList([BatchGATConv(dim+dim//2,dim) for _ in range(nly)])
        self.sps_x = nn.ModuleList([GATConv(self.conv_dim,dim,add_self_loops=False) for _ in range(nly)])
        self.sps_e = nn.ModuleList([GATConv(self.conv_dim,dim,add_self_loops=False) for _ in range(nly)])
        self.nes = nn.ModuleList([NodeEdge(dim,activation,th.abs(node_edge)) for _ in range(nly)])
        self.ens = nn.ModuleList([NodeEdge(dim,activation,th.abs(node_edge).T) for _ in range(nly)])
        self.tps = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        # self.tps_x = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        # self.tps_e = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        self.padding = [(kernel-1)*(2**i) for i in range(nly)]
        self.activation = getattr(F,activation,F.relu)
        self.n_node,self.n_edge = node_edge.shape

    def forward(self,x,e,edge_ind,node_ind):
        x,e = x.view(-1, self.n_node, self.dim),e.view(-1, self.n_edge, self.dim)
        for spx,ne,spe,en in zip(self.sps_x,self.nes,self.sps_e,self.ens):
            x,e = th.concat([x,ne(e)],dim=-1),th.concat([e,en(x)],dim=-1)
            x = self.activation(spx(x.view(-1,self.conv_dim),edge_ind))
            e = self.activation(spe(e.view(-1,self.conv_dim),node_ind))
            x,e = x.view(-1, self.n_node, self.dim),e.view(-1, self.n_edge, self.dim)
        h = th.concat([x,e],dim=-2).view(-1,self.seq,self.n_node+self.n_edge,self.dim)
        h = h.permute(0,2,3,1).reshape(-1, self.dim, self.seq)
        for tp,pad in zip(self.tps,self.padding):
            h = F.pad(h,(pad,0))
            h = self.activation(tp(h))
        h = h.view(-1, self.n_node+self.n_edge, self.dim, self.seq).permute(0,3,1,2)
        x,e = h.split([self.n_node,self.n_edge],dim=-2)
        # x,e = x.view(-1, self.seq, self.n_node, self.dim),e.view(-1, self.seq, self.n_edge, self.dim)
        # x,e = x.permute(0,2,3,1).reshape(-1, self.dim, self.seq),e.permute(0,2,3,1).reshape(-1, self.dim, self.seq)
        # for tpx,tpe,pad in zip(self.tps_x,self.tps_e,self.padding):
        #     x,e = F.pad(x,(pad,0)),F.pad(e,(pad,0))
        #     x,e = self.activation(tpx(x)),self.activation(tpe(e))
        # x = x.view(-1, self.n_node, self.dim, self.seq).permute(0,3,1,2)
        # e = e.view(-1, self.n_edge, self.dim, self.seq).permute(0,3,1,2)
        return x,e
    
#TODO: Heterogeneous Graphs for node/edge based graph
class STBlock2(nn.Module):
    def __init__(self,nly,dim,seq,kernel,activation,node_edge,*args,**kwargs):
        super(STBlock2,self).__init__()
        self.dim = dim
        self.seq = seq
        self.sps = nn.ModuleList([GATConv(dim,dim) for _ in range(nly)])
        self.tps = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        self.padding = [(kernel-1)*(2**i) for i in range(nly)]
        self.activation = getattr(F,activation,F.relu)
        self.n_node,self.n_edge = node_edge.shape

    def forward(self,x,e,edge_ind):
        h = th.concat([x,e],dim=-2).view(-1,self.n_node+self.n_edge,self.dim)
        for sp in self.sps:
            h = self.activation(sp(h,edge_ind))
        h = h.view(-1,self.seq,self.n_node+self.n_edge,self.dim)
        h = rearrange(h,'b t n d -> (b n) d t')
        for tp,pad in zip(self.tps,self.padding):
            h = F.pad(h,(pad,0))
            h = self.activation(tp(h))
        h = rearrange(h,'(b n) d t -> b t n d', n=self.n_node+self.n_edge)
        x,e = th.split(h, [self.n_node,self.n_edge], dim=-2)
        return x,e
    

class STM(nn.Module):
    def __init__(self,in_dims,node_edge,args,graph_base=False,if_flood=False):
        super(STM,self).__init__()
        nly = getattr(args,"nly",3)
        dim = getattr(args,'dim',64)
        kernel = getattr(args,"kernel",3)
        activation = getattr(args,"activation","relu")
        self.seq = getattr(args,"seq",5)
        self.encs = nn.ModuleList([nn.Linear(in_dim,dim if i < 2 else dim//2)
                      for i,in_dim in enumerate(in_dims)])
        stblock = STBlock2 if graph_base else STBlock
        self.st1 = stblock(nly, dim, self.seq, kernel, activation, node_edge)
        self.concs = nn.ModuleList([nn.Linear(dim+dim//2, dim)] * 2)
        self.st2 = stblock(nly, dim, self.seq, kernel, activation, node_edge)
        self.res = nn.ModuleList([nn.Linear(dim, dim)] * 2)
        self.outs = nn.ModuleList([nn.Linear(dim, 1),nn.Linear(dim, 2)])
        self.activation = getattr(F,activation,F.relu)
        self.if_flood = if_flood
        if self.if_flood:
            Activate = getattr(nn,activation.capitalize().replace('lu','LU'),'ReLU')
            self.flood = nn.Sequential(*[nn.Linear(dim,dim),Activate()]*nly+\
                                       [nn.Linear(dim,1)])

    def forward(self,inputs,indices):
        x,e = self.pred(inputs,indices)
        out,e = self.output(x,e)
        # flood classification
        if self.if_flood:
            out = th.concat([out,self.flood(x)],dim=-1)
        return out,e

    def pred(self,inputs,indices):
        # Encoding
        x,e,b,a = [enc(inp) for enc,inp in zip(self.encs, inputs)]
        res = [x[:,-1:,...].tile(1,self.seq,1,1),
               e[:,-1:,...].tile(1,self.seq,1,1)]
        x,e,b,a = [self.activation(inp) for inp in [x,e,b,a]]
        # Spatio-temporal block
        x,e = self.st1(x,e,*indices)
        # link st1 and st2
        x = self.activation(self.concs[0](th.concat([x,b],dim=-1)))
        e = self.activation(self.concs[1](th.concat([e,a],dim=-1)))
        x,e = self.st2(x,e,*indices)
        # resnet
        x = self.activation(th.cumsum(self.res[0](x),dim=1) + res[0])
        e = self.activation(th.cumsum(self.res[1](e),dim=1) + res[1])
        return x,e
        
    def output(self,x,e):
        x,e = self.outs[0](x),self.outs[1](e)
        return F.hardsigmoid(x),th.stack([F.hardsigmoid(e[...,0]),th.tanh(e[...,-1])],dim=-1)
        
class Emulator:
    def __init__(self,args=None,act_only=False):
        self.n_node,self.n_in = getattr(args,'state_shape',(40,4))
        self.n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.tide = getattr(args,'tide',False)
        # Runoff (tide) is boundary
        self.b_in = 2 if self.tide else 1
        self.seq = getattr(args,'seq',5)

        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 1
        self.epsilon = getattr(args,"epsilon",-1.0)
        in_dims = [self.n_in,self.e_in,self.b_in,1]

        self.graph_base = getattr(args,"graph_base",False)
        self.edges = th.LongTensor(getattr(args,"edges").T).to(device)
        self.node_index = th.LongTensor(getattr(args,"node_index").T).to(device)
        self.node_edge_index = th.LongTensor(getattr(args,"node_edge_index").T).to(device)
        self.node_edge = th.Tensor(getattr(args,"node_edge")).to(device)
        # Need to specify batch_size to change indices because of batch GATConv
        self.batch_size = getattr(args,"batch_size",1)
        self.set_indices(self.batch_size)

        self.model = STM(in_dims,self.node_edge,args,self.graph_base,self.if_flood).to(device)
        self.model_dir = getattr(args,"model_dir")
        if not act_only:
            self.optimizer = th.optim.Adam(self.model.parameters(),lr=getattr(args,"learning_rate",1e-3))
            self.mixed_precision = getattr(args,"mixed_precision",False)
            self.scaler = th.amp.GradScaler()
            self.mse = F.mse_loss
            if self.if_flood:
                self.bce = F.binary_cross_entropy_with_logits
                self.poswei = th.Tensor(getattr(args,"poswei",[1.0 for _ in range(self.n_node)])).to(device)
            # GradNorm for multi-task learning
            self.gradnorm = getattr(args,"gradnorm",False)
            if self.gradnorm:
                self.alpha_reg = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
                self.alpha_cls = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
                self.alpha_optimizer = th.optim.Adam([self.alpha_reg,self.alpha_cls],lr=1e-4)
                self.l1loss = nn.L1Loss()
                self.alpha = 0.5
            self.roll = getattr(args,"horizon",getattr(args,"roll",0)*self.seq)//self.seq
            self.nwei = th.Tensor(getattr(args,"nwei",[1.0 for _ in range(self.n_node)])).to(device)
            self.ewei = th.Tensor(getattr(args,"ewei",[1.0 for _ in range(self.n_edge)])).to(device)  

        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.act_edges = act_edges = getattr(args,"act_edges",[])
            sett = np.zeros(self.n_edge)
            sett[act_edges] = range(1,len(act_edges)+1)
            self.sett = th.LongTensor(sett).to(device)

        self.is_outfall = th.Tensor(getattr(args,"is_outfall",[0 for _ in range(self.n_node)])).to(device)
        self.pump = th.Tensor(getattr(args,"pump",[0.0 for _ in range(self.n_edge)])).to(device)
        self.has_pump = self.pump.min().item() > 0
        if self.has_pump:
            self.area = th.Tensor(getattr(args,"area",[0.0 for _ in range(self.n_node)])).to(device)
        self.hmax = th.Tensor(getattr(args,"hmax",[1.5 for _ in range(self.n_node)])).to(device)
        self.hmin = th.Tensor(getattr(args,"hmin",[0.0 for _ in range(self.n_node)])).to(device)
        self.use_head = self.hmin.max().item() > 0
        self.offset = th.Tensor(getattr(args,"offset",[0.0 for _ in range(self.n_edge)])).to(device)
        self.has_offset = self.offset.max().item() > 0

    def set_indices(self,batch_size):
        if not hasattr(self,"indices") or batch_size != self.batch_size:
            if self.graph_base:
                indices = [get_batch_index(self.node_edge_index,batch_size*self.seq)]
            else:
                indices = [get_batch_index(self.edges,batch_size*self.seq),
                           get_batch_index(self.node_index,batch_size*self.seq)]
            setattr(self,"indices",indices)
    
    def get_edge_action(self,a,g=False):
        if g:
            a = th.concat([th.ones_like(a[...,:1]),a],dim=-1)
            return th.index_select(a, -1, self.sett).unsqueeze(dim=-1)
        else:
            def set_edge_action(s):
                out = np.ones(self.n_edge)
                out[self.act_edges] = s
                return out
            return np.expand_dims(np.apply_along_axis(set_edge_action,-1,a),-1)

    @th.compile(dynamic=False)
    def _model(self,x,ex,b,ae):
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
                    x_new = th.concat([preds[...,:-1],(preds[...,-1:]>0).to(th.float32),bi[...,:1]],dim=-1)
                else:
                    x_new = th.concat([preds,bi[...,:1]],dim=-1)
                ex_new = th.concat([edge_preds,aei],dim=-1)
                x,ex = x_new,ex_new
            preds = th.concat(predss,dim=1)
            edge_preds = th.concat(edge_predss,dim=1)
        else:
            inp = [x,ex,b,ae]
            preds,edge_preds = self.model(inp,self.indices)
            preds,edge_preds = self.post_proc(preds,edge_preds,b,ae)
        preds = th.concat([preds[...,:-1].clip(0,1),preds[...,-1:]],dim=-1) # avoid large loss value
        return preds,edge_preds

    @th.compile(dynamic=False)
    def fit_eval(self,x,a,b,y,ex,ey,ini_loss=None,fit=True):
        ae = self.get_edge_action(a,g=True)
        with th.autocast(device_type='cuda',dtype=th.float16 if self.mixed_precision else th.float32):
            preds,edge_preds = self._model(x,ex,b,ae)
            # Loss funtion
            # narrow down norm range of water head
            if self.use_head:
                wei = (self.hmax-self.hmin)*(1-self.is_outfall) + (self.hmax-self.hmin).mean()*self.is_outfall
                wei = (self.norm_y[0,:,0].max()-self.norm_y[1,:,0].min())/wei
                preds = th.concat([preds[...,:1] * wei,preds[...,1:]],dim=-1)
                y = th.concat([y[...,:1] * wei,y[...,1:]],dim=-1)
            node_loss = self.mse(th.einsum('btnd,n->btnd',[preds[...,:3],self.nwei]),
                                th.einsum('btnd,n->btnd',[y[...,:3],self.nwei]))
            edge_loss = self.mse(th.einsum('btnd,n->btnd',[edge_preds,self.ewei]),
                                 th.einsum('btnd,n->btnd',[ey,self.ewei]))
            if self.if_flood:
                weight = (self.nwei*self.poswei).unsqueeze(dim=-1) * y[...,-2:-1] +\
                    self.nwei.unsqueeze(dim=-1) * (1 - y[...,-2:-1])
                flood_loss = self.bce(preds[...,-1:], y[...,-2:-1], weight = weight)
        if fit:
            # th.autograd.set_detect_anomaly(True)
            reg_loss = node_loss + edge_loss
            # --- GradNorm ---
            if self.if_flood and self.gradnorm and ini_loss is not None:
                self._fit_grad_norm(reg_loss,flood_loss,ini_loss)
            loss = (self.alpha_reg if self.gradnorm else 1) * reg_loss
            if self.if_flood:
                loss += (self.alpha_cls if self.gradnorm else 1) * flood_loss
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward() if self.mixed_precision else loss.backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer) if self.mixed_precision else self.optimizer.step()
            if self.mixed_precision:
                self.scaler.update()
        return [node_loss,flood_loss,edge_loss] if self.if_flood else [node_loss,edge_loss]

    def _fit_grad_norm(self,reg_loss,flood_loss,ini_loss):
        W = self.model.res[0].weight
        self.optimizer.zero_grad()
        grad_reg = th.autograd.grad(reg_loss,W,retain_graph=True,create_graph=True)
        grad_reg_norm = th.norm(self.alpha_reg * grad_reg[0].clone().detach())
        self.optimizer.zero_grad()
        grad_cls = th.autograd.grad(flood_loss,W,retain_graph=True,create_graph=True)
        grad_cls_norm = th.norm(self.alpha_cls * grad_cls[0].clone().detach())

        r_reg,r_cls = reg_loss.detach()/(ini_loss[0]+ini_loss[-1]), flood_loss.detach()/ini_loss[1]
        r_avg = (r_reg + r_cls) / 2
        grad_norm = th.stack([grad_reg_norm,grad_cls_norm])
        target_grad = grad_norm.mean().detach() * (th.stack([r_reg,r_cls])/ r_avg) ** self.alpha
        alpha_loss = self.l1loss(grad_norm,target_grad)

        # update alpha
        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward() if self.mixed_precision else alpha_loss.backward()
        self.scaler.step(self.alpha_optimizer) if self.mixed_precision else self.alpha_optimizer.step()
        
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
            ey = self.normalize(np.squeeze(ey,0),'ey',True)
            y = self.normalize(np.squeeze(y,0),'y',True)

            # Pumped storage depth calculation: boundary condition differs from orifice
            if self.has_pump:
                ps = (self.area * np.matmul(np.clip(self.node_edge,0,1),np.expand_dims(self.pump,axis=-1)).squeeze())>0
                h,qin,qout = [y[...,i] for i in [0,1,2]]
                de = []
                for t in range(self.seq):
                    de += [np.clip(x[-1,:,0] if t==0 else de[-1] +\
                                    (qin-qout)[t,:]/(self.area+1e-6),self.hmin,self.hmax)]
                y[...,0] = h*(1-ps) + np.stack(de,axis=0) * ps

            q_w,y = self.constrain(y,bi[...,0],x[-1:,:,0])
            y = np.concatenate([y,np.expand_dims(q_w,axis=-1)],axis=-1)
            preds.append(y)
            edge_preds.append(ey)
        return np.array(preds),np.array(edge_preds)

    @th.compile(fullgraph=True)
    def predict(self,states,edge_state,b,a,constr=True):
        x = states[:,-self.seq:,...]
        ex = edge_state[:,-self.seq:,...]
        # assert b.shape[1] == self.seq
        if self.act:
            ae = self.get_edge_action(a,g=True)
        inp = [self.normalize(x,'x'),self.normalize(ex,'e'),self.normalize(b,'b'),ae]
        y,ey = self.model(inp,self.indices)

        y,ey = self.post_proc(y,ey,self.normalize(b,'b'),ae)
        y = self.normalize(y,'yy',True)
        ey = self.normalize(ey,'ey',True)

        if constr:
            y = self.constrain(y,x[:,-1,:,0])
            y = th.concat([y,self.get_flood(y,b[...,0])],dim=-1)
        return y,ey

    def post_proc(self,y,ey,b,ae):
        # tide boundary
        if self.tide:
            h = y[...,0] * (1 - self.is_outfall) + b[...,-1]
            y = th.concat([h.unsqueeze(dim=-1), y[...,1:]],dim=-1)
        # TODO: if need to regulate pipe inflow offset
        if self.has_offset:
            inoff = th.einsum("btn,ne->bte",[self.normalize(y,'yy',True)[...,0] - self.hmin,
                                            th.clip(self.node_edge,0,1)])
            ey = ey * (ey[...,-1:]>0).type(th.float32) * ((inoff > self.offset) * (self.offset > 0)).type(th.float32).unsqueeze(dim=-1).tile(1,1,1,2) +\
                ey * (ey[...,-1:]<=0).type(th.float32) * (self.offset > 0).type(th.float32).unsqueeze(dim=-1).tile(1,2) +\
                ey * (self.offset == 0).type(th.float32).unsqueeze(dim=-1).tile(1,2)
        if self.act:
            # regulate pumping flow (rated value if there is volume in inlet tank)
            if self.has_pump>0:
                fl = self.pump * th.einsum("btn,ne->bte",[(y[...,0]>0.01).type(th.float32),th.clip(self.node_edge,0,1)])
                fl *= (self.norm_e[0,:,1]>1e-3).type(th.float32)/self.norm_e[0,:,1]
                fl = (ey[...,-1] * (fl==0).type(th.float32) + fl)*ae[...,0]
                ey = th.stack([ey[...,0],fl],dim=-1)
            else:
                ey = th.concat([ey[...,:1],ey[...,-1:]*ae],dim=-1)
        efl = self.normalize(ey,'ey',True)[...,-1:]
        node_outflow = th.einsum('ne,bted->btnd',[th.clip(self.node_edge,0,1),th.clip(efl,0,th.inf)]) +\
              th.einsum('ne,bted->btnd',[th.abs(th.clip(self.node_edge,-1,0)),-th.clip(efl,-th.inf,0)])
        node_inflow = th.einsum('ne,bted->btnd',[th.abs(th.clip(self.node_edge,-1,0)),th.clip(efl,0,th.inf)]) +\
              th.einsum('ne,bted->btnd',[th.clip(self.node_edge,0,1),-th.clip(efl,-th.inf,0)])
        node_outflow *= (self.norm_y[0,:,2:3]>1e-3).type(th.float32)/self.norm_y[0,:,2:3]
        node_inflow *= (self.norm_y[0,:,1:2]>1e-3).type(th.float32)/self.norm_y[0,:,1:2]
        y = th.concat([y[...,:1],node_inflow,node_outflow,y[...,1:]],dim=-1)
        return y,ey

    def constrain(self,y,h0=None):
        # Pumped storage depth calculation: boundary condition differs from orifice
        if self.has_pump:
            y = th.concat([self.pump_depth(y,h0),y[...,1:]],dim=-1)
        h = th.clip(y[...,0],self.hmin,self.hmax)
        if self.if_flood:
            f = (y[...,-1] > 0).type(th.float32)
            h = self.hmax * f + h * (1-f)
        y = th.concat([h.unsqueeze(dim=-1),y[...,1:]],dim=-1)
        return y

    def get_flood(self,y,b):
        h,q_us,q_ds = [y[...,i] for i in range(3)]
        # h = th.clip(h,self.hmin,self.hmax)
        q_w = th.clip(q_us + b - q_ds,0,np.inf) * (1-self.is_outfall)
        # if self.if_flood:
        #     f = (y[...,-1] > 0).type(th.float32)
        #     h = self.hmax * f + h * (1-f)
        #     y = th.stack([h,q_us,q_ds,y[...,-1]],dim=-1)
        # else:
        #     y = th.stack([h,q_us,q_ds],dim=-1)
        if self.epsilon > 0:
            q_w *= ((self.hmax - h) < self.epsilon).type(th.float32)
        elif self.epsilon == 0:
            pass
        elif self.if_flood:
            q_w *= (y[...,-1] > 0).type(th.float32)
        return q_w.unsqueeze(dim=-1)
    
    def pump_depth(self,y,x0):
        # Pumped storage depth calculation: boundary condition differs from orifice
        ps = (self.area * th.einsum('ne,e->n',th.clip(self.node_edge,0,1),self.pump))>0
        h,qin,qout = [y[...,i] for i in range(3)]
        de = th.zeros_like(h)
        for t in range(self.seq):
            de[:,t,:] = th.clip(x0 if t==0 else de[:,t-1,:] +\
                                (qin-qout)[:,t,:]/(self.area+1e-6),self.hmin,self.hmax)
        h = h*(1-ps) + de * ps
        return h

    def to_tensor(self,dat):
        return th.Tensor(dat).to(device)

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
        for item in 'xbyre':
            setattr(self,'norm_%s'%item, th.Tensor(eval('norm_%s'%item)).to(device))

    def normalize(self,dat,item,inverse=False):
        normal = getattr(self,'norm_%s'%item[:1])
        maxi,mini = normal[0,...],normal[1,...]
        if len(item) > 1:
            maxi,mini = maxi[...,:-1],mini[...,:-1]
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
        if os.path.isfile(self.model_dir):
            path = self.model_dir
            self.model_dir = os.path.dirname(self.model_dir)
        else:
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
                setattr(self,'norm_%s'%item,th.Tensor(np.load(norm_path)).to(device))