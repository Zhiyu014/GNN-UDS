import torch as th
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch_geometric.nn import SAGEConv,GraphConv,GATConv,HeteroConv
import torch.nn.functional as F
import numpy as np
import os,shutil
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

import yaml,time,matplotlib.pyplot as plt
from main import Argument,HERE
from dataloader import DataGenerator
from envs import get_env

class ArgumentPredictor(Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--norm',action='store_true',help='if norm targets individually')

def parser(config=None):
    parser = ArgumentPredictor(description='prediction')
    args = parser.parse_args()
    if config is not None:
        hyps = yaml.load(open(config,'r'),yaml.FullLoader)
        parser.set_defaults(**hyps[args.env])
    args = parser.parse_args()
    config = {k:v for k,v in args.__dict__.items() if v!=hyps[args.env].get(k,v)}
    for k,v in config.items():
        if '_dir' in k:
            setattr(args,k,os.path.join(hyps[args.env][k],v))
    print('Training configs: {}'.format(args))
    return args,config

class BLM(nn.Module):
    def __init__(self,in_dims,n_outs,args,if_flood=False):
        super(BLM,self).__init__()
        nly = getattr(args,"nly",3)
        dim = getattr(args,'dim',64)
        kernel = getattr(args,"kernel",3)
        activation = getattr(args,"activation","relu")
        Activate = getattr(nn,activation.capitalize().replace('lu','LU'),'ReLU')
        self.encs = nn.ModuleList([nn.Linear(in_dim,dim//2)
                      for in_dim in in_dims])
        self.spsx = nn.Sequential(*[nn.Linear(dim,dim),Activate()]*nly)
        self.spsb = nn.Sequential(*[nn.Linear(dim,dim),Activate()]*nly)
        # self.tps0 = nn.ModuleList([nn.Conv1d(dim,dim,kernel,dilation=2**i) for i in range(nly)])
        # self.tps1 = nn.ModuleList([nn.Conv1d(dim*2 if i==0 else dim,dim,kernel,dilation=2**i) for i in range(nly)])
        self.tps0 = nn.LSTM(dim,dim,nly,batch_first=True)
        self.tps1 = nn.LSTM(dim*2,dim,nly,batch_first=True)
        self.padding = [(kernel-1)*(2**i) for i in range(nly)]
        self.out = nn.ModuleList([nn.Linear(dim, n_outs[0])])
        self.activation = getattr(F,activation,F.relu)
        self.if_flood = if_flood
        if self.if_flood:
            self.flood = nn.Sequential(*[nn.Linear(dim,dim),Activate()]*nly+\
                                       [nn.Linear(dim,n_outs[1])])
    
    def forward(self,inps):
        x,e,b,a = [enc(inp) for enc,inp in zip(self.encs,inps)]
        x,b = th.concat([x,e],dim=-1),th.concat([b,a],dim=-1)
        x,b = self.spsx(x),self.spsb(b)
        x, (h0,c0) = self.tps0(x)
        x = th.concat([x,b],dim=-1)
        x, _ = self.tps1(x, (h0,c0))
        out = self.out[0](x)
        if self.if_flood:
            out = th.concat([out,self.flood(x)],dim=-1)
        return out

class Predictor:
    def __init__(self,args=None):
        n_node,self.n_in = getattr(args,'state_shape',(40,4))
        n_edge,self.e_in = getattr(args,'edge_state_shape',(40,4))
        self.tide = getattr(args,'tide',False)
        self.b_in = 2 if self.tide else 1
        self.seq = getattr(args,'seq',5)
        self.if_flood = getattr(args,"if_flood",False)
        if self.if_flood:
            self.n_in += 1
        self.act = getattr(args,"act",False)
        self.act = self.act and self.act != 'False'
        if self.act:
            self.n_act = len(getattr(args,"act_edges"))
            self.act_edges = getattr(args,"act_edges",[])
        in_dims = [n_node*self.n_in, n_edge*self.e_in, n_node*self.b_in, self.n_act]

        self.norm = getattr(args,'norm',False)
        self.targets = getattr(args,"performance_targets")
        self.n_out = len(self.targets)
        self.flood_nodes = [args.elements['nodes'].index(i)
                             for i,attr,_ in self.targets 
                             if attr == 'cumflooding' and i in args.elements['nodes']]
        self.flood_nodes = list(set(self.flood_nodes + np.where(args.area>0)[0].tolist()))
        self.n_flood = len(self.flood_nodes)
        n_outs = [self.n_out,self.n_flood]

        self.model = BLM(in_dims,n_outs,args,self.if_flood).to(device)
        self.optimizer = th.optim.Adam(self.model.parameters(),lr=getattr(args,"learning_rate",1e-3))
        self.mixed_precision = getattr(args,"mixed_precision",False)
        self.scaler = th.amp.GradScaler()
        self.mse = F.mse_loss
        if self.if_flood:
            self.bce = F.binary_cross_entropy_with_logits
            self.poswei = th.Tensor([weight for _,attr,weight in self.targets
                                     if attr=='cumflooding']).to(device)
        # GradNorm for multi-task learning
        self.gradnorm = getattr(args,"gradnorm",False)
        if self.gradnorm:
            self.alpha_reg = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
            self.alpha_cls = nn.Parameter(th.tensor(1.0).to(device),requires_grad=True)
            self.alpha_optimizer = th.optim.Adam([self.alpha_reg,self.alpha_cls],lr=1e-4)
            self.l1loss = nn.L1Loss()
            self.alpha = 0.5
        self.model_dir = getattr(args,"model_dir")

    @th.compile(dynamic=False)
    def fit_eval(self,x,e,b,a,objs,ini_loss=None,fit=True):
        with th.autocast(device_type='cuda',dtype=th.float16 if self.mixed_precision else th.float32):
            x,b,e = [th.flatten(dat,2) for dat in [x,b,e]]
            pred = self.model([x,e,b,a])
            # Loss funtion
            if self.if_flood:
                flood_loss = self.bce(pred[...,-self.n_flood:],
                                      (objs[...,:self.n_flood]>0).type(th.float32),
                                      pos_weight = self.poswei)
                pred = th.concat([pred[...,:self.n_flood] * (pred[...,-self.n_flood:]>0.5).type(th.float32),
                                  pred[...,self.n_flood:self.n_out]],dim=-1)
            reg_loss = self.mse(pred[...,:self.n_out],objs)
        if fit:
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
        return [reg_loss,flood_loss] if self.if_flood else [reg_loss]

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
    
    # @th.compile(fullgraph=True)
    def predict(self,state,edge_state,runoff,settings):
        x,b,e = [self.normalize(dat,item) for dat,item in zip([state,runoff,edge_state],'xbe')]
        x,b,e = [th.flatten(dat,2) for dat in [x,b,e]]
        inp = [x,e,b,settings] if self.act else [x,e,b]
        preds = self.model(inp)
        if self.if_flood:
            preds = th.concat([preds[...,:self.n_flood] * (preds[...,-self.n_flood:]>0.5).type(th.float32),
                               preds[...,self.n_flood:self.n_out]],dim=-1)
        if self.norm:
            preds = self.normalize(preds,'o',inverse=True)
        return preds
    
    def to_tensor(self,dat):
        return th.Tensor(dat).to(device)

    def set_norm(self,norm_x,norm_b,norm_y,norm_r,norm_e):
        for item in 'xbyre':
            setattr(self,'norm_%s'%item, th.Tensor(eval('norm_%s'%item)).to(device))

    def normalize(self,dat,item,inverse=False):
        normal = getattr(self,'norm_%s'%item)
        maxi,mini = normal[0,...,:dat.shape[-1]],normal[1,...,:dat.shape[-1]]
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
        for item in 'xbyreo':
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
        for item in 'xbyreo':
            norm_path = os.path.join(self.model_dir,'norm_%s.npy'%item)
            if os.path.exists(norm_path):
                setattr(self,'norm_%s'%item,th.Tensor(np.load(norm_path)).to(device))


if __name__ == '__main__':
    args,config = parser(os.path.join(HERE,'utils','config.yaml'))

    train_de = {
        # 'train':True,
        # 'env':'astlingen',
        # 'data_dir':'./envs/data/astlingen/1s_edge_conti128_rain50/',
        # 'act':'conti',
        # 'model_dir':'./model/astlingen/test/',
        # 'load_model':False,
        # 'ctrl_step':5,
        # 'batch_size':64,
        # 'epochs':10000,
        # 'norm':True,
        # 'nly':5,
        # 'seq':60,
        # 'if_flood':True,
        }
    for k,v in train_de.items():
        setattr(args,k,v)

    env = get_env(args.env)(initialize=False)
    env_args = env.get_args()
    for k,v in env_args.items():
        if k == 'act':
            v = v and args.act != 'False' and args.act
        setattr(args,k,v)
    
    dG = DataGenerator(env.config,args.data_dir,args)
    dG.load(args.data_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    n_events = int(max(dG.event_id))+1
    if os.path.isfile(os.path.join(args.data_dir,args.train_event_id)):
        train_ids = np.load(os.path.join(args.data_dir,args.train_event_id))
    elif args.load_model:
        train_ids = np.load(os.path.join(args.model_dir,'train_id.npy'))
    else:
        train_ids = np.random.choice(np.arange(n_events),int(n_events*args.ratio),replace=False)
    test_ids = [ev for ev in range(n_events) if ev not in train_ids]
    train_idxs = dG.get_data_idxs(train_ids,args.seq)
    test_idxs = dG.get_data_idxs(test_ids,args.seq)

    # Data balance: Only use flooding steps
    # nodes = [args.elements['nodes'].index(node) for node,attr,_ in args.performance_targets if 'flooding' in attr and node != 'T1']
    # iys = np.apply_along_axis(lambda t:np.arange(t,t+seq),axis=1,arr=np.expand_dims(train_idxs,axis=-1))
    # train_idxs = train_idxs[np.take(dG.perfs[:,nodes,-1].sum(axis=-1),iys,axis=0).sum(axis=-1)>0]
    # iys = np.apply_along_axis(lambda t:np.arange(t,t+seq),axis=1,arr=np.expand_dims(test_idxs,axis=-1))
    # test_idxs = test_idxs[np.take(dG.perfs[:,nodes,-1].sum(axis=-1),iys,axis=0).sum(axis=-1)>0]

    emul = Predictor(args)
    if args.load_model:
        emul.load(args.model_dir)
        args.model_dir = os.path.join(args.model_dir,'retrain')
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        if 'model_dir' in config:
            config['model_dir'] += '/retrain'
    emul.set_norm(*dG.get_norm())
    emul.norm_o = emul.to_tensor(env.get_obj_norm(emul.norm_y.cpu().numpy(), dG.perfs.max(axis=0).squeeze()))
    yaml.dump(data=config,stream=open(os.path.join(args.model_dir,'parser.yaml'),'w'))

    t0 = time.time()
    train_losses,test_losses,secs = [],[],[0]
    log_dir = "logs/model/"
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir,exist_ok=True)
    writer = SummaryWriter(log_dir)
    for epoch in range(args.epochs):
        train_dats = dG.prepare_batch(train_idxs,args.seq,args.batch_size,interval=args.ctrl_step,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in train_dats[:4]]
        ex,ey = [dat for dat in train_dats[6:8]]
        objs = env.objective_pred([y,ey],[x,ex],a,keepdim=True)
        if not args.norm:
            objs = emul.to_tensor(env.norm_obj(objs,[x,ex]))
        else:
            objs = emul.normalize(emul.to_tensor(objs),'o')
        x,ex,b,a = [emul.to_tensor(dat) for dat in [x,ex,b,a]]
        x,ex,b = [emul.normalize(dat,item) for dat,item in zip([x,ex,b],'xeb')]
        ini_loss = emul.to_tensor(np.array(train_losses[0])) if epoch > 0 else None
        train_loss = emul.fit_eval(x,ex,b,a,objs,ini_loss)
        train_loss = [los.detach().cpu().numpy() for los in train_loss]
        train_losses.append(train_loss)

        test_dats = dG.prepare_batch(test_idxs,args.seq,args.batch_size,interval=args.ctrl_step,trim=False)
        x,a,b,y = [dat if dat is not None else dat for dat in test_dats[:4]]
        ex,ey = [dat for dat in test_dats[6:8]]
        objs = env.objective_pred([y,ey],[x,ex],a,keepdim=True)
        if not args.norm:
            objs = emul.to_tensor(env.norm_obj(objs,[x,ex]))
        else:
            objs = emul.normalize(emul.to_tensor(objs),'o')
        x,ex,b,a = [emul.to_tensor(dat) for dat in [x,ex,b,a]]
        x,ex,b = [emul.normalize(dat,item) for dat,item in zip([x,ex,b],'xeb')]
        test_loss = emul.fit_eval(x,ex,b,a,objs,fit=False)
        test_loss = [los.detach().cpu().numpy() for los in test_loss]
        test_losses.append(test_loss)

        if sum(train_loss) < min([1e6]+[sum(los) for los in train_losses[:-1]]):
            emul.save('train')
        if sum(test_loss) < min([1e6]+[sum(los) for los in test_losses[:-1]]):
            emul.save('test')
        if epoch > 0 and epoch % args.save_gap == 0:
            emul.save('%s'%epoch)
            
        secs.append(time.time()-t0)

        # Log output
        log = "Epoch {}/{}  {:.4f}s Train loss: {:.4f} Test loss: {:.4f}".format(epoch,args.epochs,secs[-1]-secs[-2],sum(train_loss),sum(test_loss))
        writer.add_scalar('train loss', sum(train_loss), epoch)
        log += " ("
        log += "Obj: {:.4f}".format(test_loss[0])
        writer.add_scalar('Obj loss', test_loss[0], epoch)
        if args.if_flood:
            log += " if_flood: {:.4f}".format(test_loss[1])
            writer.add_scalar('Flood classification', test_loss[1], epoch)
        log += " )"
        print(log)

    # save
    emul.save()
    np.save(os.path.join(args.model_dir,'train_id.npy'),np.array(train_ids))
    np.save(os.path.join(args.model_dir,'test_id.npy'),np.array(test_ids))
    np.save(os.path.join(args.model_dir,'train_loss.npy'),np.array(train_losses))
    np.save(os.path.join(args.model_dir,'test_loss.npy'),np.array(test_losses))
    np.save(os.path.join(args.model_dir,'time.npy'),np.array(secs[1:]))
    # plt.plot(train_losses,label='train')
    # plt.plot(test_losses,label='test')
    # plt.legend()
    # plt.savefig(os.path.join(args.model_dir,'train.png'),dpi=300)
