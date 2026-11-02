import numpy as np
import torch
from torch import nn
from .features import PHYSICAL_DIM,MEMORY_DIM
from .data import RESPONSE_GROUPS


class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.ModuleList([nn.Linear(PHYSICAL_DIM+51,128) for _ in range(2)])
        self.mix=nn.ModuleList([nn.Linear(128+MEMORY_DIM,128) for _ in range(2)])
        self.value=nn.ModuleList([nn.Linear(128,1) for _ in range(2)])
        self.response=nn.ModuleList([nn.Linear(128,60) for _ in range(2)])

    def forward(self,state,action,memory):
        hidden=[torch.relu(e(torch.cat([state,action],-1))) for e in self.encoder]
        q=[v(torch.relu(m(torch.cat([h,memory],-1)))) for h,m,v in zip(hidden,self.mix,self.value)]
        response=[r(h) for r,h in zip(self.response,hidden)]
        return torch.cat(q,-1),torch.stack(response).mean(0)


class Agent:
    def __init__(self,seed,stable):
        torch.set_num_threads(1);torch.manual_seed(seed)
        self.net=Twin();self.optimizer=torch.optim.Adam(self.net.parameters(),lr=.0003)
        self.rng=np.random.default_rng(seed);self.stable=stable

    def train_steps(self,state,action,mem,target,response,steps):
        tensors=[torch.as_tensor(np.array(a),dtype=torch.float32) for a in [state,action,mem,target,response]]
        for _ in range(steps):
            ix=self.rng.integers(len(target),size=32);s,a,m,y,r=[t[ix] for t in tensors]
            q,pred=self.net(s,a,m);loss=((q-y[:,None])**2).mean()
            if self.stable:
                aux=torch.stack([((pred[:,lo:hi]-r[:,lo:hi])**2).mean() for lo,hi in RESPONSE_GROUPS.values()]).mean()
                loss=loss+aux
            self.optimizer.zero_grad();loss.backward();self.optimizer.step()

    def predict(self,state,action,mem):
        self.net.eval()
        with torch.no_grad():
            q,r=self.net(*[torch.as_tensor(np.asarray(a),dtype=torch.float32) for a in [state,action,mem]])
        return q.min(-1).values.numpy(),r.numpy()
