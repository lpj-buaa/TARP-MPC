import hashlib
import numpy as np
from scipy.special import expit
from tarp_sequence.baselines import select as old_select
from .data import partition,public
from .core import Readout,targets,raw_prediction,task_errors,virtual_values


class Episode:
    def __init__(self,case,encoded,reference,scale,noise,budget,seed):
        self._oracle=case;self.public=public(case);self.partition=partition(case)
        self.scale=scale;self.noise=noise;self.budget=budget;self.remaining_budget=budget
        self.seen=[];self.queries=[];self.seed=seed;self.reference=reference;self._values=None
        h=np.r_[encoded['features'],reference['features']];base=np.r_[encoded['base'],reference['base']]
        self.learner=Readout(h,base)
        for i in self.partition['initial']:self._observe(i)

    def _observe(self,i):
        c=self._oracle
        if not c['response_mask'][i].all() or not c['recovery_mask'][i].all():raise ValueError('Unknown queried answer')
        y=targets(c['response'][i:i+1],~c['recoverable'][i:i+1].astype(bool),self.scale)[0]
        self.learner.observe(i,y);self.seen.append(i);self._values=None

    @property
    def done(self):return self.remaining_budget==0 or not self.available().any()

    def available(self):return np.array([i in self.partition['query'] and i not in self.seen for i in range(32)])

    def step(self,i):
        if self.done or not self.available()[i]:raise ValueError('Unavailable or audit query')
        self._observe(int(i));self.queries.append(int(i));self.remaining_budget-=1

    def observations(self):
        c=self._oracle
        return [(i,c['response'][i],c['response_mask'][i],c['recoverable'][i],c['recovery_mask'][i]) for i in self.seen]

    def predictions(self):
        z=self.learner.mean[:32]
        return np.c_[z[:,:60]*self.scale,expit(z[:,60:]*np.log(19.))]

    def loss(self,truth):
        raw=raw_prediction(self.learner.mean[32:],self.scale)
        values=task_errors(raw,truth['response'],truth['failure'],self.scale)
        return float(np.mean([np.mean(values[np.array(truth['sources'])==s]) for s in set(truth['sources'])]))

    def values(self):
        if self._values is None:
            available=np.flatnonzero(self.available())
            seed=int(hashlib.sha256(f"{self.seed}|{self.public['key']}|{sorted(self.seen)}".encode()).hexdigest()[:8],16)
            score,uncertainty=virtual_values(self.learner,available,32+np.array(self.reference['probe']),self.noise,seed)
            values=np.zeros(32);std=np.zeros(32);values[available]=score;std[available]=uncertainty
            self._values=(values,std)
        return self._values

    def features(self):
        score,std=self.values();available=self.available();normal=max(float(np.max(np.abs(score[available]))),1e-4)
        candidates=np.c_[self.public['action'],self.learner.mean[:32,:60]/np.sqrt(60),self.predictions()[:,60:],score/normal,std]
        observed=np.zeros((32,177),np.float32);known=np.zeros(32,bool)
        for slot,(i,y,m,recovery,k) in enumerate(self.observations()):
            observed[slot]=np.r_[self.public['action'][i],y/self.scale/np.sqrt(60),m,recovery,k];known[slot]=True
        return dict(state=np.asarray(self.public['state'],np.float32),candidates=candidates.astype(np.float32),
            observations=observed,observation_mask=known,available=available,
            budget=np.array([self.remaining_budget/24,self.budget/24],np.float32),information=score.astype(np.float32))


def choose_rule(episode,method,rng):
    available=np.flatnonzero(episode.available())
    if method=='value_greedy':
        import torch
        from .policy import anchored_logits
        values=torch.as_tensor(episode.values()[0][None],dtype=torch.float32)
        logits=anchored_logits(torch.zeros_like(values),values,torch.as_tensor(episode.available()[None]))
        return int(logits.argmax(-1)[0])
    if method=='random':return int(rng.choice(available))
    return old_select(episode.public,episode,episode.predictions(),'local' if method=='local' else 'bracket_random',rng)
