import numpy as np


class AcquisitionEnv:
    """Finite knowledge-state MDP. Privileged truth is used only by query/reward."""
    def __init__(self,case,budget):self._case=case;self.budget=budget;self.reset()

    def reset(self):
        self.seen=list(self._case['initial']);self.queries=[];self.remaining_budget=self.budget
        self._valid=np.array(self._case['valid'],bool);self._d=np.asarray(self._case['descriptor'],float)
        self._pool=[i for i in range(len(self._d)) if i not in self.seen and self._valid[i]]
        self._pairs=self._case['boundary_pairs']
        self._base_distance=self._distance()
        return self

    @property
    def done(self):return self.remaining_budget==0 or len(self.seen)==len(self._d)

    def available(self):return np.array([i not in self.seen for i in range(len(self._d))])

    def _distance(self):
        known=[i for i in self.seen if self._valid[i]]
        if not self._pool:return 0.
        if not known:return 1.
        return float(np.linalg.norm(self._d[self._pool,None,:]-self._d[np.array(known)][None,:,:],axis=-1).min(1).mean())

    def boundary_recall(self):
        if not self._pairs:return None
        return sum(a in self.seen and b in self.seen for a,b in self._pairs)/len(self._pairs)

    def utility(self):
        cover=1-self._distance()/self._base_distance if self._base_distance>1e-12 else 0.
        boundary=self.boundary_recall()
        return float((cover+boundary)/2 if boundary is not None else cover)

    def step(self,index,score=True):
        if self.done or index in self.seen or index<0 or index>=len(self._d):raise ValueError('Unavailable query or exhausted budget')
        before=self.utility() if score else 0.;self.seen.append(index);self.queries.append(index);self.remaining_budget-=1
        return self.utility()-before if score else 0.

    def observations(self):
        # This is the only response accessor used by the policy feature builder.
        c=self._case
        return [(i,np.asarray(c.get('response',np.zeros((len(self._d),60))))[i],
                 np.asarray(c.get('response_mask',np.ones((len(self._d),60),bool)))[i],
                 np.asarray(c['recoverable'])[i],np.asarray(c.get('recovery_mask',np.ones((len(self._d),3),bool)))[i]) for i in self.seen]

    def metrics(self):
        return dict(utility=self.utility(),distance=self._distance(),boundary_recall=self.boundary_recall(),
                    unknown_queries=sum(not self._valid[i] for i in self.queries),queries=list(self.queries),
                    valid_candidates=len(self._pool),boundary_pairs=len(self._pairs))
