import numpy as np
from scipy.special import expit

LOGIT_TARGET=float(np.log(19.))


def targets(response,failure,scale):
    return np.c_[np.asarray(response)/scale,2*np.asarray(failure,float)-1]


def raw_prediction(z,scale):
    return np.c_[np.asarray(z)[:,:60]*scale,np.asarray(z)[:,60:]*LOGIT_TARGET]


def task_errors(prediction,response,failure,scale):
    continuous=np.max(np.abs(np.asarray(prediction)[:,:60]-response)/scale,axis=-1)
    classification=np.max(np.abs(expit(np.asarray(prediction)[:,60:])-failure),axis=-1)
    return np.maximum(continuous,classification)


def learning_reward(before,after,initial):
    return (before-after)/max(initial,1e-8)


class Readout:
    """Ridge update of the existing linear heads, represented by their kernel."""
    def __init__(self,features,base,regularization=1.):
        self.features=np.asarray(features,float);self.base=np.array(base,float)
        self.mean=self.base.copy();self.covariance=self.features@self.features.T/regularization
        self.observed=[]

    def observe(self,index,target):
        if index in self.observed:raise ValueError('Repeated observation')
        column=self.covariance[:,index].copy();denominator=1.+column[index]
        self.mean+=column[:,None]/denominator*(np.asarray(target)-self.mean[index])
        self.covariance-=np.outer(column,column)/denominator
        self.covariance=(self.covariance+self.covariance.T)/2
        self.observed.append(index)

    def copy(self):
        result=object.__new__(Readout)
        result.features=self.features;result.base=self.base;result.mean=self.mean.copy()
        result.covariance=self.covariance.copy();result.observed=self.observed.copy()
        return result


def virtual_values(state,candidate_indices,probe_indices,noise_scale,draw_seed):
    """Joint function draws. Neither audit labels nor unqueried targets are inputs."""
    candidates=np.asarray(candidate_indices,int);probes=np.asarray(probe_indices,int)
    indices=np.r_[candidates,probes];n=len(candidates)
    cov=state.covariance[np.ix_(indices,indices)]
    # The physical function is approximated only in the fixed neural feature space.
    eigenvalue,eigenvector=np.linalg.eigh((cov+cov.T)/2)
    factor=eigenvector*np.sqrt(np.maximum(eigenvalue,0.))[None,:]
    rng=np.random.default_rng(draw_seed)
    draw=rng.normal(size=(2,len(indices),63))
    fields=np.einsum('ij,mjk->mik',factor,draw)*np.asarray(noise_scale)[None,None,:]
    fields=np.concatenate([fields,-fields],axis=0)
    observation_noise=rng.normal(size=(2,n,63))*np.asarray(noise_scale)[None,None,:]
    observation_noise=np.concatenate([observation_noise,-observation_noise],axis=0)
    query_world=state.mean[candidates][None]+fields[:,:n]
    innovation=fields[:,:n]+observation_noise
    innovation[:,:,60:]=np.where((query_world+observation_noise)[:,:,60:]>0,1.,-1.)-state.mean[candidates][None,:,60:]
    reference_world=state.mean[probes][None]+fields[:,n:]
    before=np.maximum(np.max(np.abs(fields[:,n:,:60]),axis=-1),
        np.max(np.abs(expit(state.mean[probes][None,:,60:]*LOGIT_TARGET)-(reference_world[:,:,60:]>0)),axis=-1)).mean()
    gain=state.covariance[np.ix_(probes,candidates)]/(1+np.diag(state.covariance)[candidates])[None,:]
    changed=state.mean[probes][None,None]+np.einsum('rc,mck->mcrk',gain,innovation)
    continuous=np.max(np.abs(reference_world[:,None,:,:60]-changed[:,:,:,:60]),axis=-1)
    classification=np.max(np.abs(expit(changed[:,:,:,60:]*LOGIT_TARGET)-(reference_world[:,None,:,60:]>0)),axis=-1)
    values=before-np.maximum(continuous,classification).mean(axis=(0,2))
    return values,np.sqrt(np.maximum(np.diag(state.covariance)[candidates],0.))
