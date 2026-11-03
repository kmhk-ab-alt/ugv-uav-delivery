"""Small shared candidate-scoring MLP, implemented in NumPy for CPU learning."""
import json
from pathlib import Path
import numpy as np
from .dispatch_env import FEATURES

class ScoringPolicy:
    name='learned_scoring_mlp'
    def __init__(self,seed=0,hidden=32):
        rng=np.random.default_rng(seed)
        self.params=dict(w1=rng.normal(0,.18,(len(FEATURES)+6,hidden)),
                         b1=np.zeros(hidden),w2=rng.normal(0,.12,hidden))
    def forward(self,obs):
        legal=np.flatnonzero(obs['action_mask'])
        if not len(legal):raise ValueError('no legal actions')
        x=np.concatenate((obs['candidates'][legal],
            np.repeat(obs['global_state'][None,:],len(legal),axis=0)),axis=1).astype(float)
        h=np.tanh(x@self.params['w1']+self.params['b1'])
        scores=h@self.params['w2'];scores-=scores.max()
        probs=np.exp(scores);probs/=probs.sum()
        return legal,x,h,probs
    def select(self,obs):
        legal,_,_,p=self.forward(obs)
        return int(legal[np.argmax(p)])
    def sample(self,obs,rng):
        legal,_,_,p=self.forward(obs)
        return int(rng.choice(legal,p=p))
    def gradient(self,obs,ds):
        legal,x,h,p=self.forward(obs)
        dh=ds[:,None]*self.params['w2'][None,:]*(1-h*h)
        return dict(w1=x.T@dh,b1=dh.sum(axis=0),w2=h.T@ds)
    def imitation(self,obs,action):
        legal,_,_,p=self.forward(obs)
        match=np.all(np.isclose(obs['candidates'][legal],obs['candidates'][action],atol=1e-7),axis=1)
        target=match.astype(float)/match.sum()
        loss=-float(np.sum(target*np.log(np.maximum(p,1e-12))))
        return loss,self.gradient(obs,p-target)
    def reinforce(self,obs,action,advantage):
        legal,_,_,p=self.forward(obs)
        ds=p.copy();ds[np.flatnonzero(legal==action)[0]]-=1
        return self.gradient(obs,ds*advantage)
    def clone(self):
        other=ScoringPolicy();other.params={k:v.copy() for k,v in self.params.items()};return other
    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**self.params,features=np.array(FEATURES),metadata=np.array(json.dumps(metadata or {})))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if tuple(d['features'].tolist())!=FEATURES:raise ValueError('checkpoint feature schema mismatch')
            if d['w1'].ndim!=2 or d['w1'].shape[0]!=len(FEATURES)+6:raise ValueError('checkpoint shape mismatch')
            model=cls(hidden=d['w1'].shape[1]);model.params={k:d[k].copy() for k in model.params}
        if model.params['b1'].shape!=(model.params['w1'].shape[1],) or model.params['w2'].shape!=model.params['b1'].shape:
            raise ValueError('checkpoint output shape mismatch')
        if not all(np.isfinite(v).all() for v in model.params.values()):raise ValueError('nonfinite weights')
        return model

class Adam:
    def __init__(self,params,lr=.002):
        self.lr=lr;self.t=0
        self.m={k:np.zeros_like(v) for k,v in params.items()};self.v={k:np.zeros_like(v) for k,v in params.items()}
    def step(self,params,grads):
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale=min(1.,5./max(norm,1e-12));self.t+=1
        for k in params:
            g=grads[k]*scale
            self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            params[k]-=self.lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
