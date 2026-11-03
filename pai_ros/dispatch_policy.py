"""Deterministic baselines; no trained model or optimality claim."""
import numpy as np

class PriorityDeadlinePolicy:
    name='priority_deadline_eta'
    def select(self,observation):
        legal=np.flatnonzero(observation['action_mask']);legal=legal[legal!=0]
        if not len(legal):return 0
        f=observation['candidates'][legal]
        # Priority first; then deadline slack and estimated time.
        score=100*f[:,0]-10*f[:,2]-f[:,3]
        return int(legal[np.argmax(score)])

class EarliestDeadlinePolicy:
    name='earliest_deadline_eta'
    def select(self,observation):
        legal=np.flatnonzero(observation['action_mask']);legal=legal[legal!=0]
        if not len(legal):return 0
        f=observation['candidates'][legal]
        return int(legal[np.argmin(f[:,2]+.01*f[:,3])])

class RandomPolicy:
    name='random_dispatch'
    def __init__(self,seed=0):self.rng=np.random.default_rng(seed)
    def select(self,observation):
        legal=np.flatnonzero(observation['action_mask']);legal=legal[legal!=0]
        return int(self.rng.choice(legal)) if len(legal) else 0
