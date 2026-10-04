#!/usr/bin/env python3
"""Conditional ensemble sequence monitor with separate sensor/model uncertainty."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass,field
import numpy as np

@dataclass
class ConditionalWindowMonitor:
    window:int=10
    beta:float=1.0
    alarm_threshold:float=2.6228122472763062
    reset_threshold:float=1.5
    sensor_invalid_patience:int=3
    scores:deque=field(default_factory=deque)
    alarm:bool=False
    sensor_invalid_run:int=0
    def __post_init__(self):
        if self.window<1 or self.reset_threshold>=self.alarm_threshold or self.sensor_invalid_patience<1:raise ValueError('invalid monitor parameters')
        self.scores=deque(maxlen=self.window)
    def update(self,mean_lr:float,ensemble_std:float,sensor_quality:float,model_support:float):
        sensor_valid=bool(np.isfinite([mean_lr,ensemble_std,sensor_quality]).all() and sensor_quality>=.5)
        self.sensor_invalid_run=0 if sensor_valid else self.sensor_invalid_run+1
        if sensor_valid:self.scores.append(float(mean_lr-self.beta*max(ensemble_std,0.)))
        window_score=float(np.mean(self.scores)) if len(self.scores)==self.window else None
        if window_score is not None:
            if window_score>self.alarm_threshold:self.alarm=True
            elif window_score<self.reset_threshold:self.alarm=False
        # Model support is reported for abstention/coverage, but does not erase
        # robust sequence evidence and does not itself command an emergency stop.
        model_unknown=bool(not np.isfinite(model_support) or model_support<.2)
        return {'robust_step_score':self.scores[-1] if sensor_valid else None,'window_score':window_score,'execution_alarm':self.alarm,
                'sensor_protective_stop':self.sensor_invalid_run>=self.sensor_invalid_patience,'model_unknown':model_unknown,'model_support':float(model_support) if np.isfinite(model_support) else None}
    def reset(self):
        self.scores.clear();self.alarm=False;self.sensor_invalid_run=0
