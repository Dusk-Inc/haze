class Config:
    def __init__(
            self, 
            signal_threshold: float = 0.3, 
            strength_lower: float = 0.5,
            strength_upper: float = 0.9,
            epsilon_decay: float = 0.9999,
            epsilon_start: float = 0.7,
            n_aperture: float = 0.1, 
            n_nexus: float = 0.1, 
            n_terminus: float = 0.1,
            growth_rate: float = 0.01,
            prune_threshold: float = 0.2,
            growth_threshold: float = 0.5,
            audit_window: int = 10,
            relearn_limit: int = 100,
            neuron_firing_threshold: float = 0.5
        ):
        self.signal_threshold = signal_threshold
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.n_aperture = n_aperture
        self.n_nexus = n_nexus
        self.n_terminus = n_terminus
        self.growth_rate = growth_rate
        self.strength_lower = strength_lower
        self.strength_upper = strength_upper
        self.prune_threshold = prune_threshold
        self.growth_threshold = growth_threshold
        self.audit_window = audit_window
        self.relearn_limit = relearn_limit
        self.neuron_firing_threshold = neuron_firing_threshold
        self._check_config()

    def get_threshold(self):
        return self.signal_threshold
        
    def _check_config(self):
        if self.signal_threshold > 0.9 or self.signal_threshold < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.epsilon_decay > 1 or self.epsilon_decay < 0.1:
            raise Exception("Value must be between 0.1 and 1.0")
        
        if self.epsilon_start > 0.9 or self.epsilon_start < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.n_aperture > 0.9 or self.n_aperture < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.n_nexus > 0.9 or self.n_nexus < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.n_terminus > 0.9 or self.n_terminus < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.growth_rate > 1 or self.growth_rate < 0:
            raise Exception("Value must be greater than 0 and less than 1")
        
        if self.strength_lower > 0.9 or self.strength_lower < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.strength_upper > 0.9 or self.strength_upper < 0.5:
            raise Exception("Value must be between 0.5 and 0.9")
        
        if self.prune_threshold > 0.9 or self.prune_threshold < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")
        
        if self.neuron_firing_threshold > 0.9 or self.neuron_firing_threshold < 0.1:
            raise Exception("Value must be between 0.1 and 0.9")