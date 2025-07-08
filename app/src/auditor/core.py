from .models import ActivityModel, AuditResultsModel
from ..config.core import Config
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
from ..config.core import Config
from ..injector.core import Injector
from ..neuron_io.core import NeuronIO
from ..neuron_io.enums import TransformerTypes
from ..utils.calculations import clamp
from ..network.core import Network
import numpy as np

class Auditor:
    def __init__(self) -> None:
        self.activity: ActivityModel = ActivityModel()
        self.reward_history = [1]
        self.confidence_history = [1]
        self.config: Config = Injector.resolve(GlobalTypes.CONFIG)
        self._io: NeuronIO = Injector.resolve(GlobalTypes.NEURON_IO)
        self._network: Network = Injector.resolve(GlobalTypes.NETWORK)

    def get_confidence_score(self):
        np_confidence = np.array(self.confidence_history)
        return np.mean(np_confidence)
    
    def _update_error_rate(self, reward):
        self.reward_history.append(float(reward))
        if len(self.reward_history) > self.config.audit_window:
            del self.reward_history[0]

    def get_error_rate(self):
        rewards = np.array(self.reward_history)
        return 1-np.mean(rewards)
    
    def get_confidence_rate(self):
        confidence = np.array(self.confidence_history)
        return np.mean(confidence)
    
    def _update_confidence_rate(self, confidence_score):
        self.confidence_history.append(float(confidence_score))
        if len(self.confidence_history) > self.config.audit_window:
            del self.confidence_history[0]

    def stimulate_growth(self, reward: float, confidence: float):
        self._update_error_rate(reward=reward)
        self._update_confidence_rate(confidence_score=confidence)
        growth_check = self.check_growth()
        return growth_check

    def check_growth(self):
        error_rate = self.get_error_rate()
        reward_span = len(self.reward_history)
        confidence_span = len(self.confidence_history)

        if reward_span > self.config.audit_window:
            raise ValueError("Reward history is too long.")
        
        if confidence_span > self.config.audit_window:
            raise ValueError("Confidence history is too long.")

        if reward_span == self.config.audit_window\
            and confidence_span == self.config.audit_window\
            and error_rate > self.config.growth_threshold:
            return self.determine_growth()
            
        self.clear()
        clean_audit = AuditResultsModel(
            nexus_growth=0,
            terminus_growth=0
        )
        return clean_audit

    def determine_growth(self) -> AuditResultsModel:
        total_motors = self._io.get_neuron_total(transformer_type=TransformerTypes.DECODER)
        error_rate = self.get_error_rate()
        confidence = self.get_confidence_score()
        f = self.activity.features
        d = self.config.growth_rate * (1 - error_rate) + (1 - self.config.growth_rate) * (1 - confidence)
        d = max(0.0, min(1.0, d))
        nexus_growth = int(self.activity.nexus_size/(self.config.n_nexus * d * f))
        
        if total_motors > 0 and self.activity.new_motors > 0:
            terminus_growth = int(self.config.n_terminus * d * (self.activity.new_motors**2/total_motors))
        else:
            terminus_growth = 0

        self.reward_history = [1]
        self.confidence_history = [1]
        self.clear()
        return AuditResultsModel(
            nexus_growth=clamp(nexus_growth,min_val=1,max_val=3),
            terminus_growth=self.get_terminus_growth(terminus_growth, total_motors)
        )

    def get_terminus_growth(self, terminus_growth: int, total_motors: int):
        terminus_size = len(self._network.mesh.terminus.get_inters())
        if terminus_size < terminus_growth + total_motors:
            return (total_motors - terminus_size) + terminus_growth

        return terminus_growth

    def clear(self) -> None:
        self.activity = ActivityModel()