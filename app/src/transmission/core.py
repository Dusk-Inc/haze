from ..context.core import Context
from ..signal.core import Signal

class Transmission:
    def __init__(self, context: Context, signal: Signal):
        self._context: Context = context
        self._signal: Signal = signal

    def get_context(self):
        return self._context
    
    def get_signal(self):
        return self._signal
    
    def set_signal(self, signal: Signal):
        self._signal = signal