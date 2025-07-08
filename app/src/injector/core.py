class Injector:
    _instances = {}

    @classmethod
    def register(cls, name: str, instance):
        cls._instances[name] = instance

    @classmethod
    def resolve(cls, name: str):
        if name not in cls._instances:
            raise KeyError(f"Instance with name '{name}' is not registered to Injector.")
        return cls._instances.get(name)
    
    @classmethod
    def reset(cls):
        cls._instances.clear()