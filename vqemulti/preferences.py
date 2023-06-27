from dataclasses import dataclass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=False)
class Configuration(metaclass=Singleton):
    mapping: str = 'jw'  # jw: Jordan-wigner , bk: Bravyi-Kitaev
    verbose: bool = False

