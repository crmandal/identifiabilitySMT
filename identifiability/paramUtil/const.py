
class _Singleton():
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(_Singleton, cls).__new__(cls)
        return cls.__instance

class _PInf(_Singleton):
    def __neg__(self):
        return _NInf()
    def __le__(self, o):
        return isinstance(o, _PInf)
    def __lt__(self, o):
        return False
    def __gt__(self, o):
        return isinstance(o, _PInf)
    def __ge__(self, o):
        return True
    def __eq__(self, o):
        return isinstance(o, _PInf)
    def __repr__(self):
        return '+Inf'

class _NInf(_Singleton):
    def __neg__(self):
        return _PInf()
    def __lt__(self, o):
        return isinstance(o, _NInf)
    def __le__(self, o):
        return True
    def __ge__(self, o):
        return isinstance(o, _NInf)
    def __gt__(self, o):
        return False    
    def __eq__(self, o):
        return isinstance(o, _NInf)
    def __repr__(self):
        return '-Inf'

Inf = _PInf()
DEBUG = False