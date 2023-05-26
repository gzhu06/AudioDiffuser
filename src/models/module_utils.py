from typing import Optional, TypeVar
from typing_extensions import TypeGuard

# utils
T = TypeVar("T")
def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def default(val, d):
    return val if exists(val) else d
