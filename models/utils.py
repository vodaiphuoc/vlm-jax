from functools import wraps
from beartype import beartype as beartypechecker
import jaxtyping
from typing import Literal

def typechecked(mode: Literal['DEV', 'PROD']):
    """
    A wrapper decorator for type and shape checking conditionally by `mode`
    """
    
    def decorator_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if mode == "DEV":
            return jaxtyping.jaxtyped(typechecker = beartypechecker)(func)
        else:
            return wrapper
            
    return decorator_wrapper

