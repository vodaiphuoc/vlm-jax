from functools import wraps
from beartype import beartype as beartypechecker
import jaxtyping
from typing import Literal
from huggingface_hub import snapshot_download
import os

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

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


def download_hf_repo(repo_id: str)->str:
    r"""
    Download repo from huggingface to local cach dir.
    Args:
        repo_id (id): target repo id
    Returns:
        (str): download dir
    """
    return snapshot_download(repo_id, revision="main")
