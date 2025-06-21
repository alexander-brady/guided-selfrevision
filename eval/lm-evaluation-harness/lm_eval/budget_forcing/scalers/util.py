from functools import wraps


def should_scale_only(func):
    """
    Decorator for scale functions that only return a boolean 
    indicating whether to scale.
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        scale_token = kwargs.pop('scale_token')
        return func(*args, **kwargs), scale_token
    
    return wrapper

def scale_token_only(func):
    """
    Decorator for scale functions that only return the scale token.
    Returns true if a scale token is returned, false otherwise.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        scale_token = func(*args, **kwargs)
        return bool(scale_token), scale_token
    
    return wrapper