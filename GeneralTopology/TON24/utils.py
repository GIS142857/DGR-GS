import inspect

def ensure_generator(env, func, *args, **kwargs):
    '''
    Make sure that func is a generator function.  If it is not, return a
    generator wrapper
    '''
    if inspect.isgeneratorfunction(func):
        return func(*args, **kwargs)
    else:
        def _wrapper():
            func(*args, **kwargs)
            yield env.timeout(0)

        return _wrapper()


###########################################################
def distance(pos1, pos2):
    '''
        calculate the distance between two nodes
    '''
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5