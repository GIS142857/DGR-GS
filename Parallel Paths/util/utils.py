import inspect

def distance(pos1, pos2):
    sum = 0
    for i in range(len(pos1)):
        sum += (pos1[i] - pos2[i]) ** 2
    return sum ** 0.5

def ensure_generator(env, func, *args, **kwargs):
    if inspect.isgeneratorfunction(func):
        return func(*args, **kwargs)
    else:
        def _wrapper():
            func(*args, **kwargs)
            yield env.timeout(0)
        return _wrapper()

def trans_path_reverse(trans_path):
    str_list = map(str, trans_path)
    return '-'.join(str_list)