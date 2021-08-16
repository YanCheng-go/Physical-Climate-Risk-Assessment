'''
A collection of commonly-used functions throughout the EBRD data-science project
'''


def kwargs_generator(kwargs, kw, def_vals, typ=None):
    if kw not in kwargs or kwargs[kw] is None:
        kwargs[kw] = def_vals
    if typ == 'ls':
        kwargs[kw] = [kwargs[kw]] if isinstance(kwargs[kw], str) else kwargs[kw]
    return kwargs[kw]
