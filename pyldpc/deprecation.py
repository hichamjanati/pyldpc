import warnings


class deprecated_variable(object):

    def __init__(self, dic_change=None):
        self.dic_change = dic_change

    def __call__(self, fun):

        msg_template = "Function parameter '{}' is deprecated and will be "
        "removed in v0.8. Use '{}' instead. "

        def wrapped(*args, **kwargs):
            kwargs2 = dict()
            for kw in kwargs:
                if kw in self.dic_change:
                    warnings.warn(msg_template.format(kw, self.dic_change[kw]),
                                  category=DeprecationWarning)
                    kwargs2[self.dic_change[kw]] = kwargs[kw]
                else:
                    kwargs2[kw] = kwargs[kw]
            return fun(*args, **kwargs2)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = fun.__doc__

        return wrapped
