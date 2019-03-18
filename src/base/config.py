_key_cfg = '_cfg'
_key_isloading = '_isloading'
_key_isloaded = '_isloaded'


class _cfg(object):
    """
    Config class

    To be loaded exactly once.
    """

    def __init__(self):
        object.__setattr__(self, _key_cfg, dict())
        object.__setattr__(self, _key_isloading, False)
        object.__setattr__(self, _key_isloaded, False)

    def __setattr__(self, key, value):
        if key in object.__getattribute__(self, '__dict__'):
            object.__setattr__(self, key, value)
        elif key in object.__getattribute__(self, _key_cfg) or self._isloading:
            object.__getattribute__(self, _key_cfg)[key] = value
        else:
            raise Exception('Config parameters cannot be added. '
                            'To add config parameters, adjust the `config.py` file.')

    def __getattr__(self, item):
        if item in object.__getattribute__(self, '__dict__'):
            return object.__getattribute__(self, item)
        elif item in object.__getattribute__(self, _key_cfg):
            return object.__getattribute__(self, _key_cfg)[item]
        elif not self._isloaded:
            raise Exception('Config needs to be loaded first.')
        raise AttributeError("'cfg' does not have attribute '{}'".format(item))

    def __str__(self):
        """
        Pretty print
        :return: Beautiful print of all config parameters as string.
        """
        if self._isloaded:
            members = object.__getattribute__(self, _key_cfg)
            max_len = max([len(key) for key in members])
            max_len_values = max([len(str(members[key])) for key in members])
            values = ''
            for key in sorted(members):
                values += '{0: <{2}} {1}\n'.format(key, members[key], max_len + 2)
            line_key = ''.join(['-' for _ in range(max_len)])
            line_values = ''.join(['-' for _ in range(max_len_values)])
            out = 'Config:\n{0}   {1}\n{2}{0}   {1}'.format(line_key, line_values, values)
            return out
        else:
            raise Exception('Config needs to be loaded first.')

    def to_dict(self):
        """
        Config parameters to `dict`
        :return: `dict` with config params
        """
        if not self._isloaded:
            raise Exception('Config needs to be loaded first.')
        return object.__getattribute__(self, _key_cfg)

    def load(self, config):
        """
        Load all attributes of object `config` into `cfg`.
        :param config: Class with parameters as attributes
        """
        self._isloaded = True
        self._isloading = True
        # Load `base2.config.cfg`
        for e in dir(config):
            # Requires inheriting from `object`.
            if e not in dir(self):
                setattr(self, e, getattr(config, e))
        self._isloading = False


"""
Expose interface: use as `from base2.config import cfg`
Load the config via `cfg.load(Class/Object config): it will copy over all attributes,
then config attributes can be read via `cfg.x`.
You can never add new attributes, e.g., `cfg.y = 1` when they are not loaded.
You can set the value of existing attrihutes after the config was loaded.
"""
if 'cfg' not in globals() and 'cfg' not in vars():
    cfg = _cfg()
