import os
import imp


def import_from_uri(uri, absl=True):
    """
    import a module according to given path
    :param uri:
    :param absl:
    :return:
    """
    if not absl:
        uri = os.path.normpath(os.path.join(os.path.dirname(__file__), uri))
    path, fname = os.path.split(uri)
    mname, ext = os.path.splitext(fname)

    no_ext = os.path.join(path, mname)

    if os.path.exists(no_ext + '.py'):
        return imp.load_source(mname, no_ext + '.py')
