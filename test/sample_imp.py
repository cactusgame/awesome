import imp
import os
import sys


def importFromURI(uri, absl=False):
    if not absl:
        uri = os.path.normpath(os.path.join(os.path.dirname(__file__), uri))
    path, fname = os.path.split(uri)
    mname, ext = os.path.splitext(fname)

    no_ext = os.path.join(path, mname)

    if os.path.exists(no_ext + '.pyc'):
        try:
            return imp.load_compiled(mname, no_ext + '.pyc')
        except:
            pass
    if os.path.exists(no_ext + '.py'):
        try:
            return imp.load_source(mname, no_ext + '.py')
        except:
            pass


if __name__ == "__main__":
    py_mod = importFromURI(os.path.join('../src', 'extract/feature_definition.py'))
    if py_mod is not None:
        print("yyyyyy")
        print(py_mod.feature_definition)
    else:
        print("nnnn")
