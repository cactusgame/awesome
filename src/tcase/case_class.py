class A:
    def __init__(self):
        print("A create")


class B:
    def __init__(self, pa, pb):
        print("B pa is {} , B pb is {}".format(pa, pb))


class C(B):
    def __init__(self, pa, pb):
        super(C,self).__init__(pa, pb)


obj = C('hello', 'world')
