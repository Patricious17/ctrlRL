
class System(object):
    def __init__(self, F, C = None, ):
        self.status = {}
        self.F = F
        self.C = C
        pass

    '''Open loop'''
    def dxdt(self, isOL = True):
        if isOL:
            u

        return self.F(x,u)


        pass
