import numpy as np

class Models:
    def __init__(self):
        pass

    def malthus(self, x, y, *args):
        r = args[0]
        return r * y

    def verhulst(self, x, y, *args):
        r, k = args[0], args[1]
        return r * y * (1 - (y/k))

    def montroll(self, x, y, *args):
        r, k, b = args[0], args[1], args[2]
        return r * y * (1 - (y / k)**b)

    def gompertz(self, x, y, *args):
        k = args[0]
        return - y * np.log(y / k)
    
class NumericalMethods:
    def __init__(self):
        pass
    
    def euler(self, f, x0, y0, h, n, *args):
        x = np.zeros(n + 1)
        y = np.zeros(n + 1)

        x[0] = x0
        y[0] = y0

        for i in range(1, n + 1):
            x[i] = x[i - 1] + h
            y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1], *args)

        return x, y
    
    def rk4(self, f, x0, y0, h, n, *args):
        x = np.zeros(n + 1)
        y = np.zeros(n + 1)

        x[0] = x0
        y[0] = y0

        for i in range(1, n + 1):
            k1 = f(x[i - 1], y[i - 1], *args)
            k2 = f(x[i - 1] + h / 2, y[i - 1] + h / 2 * k1, *args)
            k3 = f(x[i - 1] + h / 2, y[i - 1] + h / 2 * k2, *args)
            k4 = f(x[i - 1] + h, y[i - 1] + h * k3, *args)

            y[i] = y[i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[i] = x[i - 1] + h

        return x, y