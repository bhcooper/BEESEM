# coding: utf-8
from openopt import *
from scipy import *
import csv


USE_BASELINE = bool(0)
EXTRAS = ('Potential', 'Scale') + (('Baseline',) if USE_BASELINE else ())
PRIORS = {'gaussian': 1e3, 'gamma': (1, 1e6), 'baseline': (1, 0.1)}
UPPER_BOUND = 50.


def Gradient(f, x0, h=1e-8):
    n = len(x0)
    g = zeros(n)
    for i in range(n):
        x = array(x0)
        x[i] -= h
        g[i] -= f(x)
        x = array(x0)
        x[i] += h
        g[i] += f(x)
    g /= 2 * h
    return g


class Prior(object):
    var = PRIORS['gaussian']
    k, theta = PRIORS['gamma']

    def __init__(self, n):
        self.m = n + 1

    def error(self, w):
        assert w.size == self.m + 1
        return (sum(w[:self.m] ** 2) / self.var +
                (1 - self.k) * w[-1] + exp(w[-1]) / self.theta)

    def reparameterize(self, w):
        assert w.size == self.m + 1
        out = zeros_like(w)
        out[:self.m] = w[:self.m]
        out[-1] = exp(w[-1])
        return out


class Layer(object):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n + 2

    @classmethod
    def error(cls, output, target):
        return ((output - target).A ** 2).sum() / 2

    def output(self, inputs, w):
        w = mat(w).reshape(-1, 1)
        assert w.size == self.n + 2
        pwm = w[:self.n].T
        mu = w[-2]
        scale = w[-1]
        return scale * 1.0 / (1.0 + exp(-(pwm * inputs + mu)))


class PredictionClosure(object):

    def __init__(self, NN, W, P):
        self.NN = NN
        self.W = W
        self.rW = P.reparameterize(W)

    def __call__(self, I):
        return self.NN.output(I, self.rW)


class TrainingClosure(object):

    def __init__(self, NN, I, T, IW=1.0, P=None):
        self.NN = NN  # model
        """:type: Layer"""
        self.i = I  # data
        self.t = T  # target
        self.p = P  # prior
        self.iw = IW

    def error(self, w):
        T = self.t
        O = self.output(w)
        E = self.NN.error(O, T)
        E += self.p.error(w)
        if isnan(E):
            E = Inf
        return E

    def output(self, w):
        w = self.p.reparameterize(w)
        I = self.i
        O = self.NN.output(I, w)
        return O

    def gradient(self, w):
        return Gradient(self.error, w)

    def hessian(self, w, h=1e-4):
        N = len(self.NN)
        H = zeros((N, N))
        for i in xrange(N):
            W = array(w)
            W[i] -= h
            H[:, i] = -self.gradient(W)
            W = array(w)
            W[i] += h
            H[:, i] += self.gradient(W)
        H /= (2 * h)
        H = (H + H.T) / 2
        return H


class SequenceModel(object):
    PC = PredictionClosure
    TC = TrainingClosure

    def __init__(self, I=10, H=5, O=1, dialect="excel-tab", hasheader=True):
        self.H = H
        self.I = I
        self.O = O
        self.hasheader = hasheader
        self.dialect = dialect
        self.model, self.prior = self._model()

    def _model(self):
        n = self.I
        model = Layer(n)
        self._ub = full(len(model), UPPER_BOUND)
        self._lb = -self._ub
        return model, Prior(n)

    def train(self, data, target, initW=None, instance_weight=1.0, maxIter=150,
              verbosity=-1):
        TC = self.closure(data, target, instance_weight)
        self.R = NN_train_train(
            TC, maxIter=maxIter, initW=initW, verbosity=verbosity, lb=self._lb,
            ub=self._ub
        )

    def read_data(self, myfile, targets):
        """
        Reads a file in self.dialect csv format in.
        The first column is the ID.
        The next self.I columns are data.
        The target is the last column.
        """
        IN = csv.reader(myfile, dialect=self.dialect)
        data = []
        target = []
        _id = []

        if self.hasheader:
            IN.next()

        for a in IN:
            i = a[0]
            if targets:
                t = array(
                    [float(x) for x in a[self.I + 1: self.I + 1 + self.O]])
                target.append(t)

            d = array([float(x) for x in a[1: self.I + 1]])
            data.append(d)
            _id.append(i)
        if targets:
            return (_id, mat(data).T), mat(target).T
        else:
            return _id, mat(data).T

    def closure(self, data, target, instance_weight=1.0):
        _, data = data
        return self.TC(self.model, data, target, instance_weight, self.prior)


def NN_train_train(NNC, initW=None, restarts=1, restart_callback=lambda R: None,
                   maxIter=150, solver='scipy_lbfgsb', verbosity=0, lb=None,
                   ub=None):  # changed
    SOL = []
    randomize = True if initW is None else False
    for _ in xrange(restarts):
        if randomize:
            initW = (rand(len(NNC.NN)) * 2 - 1) * 0.2
        W = initW.reshape(-1)

        p = NLP(NNC.error, W, iprint=verbosity, maxIter=maxIter)
        if (lb is not None) and (ub is not None):  # changed
            p.lb = lb
            p.ub = ub
        p.dwhole = None
        p.df = NNC.gradient
        p.n = len(NNC.NN)
        r = p.solve(solver)
        restart_callback(r)
        SOL.append((r.ff, r))

    SOL.sort()

    n = 0
    while SOL[n][0] is nan:
        n += 1
        if n == len(SOL):
            n = 0
            break

    return SOL[n][1]


__all__ = ['USE_BASELINE', 'EXTRAS', 'PRIORS', 'SequenceModel']
