# coding: utf-8
import pickle
from universal import *
from SequenceModel import USE_BASELINE


MIN_INFO = 0.25
TARGT_IC = 0.50  # mean column IC after normalizing a PWM
P_FACTOR = 1E+6  # just a very large number, or mu = -13.8
MAX_ITER = 1000
CI_LEVEL = 0.95
FRAC1 = (1, 0, 1), (1, 0, None)
FRAC2 = (1, 0, 1), (1, 1, None)


def Weights(e_facs, p_fac, n_fac, u=(1, 1, 1), d=(1, 1, 1)):
    nu = np.array(u)
    nd = np.array(d)

    for i, x in enumerate(nd):
        if x is None:
            nd[i] = len(e_facs)

    den = [nu.dot([e_fac, p_fac, n_fac]) for e_fac in e_facs]
    num = nd.dot([sum(e_facs), p_fac, n_fac])
    return np.divide(den, num)


class ModelBase(object):
    def SetData(self, seqs):
        self.uniqs = list(set(seqs))

    def EFactors(self):
        return dict(self.Prediction())

    def PFactor(self):
        return P_FACTOR

    def NFactor(self):
        return 0.0

    def Prediction(self):
        for seq in self.uniqs:
            yield seq, 1.0

    def CalWeights(self, u=(1, 1, 1), d=(1, 1, 1)):
        pred = self.EFactors()
        p_fac = self.PFactor()
        n_fac = self.NFactor()

        def Func(seqs):
            e_facs = [pred[seq] for seq in seqs]
            return Weights(e_facs, p_fac, n_fac, u, d)
        return Func

    def Consensus(self):
        self.consensus = 'T0'
        return self.consensus

    def Init(self, _PFM, extra=()):
        pwm = PFM.Shrink(_PFM, self.consensus)
        pwm.extend(extra)
        pwm.append(-np.log(P_FACTOR))
        pwm.append(0.0)  # A = exp(0) = 1
        if USE_BASELINE:
            pwm.append(-np.log(P_FACTOR))
        return np.array(pwm)


class Model(ModelBase):
    def __init__(self, model_file, Encode, pwm=None):
        """
        :param model_file: str
        :param Encode: Encoder.Encode
        :param pwm: array_like | None
        :return:
        """
        self.model = pickle.load(open(model_file))
        """:type: BaseModel"""
        self.Encode = Encode
        self.pwm = self.model.R.xf if pwm is None else np.array(pwm, float)

    def ReadData(self, path, bias=1.0):
        """
        :param path: str
        :param bias: float | array_like
        :return:
        """
        self.data, self.targ = self.model.read_data(open(path), targets=True)
        self.closure = self.model.closure(
            self.data, self.targ, instance_weight=bias
        )

    def SetData(self, seqs):
        """
        :param seqs: iterable
        :return:
        """
        uniqs = list(set(seqs))
        codes = np.mat(map(self.Encode, uniqs)).T
        self.data = uniqs, codes

    def Reparameterize(self, pwm=None):
        if pwm is None:
            pwm = self.pwm
        return self.model.prior.reparameterize(pwm)

    def Error(self, pwm=None):
        if pwm is None:
            pwm = self.pwm
        return self.closure.error(pwm)

    def Hessian(self, pwm=None, h=1e-6):
        if pwm is None:
            pwm = self.pwm
        return self.closure.hessian(pwm, h)

    def InvCov(self, pwm=None, h=1e-6):
        H = self.Hessian(pwm, h)
        m = len(self.targ.A.T) - len(self.model.model)
        sigma2 = self.Error(pwm) / m  # unbiased estimate of sigma^2
        return H / sigma2

    def Gradient(self, pwm=None):
        if pwm is None:
            pwm = self.pwm
        return self.closure.gradient(pwm)

    def Prediction(self, cutoff=0.0):
        uniqs, codes = self.data
        pc = self.model.PC(self.model.model, self.pwm, self.model.prior)
        counts, = pc(codes).A  # pc(codes) is 1-by-n matrix
        return itt.izip(uniqs, np.maximum(counts, cutoff))

    def OnlyPWM(self):
        return self.Reparameterize()[:self.model.I]

    def EFactors(self):
        """
        :return: dict
        """
        pwm = self.OnlyPWM()
        uniqs, codes = self.data  # codes is np.matrix
        energies = codes.T.A.dot(pwm)  # pwm is negative of actual PWM
        return dict(itt.izip(uniqs, np.exp(energies)))

    def PFactor(self):
        potential = self.Reparameterize()[self.model.I]  # chem potential
        return np.exp(-potential)

    def NFactor(self):  # non-specific binding
        return self.Reparameterize()[self.model.I + 2] if USE_BASELINE else 0.0


class Model2(Model):
    def __init__(self, model_file, Encode, pwm=None):
        super(Model2, self).__init__(model_file, Encode, pwm)
        self.Unravel()

    def Unravel(self):
        if USE_BASELINE:
            U, A, b = self.Reparameterize()[self.model.I:]  # A > 0, b > 0
            self.p_factor = np.exp(-U) * A / (A + b)
            self.n_factor = np.exp(-U) - self.p_factor
        else:
            U = self.Reparameterize()[self.model.I]
            self.p_factor = np.exp(-U)
            self.n_factor = 0.0

    def PFactor(self):
        return self.p_factor

    def NFactor(self):  # non-specific binding
        return self.n_factor


class PPMModel(ModelBase):
    def __init__(self, _PPM, width, pseudo, labels=BASES, extra=()):
        self.pred, self._PPM = PFM.Predictor(_PPM, width, labels, pseudo)
        self.consensus = PFM.Consensus(self._PPM, labels)
        self.x0 = self.Init(self._PPM, extra)

    def Consensus(self):
        return self.consensus

    def Prediction(self):
        for seq in self.uniqs:
            yield seq, self.pred[seq]


def MakeLib(path, width, templ, both=True):
    lib = ContigLibrary(templ.format).ReadCount(path)
    lib.Contig(width, both)
    return lib


def ContigCnt(lib, model, start=0, stop=None, u=None, d=None):
    model.SetData(contig[start:stop] for contig in lib.Iter())
    CalWeights = model.CalWeights(u, d)
    for rec in lib.itervalues():
        rec.SetWeights(CalWeights([seq[start:stop] for seq in rec.contigs]))
    return lib.New()


class PFMSelector(object):
    def __init__(self, PPMs, core_len=5):
        self.PPMs = sorted(PPMs, key=len)
        self.core_len = min(len(self.PPMs[0]), core_len)

    def get_base_PFM(self, pseudo=0.0):
        return max(
            self.PPMs, key=lambda x: self.core_MCIC(x, self.core_len, pseudo)
        )

    def select(self, limit, min_info=MIN_INFO):
        out_PFM = base_PFM = self.get_base_PFM()

        for PPM in self.PPMs:
            if len(PPM) > len(base_PFM):
                cond1 = self.is_aligned(base_PFM, PPM, self.core_len, limit)
                cond2 = self.is_aligned(
                    base_PFM, self.rc_PFM(PPM), self.core_len, limit
                )
                if cond1 or cond2:
                    ents = PFM.Entropy(PPM)
                    if 2.0 - max(ents[0], ents[-1]) > min_info:
                        out_PFM = PPM
        return out_PFM

    @classmethod
    def core_MCIC(cls, _PFM, width, pseudo=0.0):
        return 2.0 - np.mean(PFM.Entropy(PFM.Slice(_PFM, width, pseudo)))

    @classmethod
    def is_aligned(cls, xPFM, yPFM, width, limit):
        r2s = []
        for xPWM in Subset(PFM.PFM_PWM(xPFM), width):
            for yPWM in Subset(PFM.PFM_PWM(yPFM), width):
                r2s.append(cls.mean_col_r2(xPWM, yPWM))
        return max(r2s) > limit

    @classmethod
    def mean_col_r2(cls, xPFM, yPFM):
        assert len(xPFM) == len(yPFM)
        r2s = [stats.pearsonr(x, y)[0] ** 2 for x, y in zip(xPFM, yPFM)]
        return np.mean(r2s)

    @classmethod
    def rc_PFM(cls, _PFM):
        return np.fliplr(_PFM[::-1])
