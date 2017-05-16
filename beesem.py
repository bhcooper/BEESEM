# coding: utf-8
import argparse
import glob
from scipy import linalg
from tools import *
from SequenceModel import *


MIN_COREL = 0.9
PSEUDOCNT = 1E-6  # damping factor
INIT_LAMB = 0.5  # initial lambda
MAX_RATIO = 1.0  # normalization max
MODEL = Model2
UD1 = (1, 0, 1), (1, 0, None)
UD0 = (1, 1, 1), (1, 1, None)


def get_freq(path, width, templ, halve_sym=False):
        lib_long = MakeLib(path, width, templ)
        lib_long.RndWeights(normalize=False)
        lib = lib_long.New()

        if halve_sym:
            for key, rec in lib.iteritems():
                if RC.IsSym(key):
                    rec.Div(2)

        return dict(lib.Export()), lib


class RawModel(ModelBase):
    def __init__(self, path, width, pseudo=0.0, templ='{}', mismatch=1, lb=1e-3,
                 consensus=None):
        self.pseudo = pseudo
        self.width = width

        self.freq, lib = get_freq(path, width, templ, halve_sym=True)

        if consensus:
            if len(consensus) >= width:
                n = (len(consensus) - width) / 2
                self.consensus = consensus[n:n+width]
            else:
                for seq in sorted(self.freq, key=self.freq.get, reverse=True):
                    if consensus in seq:
                        self.consensus = seq
                        break
                else:
                    raise KeyError
        else:
            lib.FilterKey(lb)
            freq2 = dict(lib.Export())
            keys = sorted(freq2, key=freq2.get, reverse=True)
            self.iter = self.Iter(keys, mismatch)

    @classmethod
    def Iter(cls, keys, mismatch):

        def dist(x, y):
            return min(CmpSeq.Mismatch(x, y), max_subseq(x, y, mismatch + 1))

        prevs = []
        for key in keys:
            if any(dist(key, s) <= mismatch for s in prevs):
                continue
            else:
                prevs.append(key)
                if not RC.IsSym(key):
                    prevs.append(RC.RC(key))
                yield key

    def Consensus(self):
        try:
            self.consensus = next(self.iter)
        except AttributeError:
            print 'Using the predefined consensus.\n'

        f = np.vectorize(lambda x: self.freq.get(x, 0), otypes=[np.float])
        _PFM = f(hamming1(self.consensus, squeeze=False))
        _PPM = PFM.PWM_PPM(PFM.target_mean_IC(PFM.PFM_PWM(_PFM, 1.0), TARGT_IC))

        self.pred, self._PPM = PFM.Predictor(_PPM, pseudo=self.pseudo)
        self.x0 = self.Init(self._PPM)
        return self.consensus

    def Prediction(self):
        for seq in self.uniqs:
            yield seq, self.pred[seq]


def IN(curr_path, prev_path, train_set, enc, templ, model, reduce_size,
       base_width=None, lambda0=0.5):

    width = enc.width  # width = total length of seqL + main seq + seqR
    if base_width is None:
        base_width = width

    start = (width - base_width) / 2
    stop = start + base_width

    model.SetData(Product(BASES, base_width))
    pred = Op.Nom(model.EFactors())
    avg_prob = 1.0 / len(pred)

    lib_long = MakeLib(curr_path, base_width, templ)
    """:type: ContigLibrary"""
    n_reads = lib_long.Sum()

    for rec in lib_long.itervalues():
        f1 = np.mean([pred[seq] for seq in rec.contigs]) * lambda0
        f2 = avg_prob * (1 - lambda0)
        rec.count *= f1 / (f1 + f2)
    lambda1 = lib_long.Sum() / n_reads

    if base_width != width:
        lib_long.Contig(width)
    lib1 = ContigCnt(lib_long, model, start, stop, *UD1)

    if prev_path:
        lib0 = ContigCnt(
            MakeLib(prev_path, width, templ), model, start, stop, *UD0
        )
        lib1.Normalize(lib0)

    if MAX_RATIO:
        div = max(lib1.Get()) / float(MAX_RATIO)
        lib1.Rescale(div)

    if reduce_size:
        if enc.encoding in Encoder.decoder:  # likely extension case
            consensus = max(pred, key=pred.get)
        elif len(enc.encoding) == base_width:
            consensus = enc.encoding
        else:
            raise ValueError

        for key in lib1.keys():
            if CmpSeq.Mismatch(key[start:stop], consensus) > base_width / 2:
                del lib1[key]

    lib1.Encode(enc.Encode)

    header = ['ID'] + map(str, range(enc.n)) + ['TARGET']
    CheckDir(train_set)
    WriteCSV(train_set, lib1.Tabulate(), header)

    return lambda1


def NN(train_set, model_out, n, bias=1.0, x0=None):
    model = SequenceModel(I=n)
    data, targ = model.read_data(open(train_set), targets=True)
    model.train(data, targ, x0, bias, maxIter=MAX_ITER, verbosity=0)
    with open(model_out, 'w') as f:
        pickle.dump(model, f)


def OT(train_set, model_out, info_text, path, enc, name, bias=1.0, flag=True):
    model = MODEL(model_out, enc.Encode)
    model.ReadData(train_set, bias)

    if flag:
        return model

    pwm = model.Reparameterize()
    HES = model.InvCov()
    try:
        COV = linalg.inv(HES)
    except linalg.LinAlgError:
        return model

    cond = np.diag(COV) < 0
    var = np.where(cond, np.inf, np.diag(COV))
    std = np.sqrt(var)
    PWM = PFM.Expansion(-pwm[:enc.n], enc.encoding)
    STD = PFM.Expansion(std[:enc.n], enc.encoding)
    PPM = PFM.PWM_PPM(-np.asarray(PWM))

    with open(info_text, 'w') as f:
        f.write('uid: %s\n' % name)
        f.write('encoding: %s\n' % enc.encoding)
        f.write('motif length: %d\n' % enc.width)
        f.write('chemical potential and its standard deviation: %.3f (%.3f)\n'
                % (pwm[enc.n], std[enc.n]))
        f.write('PWM and its standard deviation:\n')
        f.write('\t'.join(BASES) + '\n')
        for row, row2 in zip(PWM, STD):
            f.write('\t'.join('%.3f (%.3f)' % x for x in zip(row, row2)) + '\n')

    with open(path, 'w') as f:
        f.write('PFM (when chemical potential = -inf):\n')
        f.write('\t'.join(BASES) + '\n')
        for row in PPM:
            f.write('\t'.join('%.3f' % x for x in row) + '\n')

    return model


class Main(object):
    def __init__(self, name, path, curr_path, prev_path, flank, width, seed,
                 num_phs, num_rep=1):

        self.name = name
        self.curr_path = curr_path
        self.prev_path = prev_path
        self.seed = seed

        self.templ = '{}'.join(flank)

        self.pseudo = PSEUDOCNT

        self.encodings = 'results',
        self.level = 1
        self.width = int(width)
        self.reduce_size = (self.width > 8)

        self.num_phs = int(num_phs)
        self.num_rep = int(num_rep)

        pwd, cwd = self.Path(path)

        self.root = pwd.GetPath()
        CheckDir(self.root)

        try:
            self.Loop(cwd)
        except AssertionError as err:
            print err.message

    def Path(self, path):
        tmp = [self.name, 'rep=%d' % self.num_rep, 'phs=%d' % self.num_phs]
        pwd = Path(os.path.join(path, '_'.join(tmp)))

        cwd = pwd / Path('%(enc)s')
        cwd.AddFile(
            File('data', 'txt'),
            File('model', 'pyp'),
            File('report', 'txt'),
            File('pfm', 'txt')
        )
        return pwd, cwd

    def Loop(self, cwd):
        nul_model = RawModel(self.curr_path, self.width, self.pseudo,
                             consensus=self.seed)

        for encoding in self.encodings:
            for rep in range(self.num_rep):
                consensus = nul_model.Consensus()
                model = nul_model
                x0 = nul_model.x0
                lambda_ = INIT_LAMB

                for phs in range(self.num_phs):
                    d = dict(rep=rep, phs=phs, enc=encoding, wid=self.width)
                    cwd.Stamp('r%(rep)d', 'p%(phs)d', 'w%(wid)d')

                    enc = Encoder(self.width, consensus, self.level)

                    lambda_ = IN(
                        self.curr_path, self.prev_path, cwd['data'] % d,
                        enc, self.templ, model, self.reduce_size,
                        lambda0=lambda_
                    )

                    NN(cwd['data'] % d, cwd['model'] % d, enc.n, x0=x0)

                    model = OT(
                        cwd['data'] % d, cwd['model'] % d, cwd['report'] % d,
                        cwd['pfm'] % d, enc, self.name,
                        flag=(phs != self.num_phs-1)
                    )
                    x0 = model.pwm

                    if os.path.isfile(cwd['data'] % d):
                        os.remove(cwd['data'] % d)

                    if os.path.isfile(cwd['model'] % d):
                        os.remove(cwd['model'] % d)


def driver():
    parser = argparse.ArgumentParser()
    parser.add_argument('uid', help='unique name of this experiment')
    parser.add_argument('inputs', nargs='+',
                        help='one or two input sequence file(s)')
    parser.add_argument('-f', '--flanking', nargs=2, default=['', ''],
                        help='two flanking sequences (optional)')
    parser.add_argument('-l', '--length', type=int,
                        help='motif length (optional)')
    parser.add_argument('-o', '--output', default=os.getcwd(),
                        help='output directory (optional, default: current working directory)')
    parser.add_argument('-p', '--phase', type=int, default=10,
                        help='number of expectation-maximization rounds (optional, default: 10)')
    parser.add_argument('-s', '--seed', help='consensus sequence (optional)')
    parser.add_argument('-v', '--version', action='version', version='1.0.0')

    kwargs = vars(parser.parse_args())

    inputs = kwargs['inputs']
    if len(inputs) == 1:
        prev_path = ''
        curr_path, = inputs
    elif len(inputs) == 2:
        prev_path, curr_path = inputs
    else:
        print 'No more than 2 input files!\nProgram aborted.\n'
        return

    def main(width):
        return Main(
            kwargs['uid'], kwargs['output'], curr_path, prev_path,
            kwargs['flanking'], width, kwargs['seed'], kwargs['phase']
        ).root

    if kwargs['length']:
        main(kwargs['length'])
    elif kwargs['seed']:
        main(len(kwargs['seed']))
    else:
        root = ''
        for length in range(7, 11):
            root = main(length)

        paths = glob.glob(os.path.join(root, '*', '*.txt'))
        selector = PFMSelector([np.loadtxt(p, skiprows=2) for p in paths
                                if os.path.basename(p).startswith('pfm')])

        opt_len = len(selector.select(MIN_COREL))
        for p in paths:
            if not p.endswith('w%d.txt' % opt_len):
                os.remove(p)


if __name__ == '__main__':
    driver()
