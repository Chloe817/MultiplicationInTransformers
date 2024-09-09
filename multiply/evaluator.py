import torch


class Evaluator_unit:
    def __init__(self,
                 n_digits=5,
                 dataset=None,
                 model=None,
                 reverse_output=False):

        # self.args = args
        self.n_digits = n_digits
        self.dataset = dataset
        self.model = model

        self.reverse_output = reverse_output

    def eval(self):
        acc_overall = 0
        acc_perdigit = [0 for i in range(self.n_digits*2)]

        test_iter = 100
        self.model.eval()
        for i in range(test_iter):
            
            tokens, _, _, _, _ = next(self.dataset)
            batch_size = tokens.shape[0]

            logits = self.model(tokens)
            logits = logits.argmax(-1)

            if self.reverse_output:
                logits[:, -self.n_digits-2:-1] = torch.flip(logits[:, -self.n_digits-2:-1], [1])
                tokens[:, -self.n_digits-1:] = torch.flip(tokens[:, -self.n_digits-1:], [1])

            xs, ys, z_pred = 0, 0, 0
            for n in range(self.n_digits):
                xs = xs * 10 + tokens[:, n]
            ys = tokens[:, self.n_digits + 1]

            for n in range(self.n_digits+1):
                col_id = -self.n_digits - 2 + n
                z_pred = z_pred * 10 + logits[:, col_id]
                acc_perdigit[n] += (logits[:, col_id] == tokens[:, col_id+1]).float().sum()
            true_zs = xs * ys
            acc_overall += (z_pred == true_zs).float().sum()

            tokens = tokens[0]
            logits = logits[0]
            x = '%d%d%d%d%d'%(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4])
            y = '%d'%(tokens[6])
            z = '%d%d%d%d%d%d'%(logits[7], *logits[8:-1])
            true_z = int(x) * int(y)
            print('test %d: %s * %s = %s (%6d):'%(i, x, y, z, true_z))
            
        acc_overall = acc_overall / (test_iter * batch_size)
        print('acc_overall: %.4f'%(acc_overall))
        for n in range(self.n_digits+1):
            acc_perdigit[n] = acc_perdigit[n] / (test_iter * batch_size)
            print('acc %d digit: %.4f'%(self.n_digits-n, acc_perdigit[n]))

            
class Evaluator_mult:
    def __init__(self,
                 n_digits=5,
                 dataset=None,
                 model=None,
                 reverse_output=False):

        # self.args = args
        self.n_digits = n_digits
        self.dataset = dataset
        self.model = model

        self.reverse_output = reverse_output
        self.reverse_output = reverse_output

    def eval(self):

        acc_overall = 0
        acc_perdigit = [0 for i in range(self.n_digits*2)]

        test_iter = 100
        self.model.eval()
        for i in range(test_iter):
            
            tokens, _, _, _, _ = next(self.dataset)
            batch_size = tokens.shape[0]

            logits = self.model(tokens)
            logits = logits.argmax(-1)

            if self.reverse_output:
                logits[:, -2*self.n_digits-1:-1] = torch.flip(logits[:, -2*self.n_digits-1:-1], [1])
                tokens[:, -2*self.n_digits:] = torch.flip(tokens[:, -2*self.n_digits:], [1])

            xs, ys, z_pred = 0, 0, 0
            for n in range(self.n_digits):
                xs = xs * 10 + tokens[:, n]
                ys = ys * 10 + tokens[:, n + self.n_digits + 1]
            for n in range(self.n_digits*2):
                col_id = -2*self.n_digits - 1 + n
                # print(n, col_id, logits.shape, tokens.shape)
                z_pred = z_pred * 10 + logits[:, col_id]
                acc_perdigit[n] += (logits[:, col_id] == tokens[:, col_id+1]).float().sum()
            true_zs = xs * ys

            acc_overall += (z_pred == true_zs).float().sum()
            # print('xs', xs[0:5], xs.shape)
            # print('ys', ys[0:5], ys.shape)
            # print('z_pred', z_pred[0:5], z_pred.shape)
            # print('true_zs', true_zs[0:5], true_zs.shape)

            tokens = tokens[0]
            logits = logits[0]
            x = '%d%d%d%d%d'%(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4])
            y = '%d%d%d%d%d'%(tokens[6], tokens[7], tokens[8], tokens[9], tokens[10])
            z = '%d%d%d%d%d%d%d%d%d%d'%(logits[11], *logits[12:-1])
            true_z = int(x) * int(y)
            print('test %d: %s * %s = %s (%10d):'%(i, x, y, z, true_z))
    
            # x = '%d%d%d'%(tokens[0], tokens[1], tokens[2])
            # y = '%d%d%d'%(tokens[4], tokens[5], tokens[6])
            # z = '%d%d%d%d%d%d'%(logits[7], *logits[8:-1])
            # true_z = int(x) * int(y)
            # print('test %d: %s * %s = %s (%6d):'%(i, x, y, z, true_z))
            
        acc_overall = acc_overall / (test_iter * batch_size)
        print('acc_overall: %.4f'%(acc_overall))
        for n in range(self.n_digits*2):
            acc_perdigit[n] = acc_perdigit[n] / (test_iter * batch_size)
            print('acc %d digit: %.4f'%(self.n_digits*2-n-1, acc_perdigit[n]))
            