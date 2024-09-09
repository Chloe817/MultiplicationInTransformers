import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import tqdm
import random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig

from multiply.data import data_generator_mult
from multiply.evaluator import Evaluator_mult
from multiply.loss import logits_to_tokens_loss, loss_fn
from multiply.loss_record import LossRecord, LossRecord_subtask_mult
from multiply.draw import draw_attn_mult


class Eval:
    def __init__(self,
                 n_digits=5,
                 batch_size=64,
                 seed=1,
                 args=None):

        self.seed = seed
        self.n_digits = n_digits
        self.batch_size = batch_size
        self.args = args
        
        self.set_seed(seed)

        # Initialise the data generator
        self.ds_test = data_generator_mult(batch_size, n_digits, seed, reverse=args.reverse_output, is_train=False)

        cfg = HookedTransformerConfig(
            n_layers = args.n_layers,
            n_heads = args.n_heads,
            d_model = args.d_model,
            d_head = args.d_head,
            d_mlp = args.d_mlp,
            act_fn = 'relu',
            normalization_type = 'LN',
            d_vocab=args.d_vocab,
            d_vocab_out=args.d_vocab,
            n_ctx=args.n_ctx,
            init_weights = True,
            device="cuda",
            seed = seed,
        )
        model = HookedTransformer(cfg)

        state = torch.load(args.ckpt)
        model.load_state_dict(state)
        model.eval()

        self.model = model.cuda()

    def set_seed(self, seed=2):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print('set random seed:', seed)

    def test(self):
        acc_overall = 0
        acc_perdigit = [0 for i in range(self.n_digits*2)]

        test_iter = self.args.test_iter
        for i in range(test_iter):

            tokens, _, _, _, _ = next(self.ds_test)
            batch_size = tokens.shape[0]

            logits = self.model(tokens)
            per_token_train_losses_raw, _ = logits_to_tokens_loss(logits, tokens, self.n_digits, unit_multiplier=False)
            per_token_train_losses = loss_fn(per_token_train_losses_raw)
            train_loss = per_token_train_losses.mean()

            logits = logits.argmax(-1)
            if self.args.reverse_output:
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
            is_correct = '√' if int(z) == true_z else 'x'
            print('test %2d: %s * %s = %s (%6d), %s,  train_loss: %.4f'%(i, x, y, z, true_z, is_correct, train_loss))
    
            
        acc_overall = acc_overall / (test_iter * batch_size)
        print('acc_overall: %.4f'%(acc_overall))
        for n in range(self.n_digits*2):
            acc_perdigit[n] = acc_perdigit[n] / (test_iter * batch_size)
            print('acc %d digit: %.4f'%(self.n_digits*2-n-1, acc_perdigit[n]))


    def show(self):
        # self.loss_record.analyse_all_cases(unit_multiplier=False)
        # self.loss_record_subtask.show()
        draw_attn_mult(self.model, n_digits=self.n_digits, n_heads=self.args.n_heads, args=self.args)



if __name__ == '__main__':
    parser = ArgumentParser(description='Argumentparser')
    parser.add_argument('--seed', type=int, default=129000)

    # Model param  
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_vocab', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)

    # Data param  
    parser.add_argument('--n_digits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--reverse_output', type=bool, default=True)
    parser.add_argument('--test_iter', type=int, default=100)

    parser.add_argument('--ckpt', type=str, default='weights/Mult_5digit_4layer_8head_200000epoch_reverse_noaug.pt')
    args = parser.parse_args()

    # Model param  
    args.d_model = (512 // args.n_heads ) * args.n_heads  # About 512, and divisible by n_heads
    args.d_head = args.d_model // args.n_heads
    args.d_mlp = 4 * args.d_model

    # Data param
    args.n_ctx = 4 * args.n_digits + 2

    print('args', args)

    evaluator = Eval(
         n_digits=args.n_digits,
         batch_size=args.batch_size,
         seed=args.seed,
         args=args)
    evaluator.test()
