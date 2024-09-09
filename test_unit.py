import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import tqdm
import random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig

from multiply.data import data_generator_unit
from multiply.evaluator import Evaluator_unit
from multiply.loss import logits_to_tokens_loss, loss_fn
from multiply.loss_record import LossRecord, LossRecord_subtask_unit
from multiply.draw import draw_attn_unit


def experiment5_hook(value, hook):
  #print( "In hook", hook_l0_z_name, experiment5_head, experiment5_ablate, experiment5_case, value.shape) # Get [64, 18, 3, 170] = batch_size, num_heads, d_model
    # Ablate the attn_z values into all the batch questions
    experiment5_head=2
    mean_attn_z = torch.mean(value, dim=0, keepdim=True)
    value[:,:,experiment5_head,:] = 0#mean_attn_z[0,:,experiment5_head,:].clone() # Mean ablate
    #value[:,:,experiment5_head,:] = 0 # Zero ablate


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
        self.ds_test = data_generator_unit(batch_size, n_digits, seed, reverse=args.reverse_output, is_train=False)

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
        print('self.model', self.model)
        
        
        self.loss_record_subtask = LossRecord_subtask_unit(
            n_digits=n_digits, vis_digit=0,
            smooth_window=9)
        

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

            tokens, base_muls, use_carry1s, make_carry1s, use_sum10s = next(self.ds_test)
            batch_size = tokens.shape[0]

            # hook_l0_z_name = 'blocks.0.attn.hook_z'
            # experiment5_fwd_hooks = [(hook_l0_z_name, experiment5_hook)]

            # self.model.reset_hooks()
            # logits = self.model.run_with_hooks(tokens, return_type="logits", fwd_hooks=experiment5_fwd_hooks)



            logits = self.model(tokens)
            per_token_train_losses_raw, _ = logits_to_tokens_loss(logits, tokens, self.n_digits, unit_multiplier=True)
            per_token_train_losses = loss_fn(per_token_train_losses_raw)
            train_loss = per_token_train_losses.mean().item()
            
            self.loss_record_subtask.insert(tokens, base_muls, use_carry1s,
                                            make_carry1s, use_sum10s, per_token_train_losses_raw)
            
            
            logits = logits.argmax(-1)

            if self.args.reverse_output:
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
            is_correct = '√' if int(z) == true_z else 'x'
            print('test %2d: %s * %s = %s (%6d), %s,  train_loss: %.4f'%(i, x, y, z, true_z, is_correct, train_loss))
            
        acc_overall = acc_overall / (test_iter * batch_size)
        print('acc_overall: %.4f'%(acc_overall))
        for n in range(self.n_digits+1):
            acc_perdigit[n] = acc_perdigit[n] / (test_iter * batch_size)
            print('acc %d digit: %.4f'%(self.n_digits-n, acc_perdigit[n]))

    def draw_attn(self):
        draw_attn_unit(self.model, n_digits=self.n_digits, n_heads=self.args.n_heads, args=self.args)

    def draw_loss(self):
        self.loss_record_subtask.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Argumentparser')
    parser.add_argument('--seed', type=int, default=114514)

    # Model param  
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--d_vocab', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)

    # Data param  
    parser.add_argument('--n_digits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--reverse_output', type=bool, default=True)
    parser.add_argument('--test_iter', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=2000)

    parser.add_argument('--ckpt', type=str, default='weights/Unit_5digit_1layer_3head_2000epoch_reverse.pt')
    args = parser.parse_args()

    # Model param  
    args.d_model = (512 // args.n_heads ) * args.n_heads  # About 512, and divisible by n_heads
    args.d_head = args.d_model // args.n_heads
    args.d_mlp = 4 * args.d_model

    # Data param
    args.n_ctx = 2 * args.n_digits + 4

    print('args', args)

    evaluator = Eval(
         n_digits=args.n_digits,
         batch_size=args.batch_size,
         seed=args.seed,
         args=args)
    evaluator.test()
    # evaluator.draw_attn()
    evaluator.draw_loss()
