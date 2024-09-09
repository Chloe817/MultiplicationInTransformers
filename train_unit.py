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


class Trainer:
    def __init__(self,
                 n_digits=5,
                 batch_size=64,
                 lr=0.0001,
                 visualized_digit=0,
                 smooth_window=1,
                 seed=1,
                 args=None):

        self.seed = seed
        self.n_digits = n_digits
        self.batch_size = batch_size
        self.vis_digit = visualized_digit
        self.args = args
        
        self.set_seed(seed)

        # Initialise the data generator
        self.ds = data_generator_unit(batch_size, n_digits, seed, reverse=args.reverse_output, is_train=True)
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
        # print('model', model)
        self.optimizer = optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=0.1,
                                betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1))
        self.model = model.cuda()

        self.evaluator = Evaluator_unit(self.n_digits, self.ds_test, self.model, args.reverse_output)
        self.loss_record = LossRecord(
            n_digits=n_digits, vis_digit=visualized_digit,
            smooth_window=smooth_window)
        self.loss_record_subtask = LossRecord_subtask_unit(
            n_digits=n_digits, vis_digit=visualized_digit,
            smooth_window=smooth_window)

    def set_seed(self, seed=2):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print('set random seed:', seed)

    def train(self):

        self.model.train()
        for epoch in tqdm.tqdm(range(self.args.n_epochs)):

            tokens, base_muls, use_carry1s, make_carry1s, use_sum10s = next(self.ds)

            logits = self.model(tokens)
            per_token_train_losses_raw, _ = logits_to_tokens_loss(logits, tokens, self.n_digits, unit_multiplier=True)
            per_token_train_losses = loss_fn(per_token_train_losses_raw)

            train_loss = per_token_train_losses.mean()
            train_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.loss_record.insert(per_token_train_losses, train_loss)
            self.loss_record_subtask.insert(tokens, base_muls, use_carry1s,
                                            make_carry1s, use_sum10s, per_token_train_losses_raw)

        if self.args.reverse_output:
            save_path = 'weights/Unit_%ddigit_%dlayer_%dhead_%depoch_reverse.pt'%(self.n_digits, self.args.n_layers, self.args.n_heads, self.args.n_epochs)
        else:
            save_path = 'weights/Unit_%ddigit_%dlayer_%dhead_%depoch.pt'%(self.n_digits, self.args.n_layers, self.args.n_heads, self.args.n_epochs)
        torch.save(self.model.state_dict(), save_path)

        self.evaluator.eval()

    def show(self):

        self.loss_record.analyse_all_cases(unit_multiplier=True)
        self.loss_record_subtask.show()
        draw_attn_unit(self.model, n_digits=self.n_digits, n_heads=self.args.n_heads, args=self.args)



if __name__ == '__main__':
    parser = ArgumentParser(description='Argumentparser')
    parser.add_argument('--seed', type=int, default=114514)

    # Model param  
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--d_vocab', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)

    # Data param  
    parser.add_argument('--n_digits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualized_digit', type=int, default=0)

    parser.add_argument('--smooth_window', type=int, default=5)
    parser.add_argument('--reverse_output', type=bool, default=False)

    # Optimizer param
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--n_epochs', type=int, default=2000)

    args = parser.parse_args()

    # Model param  
    args.d_model = (512 // args.n_heads ) * args.n_heads  # About 512, and divisible by n_heads
    args.d_head = args.d_model // args.n_heads
    args.d_mlp = 4 * args.d_model

    # Data param
    args.n_ctx = 2 * args.n_digits + 4

    print('args', args)

    trainer = Trainer(
         n_digits=args.n_digits,
         batch_size=args.batch_size,
         lr=args.lr,
         visualized_digit=args.visualized_digit,
         smooth_window=args.smooth_window,
         seed=args.seed,
         args=args)
    trainer.train()
    # trainer.show()
