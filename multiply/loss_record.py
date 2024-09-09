import torch
import numpy as np
import transformer_lens.utils as utils

from multiply.loss import BM_loss, CM_loss, UC_loss, US10_loss, Carry_loss
from multiply.utils import smooth, draw
from multiply.analysis import line, lines, graph_perdigit


class LossRecord:
    def __init__(self,
                 n_digits=5,
                 vis_digit=0,
                 smooth_window=3):

        self.n_digits = n_digits
        self.vis_digit = vis_digit
        self.smooth_window = smooth_window

        self.train_losses_list = []
        self.per_token_train_losses_list = []

    def insert(self, per_token_train_losses, train_loss):

        self.per_token_train_losses_list.append(utils.to_numpy(per_token_train_losses))
        self.train_losses_list.append(train_loss.item())

    def analyse_all_cases(self, unit_multiplier=False):
        
        train_losses_list = np.array(self.train_losses_list)
        per_token_losses = np.stack(self.per_token_train_losses_list, axis=1)

        print('smooth_window', self.smooth_window)
        # smooth loss
        train_losses_list = smooth(train_losses_list, self.smooth_window)
        for i in range(per_token_losses.shape[0]):
            per_token_losses[i] = smooth(per_token_losses[i], self.smooth_window)
        print('train_losses_list', train_losses_list.shape)
        print('per_token_losses1', per_token_losses.shape)


        title_suffix = ' Loss Curves for ' + str(self.n_digits) + ' digit multiply'
        # line(train_losses_list, title=title_suffix)

        answer_digits = self.n_digits + 1 if unit_multiplier else self.n_digits *2
        print('answer_digits', answer_digits)
        lines([per_token_losses[i] for i in range(answer_digits)]+[train_losses_list],
            labels = [f'A{answer_digits-1-i}' for i in range(answer_digits)]+['All'],
            title='Per digit'+title_suffix,
            all_epochs=True)

        # lines([per_token_losses[i] for i in range(answer_digits)]+[train_losses_list],
        #     labels = [f'A{answer_digits-1-i}' for i in range(answer_digits)]+['All'],
        #     title='Per digit'+title_suffix,
        #     all_epochs=True,
        #     log_y=True)

        for i in range(answer_digits):
            print('Final Loss for A' + str(self.n_digits-i) + ' is ', per_token_losses[-1, i])


class LossRecord_subtask_unit:
    def __init__(self,
                 n_digits=5,
                 vis_digit=0,
                 smooth_window=3):

        self.n_digits = n_digits
        self.vis_digit = vis_digit
        self.smooth_window = smooth_window

        # loss analyse
        self.bm_loss = BM_loss(n_digits)
        self.cm_loss = CM_loss(n_digits)
        self.carry_loss = Carry_loss(n_digits)
        self.uc_loss = UC_loss(n_digits)
        self.us10_loss = US10_loss(n_digits)
        # print('model', model)

    def insert(self, tokens, base_muls, use_carry1s, make_carry1s, use_sum10s, per_token_train_losses_raw):

        self.bm_loss.calculate_loss(tokens, per_token_train_losses_raw, base_muls, use_carry1s)
        self.cm_loss.calculate_loss(tokens, per_token_train_losses_raw, base_muls, use_carry1s)
        self.carry_loss.calculate_loss(tokens, per_token_train_losses_raw, use_carry1s)

        self.uc_loss.calculate_loss(tokens, per_token_train_losses_raw, use_carry1s, use_sum10s)
        self.us10_loss.calculate_us10_loss(tokens, per_token_train_losses_raw, use_carry1s, use_sum10s)

    def compute_total_cases(self):
        bm_total_cases = self.bm_loss.total_cases
        cm_total_cases = self.cm_loss.total_cases
        uc1_total_cases = self.uc_loss.total_cases
        us10_total_cases = self.us10_loss.total_cases
        carry_total_cases = self.carry_loss.total_cases
        print('\nbm_total_cases', bm_total_cases)
        print('cm_total_cases', cm_total_cases)
        print('carry_total_cases', carry_total_cases)
        print('uc1_total_cases', uc1_total_cases)
        print('us10_total_cases', us10_total_cases)
        total_case = bm_total_cases + cm_total_cases + uc1_total_cases + us10_total_cases + carry_total_cases
        print('total_case', total_case)
        print()
        self.total_case = total_case
        return total_case
    
    def show(self):
        print('smooth_window', self.smooth_window)
        total_case = self.compute_total_cases()
        draw(self.bm_loss.total_cases, total_case, self.n_digits, self.bm_loss.alldigits_loss, 
             self.bm_loss.perdigit_loss, title='Base Mul Loss', smooth_window=self.smooth_window)
        draw(self.cm_loss.total_cases, total_case, self.n_digits, self.cm_loss.alldigits_loss, 
             self.cm_loss.perdigit_loss, title='Carry Mul Loss', smooth_window=self.smooth_window)
        draw(self.carry_loss.total_cases, total_case, self.n_digits, self.carry_loss.anydigits_loss,
             self.carry_loss.perdigit_loss, title='Carry Loss', smooth_window=self.smooth_window)
        draw(self.uc_loss.total_cases, total_case, self.n_digits, self.uc_loss.anydigits_loss,
             self.uc_loss.perdigit_loss, title='Use Carry Loss', smooth_window=self.smooth_window)
        draw(self.us10_loss.total_cases, total_case, self.n_digits, self.us10_loss.anydigits_loss,
             self.us10_loss.perdigit_loss, title='Use Carry10 Loss', smooth_window=self.smooth_window)

class LossRecord_subtask_mult:
    def __init__(self,
                 n_digits=5,
                 vis_digit=0,
                 smooth_window=3):
        self.n_digits = n_digits
        self.vis_digit = vis_digit
        self.smooth_window = smooth_window

        # loss analyse
        self.bm_loss = BM_loss(n_digits)
        self.cm_loss = CM_loss(n_digits)
        self.carry_loss = Carry_loss(n_digits)
        self.uc_loss = UC_loss(n_digits)
        self.us10_loss = US10_loss(n_digits)

    def insert(self, tokens, base_muls, use_carry1s, make_carry1s, use_sum10s, per_token_train_losses_raw):
        base_muls = base_muls[:, self.vis_digit, :]
        use_carry1s = use_carry1s[:, self.vis_digit, :]
        make_carry1s = make_carry1s[:, self.vis_digit, :]
        use_sum10s = use_sum10s[:, self.vis_digit, :]
        per_token_train_losses_raw = per_token_train_losses_raw[:, self.n_digits-self.vis_digit-1:self.n_digits*2-self.vis_digit]

        self.bm_loss.calculate_loss(tokens, per_token_train_losses_raw, base_muls, use_carry1s)
        self.cm_loss.calculate_loss(tokens, per_token_train_losses_raw, base_muls, use_carry1s)
        self.carry_loss.calculate_loss(tokens, per_token_train_losses_raw, use_carry1s)

        self.uc_loss.calculate_loss(tokens, per_token_train_losses_raw, use_carry1s, use_sum10s)
        self.us10_loss.calculate_us10_loss(tokens, per_token_train_losses_raw, use_carry1s, use_sum10s)

    def compute_total_cases(self):
        bm_total_cases = self.bm_loss.total_cases
        cm_total_cases = self.cm_loss.total_cases
        uc1_total_cases = self.uc_loss.total_cases
        us10_total_cases = self.us10_loss.total_cases
        carry_total_cases = self.carry_loss.total_cases
        print('\nbm_total_cases', bm_total_cases)
        print('cm_total_cases', cm_total_cases)
        print('carry_total_cases', carry_total_cases)
        print('uc1_total_cases', uc1_total_cases)
        print('us10_total_cases', us10_total_cases)
        total_case = bm_total_cases + cm_total_cases + uc1_total_cases + us10_total_cases + carry_total_cases
        print('total_case', total_case)
        print()
        self.total_case = total_case
        return total_case
    
    def show(self):
        total_case = self.compute_total_cases()
        draw(self.bm_loss.total_cases, total_case, self.n_digits, self.bm_loss.alldigits_loss, 
             self.bm_loss.perdigit_loss, title='Base Mul Loss', smooth_window=self.smooth_window)
        draw(self.cm_loss.total_cases, total_case, self.n_digits, self.cm_loss.alldigits_loss, 
             self.cm_loss.perdigit_loss, title='Carry Mul Loss', smooth_window=self.smooth_window)
        draw(self.carry_loss.total_cases, total_case, self.n_digits, self.carry_loss.anydigits_loss,
             self.carry_loss.perdigit_loss, title='Carry Loss', smooth_window=self.smooth_window)
        draw(self.uc_loss.total_cases, total_case, self.n_digits, self.uc_loss.anydigits_loss,
             self.uc_loss.perdigit_loss, title='Use Carry Loss', smooth_window=self.smooth_window)
        draw(self.us10_loss.total_cases, total_case, self.n_digits, self.us10_loss.anydigits_loss,
             self.us10_loss.perdigit_loss, title='Use Carry10 Loss', smooth_window=self.smooth_window)
