import numpy as np
from multiply.analysis import line, lines, graph_perdigit
import torch


def tokens_to_int_unit(x_tokens, y_tokens, n_digits=3):
    # print('x_tokens', x_tokens[0])
    # print('y_tokens', y_tokens[0])
    x = torch.zeros(x_tokens.shape[0]).long()
    y = torch.zeros(x_tokens.shape[0]).long()

    for i in range(n_digits):
        x = x*10 + x_tokens[:, i]
    y = y_tokens[:, 0]
    z = x * y

    res = torch.zeros((x.shape[0], n_digits+1)).to(torch.int64)
    for i in range(n_digits+1):
        res[:, -i-1] = z % 10
        z = z // 10
    return res


def tokens_to_int_ds(x_tokens, y_tokens, n_digits=3):
    # print('x_tokens', x_tokens[0])
    # print('y_tokens', y_tokens[0])
    x = torch.zeros(x_tokens.shape[0]).long()
    y = torch.zeros(x_tokens.shape[0]).long()

    for i in range(n_digits):
        x = x*10 + x_tokens[:, i]
        y = y*10 + y_tokens[:, i]
    z = x * y

    res = torch.zeros((x.shape[0], n_digits*2)).to(torch.int64)
    for i in range(2*n_digits):
        res[:, -i-1] = z % 10
        z = z // 10
    return res


def smooth(x, window_size=1):
    x = np.array(x)
    kernel = np.ones(window_size) / window_size
    x_smooth = np.convolve(x, kernel, mode='same')
    if window_size > 1:
        x_smooth[0:window_size//2] = x[0:window_size//2]
        x_smooth[-window_size//2:] = x[-window_size//2:]
    return x_smooth


def draw(task_case, total_case, n_digits, alldigits_loss, perdigit_loss, title='', smooth_window=3):
    print()
    perc = (int)(100 * task_case / total_case)
    print(title + ' (' + str(task_case) + ' cases, ' + str(perc) + '%)')

    alldigits_loss = np.array(alldigits_loss)
    perdigit_loss = np.array(perdigit_loss)
    if smooth_window > 1:        
        alldigits_loss = smooth(alldigits_loss, smooth_window)
        # print('alldigits_loss', alldigits_loss.shape, 'perdigit_loss', perdigit_loss.shape)
        for i in range(perdigit_loss.shape[0]):
            perdigit_loss[i] = smooth(perdigit_loss[i], smooth_window)

    # line(alldigits_loss, title='AllDigits '+ title)
    graph_perdigit(perdigit_loss, n_digits, title, showlog=False, all_epochs=True)
