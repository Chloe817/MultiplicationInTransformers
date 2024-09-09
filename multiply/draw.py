import torch
import circuitsvis as cv
import transformer_lens.utils as utils

from multiply.data import data_generator_unit, data_generator_mult


def draw_attn_unit(model, n_digits=5, n_heads=5, args=None):

    ds = data_generator_unit(batch_size=128, n_digits=n_digits, seed=1)
    tokens, _, _, _, _ = next(ds)

    tokens_str = []
    for i in range(tokens.shape[0]):
        one_token_str = []
        for j in tokens[i]:
            one_token_str.append(str(utils.to_numpy(j)))
        one_token_str[5] = '*'
        one_token_str[7] = '='
        tokens_str.append(one_token_str)

    print('tokens_str', tokens_str)
    print('tokens', tokens.shape)
    # Refer https://github.com/callummcdougall/CircuitsVis/blob/main/python/circuitsvis/circuitsvis_demo.ipynb

    original_logits, cache = model.run_with_cache(tokens)
    html_object = cv.attention.from_cache(
        cache = cache,
        tokens = tokens_str, # list of list of strings
        return_mode = "html",
    )

    # Create a CoLab file containing the 6 question attention pattern(s) in HTML
    if args.reverse_output:
        htmlfilename = "draw/AttnMap_Unit_%dDigits_%dLayers_%dheads_%dEpoch_reverse.html"%(n_digits, args.n_layers, n_heads, args.n_epochs)
    else:
        htmlfilename = "draw/AttnMap_Unit_%dDigits_%dLayers_%dheads_%dEpoch.html"%(n_digits, args.n_layers, n_heads, args.n_epochs)
    with open(htmlfilename, "w") as f:
        f.write(html_object.data)


def draw_attn_mult(model, n_digits=5, n_heads=5, args=None):

    ds = data_generator_mult(batch_size=2, n_digits=n_digits, seed=1)
    tokens, _, _, _, _ = next(ds)

    tokens_str = []
    for i in range(tokens.shape[0]):
        one_token_str = []
        for j in tokens[i]:
            one_token_str.append(str(utils.to_numpy(j)))
        one_token_str[5] = '*'
        one_token_str[11] = '='
        tokens_str.append(one_token_str)

    print('tokens_str', tokens_str)
    print('tokens', tokens.shape)
    # Refer https://github.com/callummcdougall/CircuitsVis/blob/main/python/circuitsvis/circuitsvis_demo.ipynb

    original_logits, cache = model.run_with_cache(tokens)
    html_object = cv.attention.from_cache(
        cache = cache,
        tokens = tokens_str, # list of list of strings
        return_mode = "html",
    )

    if args.reverse_output:
        htmlfilename = "draw/AttnMap_Mult_%dDigits_%dLayers_%dheads_%dEpoch_reverse.html"%(n_digits, args.n_layers, n_heads, args.n_epochs)
    else:
        htmlfilename = "draw/AttnMap_Mult_%dDigits_%dLayers_%dheads_%dEpoch.html"%(n_digits, args.n_layers, n_heads, args.n_epochs)
    with open(htmlfilename, "w") as f:
        f.write(html_object.data)