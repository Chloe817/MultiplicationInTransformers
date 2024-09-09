
import torch
import transformer_lens.utils as utils
from multiply.utils import tokens_to_int_ds, tokens_to_int_unit


# Special tokens
TIMES_INDEX = 10
EQUALS_INDEX = 11


def data_generator_unit(batch_size, n_digits, seed, reverse=False, is_train=False):
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, 2*n_digits+4)).to(torch.int64)
        x = torch.randint(0, 10, (batch_size, n_digits))
        y = torch.randint(0, 10, (batch_size, 1))

        # x[0, 0] = 1
        # x[0, 1] = 2
        # x[0, 2] = 3
        # x[0, 3] = 2
        # x[0, 4] = 4
        # y[0, 0] = 2

        # x[1, 0] = 4
        # x[1, 1] = 0
        # x[1, 2] = 7
        # x[1, 3] = 0
        # x[1, 4] = 6
        # y[1, 0] = 6

        # x[2, 0] = 4
        # x[2, 1] = 7
        # x[2, 2] = 1
        # x[2, 3] = 3
        # x[2, 4] = 4
        # y[2, 0] = 9

        if is_train:
            x_flat = x.view(-1)
            num_elements_to_modify = int(0.1 * x.numel())
            indices_to_modify = torch.randperm(x_flat.numel())[:num_elements_to_modify]
            x_flat[indices_to_modify] = 0

        batch[:, :n_digits] = x
        batch[:, n_digits] = TIMES_INDEX
        batch[:, n_digits+1] = y.squeeze(-1)
        batch[:, n_digits+2:] = EQUALS_INDEX

        res = tokens_to_int_unit(x, y, n_digits)
        batch[:, n_digits+3:] = res

        # These attributes are used for testing the model training progress
        base_muls = torch.zeros((batch_size,n_digits)).to(torch.int64)
        make_carry1s = torch.zeros((batch_size,n_digits)).to(torch.int64)
        use_carry1s = torch.zeros((batch_size,n_digits)).to(torch.int64)
        carry1s = torch.zeros((batch_size,n_digits)).to(torch.int64)
        use_sum10s = torch.zeros((batch_size,n_digits)).to(torch.int64)

        # generate the multiply question answers & other info for testing
        for i in range(n_digits):
            test_col = n_digits-1-i

            base_mul = batch[:, n_digits-1-i] * batch[:, n_digits+1]
            base_muls[:, test_col] = base_mul

            if i > 0:
                carry1s[:, test_col] = make_carry1s[:, test_col+1]
                use_sum10s[:, test_col] = (base_mul%10 + carry1s[:, test_col]) > 9

            digit_sum = base_mul + carry1s[:, test_col]
            make_carry1s[:, test_col] = digit_sum // 10

            batch[:, -1-i] = (digit_sum % 10)

        use_carry1s = (carry1s > 0).long()
        batch[:, -1-n_digits] = make_carry1s[:, 0]
        
        if reverse:
            batch[:, -1-n_digits:] = torch.flip(batch[:, -1-n_digits:], [1])

        yield batch.cuda(), base_muls.cuda(), use_carry1s.cuda(), make_carry1s.cuda(), use_sum10s.cuda()



# Define "iterator" data generator function. Invoked using next().
# Batch entries are in format XXXXX*YYYYY=ZZZZZZ e.g. 55003*80002=4,400,350,006
def data_generator_mult(batch_size, n_digits, seed, reverse=False, is_train=False):
    torch.manual_seed(seed)
    while True:
        #generate a batch of addition questions (answers calculated below)
        batch = torch.zeros((batch_size, 4*n_digits+2)).to(torch.int64)
        x = torch.randint(0, 10, (batch_size, n_digits))
        y = torch.randint(0, 10, (batch_size, n_digits))

        # y[:, 0:4] = 0

        # # augment samples 
        proportion = 0.45
        if is_train:
            num_elements_to_modify = int(y.shape[0] * proportion)
            indices_to_modify = torch.randperm(y.shape[0])[:num_elements_to_modify]
            mask = torch.ones(y.shape)
            mask[indices_to_modify] = 0

            inds = torch.randint(0, 5, (num_elements_to_modify,))
            mask[indices_to_modify, inds] = 1
            y = y * mask
            # print('proportion', proportion)
            # print('y', y)
        else:
            print('proportion', proportion)

        batch[:, :n_digits] = x
        batch[:, n_digits] = TIMES_INDEX
        batch[:, 1+n_digits:1+n_digits*2] = y
        batch[:, 1+n_digits*2] = EQUALS_INDEX

        res = tokens_to_int_ds(x, y, n_digits)
        batch[:, 2+n_digits*2:] = res

        # These attributes are used for testing the model training progress
        base_muls = torch.zeros((batch_size, n_digits, n_digits)).to(torch.int64)
        make_carry1s = torch.zeros((batch_size, n_digits, n_digits)).to(torch.int64)
        carry1s = torch.zeros((batch_size, n_digits, n_digits)).to(torch.int64)
        use_sum10s = torch.zeros((batch_size, n_digits, n_digits)).to(torch.int64)
        multi_results = torch.zeros((batch_size, n_digits, n_digits*2)).to(torch.int64)

        # generate the multiply question answers & other info for testing
        for i in range(n_digits): # for y
            for j in range(n_digits): # for x
            
                col_y = n_digits-1-i # the column id for y
                col_x = n_digits-1-j # the column id for x

                base_mul = x[:, n_digits-1-j] * y[:, n_digits-1-i]
                base_muls[:, i, col_x] = base_mul

                if j > 0:
                    carry1s[:, i, col_x] = make_carry1s[:, i, col_x+1]
                    use_sum10s[:, i, col_x] = (base_mul%10 + carry1s[:, i, col_x]) > 9

                digit_sum = base_mul + carry1s[:, i, col_x]
                make_carry1s[:, i, col_x] = digit_sum // 10
                multi_results[:, i, col_x + col_y+1] = digit_sum % 10

            multi_results[:, i, col_x + col_y] = make_carry1s[:, i, col_x]
        
        # final result
        final_carrys = torch.zeros((batch_size)).to(torch.int64)
        for i in range(2*n_digits): # for x
            count = torch.zeros((batch_size,)).to(torch.int64)
            for j in range(n_digits): # for y
                count += multi_results[:, n_digits-j-1, 2*n_digits-i-1]
            batch[:, -1-i] = (count + final_carrys) % 10
            final_carrys = (count + final_carrys) // 10

        use_carry1s = (carry1s > 0).long()

        if reverse:
            batch[:, -2*n_digits:] = torch.flip(batch[:, -2*n_digits:], [1])
        yield batch.cuda(), base_muls.cuda(), use_carry1s.cuda(), make_carry1s.cuda(), use_sum10s.cuda()


# Embedding / Unembedding
def tokens_to_string(tokens, n_digits):
    tokens = utils.to_numpy(tokens)
    x = "".join([str(i) for i in tokens[:n_digits]])
    y = "".join([str(i) for i in tokens[n_digits+1:n_digits*2+1]])
    z = "".join([str(i) for i in tokens[n_digits*2+2:]])
    return f"{x}+{y}={z}"


def string_to_tokens(string, batch=False):
    lookup = {str(i):i for i in range(10)}
    lookup['+']=TIMES_INDEX
    lookup['=']=EQUALS_INDEX
    tokens = [lookup[i] for i in string if i not in '\n ']
    if batch:
        return torch.tensor(tokens)[None, :]
    else:
        return torch.tensor(tokens)

