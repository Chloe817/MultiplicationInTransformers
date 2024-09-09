import torch
import torch.nn.functional as F
import transformer_lens.utils as utils


# Loss functions
# Calculate the per-token probability by comparing a batch of prediction "logits" to answer "tokens"
def logits_to_tokens_loss(logits, tokens, n_digits, unit_multiplier=False):

    if unit_multiplier:
        ans_logits = logits[:, -(n_digits+2):-1]
        ans_tokens = tokens[:, -(n_digits+1):]
    else:
        ans_logits = logits[:, -(n_digits*2+1):-1]
        ans_tokens = tokens[:, -(n_digits*2):]

    # Convert raw score (logits) vector into a probability distribution.
    ans_probs = F.log_softmax(ans_logits.to(torch.float64), dim=-1)
    max_indices = torch.argmax(ans_probs, dim=-1)
    
    # Extract values from the ans_probs tensor, based on indices from the ans_tokens tensor
    ans_loss = torch.gather(ans_probs, -1, ans_tokens[:, :, None])[..., 0]
    return ans_loss, max_indices

# Calculate loss as negative of average per-token mean probability
def loss_fn(ans_loss):
    return -ans_loss.mean(0)


class BM_loss():
    def __init__(self, n_digits) -> None:
        # Base-mul-only loss
        # Identify the subset of (simple) tokens that only require BM (not carry) to get the correct answer
        # Array index 0 is the 'Units' digit. Array index 3 is the 'Thousands' digit.
        self.alldigits_loss = []
        self.perdigit_loss = []
        self.perdigit_cases = 0
        self.total_cases = 0

        self.n_digits = n_digits

    # Base Mul AllDigits
    def calculate_oneloss(self, tokens, per_token_losses, base_muls, use_carry1s):
        any_use_carry1s = torch.any(use_carry1s.bool(), dim=1)
        no_use_carry1s = ~ any_use_carry1s

        base_muls_no_carry = base_muls < 10
        base_muls_no_carry = torch.sum(base_muls_no_carry, dim=1)
        base_muls_no_carry = torch.where(base_muls_no_carry > 0, 1, 0 )

        filtered_cases = no_use_carry1s & base_muls_no_carry
        num_cases = utils.to_numpy(torch.sum(filtered_cases))

        answer = 0
        if num_cases > 0 :
            filtered_loss = per_token_losses[:, -self.n_digits:] * filtered_cases[:, None]
            sum_loss = torch.sum(filtered_loss)
            answer = - utils.to_numpy(sum_loss) / num_cases
            answer = answer / self.n_digits    # Approximately align the scale of alldigits_loss to perdigit_loss
        return answer

    def calculate_loss(self, tokens, per_token_losses, base_muls, use_carry1s):
        # Base Mul All Digits
        alldigits_oneloss = self.calculate_oneloss(tokens, per_token_losses, base_muls, use_carry1s)
        self.alldigits_loss.append(alldigits_oneloss)

        base_muls_no_carry = (base_muls < 10).float()
        # Base Mul Per Digit
        perdigit_cases = 0
        for digit_num in range(self.n_digits):
            no_use_carry = 1 - use_carry1s[:, -1-digit_num]
            base_mul_no_carry = base_muls_no_carry[:, -1-digit_num]
            filtered_cases = no_use_carry * base_mul_no_carry

            num_cases = utils.to_numpy(torch.sum(filtered_cases))
            perdigit_cases += num_cases
            self.total_cases += num_cases

            answer = 0
            if num_cases > 0 :
                filtered_loss = per_token_losses[:, -1-digit_num] * filtered_cases
                sum_loss = torch.sum(filtered_loss)
                answer = - utils.to_numpy(sum_loss) / num_cases
            if len(self.perdigit_loss) <= digit_num:
                self.perdigit_loss.append([])
            # Use the previous step's loss. Improves graph
            if (num_cases == 0) & (len(self.perdigit_loss[digit_num]) > 0) :
                answer = self.perdigit_loss[digit_num][-1]
            self.perdigit_loss[digit_num].append(answer)


class CM_loss():
    def __init__(self, n_digits) -> None:
        # Base-mul-with-carry loss
        # Identify the subset of (simple) tokens that only require BM (not carry) to get the correct answer
        # Array index 0 is the 'Units' digit. Array index 3 is the 'Thousands' digit.
        self.alldigits_loss = []
        self.perdigit_loss = []
        self.perdigit_cases = 0
        self.total_cases = 0

        self.n_digits = n_digits

    # Carry Mul AllDigits
    def calculate_oneloss(self, tokens, per_token_losses, base_muls, use_carry1s):
        any_use_carry1s = torch.any(use_carry1s.bool(), dim=1)
        no_use_carry1s = ~ any_use_carry1s

        base_muls_carry = base_muls >= 10
        base_muls_carry = torch.sum(base_muls_carry, dim=1)
        base_muls_carry = torch.where(base_muls_carry > 0, 1, 0 )

        filtered_cases = no_use_carry1s & base_muls_carry
        num_cases = utils.to_numpy(torch.sum(filtered_cases))

        answer = 0
        if num_cases > 0 :
            filtered_loss = per_token_losses[:, -self.n_digits:] * filtered_cases[:, None]
            # print('filtered_loss', filtered_loss)
            sum_loss = torch.sum(filtered_loss)
            answer = - utils.to_numpy(sum_loss) / num_cases
            answer = answer / self.n_digits
        if (num_cases == 0) & (len(self.alldigits_loss) > 0) :
            answer = self.alldigits_loss[-1]
        return answer

    def calculate_loss(self, tokens, per_token_losses, base_muls, use_carry1s):
        # Base Mul All Digits
        alldigits_oneloss = self.calculate_oneloss(tokens, per_token_losses, base_muls, use_carry1s)
        self.alldigits_loss.append(alldigits_oneloss)

        base_muls_carry = (base_muls >= 10).float()

        # Base Mul Per Digit
        perdigit_cases = 0
        for digit_num in range(self.n_digits):
            no_use_carry = 1 - use_carry1s[:, -1-digit_num]
            base_mul_carry = base_muls_carry[:, -1-digit_num]
            filtered_cases = no_use_carry * base_mul_carry

            num_cases = utils.to_numpy(torch.sum(filtered_cases))
            perdigit_cases += num_cases
            self.total_cases += num_cases

            answer = 0
            if num_cases > 0 :
                filtered_loss = per_token_losses[:, -1-digit_num] * filtered_cases
                sum_loss = torch.sum(filtered_loss)
                answer = - utils.to_numpy(sum_loss) / num_cases
            if len(self.perdigit_loss) <= digit_num:
                self.perdigit_loss.append([])
            # Use the previous step's loss. Improves graph
            if (num_cases == 0) & (len(self.perdigit_loss[digit_num]) > 0) :
                answer = self.perdigit_loss[digit_num][-1]
            self.perdigit_loss[digit_num].append(answer)


class Carry_loss():
    def __init__(self, n_digits) -> None:
        # Carry_loss loss
        self.anydigits_loss = []
        self.perdigit_loss = []
        self.perdigit_cases = 0
        self.total_cases = 0

        self.n_digits = n_digits

    def calculate_oneloss(self, tokens, per_token_losses, use_carry1s, x_zeros):
        num_carry = torch.sum(use_carry1s & x_zeros, dim=1)
        filtered_num = torch.where(num_carry != 0, 1, 0 ) # At least OneDigit uses US10

        filtered_indices = torch.nonzero(filtered_num).squeeze()
        filtered_token_losses = per_token_losses[filtered_indices]
        answer = - filtered_token_losses.mean()
        return utils.to_numpy(answer)

    def calculate_loss(self, tokens, per_token_losses, use_carry1s):
        input_xs = tokens[:, 0:self.n_digits]
        x_zeros = (input_xs == 0)
        anydigits_oneloss = self.calculate_oneloss(tokens, per_token_losses, use_carry1s, x_zeros)
        self.anydigits_loss.append(anydigits_oneloss)

        # For each token in the batch, identity the digit columns (e.g. 3) where US10 is used
        perdigit_cases = 0
        for digit_num in range(self.n_digits):
            use_carry = use_carry1s[:, -1-digit_num]
            x_zero = x_zeros[:, -1-digit_num]
            filtered_cases = use_carry & x_zero

            num_cases = utils.to_numpy(torch.sum(filtered_cases))
            perdigit_cases += num_cases
            self.total_cases += num_cases

            answer = 0
            if num_cases > 0 :
                filtered_loss = per_token_losses[:, -1-digit_num] * filtered_cases
                sum_loss = torch.sum(filtered_loss)
                answer = - utils.to_numpy(sum_loss) / num_cases
            if len(self.perdigit_loss)<=digit_num:
                self.perdigit_loss.append([])
            # Use the previous step's loss. Improves graph
            if (num_cases==0) & (len(self.perdigit_loss[digit_num]) > 0) :
                answer = self.perdigit_loss[digit_num][-1]
            self.perdigit_loss[digit_num].append(answer)


class UC_loss():
    def __init__(self, n_digits) -> None:
        # Use Carry loss
        self.anydigits_loss = []
        self.perdigit_loss = []
        self.perdigit_cases = 0
        self.total_cases = 0
        
        self.n_digits = n_digits

    # UC1 AnyDigits (exclude Sum10 and x_digit==0)
    def calculate_oneloss(self, tokens, per_token_losses, use_carry1s, use_sum10s, x_nonzeros):
        num_use_carry1s = torch.sum(use_carry1s, dim=1)
        any_use_carry1s = torch.where(num_use_carry1s != 0, 1, 0 ) # At least one digit uses UC1

        num_sum10s = torch.sum(use_sum10s, dim=1)
        no_sum10s = torch.where(num_sum10s == 0, 1, 0 ) # No digits have Sum10 true

        num_xnonzeros = torch.sum(x_nonzeros, dim=1)
        any_xnonzeros = torch.where(num_xnonzeros != 0, 1, 0 )

        filtered_cases = any_use_carry1s & no_sum10s & any_xnonzeros

        filtered_indices = torch.nonzero(filtered_cases).squeeze()
        filtered_token_losses = per_token_losses[filtered_indices]
        answer = - filtered_token_losses.mean()
        return utils.to_numpy(answer)

    def calculate_loss(self, tokens, per_token_losses, use_carry1s, sum10s):
        # UC1 AnyDigits (exclude Sum10 and x_digit==0)
        input_xs = tokens[:, 0:self.n_digits]
        x_nonzeros = (input_xs > 0)

        anydigits_oneloss = self.calculate_oneloss(tokens, per_token_losses, use_carry1s, sum10s, x_nonzeros)
        self.anydigits_loss.append(anydigits_oneloss)

        # UC1 PerDigit (exclude Sum10 and x_digit==0)
        # For each token in the batch, identity the digit columns (e.g. 3) where UC1 is used on the columns & Sum10 is not true
        perdigit_cases = 0
        for digit_num in range(self.n_digits):
            use_carry = use_carry1s[:, -1-digit_num]
            no_sum10 = 1 - sum10s[:, -1-digit_num]
            x_nonzero = x_nonzeros[:, -1-digit_num]

            filtered_cases = use_carry & no_sum10 & x_nonzero
            num_cases = utils.to_numpy(torch.sum(filtered_cases))
            perdigit_cases += num_cases
            self.total_cases += num_cases

            answer = 0
            if num_cases > 0 :
                filtered_loss = per_token_losses[:, -1-digit_num] * filtered_cases
                sum_loss = torch.sum(filtered_loss)
                answer = - utils.to_numpy(sum_loss) / num_cases
            if len(self.perdigit_loss)<=digit_num:
                self.perdigit_loss.append([])
            # Use the previous step's loss. Improves graph
            if (num_cases==0) & (len(self.perdigit_loss[digit_num]) > 0) :
                answer = self.perdigit_loss[digit_num][-1]
            self.perdigit_loss[digit_num].append(answer)


class US10_loss():
    def __init__(self, n_digits) -> None:
        # Use Sum 10 loss
        self.anydigits_loss = []
        self.perdigit_loss = []
        self.perdigit_cases = 0
        self.total_cases = 0

        self.n_digits = n_digits

    # US10 OneDigit
    # Identity the tokens in the batch where US10 is used at least once over the columns
    def calculate_oneloss(self, tokens, per_token_losses, use_sum10s):
        num_use_sum10s = torch.sum(use_sum10s, dim=1)
        filtered_num_use_sum10s = torch.where( num_use_sum10s != 0, 1, 0 ) # At least OneDigit uses US10

        filtered_indices = torch.nonzero(filtered_num_use_sum10s).squeeze()
        filtered_token_losses = per_token_losses[filtered_indices]
        answer = - filtered_token_losses.mean()
        return utils.to_numpy(answer)

    def calculate_us10_loss(self, tokens, per_token_losses, use_carry1s, use_sum10s):
        # US10 OneDigit
        # Identity the tokens in the batch where US9 is used at least once over the columns
        anydigits_oneloss = self.calculate_oneloss(tokens, per_token_losses, use_sum10s)
        self.anydigits_loss.append(anydigits_oneloss)

        # For each token in the batch, identity the digit columns (e.g. 3) where US10 is used
        perdigit_cases = 0
        for digit_num in range(self.n_digits):
            use_carry = use_carry1s[:, -1-digit_num]
            use_sum10 = use_sum10s[:, -1-digit_num]

            filtered_cases = use_carry & use_sum10
            num_cases = utils.to_numpy(torch.sum(filtered_cases))
            perdigit_cases += num_cases
            self.total_cases += num_cases

            answer = 0
            if num_cases > 0 :
                filtered_loss = per_token_losses[:, -1-digit_num] * filtered_cases
                sum_loss = torch.sum(filtered_loss)
                answer = - utils.to_numpy(sum_loss) / num_cases
            if len(self.perdigit_loss)<=digit_num:
                self.perdigit_loss.append([])
            if (num_cases==0) & (len(self.perdigit_loss[digit_num]) > 0) :
                answer = self.perdigit_loss[digit_num][-1] # Use the previous step's loss. Improves graph
            self.perdigit_loss[digit_num].append(answer)
