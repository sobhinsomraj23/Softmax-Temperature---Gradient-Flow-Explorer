import numpy as np

# compute log(sum(exp(logits))) in numericallyy stable manner
# for higher logit values like 1000, np.exp(1000) becomes infinity hence compute log sum exp
# The exponential space is being recentered by preserving the relative differencess
def log_sum_exp(logits):
    max_logit =np.max(logits)
    shifted_logits = logits - max_logit

    return max_logit + np.log(np.sum(np.exp(shifted_logits)))

