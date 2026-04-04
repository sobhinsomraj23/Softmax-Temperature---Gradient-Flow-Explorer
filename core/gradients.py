import numpy as np

# Stable softmax
def softmax(logits):
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

# compute jacobian matrix of softmax, takes args as softmax probabilities and gives jacobian matrix as o/p
def softmax_jacobian(probs):
    n = probs.shape[0]
    jacobian = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # derivative of probability of a class / derivate of logit of that class = y*(1-y)
                jacobian[i][j] = probs[i] * (1-probs[i])
            else:
                jacobian[i][j] = -probs[i] * probs[j]

    return jacobian

# computes cross entropy loss
def cross_entropy_loss(probs, target_index):
    return -np.log(probs[target_index])

# gradient of cross entropy loss w.r.t logits
# grad = probs - one hot(target) which gives gradient vector
def cross_entropy_gradient(probs, target_index):
    grad = probs.copy()
    grad[target_index] -=1
    return grad

if __name__ =="__main__":
    logits=np.array([2.0, 1.0, 0.1])
    target = 0

    probs = softmax(logits)

    print("Probabilities:", probs)

    print("\nSoftmax Jacobian:")
    print(softmax_jacobian(probs))

    print("\nCross Entropy Loss:")
    print(cross_entropy_loss(probs, target))

    print("\nGradient (Softmax + Cross-Entropy):")
    print(cross_entropy_gradient(probs, target))