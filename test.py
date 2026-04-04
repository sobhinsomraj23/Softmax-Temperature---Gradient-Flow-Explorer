import numpy as np


def softmax(logits):
    """
    Stable softmax (reused for gradient computation)
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def softmax_jacobian(probs):
    """
    Compute the Jacobian matrix of softmax.

    Args:
        probs (np.ndarray): Softmax probabilities (n,)

    Returns:
        np.ndarray: Jacobian matrix (n x n)
    """
    n = probs.shape[0]
    jacobian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i][j] = probs[i] * (1 - probs[i])
            else:
                jacobian[i][j] = -probs[i] * probs[j]

    return jacobian


def cross_entropy_loss(probs, target_index):
    """
    Compute cross-entropy loss.

    Args:
        probs (np.ndarray): Softmax probabilities
        target_index (int): Correct class index

    Returns:
        float: Loss value
    """
    return -np.log(probs[target_index])


def cross_entropy_gradient(probs, target_index):
    """
    Gradient of cross-entropy loss w.r.t logits.

    Key result: grad = probs - one_hot(target)

    Args:
        probs (np.ndarray): Softmax probabilities
        target_index (int): Correct class index

    Returns:
        np.ndarray: Gradient vector
    """
    grad = probs.copy()
    grad[target_index] -= 1
    return grad


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1])
    target = 0  # correct class

    probs = softmax(logits)

    print("Probabilities:", probs)

    print("\nSoftmax Jacobian:")
    print(softmax_jacobian(probs))

    print("\nCross-Entropy Loss:")
    print(cross_entropy_loss(probs, target))

    print("\nGradient (Softmax + Cross-Entropy):")
    print(cross_entropy_gradient(probs, target))