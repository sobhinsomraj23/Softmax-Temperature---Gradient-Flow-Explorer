import numpy as np


def log_sum_exp(logits):
    """
    Compute log(sum(exp(logits))) in a numerically stable way.

    Args:
        logits (np.ndarray): Input array

    Returns:
        float: log-sum-exp value
    """
    max_logit = np.max(logits)
    shifted_logits = logits - max_logit

    return max_logit + np.log(np.sum(np.exp(shifted_logits)))


def stable_softmax(logits):
    """
    Stable softmax using log-sum-exp trick.

    Args:
        logits (np.ndarray): Input array

    Returns:
        np.ndarray: Softmax probabilities
    """
    lse = log_sum_exp(logits)
    return np.exp(logits - lse)


def naive_softmax(logits):
    """
    Unstable (naive) softmax for comparison.

    Args:
        logits (np.ndarray): Input array

    Returns:
        np.ndarray: Softmax probabilities
    """
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


if __name__ == "__main__":
    # Example showing instability
    logits = np.array([1000, 1001, 1002])

    print("Naive Softmax:")
    try:
        print(naive_softmax(logits))
    except Exception as e:
        print("Failed due to:", e)

    print("\nStable Softmax:")
    print(stable_softmax(logits))

    print("\nLog-Sum-Exp Value:")
    print(log_sum_exp(logits))