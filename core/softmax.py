import numpy as np

# To compute softmax probabilities from logits
def softmax(logits):
    shifted_logits = logits - np.max(logits)
    exp_values = np.exp(shifted_logits)
    probabilties = exp_values / np.sum(exp_values)

    return probabilties

# To computer softmax with temperature scaling
def softmax_with_temp(logits, temperature=1.0):
    if temperature <=0:
        raise ValueError("Temperature must be positive")
    
    scaled_logits = logits / temperature
    return softmax(scaled_logits)

if __name__ =="__main__":
    logits = np.array([8.0, 3.0, -1.0])
    print("Logits:", logits)

    for T in [0.5, 1.0, 2.0, 10.0]:
        probs = softmax_with_temp(logits, T)
        print(f"Temperature = {T} -> Probabilities={probs}")