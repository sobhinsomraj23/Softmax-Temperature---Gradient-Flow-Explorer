import numpy as np
import matplotlib.pyplot as plt

from core.softmax import softmax_with_temperature
from core.gradients import cross_entropy_gradient


def compute_gradient_magnitude(logits, target, temperature):
    """
    Compute L2 norm of gradient for given temperature.
    """
    probs = softmax_with_temperature(logits, temperature)
    grad = cross_entropy_gradient(probs, target)
    return np.linalg.norm(grad)


def run_temperature_experiment():
    logits = np.array([8.0, 3.0, -1.0])
    target = 0  # cat

    temperatures = np.linspace(0.1, 10, 100)

    prob_history = []
    grad_magnitudes = []

    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        grad_mag = compute_gradient_magnitude(logits, target, T)

        prob_history.append(probs)
        grad_magnitudes.append(grad_mag)

    prob_history = np.array(prob_history)

    plot_probabilities(temperatures, prob_history)
    plot_gradient_magnitude(temperatures, grad_magnitudes)


def plot_probabilities(temperatures, prob_history):
    plt.figure()

    plt.plot(temperatures, prob_history[:, 0], label="cat")
    plt.plot(temperatures, prob_history[:, 1], label="dog")
    plt.plot(temperatures, prob_history[:, 2], label="lion")

    plt.xlabel("Temperature")
    plt.ylabel("Probability")
    plt.title("Softmax Probabilities vs Temperature")
    plt.legend()
    plt.grid()

    plt.show()


def plot_gradient_magnitude(temperatures, grad_magnitudes):
    plt.figure()

    plt.plot(temperatures, grad_magnitudes)

    plt.xlabel("Temperature")
    plt.ylabel("Gradient Magnitude (L2 Norm)")
    plt.title("Gradient Magnitude vs Temperature")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    run_temperature_experiment()