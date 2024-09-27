import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

FILE_PATH = "primes.txt"

@njit
def sieve(n):
    """Generate all primes up to n using the sieve of Eratosthenes."""
    is_prime = np.ones(n + 1, dtype=np.bool_)
    is_prime[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i:n + 1:i] = False
    primes = np.flatnonzero(is_prime)
    return primes

@njit
def segmented_sieve(low, high, primes):
    """Segmented sieve for the range (low, high)."""
    segment_size = high - low
    is_prime = np.ones(segment_size, dtype=np.bool_)
    for p in primes:
        start = max(p * p, ((low + p - 1) // p) * p)
        for multiple in range(start, high, p):
            is_prime[multiple - low] = False
    primes_in_segment = np.flatnonzero(is_prime) + low
    return primes_in_segment

def bulk_save_primes(primes):
    with open(FILE_PATH, "a") as file:
        file.write("\n".join(map(str, primes)) + "\n")

def get_last_prime():
    try:
        with open(FILE_PATH, "r") as file:
            lines = file.readlines()
            if lines:
                return int(lines[-1].strip())
    except FileNotFoundError:
        return 1
    return 1

def visualize_prime_progress(axs, x_data, y_data, total_primes_list, longest_prime):
    # Clear the axes
    axs[0].clear()
    axs[1].clear()

    # Plot Primes Found per Batch
    axs[0].plot(x_data, y_data, label="Primes Found per Batch", color="blue")
    axs[0].set_title("Primes Found per Batch")
    axs[0].set_xlabel("Batches")
    axs[0].set_ylabel("Primes Found per Batch")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Total Primes Found Over Time
    axs[1].plot(x_data, total_primes_list, label="Total Primes Found", color="green")
    axs[1].set_title(f"Total Primes Found: {total_primes_list[-1]}, Longest Prime Digits: {longest_prime}")
    axs[1].set_xlabel("Batches")
    axs[1].set_ylabel("Total Primes Found")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def find_primes(batch_size=1000000):
    current = get_last_prime() + 1
    total_primes = 0
    max_digits = 0
    x_data = []
    y_data = []
    total_primes_list = []
    batch_counter = 0

    # Initialize the plot once
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    running = True

    def handle_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', handle_close)

    while running:
        high = current + batch_size
        sqrt_high = int(high ** 0.5) + 1
        primes_up_to_sqrt_high = sieve(sqrt_high)

        primes_in_segment = segmented_sieve(current, high, primes_up_to_sqrt_high)

        primes_found_in_batch = len(primes_in_segment)
        total_primes += primes_found_in_batch

        if primes_found_in_batch > 0:
            longest_prime = primes_in_segment[-1]
            digits = len(str(longest_prime))
            if digits > max_digits:
                max_digits = digits

            bulk_save_primes(primes_in_segment)

        batch_counter += 1
        x_data.append(batch_counter)
        y_data.append(primes_found_in_batch)
        total_primes_list.append(total_primes)
        visualize_prime_progress(axs, x_data, y_data, total_primes_list, max_digits)

        print(f"Batch {batch_counter}: Found {primes_found_in_batch} primes. | Total Primes: {total_primes}")

        current = high

if __name__ == "__main__":
    find_primes()