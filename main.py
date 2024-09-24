import matplotlib.pyplot as plt
from numba import cuda, int32
import numpy as np
from tqdm import tqdm

FILE_PATH = "primes.txt"

@cuda.jit
def check_prime(numbers, results):
    idx = cuda.grid(1)
    if idx < numbers.size:
        num = numbers[idx]
        is_prime = 1
        if num < 2:
            is_prime = 0
        else:
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    is_prime = 0
                    break
        results[idx] = is_prime

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

def visualize_prime_progress(x_data, y_data, total_primes, longest_prime):
    plt.clf()
    plt.plot(x_data, y_data, label="Primes Found", color="blue")
    plt.title(f"Prime Numbers Found: {total_primes}, Longest Prime: {longest_prime}")
    plt.xlabel("Batches")
    plt.ylabel("Primes Found per Batch")
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def find_primes(batch_size=1000000):
    current = get_last_prime() + 1
    total_primes = 0
    max_digits = 0
    x_data = []
    y_data = []
    batch_counter = 0
    plt.ion()
    fig = plt.figure()
    running = True

    def handle_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', handle_close)

    while running:
        with tqdm(total=batch_size, desc="Searching for primes", unit=" numbers") as progress_bar:
            numbers = np.arange(current, current + batch_size, dtype=np.int32)
            results = np.zeros(batch_size, dtype=np.int32)
            threadsperblock = 256
            blockspergrid = (numbers.size + (threadsperblock - 1)) // threadsperblock
            check_prime[blockspergrid, threadsperblock](numbers, results)
            cuda.synchronize()

            primes_in_batch = []
            primes_found_in_batch = 0
            for i in range(batch_size):
                if results[i] == 1:
                    prime = numbers[i]
                    primes_in_batch.append(prime)
                    primes_found_in_batch += 1
                    total_primes += 1
                    digits = len(str(prime))
                    if digits > max_digits:
                        max_digits = digits

            if primes_in_batch:
                bulk_save_primes(primes_in_batch)

            batch_counter += 1
            x_data.append(batch_counter)
            y_data.append(primes_found_in_batch)
            visualize_prime_progress(x_data, y_data, total_primes, max_digits)

            progress_bar.set_postfix({
                "Total Primes Found": total_primes,
                "Longest Digits": max_digits
            })
            progress_bar.update(batch_size)
            current += batch_size

if __name__ == "__main__":
    find_primes()