import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import multiprocessing
import time
import os

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
        is_prime[start - low::p] = False
    primes_in_segment = np.flatnonzero(is_prime) + low
    return primes_in_segment

def bulk_save_primes(primes):
    with open(FILE_PATH, "ab") as file:
        np.savetxt(file, primes, fmt='%d')

def get_last_prime():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "rb") as file:
            try:
                file.seek(-2, os.SEEK_END)
                while file.read(1) != b'\n':
                    file.seek(-2, os.SEEK_CUR)
                last_line = file.readline().decode()
                return int(last_line.strip())
            except OSError:
                file.seek(0)
                last_line = file.readline().decode()
                return int(last_line.strip()) if last_line else 1
    return 1

def compute_primes(queue, batch_size=1000000):
    current = get_last_prime() + 1
    total_primes = 0
    max_digits = 0
    batch_counter = 0

    while True:
        batch_start_time = time.time()

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
        batch_end_time = time.time()
        batch_runtime = batch_end_time - batch_start_time

        # Send data to the plotting process
        queue.put({
            'batch': batch_counter,
            'primes_found': primes_found_in_batch,
            'total_primes': total_primes,
            'max_digits': max_digits,
            'batch_runtime': batch_runtime
        })

        print(f"Batch {batch_counter}: Found {primes_found_in_batch} primes. | Total Primes: {total_primes} | Batch Runtime: {batch_runtime:.2f} seconds")

        current = high

def plot_progress(queue):
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    x_data = []
    y_data = []
    total_primes_list = []
    runtime_list = []
    max_digits = 0

    start_time = time.time()

    def handle_close(event):
        # Stop the plotting process when the window is closed
        plt.close('all')
        os._exit(0)  # Forcefully exit all processes when the window is closed

    fig.canvas.mpl_connect('close_event', handle_close)

    while True:
        try:
            data = queue.get(timeout=1)
            batch = data['batch']
            primes_found = data['primes_found']
            total_primes = data['total_primes']
            batch_runtime = data['batch_runtime']
            max_digits = data['max_digits']

            x_data.append(batch)
            y_data.append(primes_found)
            total_primes_list.append(total_primes)
            runtime_list.append(batch_runtime)

            # Clear the axes
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()

            # Get the elapsed time
            elapsed_time = time.time() - start_time
            elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)

            # Plot Primes Found per Batch
            axs[0].plot(x_data, y_data, label="Primes Found per Batch", color="blue")
            axs[0].set_title(f"Primes Found per Batch (Runtime: {int(elapsed_minutes)}m {int(elapsed_seconds)}s)")
            axs[0].set_xlabel("Batches")
            axs[0].set_ylabel("Primes Found per Batch")
            axs[0].legend()
            axs[0].grid(True)

            # Plot Total Primes Found Over Time
            axs[1].plot(x_data, total_primes_list, label="Total Primes Found", color="green")
            axs[1].set_title(f"Total Primes Found: {total_primes_list[-1]}, Longest Prime Digits: {max_digits}")
            axs[1].set_xlabel("Batches")
            axs[1].set_ylabel("Total Primes Found")
            axs[1].legend()
            axs[1].grid(True)

            # Plot Runtime per Batch
            axs[2].plot(x_data, runtime_list, label="Runtime per Batch (s)", color="red")
            axs[2].set_title("Runtime per Batch")
            axs[2].set_xlabel("Batches")
            axs[2].set_ylabel("Runtime (s)")
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        except Exception:
            continue

if __name__ == "__main__":
    queue = multiprocessing.Queue()

    compute_process = multiprocessing.Process(target=compute_primes, args=(queue,))
    plot_process = multiprocessing.Process(target=plot_progress, args=(queue,))

    compute_process.start()
    plot_process.start()

    compute_process.join()
    plot_process.terminate()