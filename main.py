from numba import cuda, int32
import numpy as np
import time
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

def save_prime(prime):
    with open(FILE_PATH, "a") as file:
        file.write(f"{prime}\n")

def find_primes(max_number=10000000, batch_size=1000000):
    current = 2
    total_primes = 0
    max_digits = 0
    with tqdm(total=max_number, desc="Searching for primes", unit=" numbers") as progress_bar:
        while current < max_number:
            numbers = np.arange(current, current + batch_size, dtype=np.int32)
            results = np.zeros(batch_size, dtype=np.int32)
            threadsperblock = 256
            blockspergrid = (numbers.size + (threadsperblock - 1)) // threadsperblock
            check_prime[blockspergrid, threadsperblock](numbers, results)
            cuda.synchronize()

            primes_found_in_batch = 0
            for i in range(batch_size):
                if results[i] == 1:
                    prime = numbers[i]
                    save_prime(prime)
                    primes_found_in_batch += 1
                    total_primes += 1
                    digits = len(str(prime))
                    if digits > max_digits:
                        max_digits = digits

            progress_bar.set_postfix({
                "Total Primes Found": total_primes,
                "Longest Digits": max_digits
            })
            progress_bar.update(batch_size)
            current += batch_size

if __name__ == "__main__":
    find_primes(10000000, 1000000)