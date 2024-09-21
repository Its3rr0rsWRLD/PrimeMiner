# GPU-Accelerated Prime Number Finder

This Python script efficiently finds prime numbers using GPU acceleration with the `numba` library. It also utilizes `tqdm` for progress display, keeping the terminal output clean. Found prime numbers are saved to a text file and display the number of digits and the total primes found in real-time.

## Features
- Utilizes GPU via `numba` for parallel prime number checking.
- Real-time progress display using `tqdm`.
- Saves all found prime numbers to a file.
- Displays details for each prime, including the number of digits and total primes found so far.

## Requirements

- Python 3.6+
- CUDA-compatible GPU
- `numba` and `tqdm` libraries

## Installation

First, ensure that you have Python and a CUDA-compatible GPU setup.

1. Clone this repository or download the script.

2. Install the required dependencies:

    ```bash
    pip install numba tqdm
    ```

## Usage

Run the script to start finding prime numbers. By default, it will search for primes up to `10,000,000` in batches of `1,000,000`.

```bash
python main.py
```