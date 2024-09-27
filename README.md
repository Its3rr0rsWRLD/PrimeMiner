# PrimeMiner

**Developer:** [Its3rr0rsWRLD/3rr0r](https://github.com/Its3rr0rsWRLD)

PrimeMiner is a highly efficient prime number generator and visualizer that uses advanced algorithms and optimization techniques to find prime numbers in large numerical ranges. It leverages the segmented sieve of Eratosthenes algorithm, optimized with Numba's Just-In-Time (JIT) compilation, to achieve high performance. The program also provides real-time visualization of the primes found per batch and the cumulative total of primes discovered.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Visualization](#visualization)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Efficient Prime Generation**: Utilizes the segmented sieve of Eratosthenes algorithm for fast prime number generation over large ranges.
- **Optimized Performance**: Uses Numba's JIT compilation to accelerate computational functions.
- **Real-Time Visualization**: Provides live graphical updates showing primes found per batch and the total primes found.
- **Persistence**: Saves discovered primes to a text file (`primes.txt`) and resumes from the last prime found on restart.
- **Scalable**: Allows adjustment of batch sizes to balance between performance and resource usage.
- **Interactive**: The program can be gracefully stopped by closing the visualization window.

---

## How It Works

PrimeMiner operates by processing numbers in batches and applying the segmented sieve of Eratosthenes to find all prime numbers within each batch. The program keeps track of the total primes found and updates two real-time graphs:

1. **Primes Found per Batch**: Shows the number of primes discovered in each processed batch.
2. **Total Primes Found Over Time**: Displays the cumulative total number of primes found after each batch.

The use of Numba's `@njit` decorator compiles the sieve functions to optimized machine code, significantly speeding up execution time.

---

## Prerequisites

Before running PrimeMiner, ensure that you have the following installed on your system:

- **Python 3.6 or higher**
- **NumPy**: `pip install numpy`
- **Matplotlib**: `pip install matplotlib`
- **Numba**: `pip install numba`
- **tqdm**: `pip install tqdm`

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Its3rr0rsWRLD/PrimeMiner.git
   cd PrimeMiner
   ```

2. **Install Dependencies**

   Ensure all required Python packages are installed:

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, install packages individually if a `requirements.txt` is not provided:*

   ```bash
   pip install numpy matplotlib numba tqdm
   ```

---

## Usage

To run PrimeMiner, execute the Python script:

```bash
python primeminer.py
```

*Replace `primeminer.py` with the actual filename if different.*

---

## Configuration

### Batch Size

The `batch_size` parameter determines how many numbers are processed in each batch. A larger batch size may improve performance but will use more memory.

You can adjust the batch size by modifying the `find_primes` function call:

```python
find_primes(batch_size=1000000)
```

---

## Output

### Primes File

All prime numbers found are saved to `primes.txt` in the working directory. Each prime is written on a new line. If the program is restarted, it will resume from the last prime number found in this file.

### Console Output

After each batch, the program prints:

```
Batch <batch_number>: Found <primes_found_in_batch> primes. | Total Primes: <total_primes>
```

---

## Visualization

The program provides real-time visualization using Matplotlib, displaying two graphs in the same window:

1. **Primes Found per Batch**: Plotted in the top subplot.
2. **Total Primes Found Over Time**: Plotted in the bottom subplot.

#### Controls

- **Stop the Program**: Close the visualization window to gracefully terminate the program.

#### Example Visualization

![PrimeMiner Visualization](visualization_example.png)

*Note: The image above is a placeholder. Replace with an actual screenshot if available.*

---

## Performance Considerations

- **Memory Usage**: Larger batch sizes consume more memory. Adjust `batch_size` based on your system's capabilities.
- **CPU Usage**: The program is CPU-intensive due to the nature of prime number computation.
- **Numba Optimization**: Ensure Numba is properly installed to benefit from JIT compilation. Without Numba, the performance will significantly degrade.

---

## Troubleshooting

### Common Issues

1. **Visualization Window Not Displaying**

   - Ensure that you are running the script in an environment that supports GUI operations (not in a headless server mode).
   - Check for any errors related to Matplotlib backend. You may need to install a GUI toolkit or configure Matplotlib to use an appropriate backend.

2. **High Memory Usage**

   - Reduce the `batch_size` parameter in the `find_primes` function to lower memory consumption.

3. **Slow Performance**

   - Verify that Numba is installed and functioning correctly. The first run may take longer due to function compilation.
   - Ensure that your system meets the prerequisites and has sufficient processing power.

4. **Permission Errors When Saving Primes**

   - Check that you have write permissions in the directory where `primes.txt` is located.
   - Run the script with appropriate permissions or change the `FILE_PATH` to a writable location.

5. **Program Doesn't Resume from Last Prime**

   - Ensure that `primes.txt` exists and contains valid prime numbers.
   - Check for any file read/write errors in the console output.

### Getting Help

If you encounter issues not covered here, consider reaching out through the project's [GitHub Issues](https://github.com/Its3rr0rsWRLD/PrimeMiner/issues) page.

---

## License

PrimeMiner is released under the [MIT License](LICENSE).

---

## Acknowledgments

- **Developer**: [Its3rr0rsWRLD/3rr0r](https://github.com/Its3rr0rsWRLD)
- **Algorithms Used**:
  - Segmented Sieve of Eratosthenes
- **Libraries and Tools**:
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [Numba](https://numba.pydata.org/)
  - [tqdm](https://tqdm.github.io/)
- **Inspiration**: The need for an efficient prime number generator with real-time visualization capabilities.

---

## Detailed Code Overview

Below is a detailed explanation of the core components of PrimeMiner.

### 1. Import Statements

```python
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
```

- **NumPy**: Used for efficient array operations.
- **Matplotlib**: For real-time data visualization.
- **Numba**: To compile Python functions to machine code for performance.
- **tqdm**: Provides progress bars for loops (not heavily utilized in current code).

### 2. Constants

```python
FILE_PATH = "primes.txt"
```

- **FILE_PATH**: Specifies where to save the primes found.

### 3. Sieve Functions

#### Sieve of Eratosthenes

```python
@njit
def sieve(n):
    ...
```

- **Purpose**: Generates all prime numbers up to `n`.
- **Optimization**: Decorated with `@njit` for performance.

#### Segmented Sieve

```python
@njit
def segmented_sieve(low, high, primes):
    ...
```

- **Purpose**: Finds primes in the range `[low, high)` using the base primes provided.
- **Optimization**: Also decorated with `@njit`.

### 4. File Operations

#### Saving Primes

```python
def bulk_save_primes(primes):
    with open(FILE_PATH, "a") as file:
        file.write("\n".join(map(str, primes)) + "\n")
```

- **Purpose**: Appends the list of primes to the `primes.txt` file.

#### Retrieving Last Prime

```python
def get_last_prime():
    try:
        with open(FILE_PATH, "r") as file:
            lines = file.readlines()
            if lines:
                return int(lines[-1].strip())
    except FileNotFoundError:
        return 1
    return 1
```

- **Purpose**: Reads the last prime from the file to resume computation.

### 5. Visualization

#### Visualization Function

```python
def visualize_prime_progress(axs, x_data, y_data, total_primes_list, longest_prime):
    ...
```

- **Purpose**: Updates the two subplots with new data.
- **Parameters**:
  - `axs`: Array of axes objects for the subplots.
  - `x_data`: List of batch numbers.
  - `y_data`: Primes found per batch.
  - `total_primes_list`: Cumulative total primes found.
  - `longest_prime`: Number of digits in the largest prime found.

- **Functionality**:
  - Clears previous data from axes.
  - Plots new data.
  - Updates titles and labels.
  - Redraws the canvas.

### 6. Main Function

#### `find_primes`

```python
def find_primes(batch_size=1000000):
    ...
```

- **Purpose**: Orchestrates the prime finding and visualization.
- **Flow**:
  1. Initializes variables and plots.
  2. Enters a loop that runs until the visualization window is closed.
  3. Calculates the high limit for the current batch.
  4. Generates base primes up to `sqrt(high)` using the sieve.
  5. Finds primes in the current batch using the segmented sieve.
  6. Saves the primes found.
  7. Updates data lists for visualization.
  8. Calls the visualization function.
  9. Prints batch information to the console.
  10. Increments the current number for the next batch.

#### Event Handling

```python
def handle_close(event):
    nonlocal running
    running = False
```

- **Purpose**: Allows the program to exit gracefully when the plot window is closed.

---

## Customization

### Changing the Output File Path

Modify the `FILE_PATH` constant at the beginning of the script:

```python
FILE_PATH = "path/to/your/desired/location/primes.txt"
```

### Adjusting Visualization

- **Colors and Styles**: Change the `color` parameter in plot functions.
- **Figure Size**: Modify the `figsize` parameter in `plt.subplots`.
- **Titles and Labels**: Edit the strings in `set_title`, `set_xlabel`, and `set_ylabel`.

### Extending Functionality

- **Add Progress Bars**: Integrate `tqdm` progress bars within loops for more granular feedback.
- **Parallel Processing**: Explore multiprocessing or GPU acceleration for further performance gains.

---

## Contribution

Contributions are welcome! Feel free to submit a pull request or open an issue on the [GitHub repository](https://github.com/Its3rr0rsWRLD/PrimeMiner) to report bugs or suggest enhancements.

---

## Contact

For questions or support, please contact [Its3rr0rsWRLD](mailto:socketerror404@proton.me).