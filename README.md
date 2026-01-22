# A Hybrid Genetic Algorithm with Elite-Based Iterated Local Search for the p-Median Problem

## Overview

This project implements a **Hybrid Genetic Algorithm enhanced with Elite-Based Iterated Local Search** for solving the classical p-Median Problem. The methodology combines the global exploration capability of genetic algorithms with the local exploitation capability of iterated local search techniques (ILS), achieving high-quality solutions in reasonable computational times.


### Key Features

- **Hybrid Algorithm**: Integration of Genetic Algorithm + Iterated Local Search
- **JIT Optimization**: Critical functions compiled with Numba for high performance
- **Adaptive Strategies**: Intelligent flow control with adaptive stopping and restart criteria
- **Unique Population**: Generation of unique combinations to avoid redundancy
- **Parallel Evaluation**: Population evaluation with parallel processing
- **Scalability**: Optimized for large instances of the p-Median Problem

---

## Project Structure

```
HGA-EBILS/
├── pMP_hybrid_GA.py                # Orchestrator of the hybrid algorithm
├── Algorith_pMP.py                 # Core algorithm implementation
├── GeneticAlgorithm_V2.py          # Genetic Algorithm base class
├── p_median_problem.py             # Problem definition and benchmark instances
├── poblacion_combinaciones.py      # Combination manager and population generation
├── test_runner.py                  # Main test and experimentation script
├── Hiperparametrizacion_V2.py      # Hyperparameter optimization
├── Extract_Data.py                 # Data extraction and processing utilities
├── datasets/                       # Cost matrices (.npy format)
│   ├── pmed1.npy
│   ├── pmed2.npy
│   └── ... (40 instances total)
├── resultados/                     # Results from execution runs (Excel files)
├── hyperopt_results/               # Hyperparameter optimization results
└── README.md                       # This file
```

---

## File Descriptions

### 1. **pMP_hybrid_GA.py**
**Central orchestration component of the algorithm**

- Defines the main function `pmedian_hybrid()` that encapsulates all hybrid algorithm logic
- Full flexibility: allows passing genetic operators as strings or custom functions
- Facilitates experimentation without modifying the core algorithm

**Key Functions:**
- `pmedian_hybrid()`: Main function that executes the complete hybrid algorithm

---

### 2. **Algorith_pMP.py**
**Core implementation of the genetic algorithm for p-Median**

Module containing all operations specific to the p-Median problem:

**Initialization:**
- `poblacion_inicial_combinaciones()`: Generates initial population with guaranteed unique individuals

**Evaluation:**
- `evaluar_poblacion()`: Evaluates the cost of all individuals (parallel, optimized with Numba)
- `evaluar_individuo_rapido()`: Fast evaluation of a single individual

**Genetic Operators:**
- `selecciona_torneo()`: Tournament selection (tournament selection)
- `cruzamiento_intercambio()`: Specialized crossover operator for combinations
- `mutacion_simple_Swap()`: Mutation based on facility exchange

**Local Search:**
- `_local_search_1_Swap_jit()`: Local search by exchange (compiled JIT)
- `Iterated_Local_Search()`: Iterated Local Search metaheuristic

**Flow Control:**
- `criterio_parada_estancamiento()`: Detects algorithm stagnation
- `criterio_reinicio_adaptive()`: Adaptive criterion for population restart
- `accion_reinicio()`: Restarts population while preserving best individuals

---

### 3. **GeneticAlgorithm_V2.py**
**Base Genetic Algorithm class**

Implements the general flow of a Genetic Algorithm:

- `AlgoritmoGenetico`: Main class that encapsulates all GA logic
  - Population management
  - Iterative execution with stopping criteria
  - Integration of customizable genetic operators
  - Support for elite search with Iterated Local Search


**Features:**
- Flexible parameter configuration
- Customizable stopping and restart criteria
- Support for optimization (maximize/minimize)


---

### 4. **p_median_problem.py**
**Definition of the p-Median Problem**

- `PMedian`: Class that encapsulates the p-Median problem
  - Access to standard benchmark instances (40 instances available)
  - Information about dimensions and known optimal solutions
  - Loading of cost matrices from .npy files



### 5. **poblacion_combinaciones.py**
**Combination Manager and Population Generator**

- `Combinaciones`: Specialized class for generating and managing unique combinations
  - Uses efficient structures (set of tuples) to guarantee uniqueness
  - Intelligent initial population generation
  - Methods for adding, generating, and validating combinations
  - Support for batch processing

**Features:**
- Generation of combinations without repetition
- Avoids duplicates in initial population
- Optimized for large search spaces

---

### 6. **test_runner.py**
**Main test and experimentation script**

Primary script for running complete experiments across multiple instances:

**Configuration:**
```python
REPLICAS = 10                    # Number of replicas per instance
NUM_ITERACIONES = 400            # GA generations
POP_SIZE = 500                   # Population size
PROB_CRUZAMIENTO = 0.95          # Crossover probability
PROB_MUTACION = 0.05             # Mutation probability
MAX_GEN = 100                    # Generations without improvement before stopping
MIN_GENER = 0                    # Minimum number of generations before stopping is allowed
MAX_ESTANCAMIENTO = 30           # Generations before population restart
MAXIMIZAR = False                # Whether to maximize the objective function
```

**Functionalities:**
- Loads test instances from .npy files
- Executes multiple replicas per instance
- Stores results in Excel (.xlsx) format
- Calculates statistics: Average fitness, Standard deviation of fitness, Median fitness, Best fitness found, Execution time


---

### 7. **Hiperparametrizacion_V2.py**
**Automatic Hyperparameter Optimization**

- Exhaustive or Bayesian search of optimal parameters
- Automatic evaluation of configurations
- Stores results in `hyperopt_results/` folder
- Facilitates parameter tuning for different instances

---

### 8. **Extract_Data.py**
**Data extraction and processing utilities**

- Utilities to extract data from benchmark files
- Cost matrix generation
- Data processing and validation


---

## Requirements

### Python Version and Dependencies

```
Python                3.12.10
numpy                 1.26.4
numba                 0.62.1
openpyxl              3.1.5
hyperopt              0.2.7
```


---

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/ferminriv20/HGA_EBILS.git
cd HGA-EBILS

```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` file with the following content:

```txt
numpy==1.26.4
numba==0.62.1
openpyxl==3.1.5
hyperopt==0.2.7
pandas>=1.3.0
scipy>=1.7.0
```

Then install:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy==1.26.4 numba==0.62.1 openpyxl==3.1.5 hyperopt==0.2.7
```

---

## Usage

### Basic Execution

Run the test script with predefined configuration:

```bash
python test_runner.py
```

**Expected output:**
- Excel file in `resultados/` with results from all replicas
- Consolidated statistics per instance

### Custom Execution

Import and use the algorithm directly in Python:

```python
import numpy as np
from p_median_problem import PMedian
from pMP_hybrid_GA import pmedian_hybrid

# Load an instance
pmed = PMedian(pmed_index=9)
cost_matrix = pmed.cost_matrix
p = pmed.p

# Run the hybrid algorithm
resultado = pmedian_hybrid(
    cost_matrix=cost_matrix,
    p=p,
    num_iteraciones=400,
    pop_size=500,
    seleccion='selecciona_torneo',
    cruzamiento='cruzamiento_intercambio',
    mutacion='mutacion_simple_Swap',
    para_seleccion={"num_competidores": 8},
    para_cruzamiento={},
    para_mutacion={},
    prob_cruzamiento=0.95,
    prob_mutacion=0.05,
    max_estancamiento=30,
    max_gen=100,
    min_gener=0,
    maximizar=False
)

print(f"Best solution: {resultado['mejor_individuo']}")
print(f"Best fitness: {resultado['mejor_fitness']}")
```

### Hyperparameter Optimization

```bash
python Hiperparametrizacion_V2.py
```

Results are saved in `hyperopt_results/` for later analysis.

---


## Optimization and Performance

### JIT Compilation with Numba

Critical functions use Numba's `@njit` decorators:

- `evaluar_poblacion()`: Parallelized with `@njit(parallel=True)`
- `_local_search_1_Swap_jit()`: Compiled JIT for fast local search
- `evaluar_individuo_rapido()`: Vectorized evaluation

**Benefit:** Typical speedup compared to pure Python

### Optimization Strategies

1. **Unique Population**: Avoids redundant evaluations
2. **Evaluation Cache**: Reuses previous evaluations
3. **Parallelization**: Simultaneous evaluation on multiple cores
4. **Adaptive Control**: Intelligent restart and stopping

---


## Troubleshooting

### Error: "Module 'numba' not found"
**Solution:** Install numba
```bash
pip install numba==0.62.1
```

### Error: "Dataset not found"
**Solution:** Ensure .npy files are in the `datasets/` folder

### Slow execution
**Solution:** 
- Reduce `POP_SIZE` or `NUM_ITERACIONES`
- Numba compilation takes time on first execution
- Subsequent executions will be faster (cached)

### Memory overflow on large instances
**Solution:**
- Reduce `pop_size`
- Use smaller instances for testing
- Increase available RAM

---


## License

This project is provided under the **MIT License**.

---

## References

- Beasley, J.E. (1990). *OR-Library: P-Median Problem Instances*. Available at: [https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html)

- Bergstra, J., Yamins, D., & Cox, D.D. (2013). *Hyperopt: A Python Library for Optimizing Machine Learning Algorithms*.  GitHub repository: [https://github.com/hyperopt/hyperopt](https://github.com/hyperopt/hyperopt)

---

## Authors

This research project was developed by:

- **Fermín Rivero Sotelo** - [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--8785--2297-green)](https://orcid.org/0009-0000-8785-2297)
- **Nelson Montes Villalba** - [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6306--2773-green)](https://orcid.org/0000-0002-6306-2773)
- **Jorge Lopez Pereira** - [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7317--8772-green)](https://orcid.org/0000-0001-7317-8772)
- **Helman Hernández Riaño** - [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--3042--2573-green)](https://orcid.org/0000-0003-3042-2573)

---
