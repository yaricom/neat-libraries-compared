# Overview
The repository with source code which can be used to compare performance of different implementations of the [NEAT][1] algorithm and its derivatives. 

The main goal is compare NEAT libraries written in different programming languages and to find the most performant one. The performance of the libraries is evaluated using standard benchmarks such as:
* XOR problem
* Pole balancing experiments
* Maze navigation

The corresponding NEAT libraries is mentioned in the [NEAT Software Catalog][2]

# Reguirements

In order to be able to run experiments presented in this repository you need to setup working environment appropriately.

The simplest way to do this is by using Anaconda Distribution to manage all dependencies required. The binary suitable for your operating system can be downloaded at: https://www.anaconda.com/distribution/#download-section

## Environment Setup

After installation of appropriate version of the Anaconda Distribution you are ready for environment setup.

First, we need to create Python3.5 based virtual environment and activate it:

```bash
$ conda create —name neat python=3.5
$ conda activate neat
```

### After that, we are ready to install all project dependencies

* The NEAT-Python library:

```bash
$ pip install neat-python==0.92
```
* The MultiNEAT Python library:
```bash
$ conda install -c conda-forge multineat
```
* The Matplotlib used for results rendering:
```bash
$ conda install matplotlib
```
* The Graphviz library with Python bindings for neural network graphs visualization:
```bash
$ conda install graphviz
$ conda install python-graphviz
```

# The Performance Metrics

In order to be able to find the most performant library we need to define the performance evaluation metrics that will be used to compare different libraries. 

In this experiment the following metrics will be used:
 - success rate (the number of successful trials among total trials)
 - the min/max range of evolution generations in which successful solver found 
 - the average complexity of found solutions 
 - the total complexity of best individuals in evaluated populations 
 - the elapsed time for specified number of experiment trials

# The XOR Problem Benchmark
The XOR problem solver is a classic computer science experiment in the field of reinforcement learning, which can not be solved without introducing non-linear execution to the solver algorithm. 

The two inputs to the XOR solver must be combined at some hidden unit, as opposed to only at the output node, because there is no function over a linear combination of the inputs that can separate the inputs into the proper classes. Thus, the successful XOR solver must be able to create at least one hidden unit in order to solve this problem.

The XOR problem search space can be defined as following:

| Input 1 | Input2 | Output |
|:-------:|:------:|:------:|
| 1       | 1      | 0      |
| 1       | 0      | 1      |
| 0       | 1      | 1      |
| 0       | 0      | 0      |

The evaluation is done in 100 trials over 100 generations. 
The fitness threshold for successful solver is 15.5 and it is based on the fitness function used in the experiment. 

The maximal possible fitness score is 16.0 in accordance with the following fitness function implementation:

```Python
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

error_sum = 0.0
for xi, xo in zip(xor_inputs, xor_outputs):
    output = net.activate(xi)
    error_sum += abs(output[0] - xo[0])
# Calculate amplified fitness
fitness = (4 - error_sum) ** 2
```
We decided to go with slightly reduced fitness threshold value to adjust for the floating point arithmetics used in the algorithm.

### The NEAT-Python Library Results

The [NEAT-Python][4] library is a pure Python implementation of the NEAT algorithm. In XOR experiment it demonstrated the worst performance as in terms of success rate as in terms of execution time.

The XOR experiment with NEAT-Python library can be started with following commands:

```bash
$ conda activate neat
$ cd src
$ python xor_experiment_neat.py -t 100 
```
The output of the command above is similar to the following:

```txt
XOR solutions found within generations min: 43, max: 94
Success runs 23 from 100, success rate: 0.230000
Average winners complexity: 12.522, average complexity: 10.180
Experiment's elapsed time: 432.697 sec
```

### The MultiNEAT Library Results

The [MultiNEAT][5] library is written in C++ but provides coresponding Python bindings. In XOR experiment it demonstrates excellent performance.

The XOR experiment with MultiNEAT library can be executed as following:

```bash
$ conda activate neat
$ cd src
$ python xor_experiment_multineat.py -t 100 
```
The command above will be produce output similar to the followoing:
```txt
XOR solutions found within generations min: 8, max: 91
Success runs 99 from 100, success rate: 0.990000
Average winners complexity: 16.818, average complexity: 16.770
Experiment's elapsed time: 37.223 sec
```


# Credits
This source code maintained and managed by [Iaroslav Omelianenko][3]

[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:https://io42.space
[4]:https://github.com/CodeReclaimers/neat-python
[5]:https://github.com/peter-ch/MultiNEAT

