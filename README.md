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

To find the most performant library, we need to determine the performance evaluation metrics that will be used to compare the efficiency of different libraries when searching for solvers of particular CS benchmark problems.

In our experiments, the following metrics are used:
 - success rate (the number of successful trials among total trials)
 - the average duration of one epoch of evolution among all trials/generations
 - the average number of generations per trial
 - average complexity of found solutions (the complexity of winner genomes)
 - the average fitness of found solutions

 To aggregate all the abovementioned metrics into a single unified score, we introduce the complex metric named *efficiency score*. This metric allows comparing the efficiency of each library in terms of:
 - how efficient it in finding the solution of the given problem (success rate)
 - how fast it was able to find a solution (average number of generations per trial)
 - the efficiency of the library algorithm in terms of execution speed (average epoch duration)
 - the efficiency of found solutions in terms of its topological complexity (average winner complexity)
 - the efficiency of found solutions in terms of its ability to meet the goal (average fitness score among winners)

The efficiency score metric can be estimated as follows:

```
score = success_rate * avg_fitness_score / log(avg_epoch_duration * avg_winner_complexity * avg_generations_trial)
```

We use the natural logarithm in the denominator to clamp down the denominator value to a range consistent with the value of fitness score in the numerator. The success rate, which is in range (0,1], effectively adjust the value of the efficiency score to reflect algorithm efficiency in finding a solution.

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

The evaluation is done in 100 trials over 100 generations. The fitness threshold for the successful solver is 15.5, and it is slightly less than the maximal fitness score that can be produced by the fitness function implementation used in the experiment. The maximal possible fitness value is 16.0, as given by the following  fitness function implementation:

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
We decided to use a slightly reduced value of the fitness threshold to adjust for the floating-point arithmetics used in the algorithm.

### The NEAT-Python Library Results

The [NEAT-Python][4] library is a pure Python implementation of the NEAT algorithm. In XOR experiment, it demonstrates the worst performance with the lowest success rate and the longest execution time.

The XOR experiment with NEAT-Python library can be started with following commands:

```bash
$ conda activate neat
$ cd src
$ python xor_experiment_neat.py -t 100 
```
The output of the command above is similar to the following:

```txt
Solved 19 trials from 100, success rate: 0.190000
Average
	trial duration:		4545.045724 ms
	epoch duration:		47.644999 ms
	generations/trial:	92.5

Average among winners
	Complexity:		14.000000
	Fitness:		15.789716
	generations/trial:	60.4

Average for all organisms evaluated during experiment
	Complexity:		10.960000
	Fitness:		10.709556

Efficiency score:		0.271994
```

The output above demonstrates that NEAT-Python library lagged behind other compared libraries in terms of execution speed and general performance. It can find successful solvers of the XOR problem only in about 20% of trials, which is very low compared to the other studied libraries.

### The MultiNEAT Library Results

The [MultiNEAT][5] library is written in C++ but provides corresponding Python bindings. In XOR experiment it demonstrates excellent performance, especially in terms of ability to find a problem solution. Most of the runs, the success rate is 1.0, which means that a solution was found in all experiment trials.

The XOR experiment with MultiNEAT library can be executed as following:

```bash
$ conda activate neat
$ cd src
$ python xor_experiment_multineat.py -t 100 
```
The command above will be produce output similar to the following:
```txt
Solved 100 trials from 100, success rate: 1.000000
Average
	trial duration:		378.530622 ms
	epoch duration:		10.960883 ms
	generations/trial:	35.0

Average among winners
	Complexity:		18.540000
	Fitness:		15.774268
	generations/trial:	35.0

Average for all organisms evaluated during experiment
	Complexity:		18.540000
	Fitness:		15.774268

Efficiency score:		1.778749
```

The MultiNEAT library demonstrates exceptional performance in the number of successful XOR problem solvers among all experiment trials. It has a success rate close to 100% and its execution speed almost for times better than of NEAT-Python library. All this combined gives it a much higher efficiency score in finding successful XOR solvers.

### The goNEAT Library Results

The [goNEAT][6] library is written in GO programming language and doesn't provide Python bindings. In XOR experiment, it demonstrates outstanding performance in terms of execution speed but trails the MultiNEAT library in terms of success rate.

For the instruction of how to install and use a library, please refer to the [goNEAT][6] GitHub repository.

After GO language environment is ready and the library is installed, you can run XOR experiment with the following command:

```bash
cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/xor -context ./data/xor.neat -genome ./data/xorstartgenes -experiment XOR
```

The command will produce the output similar to the following:

```text
Solved 93 trials from 100, success rate: 0.930000
Average
	Trial duration:		251.067745ms
	Epoch duration:		1.790507ms
	Generations/trial:	55.2

Champion found in 18 trial run
	Winner Nodes:		7
	Winner Genes:		16
	Winner Evals:		10150

	Diversity:		47
	Complexity:		23
	Age:			14
	Fitness:		16.000000

Average among winners
	Winner Nodes:		7.0
	Winner Genes:		13.9
	Winner Evals:		10282.8
	Generations/trial:	51.9

	Diversity:		40.096774
	Complexity:		20.978495
	Age:			25.397849
	Fitness:		15.842292

Averages for all organisms evaluated during experiment
	Diversity:		23.227037
	Complexity:		14.279039
	Age:			14.253140
	Fitness:		7.974019

Efficiency score:		1.929031

```

The output produced by goNEAT library put it in the first place among the studied libraries by solution search efficiency score. This result is achieved due to the fastest execution speed, which almost five times higher than of MultiNEAT, and the best average fitness score of the found solutions.

## The XOR Problem Results

Further, we present the results of the evaluation of the different NEAT libraries in the task of finding successful XOR solvers.

| Library | Efficiency Score | Success Rate | Avg Solution Fitness | Avg Epoch Duration | Avg Solution Complexity | Avg Generations per Trial | Platform |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| goNEAT | 1.93 | 0.93 | 15.84 | 1.79 | 20.98 | 55.2 | GO |
| MultiNEAT Python | 1.78 | 1.0 | 15.77 | 10.96 | 18.54 | 35.0 | C++, Python |
| NEAT-Python | 0.27 | 0.19 | 15.79 | 47.64 | 14.0 | 92.5 | Python |

In the results table, the libraries are ordered in descending order based on their efficiency score value. Thus, at the top row placed the most efficient library, and at the bottom row is the least efficient one.

### The XOR Problem Conclusion

The NEAT-Python library get the lowest efficiency score, but the solutions it was able to produce are the less complex ones. The goNEAT and MultiNEAT libraries are on par in terms of complexity and success rate of found solutions, but the former is much faster in terms of execution speed (almost 5x).

# Credits
The source code is maintained and managed by [Iaroslav Omelianenko][3]

[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:https://io42.space
[4]:https://github.com/CodeReclaimers/neat-python
[5]:https://github.com/peter-ch/MultiNEAT
[6]:https://github.com/yaricom/goNEAT

