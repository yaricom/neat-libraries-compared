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
 - average complexity of found solutions (the complexity of winner genomes - sum of nodal and link genomes)
 - the average fitness of found solutions

 To aggregate all the abovementioned metrics into a single unified score, we introduce the complex metric named *efficiency score*. This metric allows comparing the efficiency of each library in terms of:
 - how efficient it in finding the solution of the given problem (success rate)
 - how fast it was able to find a solution (average number of generations per trial)
 - the efficiency of the library algorithm in terms of execution speed (average epoch duration)
 - the efficiency of found solutions in terms of its topological complexity (average winner complexity)
 - the efficiency of found solutions in terms of its ability to meet the goal (average fitness score among winners)

The efficiency score metric can be estimated as follows:

```
score = success_rate * normalized_fitness_score / log(avg_epoch_duration * avg_winner_complexity * avg_generations_trial)
```

We use the natural logarithm in the denominator to clamp down the denominator value to a range consistent with the value of fitness score in the numerator. The success rate, which is in range (0,1], effectively adjust the value of the efficiency score to reflect algorithm efficiency in finding a solution.

The normalized fitness score can be estimated as follows:

```
normalized_fitness_score = 100 * avg_fitness_score / max_fitness_score
```

Where the *max_fitness_score* value is determined by the fitness function used in the problem. Also we scale normalized fitness score by the factor of 100 to equalize the nominator and denominator scales in the effitiency score formula.

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

The evaluation is done in 100 experiment trials over a maximum of 100 epochs (generations) of evolution each. The fitness threshold for the successful solver is 15.5, and it is slightly less than the maximal fitness score that can be produced by the fitness function implementation used in the experiment. The maximal possible fitness value is 16.0, as given by the following  fitness function implementation:

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
Solved 16 trials from 100, success rate: 0.160000
Average
	trial duration:		4714.596846 ms
	epoch duration:		48.995008 ms
	generations/trial:	93.5

Average among winners
	Complexity:		12.687500
	Fitness:		15.691308
	generations/trial:	59.2

Average for all organisms evaluated during experiment
	Complexity:		10.790000
	Fitness:		10.350439

Efficiency score:		1.430373
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
	trial duration:		382.263935 ms
	epoch duration:		10.807755 ms
	generations/trial:	35.5

Average among winners
	Complexity:		18.480000
	Fitness:		15.727424
	generations/trial:	35.5

Average for all organisms evaluated during experiment
	Complexity:		18.480000
	Fitness:		15.727424

Efficiency score:		11.088049
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
Solved 91 trials from 100, success rate: 0.910000
Average
	Trial duration:		233.568394ms
	Epoch duration:		1.738699ms
	Generations/trial:	53.3

Champion found in 99 trial run
	Winner Nodes:		7
	Winner Genes:		15
	Winner Evals:		9261

	Diversity:		41
	Complexity:		22
	Age:			35
	Fitness:		16.000000

Average among winners
	Winner Nodes:		7.0
	Winner Genes:		13.9
	Winner Evals:		9633.3
	Generations/trial:	48.7

	Diversity:		38.076923
	Complexity:		20.846154
	Age:			25.439560
	Fitness:		15.832682

Averages for all organisms evaluated during experiment
	Diversity:		22.478840
	Complexity:		14.251525
	Age:			13.704519
	Fitness:		7.975999

Efficiency score:		11.901923
```

The output produced by goNEAT library put it in the first place among the studied libraries by solution search efficiency score. This result is achieved due to the fastest execution speed, which almost five times higher than of MultiNEAT, and the best average fitness score of the found solutions.

## The XOR Problem Results

Further, we present the results of the evaluation of the different NEAT libraries in the task of finding successful XOR solvers.

| Library | Efficiency Score | Success Rate | Avg Solution Fitness | Avg Epoch Duration | Avg Solution Complexity | Avg Generations per Trial | Platform |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| goNEAT | 11.90 | 0.91 | 15.83 | 1.74 | 20.85 | 53.3 | GO |
| MultiNEAT Python | 11.09 | 1.0 | 15.73 | 10.81 | 18.48 | 35.5 | C++, Python |
| NEAT-Python | 1.43 | 0.16 | 15.69 | 49.00 | 12.69 | 93.5 | Python |

In the results table, the libraries are ordered in descending order based on their efficiency score value. Thus, at the top row placed the most efficient library, and at the bottom row is the least efficient one.

### The XOR Problem Conclusion

The NEAT-Python library get the lowest efficiency score, but the solutions it was able to produce are the less complex ones. The goNEAT and MultiNEAT libraries are on par in terms of complexity and success rate of found solutions, but the former is much faster in terms of execution speed (almost 5x).

# The Single Pole-Balancing Problem Benchmark

The single-pole balancer (a.k.a. *inverted pendulum*) is an unstable pendulum that has its center of mass above its pivot point. It can be stabilized by applying external forces under control of a specialized system that monitors the angle of the pole and moves the pivot point horizontally back and forth under the center of mass as it starts to fall. The single-pole balancer is a classic problem in dynamics and control theory that is used as a benchmark for testing control strategies, including the strategies based on the Reinforcement Learning methods. We are particularly interested in the implementation of the specific control algorithm that use neuroevolution-based methods to stabilize the inverted pendulum for a given amount of time.

The experiment described here considers the simulation of the inverted pendulum implemented as a cart that can move horizontally with a pivot point mounted on top of it, i.e., the cart and pole apparatus. The scheme of the described apparatus is shown in the following image:

![Single Pole-Balancing Scheme][single_pole-balancing_scheme]

The goal of the controller is to exert a sequence of forces, Fx, to the center of mass of the cart such that the pole balanced for a specific (or infinite) amount of time and the cart stays within the track, i.e., doesn’t hit left or right walls. Thus, we can say that the state of the cart-pole system must be kept to avoid certain regions of the state space, qualifying the balancer’s task as an avoidance control problem. There is no unique solution, and any trajectory drawn through the state space that miss regions to be avoided is acceptable.

The learning algorithm needs to receive a minimal amount of knowledge about the task from the environment to train pole-balancing controller. Such knowledge should reflect how close is our controller to the goal. The goal of the pole-balancing problem is to stabilize an inherently unstable
system and keep it balanced as long as possible but at least the expected number of time steps as specified in the experiment configuration (500 000). Thus, the objective function must optimize the duration of stable pole-balancing and can be defined as the logarithmic difference between the expected number of steps and the actual number of steps that obtained during the evaluation of the phenotype ANN. The loss function is given as follows:
```
loss = (log(t_max) - log(t_eval)) / log(t_max)
```
Where t_max is an expected number of time steps from the configuration of the experiment, and t_eval is the actual number of time steps during which the controller was able to maintain a stable state of the pole-balancer within bounds. Please note that loss value is in range [0.0, 1.0]

And the fitness score is:
```
fitness = 1.0 - loss
```
The evaluation is done in 100 experiment trials over a maximum of 100 epochs (generations) of evolution each. The fitness threshold for the successful solver is 1.0, which is also the maximal possible fitness value with a given fitness function.

### The NEAT-Python Library Results

In the Single Pole-Balancing experiment, the NEAT-Python library gets the efficiency score on par with MultiNEAT Python Library, but significantly lower than of goNEAT library.

The Single Pole-Balancing experiment with NEAT-Python library can be started with the following commands:

```bash
$ conda activate neat
$ cd src
$ python single_pole_experiment_neat.py -t 100
```
The output of the command above is similar to the following:

```txt
Solved 100 trials from 100, success rate: 1.000000
Average
	trial duration:		7661.165447 ms
	epoch duration:		2191.749573 ms
	generations/trial:	3.9

Average among winners
	Complexity:		13.470000
	Fitness:		1.000000
	generations/trial:	3.9

Average for all organisms evaluated during experiment
	Complexity:		13.470000
	Fitness:		1.000000

Efficiency score:		8.575182
```
As it can be seen from the results above, the NEAT-Python library demonstrates very long execution time of one epoch but can find a solution within a small number of epochs and the complexity of the produced solution is the lowest among tested libraries.

### The MultiNEAT Library Results

In the Single Pole-Balancing experiment, the MultiNEAT Python library gets the efficiency score on par with the NEAT-Python Library, but significantly lower than of goNEAT library.

The Single Pole-Balancing experiment with MultiNEAT library can be executed as follows:

```bash
$ conda activate neat
$ cd src
$ python single_pole_experiment_multineat.py -t 100
```
The command above will produce output similar to the following:

```txt
Solved 100 trials from 100, success rate: 1.000000
Average
	trial duration:		3930.903034 ms
	epoch duration:		1369.127191 ms
	generations/trial:	4.6

Average among winners
	Complexity:		17.170000
	Fitness:		1.000000
	generations/trial:	4.6

Average for all organisms evaluated during experiment
	Complexity:		17.170000
	Fitness:		1.000000

Efficiency score:		8.630517
```
The MultiNEAT Python library demonstrates execution speed improvements (almost 2x) compared to the NEAT-Python library, but it takes more epochs to find a solution and found solutions is more complex. This results in the efficiency score value comparable with the one obtained for NEAT-Python library.

### The goNEAT Library Results

In Single Pole-Balancing experiment, the goNEAT obtains the higher efficiency score value due to its outstanding execution speed compared to its Python counterparts.

For the instruction of how to install and use a library, please refer to the [goNEAT][6] GitHub repository.

After the GO language environment is ready and the library is installed, you can run a  Single Pole-Balancing experiment with the following command:

```bash
cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole1 -context ./data/pole1_150.neat -genome ./data/pole1startgenes -experiment cart_pole
```
The command will produce the output similar to the following:

```txt
Solved 100 trials from 100, success rate: 1.000000
Average
	Trial duration:		207.348488ms
	Epoch duration:		31.933857ms
	Generations/trial:	12.5

Champion found in 57 trial run
	Winner Nodes:		7
	Winner Genes:		9
	Winner Evals:		1112

	Diversity:		73
	Complexity:		16
	Age:			3
	Fitness:		1.000000

Average among winners
	Winner Nodes:		7.2
	Winner Genes:		11.9
	Winner Evals:		1794.0
	Generations/trial:	12.5

	Diversity:		64.100000
	Complexity:		18.980000
	Age:			3.710000
	Fitness:		1.000000

Averages for all organisms evaluated during experiment
	Diversity:		41.629055
	Complexity:		18.282251
	Age:			2.633224
	Fitness:		0.170506

Efficiency score:		11.195725
```

The goNEAT library spent more epochs to find winner solutions and produced the most complex ones compared to its Python counterparts. But it still received the highest efficiency score due to outstanding execution speed (almost 70x better than NEAT-Python).

## The Single Pole-Balancing Problem Results

Further, we present the results of the evaluation of the different NEAT libraries in the task of finding successful Single Pole-Balancer controllers.

| Library | Efficiency Score | Success Rate | Avg Solution Fitness | Avg Epoch Duration | Avg Solution Complexity | Avg Generations per Trial | Platform |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| goNEAT | 11.20 | 1.0 | 1.0  | 31.93 | 18.98 | 12.5 | GO |
| MultiNEAT Python | 8.63 | 1.0 | 1.0  | 1369.13 | 17.17 | 4.6 | C++, Python |
| NEAT-Python | 8.58 | 1.0 | 1.0  | 2191.75 | 13.47 | 3.9 | Python |

In the results table, the libraries are ordered in descending order based on their efficiency score value. Thus, at the top row placed the most efficient library, and at the bottom row is the least efficient one.

### The Single Pole-Balancing Problem Conclusion

The NEAT-Python and MultiNEAT libraries obtained similar efficiency scores and were able to find the successful single pole-balancing controllers within the smallest number of evolution epochs. The NEAT-Python library, as well as in the previous experiment produced the less complex ANN of successful controllers due to its way to work with BIAS nodes. In the NEAT Python library, the BIAS values are integrated into hidden and output nodes, rather than presented as separate nodes as it is done in other libraries. As a result, the produced controller ANN has fewer nodes and links.

The goNEAT library received the highest efficiency score value mostly due to its outstanding execution speed, which is almost 70x times faster than of NEAT-Python library.

# The Double Pole-Balancing Problem Benchmark

The single-pole balancing problem is easy enough for the NEAT algorithm, which can quickly find the optimal control strategy to maintain a stable system state. To make the experiment more challenging, we present a more advanced version of the cart-pole balancing problem. In this version, the two poles are connected to the moving cart by a hinge. A schema of the new cart-poles apparatus is shown in the following image:

![Double Pole-Balancing Scheme][double_pole-balancing_scheme]

The goal of the controller is to apply the force to the cart keeping two poles balanced as long as possible. At the same time, the cart should stay on track within the defined boundaries. As with single-pole balancing problem discussed before the control strategy can be defined as an avoidance control problem. Which means that the controller must maintain stable system state avoiding the danger zones when cart moves outside the track boundaries or either of the poles fall out of allowed vertical angle. There is no unique solution for this problem, but an appropriate control strategy can be found because the poles have different lengths and mass. Therefore they respond differently to the control inputs.

The goal of the pole-balancing problem is to stabilize an inherently unstable
system and keep it balanced as long as possible but at least the expected number of time steps as specified in the experiment configuration (100 000). Thus, the objective function must optimize the duration of stable pole-balancing and can be defined as the logarithmic difference between the expected number of steps and the actual number of steps that obtained during the evaluation of the phenotype ANN. The loss function is given as follows:
```
loss = (log(t_max) - log(t_eval)) / log(t_max)
```
Where t_max is an expected number of time steps from the configuration of the experiment, and t_eval is the actual number of time steps during which the controller was able to maintain a stable state of the pole-balancer within bounds. Please note that loss value is in range [0.0, 1.0]

And the fitness score is:
```
fitness = 1.0 - loss
```

The evaluation is done in 100 experiment trials over a maximum of 100 epochs (generations) of evolution each. The fitness threshold for the successful solver is 1.0, which is also the maximal possible fitness value with a given fitness function. This experiment will use increased population size of the genomes compared to the previous experiments. We will experiment with population of 1000 genomes during the evolution.

### The NEAT-Python Library Results

In the Double Pole-Balancing experiment, the NEAT-Python library gets the efficiency score better than of MultiNEAT Python Library, but significantly lower than of goNEAT library.

The Double Pole-Balancing experiment with NEAT-Python library can be started with the following commands:
```bash
$ conda activate neat
$ cd src
$ python two_pole_experiment_neat.py -t 100
```

The output of the command above is similar to the following:
```txt
Solved 37 trials from 100, success rate: 0.370000
Average
	trial duration:		223304.507408 ms
	epoch duration:		2648.731512 ms
	generations/trial:	82.3

Average among winners
	Complexity:		12.081081
	Fitness:			1.000000
	generations/trial:	52.2

Average for all organisms evaluated during experiment
	Complexity:		12.300000
	Fitness:			0.805440

Efficiency score:		2.502670

Experiment's elapsed time:	22330.452 sec
```

The success rate of the NEAT-Python library with this experiment dropped significantly, and execution time increased dramatically due to an increase in the duration of one trial. The increase in the duration of the trial was caused by low success rate (most trials executed about 100 generations), and also because the solutions were found mainly in later generations of evolution (on average after 52 generations), and both of these factors led to the average number of generations executed per trial up to about 82.

### The MultiNEAT Library Results

In the Double Pole-Balancing experiment, the MultiNEAT Python library gets the worst efficiency score among all tested libraries.

The Double Pole-Balancing experiment with MultiNEAT library can be executed as follows:
```bash
$ conda activate neat
$ cd src
$ python two_pole_experiment_multineat.py -t 100
```

The command above will be produce output similar to the following:
```txt
Solved 8 trials from 100, success rate: 0.080000
Average
	trial duration:		1080223.639586 ms
	epoch duration:		10968.178278 ms
	generations/trial:	95.7

Average among winners
	Complexity:		30.000000
	Fitness:			1.000000
	generations/trial:	57.1

Average for all organisms evaluated during experiment
	Complexity:		31.820000
	Fitness:			0.577561

Efficiency score:		0.463375

Experiment's elapsed time:	108022.365 sec
```

The MultiNEAT Python library demonstrates the worst performance among all the libraries tested in this experiment. It showed the lowest success rate (8 successful trials per 100) with a duration of one epoch almost 5 times higher than the NEAT-Python library, and 25 times higher than the goNEAT library. Such, a profound drop in execution speed may be caused by the increase in population size used during this experiment. The MultiNEAT Python library seems to handle substantial sizes of genomes populations very poorly.

### The goNEAT Library Results

In Double Pole-Balancing experiment, the goNEAT obtains the higher efficiency score value due to its outstanding execution speed compared to its Python counterparts.

For the instruction of how to install and use a library, please refer to the [goNEAT][6] GitHub repository.

After the GO language environment is ready and the library is installed, you can run a  Single Pole-Balancing experiment with the following command:

```bash
cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole1 -context ./data/pole1_150.neat -genome ./data/pole1startgenes -experiment cart_pole
```
The command will produce the output similar to the following:
```txt
Solved 90 trials from 100, success rate: 0.900000
Average
	Trial duration:		35.435151438s
	Epoch duration:		401.225213ms
	Generations/trial:	44.4

Champion found in 87 trial run
	Winner Nodes:		8
	Winner Genes:		7
	Winner Evals:		6607

	Diversity:			679
	Complexity:		15
	Age:				2
	Fitness:			1.000000

Average among winners
	Winner Nodes:		18.2
	Winner Genes:		39.9
	Winner Evals:		37749.9
	Generations/trial:	38.2

	Diversity:			639.566667
	Complexity:		57.933333
	Age:				2.811111
	Fitness:			1.000000

Averages for all organisms evaluated during experiment
	Diversity:			669.277647
	Complexity:		38.990155
	Age:				2.817825
	Fitness:			0.010878

Efficiency score:		6.499683
```

The goNEAT library demonstrate the best efficiency score among all tested libraries due to highest success rate (90 successful trials per 100) and outstanding execution speed. Also this library is able to find successful double pole-balancing controllers at the earlier stages of the evolutionary process, which resulted in lowest average number of generations executed per trial (about 44). That number is almost two times lower than corresponding value for the NEAT-Python library. All of this resulted in lowest average trial duration.

## The Double Pole-Balancing Problem Results

Further, we present the results of the evaluation of the different NEAT libraries in the task of finding successful Double Pole-Balancer controllers.

| Library | Efficiency Score | Success Rate | Avg Solution Fitness | Avg Epoch Duration | Avg Solution Complexity | Avg Generations per Trial | Platform |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| goNEAT | 6.50 | 0.9 | 1.0  | 401.23 | 57.93 | 44.4 | GO |
| NEAT-Python | 2.50 | 0.37 | 1.0  | 2648.73 | 12.08 | 82.3 | C++, Python |
| MultiNEAT Python | 0.46 | 0.08 | 1.0  | 10968.18 | 30 | 95.7 | Python |

In the results table, the libraries are ordered in descending order based on their efficiency score value. Thus, at the top row placed the most efficient library, and at the bottom row is the least efficient one.

### The Double Pole-Balancing Problem Conclusion

This experiment was dealing with significantly more hard-to-solve problems than considered earlier. To increase the chance of finding a solution within a limited number of evolution epochs, we increased the size of genomes population to expand the solution search space. As an additional benefit, we were able to test how each library handles the significant population sizes during the evolutionary process. Such amendment paid off, and as a result, we have found that MultiNEAT Python library is very bad in handling large populations of genomes. As a result, the MultiNEAT Python library obtained the lowest efficiency score due to the lowest success rate and extremely slow execution speed.

The NEAT-Python library in this experiment gained second place with an efficiency score almost 5 times higher than the MultiNEAT Python library. Such achievement can be explained by its higher success rate and the lower value of the average epoch duration compared to the MultiNEAT Python library. Also, the NEAT-Python library was able to find less complex solutions among all libraries. The low complexity of solutions caused by internal representation of bias nodes used in NEAT-Python library, which is part of standard nodes activation function rather than separate nodes as in other tested libraries.

The goNEAT library, as with previous experiments, demonstrated the best efficiency score due to the highest success rate (90%), fastest execution speed and ability to find solutions in the early stages of evolution.

# Credits
The source code is maintained and managed by [Iaroslav Omelianenko][3]

[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:https://io42.space
[4]:https://github.com/CodeReclaimers/neat-python
[5]:https://github.com/peter-ch/MultiNEAT
[6]:https://github.com/yaricom/goNEAT

[single_pole-balancing_scheme]: https://github.com/yaricom/goNEAT/blob/master/contents/single_pole-balancing.jpg "The single pole-balancing experimental setup"
[double_pole-balancing_scheme]: https://github.com/yaricom/goNEAT/blob/master/contents/double_pole-balancing.png "The double pole-balancing experimental setup"
