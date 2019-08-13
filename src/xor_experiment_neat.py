#
# This file provides source code of XOR experiment using on NEAT-Python library
#

# The Python standard library import
import os
import random
import shutil
import time
import argparse

import numpy as np

# The NEAT-Python library imports
import neat

# The helper used to visualize experiment results
import utils.visualize as visualize
import utils

from experiment import evaluate_experiment

# The XOR inputs and expected corresponding outputs for fitness evaluation
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def eval_fitness(net):
    """
    Evaluates fitness of the genome that was used to generate 
    provided net
    Arguments:
        net: The feed-forward neural network generated from genome
    Returns:
        The fitness score - the higher score the means the better 
        fit organism. Maximal score: 16.0
    """
    error_sum = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error_sum += abs(output[0] - xo[0])
    # Calculate amplified fitness
    fitness = (4 - error_sum) ** 2
    return fitness

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    XOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    for _, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)

def run_experiment(config_file, trial_id, n_generations, out_dir, view_results=False, save_results=True):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file:    the path to the file with experiment 
                        configuration
        trial_id:       the id of current trial run
        n_generations:  the number of evolutionary generations
        out_dir:        the directory to store experiment outputs
        view_results:   the flag to control whether to view result visualizations
        save_results:   the flag to control whether to save resulting stats into files
    """
    # set random seed
    seed = int(time.time())
    random.seed(seed)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to n_generations generations.
    best_genome = p.run(eval_genomes, n_generations)
    
    # Check if the best genome is an adequate XOR solver
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    best_genome_fitness = best_genome.fitness# eval_fitness(net)

    # Find best genome complexity
    complexity = len(best_genome.connections) + len(best_genome.nodes) + 2 # two input nodes

    solution_found = best_genome_fitness > config.fitness_threshold
    if solution_found:
        print("Trial: %2d\tgeneration: %d\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, p.generation, best_genome_fitness, complexity, seed))
    else:
        print("Trial: %2d\tFAILED\t\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, best_genome_fitness, complexity, seed))

    # Visualize the experiment results
    if save_results:
        node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        visualize.draw_net(config, best_genome, view=view_results, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=view_results, filename=os.path.join(out_dir, 'avg_fitness.svg'))
        visualize.plot_species(stats, view=view_results, filename=os.path.join(out_dir, 'speciation.svg'))

    return solution_found, p.generation, complexity, best_genome_fitness


if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The XOR experiment runner (NEAT-Python).")
    parser.add_argument('-g', '--generations', default=100, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=10,
                        help="The number of experiment trials.")
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'xor/xor_config.ini')

    # The directory to store experiment outputs
    out_dir = os.path.join(local_dir, '../out/xor/neat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir=out_dir)

    # Run the experiment for a number of trials
    print("\n**************************")
    print("  NEAT-Python Library")
    print("  XOR Experiment")
    print("**************************\n")
    results = evaluate_experiment(args, 
                        eval_function=run_experiment, 
                        config=config_path, 
                        max_fitness=16.0, # The maximal fitness score in accordance with fitness function definition
                        out_dir=out_dir, 
                        save_results=False)
    
    results.print_statistics()