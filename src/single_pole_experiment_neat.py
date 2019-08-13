#
# This file provides the source code of the Single-Pole balancing experiment using on NEAT-Python library
#

# The Python standard library import
import os
import random
import time
import argparse

# The NEAT-Python library imports
import neat

# The helper used to visualize experiment results
import utils.visualize as visualize
import utils

# The cart-pole simulator
import pole.cart_pole as cart

from experiment import evaluate_experiment

# The maximal number of balancing steps
max_balancing_steps_num = 500000

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    for _, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = cart.eval_fitness(net, max_bal_steps=max_balancing_steps_num)
        genome.fitness = fitness

def run_experiment(config_file, trial_id, n_generations, out_dir, view_results=False, save_results=True):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
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

    # Run for up to N generations.
    best_genome = p.run(eval_genomes, n=n_generations)

    # Check if the best genome is a winning Sinle-Pole balancing controller 
    #net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Find best genome complexity
    complexity = len(best_genome.connections) + len(best_genome.nodes) + 4 # four input nodes

    # Test if solution was found
    best_genome_fitness = best_genome.fitness # cart.eval_fitness(net, max_bal_steps=max_balancing_steps_num)#
    solution_found = (best_genome_fitness >= config.fitness_threshold)
    if solution_found:
        print("Trial: %2d\tgeneration: %d\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, p.generation, best_genome_fitness, complexity, seed))
    else:
        print("Trial: %2d\tFAILED\t\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, best_genome_fitness, complexity, seed))

    # Visualize the experiment results
    if save_results:
        node_names = {-1:'x', -2:'dot_x', -3:'θ', -4:'dot_θ', 0:'action'}
        visualize.draw_net(config, best_genome, view=view_results, node_names=node_names, directory=out_dir, fmt='svg')
        visualize.plot_stats(stats, ylog=False, view=view_results, filename=os.path.join(out_dir, 'avg_fitness.svg'))
        visualize.plot_species(stats, view=view_results, filename=os.path.join(out_dir, 'speciation.svg'))

    return solution_found, p.generation, complexity, best_genome_fitness

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The Single Pole-Balancing experiment runner (NEAT-Python).")
    parser.add_argument('-g', '--generations', default=100, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=10,
                        help="The number of experiment trials.")
    parser.add_argument('-s', '--save_results', type=bool, default=False,
                        help="Controls whether to save intermediate execution results.")
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'pole/single_pole_config.ini')

    # The directory to store experiment outputs
    out_dir = os.path.join(local_dir, '../out/pole/neat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir=out_dir)

    # Run the experiment for a number of trials
    print("\n************************************")
    print("  NEAT-Python Library")
    print("  Single Pole-Balancing Experiment")
    print("************************************\n")
    results = evaluate_experiment(args, 
                        eval_function=run_experiment, 
                        config=config_path, 
                        max_fitness=1.0, # the maximal allowed fitness value as given by fitness function
                        out_dir=out_dir, 
                        save_results=args.save_results)
    
    results.print_statistics()