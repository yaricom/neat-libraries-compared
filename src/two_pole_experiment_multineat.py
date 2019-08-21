#
# This file provides the source code of the Two Pole-Balancing experiment using the MultiNEAT Python library
#

# The Python standard library import
import os
import random
import time
import argparse

# The MultiNEAT imports
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial
from MultiNEAT import GetGenomeList, ZipFitness

# The helper used to visualize experiment results
import utils.visualize as visualize
import utils

# The cart-pole simulator
import pole.cart_two_pole as cart

from experiment import evaluate_experiment
from experiment import ANNWrapper

def evaluate(genome):
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)

    multi_net.Flush()
    fitness = cart.eval_fitness(net=ANNWrapper(multi_net))
    return fitness


def get_fitness(genome):
    return genome.GetFitness()

def run_experiment(config_file, trial_id, n_generations, out_dir, view_results=False, save_results=True):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file:    The path to the file with experiment 
                        configuration
        trial_id:       The ID of current trial
        n_generations:  The number of evolutionary generations
        out_dir:        The directory to save intermediate results.
        view_results:   The flag to control if intermediate results should be displayed after each trial
        save_results:   The flag to control whether intermediate results should be saved after each trial.
    Returns:
        The tuple (solution_found, generation, complexity, best_genome_fitness) that has flag indicating whether
        solution was found, the generation when solution was found, the complextity of best genome, and the fitness
        of best genome.
    """
    g = NEAT.Genome(0, 6+1, 0, 1, False, NEAT.ActivationFunction.TANH, 
                NEAT.ActivationFunction.TANH, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, trial_id)

     # set random seed
    seed = int(time.time())
    pop.RNG.Seed(seed)

    generations = 0
    solved = False
    best_trial_fitness = 0
    best_trial_complexity = 0
    for generation in range(n_generations):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, evaluate, display=view_results)
        NEAT.ZipFitness(genome_list, fitness_list)
        generations = generation
        best = max(genome_list, key=get_fitness)
        best_fitness = best.GetFitness()
        complexity = best.NumNeurons() + best.NumLinks()
        solved = best_fitness >= cart.MAX_FITNESS # Changed to correspond limit used with other tested libraries
        if solved:
            best_trial_fitness = best_fitness
            best_trial_complexity = complexity
            print("Trial: %2d\tgeneration: %d\tfitness: %f\tcomplexity: %d\tseed: %d" % 
                    (trial_id, generations, best_trial_fitness, complexity, seed))
            break
        # check if best fitness in this generation is better than current maximum
        if best_fitness > best_trial_fitness:
            best_trial_complexity = complexity
            best_trial_fitness = best_fitness

        # move to the next epoch
        pop.Epoch()
            
    if not solved:
        print("Trial: %2d\tFAILED\t\tfitness: %f\tcomplexity: %d\tseed: %d" % 
                (trial_id, best_trial_fitness, best_trial_complexity, seed))

    return solved, generations, best_trial_complexity, best_trial_fitness

def build_parameters():
    params = NEAT.Parameters()
    params.PopulationSize = 1000
    params.DynamicCompatibility = True
    params.AllowClones = False
    params.CompatTreshold = 5.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 3
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.02
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    params.MaxWeight = 20
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.02
    params.MutateRemLinkProb = 0.00
    params.CrossoverRate = 0.5
    params.MutateWeightsSevereProb = 0.01
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The Two Pole-Balancing experiment runner (MultiNEAT-Python).")
    parser.add_argument('-g', '--generations', default=100, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=10,
                        help="The number of experiment trials.")
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)

    # The directory to store experiment outputs
    out_dir = os.path.join(local_dir, '../out/pole/multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir=out_dir)

    # Prepare hyper-parameters
    params = build_parameters()

    # Run the experiment for a number of trials
    print("\n************************************")
    print("  MultiNEAT Library")
    print("  Two Pole-Balancing Experiment")
    print("************************************\n")
    results = evaluate_experiment(args, 
                        eval_function=run_experiment, 
                        config=params, 
                        max_fitness=cart.MAX_FITNESS, # The maximal fitness score in accordance with fitness function definition
                        out_dir=out_dir)
                        
    results.print_statistics()