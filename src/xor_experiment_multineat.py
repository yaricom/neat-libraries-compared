# ///////////////////////////////////////////////////////////////////////////////////////////
# //    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
# //
# //    Copyright (C) 2012 Peter Chervenski
# //
# //    This program is free software: you can redistribute it and/or modify
# //    it under the terms of the GNU Lesser General Public License as published by
# //    the Free Software Foundation, either version 3 of the License, or
# //    (at your option) any later version.
# //
# //    This program is distributed in the hope that it will be useful,
# //    but WITHOUT ANY WARRANTY; without even the implied warranty of
# //    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# //    GNU General Public License for more details.
# //
# //    You should have received a copy of the GNU Lesser General Public License
# //    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
# //
# //    Contact info:
# //
# //    Peter Chervenski < spookey@abv.bg >
# //    Shane Ryan < shane.mcdonald.ryan@gmail.com >
# ///////////////////////////////////////////////////////////////////////////////////////////

# 
#  The XOR experiment adapted from original MultiNEAT source code available at: 
#  https://github.com/peter-ch/MultiNEAT/blob/master/examples/TestNEAT_xor.py
# 
#  All hyper-parameters taken as is from original file.

import os
import sys
import argparse

#sys.path.insert(0, '/home/peter/code/projects/MultiNEAT') # duh
import time
import random as rnd
import numpy as np
import pickle as pickle

import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial
from MultiNEAT import GetGenomeList, ZipFitness

from concurrent.futures import ProcessPoolExecutor, as_completed

import utils
from experiment import evaluate_experiment

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0

    # do stuff and return the fitness
    net.Flush()
    net.Input(np.array([1., 0., 1.]))  # can input numpy arrays, too
    # for some reason only np.float64 is supported
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([0, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([1, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    net.Flush()
    net.Input([0, 0, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    return (4 - error) ** 2

def build_parameters():
    params = NEAT.Parameters()
    params.PopulationSize = 100
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = True
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 1.0

    params.ArchiveEnforcement = False

    params.MutateWeightsProb = 0.05

    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.25
    params.WeightReplacementRate = 0.9

    params.MaxWeight = 8

    params.MutateAddNeuronProb = 0.001
    params.MutateAddLinkProb = 0.3
    params.MutateRemLinkProb = 0.0

    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9

    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 0.0

    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    params.AllowLoops = True
    params.AllowClones = True

    return params

def get_fitness(genome):
    return genome.GetFitness()

def run_experiment(params, trial_id, n_generations, out_dir=None, view_results=False, save_results=True):
    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, trial_id)

    # set random seed
    seed = int(time.time())
    pop.RNG.Seed(seed)

    generations = 0
    solved = False
    best = None
    complexity = 0
    for generation in range(n_generations):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, evaluate, display=view_results)
        NEAT.ZipFitness(genome_list, fitness_list)
        generations = generation
        best = max(genome_list, key=get_fitness)
        complexity = best.NumNeurons() + best.NumLinks()
        solved = best.GetFitness() > 15.5 # Changed to correspond limit used with other tested libraries
        if solved:
            print("Trial: %d\tgeneration: %d\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, generations, best.GetFitness(), complexity, seed))
            break
        # move to the next epoch
        pop.Epoch()
            
    if not solved:
        print("Trial: %d\tFAILED\t\tfitness: %f\tcomplexity: %d\tseed: %d" % (trial_id, best.GetFitness(), complexity, seed))

    return solved, generations, complexity

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The XOR experiment runner (MultiNEAT-Python).")
    parser.add_argument('-g', '--generations', default=100, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=10,
                        help="The number of experiment trials.")
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)

    # The directory to store experiment outputs
    out_dir = os.path.join(local_dir, '../out/xor/multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir=out_dir)

    # Prepare hyper-parameters
    params = build_parameters()

    # Run the experiment for a number of trials
    print("\n**************************")
    print("  MultiNEAT Library")
    print("  XOR Experiment")
    print("**************************\n")
    evaluate_experiment(args, 
                        eval_function=run_experiment, 
                        config=params, 
                        out_dir=out_dir)