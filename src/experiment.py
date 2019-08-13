import os
import time
import math

import numpy as np

class ExperimentEvaluationResults:
    """
    The class to hold experiment evaluation results
    """
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.results = np.zeros((n_trials,), dtype=bool)
        self.generations = np.zeros(n_trials)
        self.complexity = np.zeros(n_trials)
        self.fitness = np.zeros(n_trials)
        self.trial_durations = np.zeros(n_trials)
        self.avg_epoch_durations = np.zeros(n_trials)
        self.elapsed_time = 0
        self.success_run = 0
        self.success_rate = 0
        self.efficiency_score = 0

    def calculate_statistics(self, max_fitness):
        """
        The function to calculate the agregate statistics over collected experiment reults.
        """
        self.success_run = np.count_nonzero(self.results)
        self.success_rate = float(self.success_run) / float(self.n_trials)

        # Build averages
        self.avg_trial_duration = np.average(self.trial_durations)
        self.avg_epoch_duration = np.average(self.avg_epoch_durations)
        self.avg_trial_generations = np.average(self.generations)

        self.avg_complexity = np.average(self.complexity)
        self.avg_fitness = np.average(self.fitness)

        self.avg_winner_complexity = np.average(self.complexity[self.results])
        self.avg_winner_fitness = np.average(self.fitness[self.results])
        self.avg_winner_trial_generations = np.average(self.generations[self.results])

        # Find solution's efficiency score
        # We are interested in efficient solver search solution that take 
        # less time per epoch, less generations per trial, and produce less complicated winner genomes.
        # At the same time it should have maximal fitness score and maximal success rate among trials.
        fitness_score = self.avg_winner_fitness
        if max_fitness > 0:
            fitness_score /= max_fitness
            fitness_score *= 100

        self.efficiency_score = self.avg_epoch_duration * self.avg_trial_generations * self.avg_winner_complexity
        if self.efficiency_score > 0:
            self.efficiency_score = self.success_rate * fitness_score / math.log(self.efficiency_score)

    def print_statistics(self):
        """
        The function to print collected statistics about experiment to the standard out as easy to read formatted text.
        """
        print("\nSolved %d trials from %d, success rate: %f" % (self.success_run, self.n_trials, self.success_rate))
        print("Average\n\ttrial duration:\t\t%f ms\n\tepoch duration:\t\t%f ms\n\tgenerations/trial:\t%.1f\n" %
            (self.avg_trial_duration, self.avg_epoch_duration, self.avg_trial_generations))
        print("Average among winners\n\tComplexity:\t\t%f\n\tFitness:\t\t%f\n\tgenerations/trial:\t%.1f\n" % 
            (self.avg_winner_complexity, self.avg_winner_fitness, self.avg_winner_trial_generations))
        print("Average for all organisms evaluated during experiment\n\tComplexity:\t\t%f\n\tFitness:\t\t%f\n" %
            (self.avg_complexity, self.avg_fitness))
        print("Efficiency score:\t\t%f\n" % self.efficiency_score)
        print("Experiment's elapsed time:\t%.3f sec\n" % (self.elapsed_time))

#
# The common experiment evaluator code
#
def evaluate_experiment(args, eval_function, config, out_dir, max_fitness=-1, save_results=False, view_results=False):
    """
    The function to evaluate given experiment specified by provided evaluation function. The evaluation
    results will be returned as data object.
    Arguments:
        args:           The command line arguments
        eval_function:  The evaluation function running one trial of experiment
        config:         The algorithm-specific configuration parameters
        out_dir:        The directory to store ouput results if any
        max_fitness:    The maximal fitness score value for experiment or -1 if not defined.
        save_results:   The flag to control if output results should be saved into output directory
        view_results:   The flag to control whether intermediate output reults should be printed.
    Returns:
        The ExperimentEvaluationResults holding statistics about experiment results.
    """
    experiment = ExperimentEvaluationResults(args.trials)
    start_time = time.time()
    for i in range(args.trials):
        trial_start_time = time.time()
        trial_out_dir = os.path.join(out_dir, "%d" % i)
        experiment.results[i], generation, experiment.complexity[i], experiment.fitness[i] = eval_function(config, 
                                                                        trial_id=i, 
                                                                        n_generations=args.generations,
                                                                        out_dir=trial_out_dir,
                                                                        save_results=save_results,
                                                                        view_results=view_results)
        experiment.trial_durations[i] = (time.time() - trial_start_time) * 1000 # ms
        experiment.avg_epoch_durations[i] = experiment.trial_durations[i] / float(generation + 1)
        experiment.generations[i] = generation

    experiment.elapsed_time = time.time() - start_time
    experiment.calculate_statistics(max_fitness=max_fitness)
    return experiment