import os
import time

import numpy as np

#
# The common experiment evaluator code
#
def evaluate_experiment(args, eval_function, config, out_dir, save_results=False, view_results=False):
    start_time = time.time()
    results = np.zeros((args.trials,), dtype=int)
    min_generation = args.generations
    max_generation = 0
    avg_win_complexity = 0
    avg_complexity = 0
    for i in range(args.trials):
        trial_out_dir = os.path.join(out_dir, "%d" % i)
        success, generation, complexity = eval_function(config, 
                                                        trial_id=i, 
                                                        n_generations=args.generations,
                                                        out_dir=trial_out_dir,
                                                        save_results=save_results,
                                                        view_results=view_results)
        avg_complexity += complexity
        if success:
            results[i] = 1
            avg_win_complexity += complexity
            if generation < min_generation:
                min_generation = generation
            if generation > max_generation:
                max_generation = generation

    success_run = np.count_nonzero(results)
    success_rate = float(success_run) / float(args.trials)
    elapsed_time = time.time() - start_time
    avg_complexity = float(avg_complexity) / float(args.trials)
    if success_run > 0:
        avg_win_complexity = float(avg_win_complexity) / float(success_run)
    else:
        avg_win_complexity = -1

    if success_run > 0:
        print("XOR solutions found within generations min: %d, max: %d" % (min_generation, max_generation))
    else:
        print("XOR solution was not found!")
    print("Success runs %d from %d, success rate: %f" % (success_run, args.trials, success_rate))
    print("Average winners complexity: %.3f, average complexity: %.3f" % (avg_win_complexity, avg_complexity))
    print("Experiment's elapsed time: %.3f sec" % (elapsed_time))