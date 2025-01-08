from itertools import product
from random import sample

arg_permutations = {
    "rm_empty": [{'threshold': [0.1]}, 'off'],
    "smote": ['off', 'on'],
    "fill_nan": [{'method': ['mean', 'median', 'zero']}],
    # Through testing, those 2 methods were found to be deffective
    # "binning": ['off', {'threshold': [0.1], 'bins': [3]}],
    # "high_zero": ['off', {'threshold': [0.1]}],
    "correlation": ['off', {'threshold': [0.999, 0.99], 'method': ['pearson']}],
    "outliers": ['off', {'threshold': [2.5, 3.0, 3.5]}],
    "deskew": ['off', 'on'],
    "normalize": ['off', {'method': ['z-score', 'minmax', 'robust']}],
    # "data_manipulation": ['off', {'method': ['BestK', 'PCA', 'Poly']}]
}

# Helper function to expand the inner lists
def expand_options(options):
    if isinstance(options, list):
        expanded = []
        for opt in options:
            if isinstance(opt, dict):
                # Convert lists in the dictionary to individual combinations
                keys, values = zip(*[
                    (k, v if isinstance(v, list) else [v])
                    for k, v in opt.items()
                ])
                for combination in product(*values):
                    expanded.append(dict(zip(keys, combination)))
            else:
                expanded.append(opt)
        return expanded
    return options

first_layer_combinations = {key: expand_options(value) for key, value in arg_permutations.items()}

# Prepare the expanded arguments
expanded_args = {key: expand_options(value) for key, value in arg_permutations.items()}

# Generate all combinations
keys, values = zip(*expanded_args.items())
all_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

def random_combinations(n=10):
    return sample(all_combinations, n)


__all__ = ['all_combinations', 'random_combinations', 'first_layer_combinations']