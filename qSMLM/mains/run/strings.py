import numpy as np
import itertools

def generate_binary_strings(length, num_ones):
    binary_strings = set()
    binary_array = np.array([1] * num_ones + [0] * (length - num_ones), dtype=int)
    for perm in set(itertools.permutations(binary_array)):
        binary_strings.add(tuple(perm))
    return binary_strings

def non_overlapping_combinations_(M,N1,N2):

    # Generate all possible binary strings with N1 1's and N2 1's
    binary_strings_N1 = generate_binary_strings(M, N1)
    binary_strings_N2 = generate_binary_strings(M, N2)

    # Find non-overlapping combinations of the two binary strings
    non_overlapping_combinations = []
    for string1 in binary_strings_N1:
        for string2 in binary_strings_N2:
            if not any(a and b for a, b in zip(string1, string2)):
                string1 = list(string1); string2 = list(string2)
                non_overlapping_combinations.append([string1, string2])

    non_overlapping_combinations = np.array(non_overlapping_combinations)
    return non_overlapping_combinations

maxM = 10
maxN1 = 3
maxN2 = 3

path = '/Users/cwseitz/Desktop/States/'

for m in range(1,maxM):
    for n1 in range(maxN1):
        for n2 in range(maxN2):
            if n1+n2 <= m:
                print(f'Generating M={m},N1={n1},N2={n2}')
                non_overlapping_combinations = non_overlapping_combinations_(m,n1,n2)
                np.savez(path+f'bin_{m}{n1}{n2}.npz',B=non_overlapping_combinations)


