from qecc.StabilizerCode import StabilizerCode
from typing import Callable

import numpy as np
import csv


def generate_training_data(file_name: str, n: int, code: StabilizerCode, error_correction_method: str,
                           error_model: Callable, *args, **kwds):
    """
    Generate [syndrome history, logical error] as training data for a stabilizer code given error model.

    Args:
        file_name: The file name where the histories are saved as a csv file.
        n: Number of different histories to be generated.
        code: The stabilizer code based on which the history is to be generated.
        error_correction_method: The logical error is only well-defined for physical errors with no syndromes. This is
                                 the method name of code that is used to first eliminated the last syndrome.
        error_model: The callable that generates the error history.
        *args, **kwds: Positional or keyword arguments that are passed to the error model.
    """
    last_phys_err = np.zeros([2 * code.n, 1], dtype=np.int)
    with open(file_name, 'w') as file:
        csv_writer = csv.writer(file)
        for i in range(n):
            phys_error_his = error_model(*args, **kwds)
            last_phys_err[:] = np.reshape(phys_error_his[:, -1], [-1, 1])
            last_phys_err += code.__getattribute__(error_correction_method)(code.phys_err_to_syndrome(last_phys_err))
            csv_writer.writerows([np.transpose(code.phys_err_to_syndrome(phys_error_his)).flatten().tolist()
                                  + [code.phys_err_to_logic_err(last_phys_err)]])


def last_syndrome_analyze_table(file_name: str, num: int, code: StabilizerCode):
    """
    Given syndrome history read from a file, count and generate last syndrome ~ logical error table of size
    2 ** (n - k) x 4 ** k.

    Args:
        file_name: The file name where the histories are to be imported.
        num: Last num syndromes to analyze. Defaults to 1.
        code: The stabilizer code based on which the history is to be generated.
    """
    def logic_err_str_to_idx(logic: str):
        idx, multiplier = 0, 0
        for i in range(len(logic)):
            if logic[i] == 'I':
                multiplier = 0
            elif logic[i] == 'X':
                multiplier = 1
            elif logic[i] == 'Z':
                multiplier = 2
            elif logic[i] == 'Y':
                multiplier = 3
            idx += multiplier * (4 ** i)
        return idx

    n_syndrome = code.n - code.k
    result = np.zeros([2 ** (num * n_syndrome), 4 ** code.k], dtype=np.int)
    binary = np.transpose(2 ** np.arange(num * n_syndrome)[::-1])

    # infer history length
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        line = csv_reader.__next__()
    l = (len(line) - 1) // (code.n - code.k)

    syndromes = np.loadtxt(file_name, dtype=np.int, usecols=(x for x in range((l - num) * n_syndrome,
                                                                              l * n_syndrome)), delimiter=",")
    logical_err = np.loadtxt(file_name, dtype=np.str, usecols=l * n_syndrome, delimiter=",")
    for i in range(np.shape(syndromes)[0]):
        result[np.dot(syndromes[i], binary), logic_err_str_to_idx(logical_err[i])] += 1

    n = np.sum(result)
    naive = np.sum(result, axis=0)[0] / n
    best = np.sum(np.amax(result, axis=1)) / n
    print("Naive correction: {:4}%. Upper bound: {:4}%".format(100.0 * naive, 100.0 * best))
