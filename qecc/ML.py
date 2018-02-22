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