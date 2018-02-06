import numpy as np
import random
from typing import List


def mod2(x: int):
    return x % 2

vectorized_mod2 = np.vectorize(mod2)


def base_b(n: int, b: int):
    """
    Return the string representation of n in base-b.

    Example:
        base_b(11, 3) = '102'
    """

    e = n // b
    q = n % b
    if n == 0:
        return '0'
    elif e == 0:
        return str(q)
    else:
        return base_b(e, b) + str(q)


def binomial_list(n: int, k: int) -> List[List[int]]:
    """
    Produce a list of all possible combinations of picking k elements out of n. All elements are different.

    Args:
        n: Total number of elements.
        k: Number of elements to pick.

    Returns:
        blist: List of list of combinations.

    Example:
        binomial_list(5, 3) =
        [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    """

    blist, stack = [], []

    for i in range(k):
        stack.append(i)

    while 1:
        # [:] ensures append by value instead of by reference
        blist.append(stack[:])

        while len(stack) > 0 and stack[-1] == len(stack) - k + n - 1:
            stack.pop()
        if len(stack) == 0:
            break
        stack[-1] += 1
        for i in range(k - len(stack)):
            stack.append(stack[-1] + 1)

    return blist


def z2_gaussian_elimination(mat: np.ndarray, back_substitution=False) -> np.ndarray:
    """
    Gaussian elimination to reduced row echelon form and back substitution over field Z_2.

    Args:
        mat: Matrix to be Gaussian eliminated to reduced row echelon form.
        back_substitution: If True, perform back substitution after Gaussian elimination and return solution. Defaults
                           to False.

    Returns:
        y: The solution after back substitution. Only return when back_substitution=True.
    """

    def swap_rows(arr, frm, to):
        arr[[frm, to], :] = arr[[to, frm], :]

    [m, n] = np.shape(mat)

    for k in range(min([m, n])):
        s = 0
        # find pivot
        while np.max(mat[k:m, k + s]) == 0 and k + s < n - 1:
            s += 1
        swap_rows(mat, np.argmax(mat[k:m, k + s]) + k, k)
        # process rows below pivot
        for i in range(k + 1, m):
            mat[i, :] = vectorized_mod2(mat[i, :] + mat[k, :] * mat[i, k + s])

    if back_substitution:
        assert m + 1 == n
        y = np.zeros([m, 1], dtype=np.int)
        for k in range(m):
            y[-1 - k, 0] = mat[-1 - k, -1]
            for j in range(k):
                y[-1 - k, 0] -= y[-k + j, 0] * mat[-1 - k, -1 - k + j]
            y[-1 - k, 0] = mod2(y[-1 - k, 0])
        return y


def z2_find_basis(mat: np.ndarray) -> np.ndarray:
    """
    Given a matrix of size m x n, m < n. The first m rows constitute a set of linearly independent basis.
    Return a n x n matrix that consists of the remaining n - m basis that spans the n-dimensional linear space.

    Args:
        mat: The incomplete set of linearly independent basis.

    Returns:
        basis: The complete set of linearly independent basis.
    """

    [m, n] = np.shape(mat)
    assert m <= n

    # check the incomplete basis are already linearly independent
    mat2 = mat.copy()
    z2_gaussian_elimination(mat2)
    assert np.count_nonzero(mat2[-1, :]) != 0, ''

    basis = mat.copy()
    idd = np.identity(n, dtype=np.int)

    for i in range(n):
        mat2 = np.concatenate([basis, np.reshape(idd[i, :], [1, -1])], axis=0)
        z2_gaussian_elimination(mat2)
        if np.count_nonzero(mat2[-1, :]) != 0:
            basis = np.concatenate([basis, np.reshape(idd[i, :], [1, -1])], axis=0)
        if np.shape(basis)[0] == n:
            break

    return basis


class StabilizerCode:
    def __init__(self, stabilizer_str: List[str], logic_str: List[str]):
        """
        Initialize the stabilizer code based on stabilizer strings and logic operator strings. The logic operator string
        should be in order [X1, Z1, X2, Z2, ...] so that the logical error will be identified correctly.
        """
        random.seed()

        self.H = self.__stabilizer_str_to_check_mat(stabilizer_str)
        self.L = self.__stabilizer_str_to_check_mat(logic_str)
        assert np.shape(self.H)[1] == np.shape(self.L)[1]

        self.n = len(stabilizer_str[0])
        self.k = self.n - len(stabilizer_str)
        print("This is a [[", self.n, ",", self.k, "]] code. ")

        self.basis = None
        self.syndrome_table = None
        self.syndrome_basis = None

    @staticmethod
    def __stabilizer_str_to_check_mat(stabilizers: List[str]):
        """
        Convert a string of stabilizers to a check matrix. The convention of check matrix is H = [X | Z].

        Args:
            stabilizers: List of strings of equal length composed of 'I', 'X', 'Y' and 'Z'.

        Returns:
            h: Check matrix of size (n - k) * 2n.
        """

        # check dimension of the strings
        n_stab = len(stabilizers)
        assert n_stab > 0

        n = len(stabilizers[0])
        assert n > 0

        stab_len = list(map(len, stabilizers))
        for x in stab_len:
            assert x == n

        h = np.zeros([n_stab, 2 * n], dtype=np.int)

        for i in range(n_stab):
            for j in range(n):
                if stabilizers[i][j] == "X":
                    h[i, j] = 1
                elif stabilizers[i][j] == "Z":
                    h[i, j + n] = 1
                elif stabilizers[i][j] == "Y":
                    h[i, j] = 1
                    h[i, j + n] = 1

        return h

    def phys_err_to_syndrome(self, phy_err: np.ndarray) -> np.ndarray:
        """
        Compute syndrome from physical error.

        Args:
            phy_err: A 2n x 1 array of physical errors.

        Returns:
            A (n - k) x 1 array of syndromes.
        """
        new_phys = np.zeros([2 * self.n, 1], dtype=np.int)
        new_phys[:self.n] = np.reshape(phy_err[-self.n:], [-1, 1])
        new_phys[-self.n:] = np.reshape(phy_err[:self.n], [-1, 1])

        return vectorized_mod2(np.dot(self.H, new_phys))

    def phys_err_to_logic_err(self, phy_err: np.ndarray) -> np.ndarray:
        """
        Compute the logical error from physical error. The physical error must be syndrome free.

        Args:
            phy_err: a 2n x 1 array of physical errors

        Returns:
            A string represent the logical error.
        """
        # check shape
        assert np.shape(phy_err)[0] == 2 * self.n and np.shape(phy_err)[1] == 1
        # check syndrome free
        assert np.all(self.phys_err_to_syndrome(phy_err) == 0)

        if self.basis is None:
            self.basis = z2_find_basis(np.concatenate((self.H, self.L), axis=0))

        y = z2_gaussian_elimination(np.concatenate((np.transpose(self.basis), phy_err), axis=1), back_substitution=True)

        logic_err = []
        for i in range(self.k):
            if y[self.n - self.k + 2 * i] == 0 and y[self.n - self.k + 2 * i + 1] == 0:
                logic_err.append('I')
            if y[self.n - self.k + 2 * i] == 1 and y[self.n - self.k + 2 * i + 1] == 0:
                logic_err.append('X')
            if y[self.n - self.k + 2 * i] == 0 and y[self.n - self.k + 2 * i + 1] == 1:
                logic_err.append('Z')
            if y[self.n - self.k + 2 * i] == 1 and y[self.n - self.k + 2 * i + 1] == 1:
                logic_err.append('Y')
        return ''.join(logic_err)

    def syndrome_lookup_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Eliminate the syndrome according to the syndrome lookup table.

        Args:
            syndrome: A (n - k) x 1 array of syndromes.

        Returns:
            A 2n x 1 array of physical operators
        """
        assert len(syndrome) == self.n - self.k

        if self.syndrome_table is None:
            self.__generate_syndrome_table()

        return self.syndrome_table[np.array2string(syndrome)]

    def syndrome_basis_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Eliminate the syndrome according to the syndrome basis.

        Args:
            syndrome: A (n - k) x 1 array of syndromes.

        Returns:
            A 2n x 1 array of physical operators.
        """
        assert len(syndrome) == self.n - self.k

        if self.syndrome_basis is None:
            self.__generate_syndrome_basis()

        return np.dot(self.syndrome_basis, syndrome)

    def __generate_syndrome_table(self, rule="qubit_weight"):
        """
        Generate a syndrome lookup table to self.syndrome_table for syndrome elimination according to minimum weight
        principle. The table should be a dictionary of 2 ** (n - k) keys.

        Warning 1: The size of the table grows exponentially with n - k. So this method only works for small code.
        Warning 2: For some code, some syndrome will never happen for topological reasons (e.g. in toric code, the
                   syndrome always appears in pairs). The code will run indefinitely in such scenario.

        Args:
            rule: "XZ_weight" or "qubit_weight". For "XZ_weight", error weight = #X errors + #Z errors. Note that a
                  Y error has weight 2. For "qubit_weight", every X, Y, Z error has weight 1.

        Returns:
            self.syndrome_table: A dictionary with 2 ** (n - k) elements. Keys (syndromes) are length-(n - k) strings.
                                 Values (physical errors) are 2n x 1 numpy vectors.
        """
        self.syndrome_table = dict()
        phys_err = np.zeros([2 * self.n, 1], dtype=np.int)
        self.syndrome_table[np.array2string(self.phys_err2syndrome(phys_err))] = phys_err

        if rule == "XZ_weight":
            for i in range(1, 2 * self.n):
                # search for weight-i error
                phys_err_list = binomial_list(2 * self.n, i)
                for j in phys_err_list:
                    phys_err = np.zeros([2 * self.n, 1], dtype=np.int)
                    for k in range(i):
                        phys_err[j[k]] = 1
                    synd_key = np.array2string(self.phys_err2syndrome(phys_err))
                    if synd_key not in self.syndrome_table:
                        self.syndrome_table[synd_key] = phys_err
                        if len(self.syndrome_table) == 2 ** (self.n - self.k):
                            break
                if len(self.syndrome_table) == 2 ** (self.n - self.k):
                    break
        elif rule == "qubit_weight":
            for i in range(1, self.n):
                # search for weight-i error
                phys_err_list = binomial_list(self.n, i)
                for j in phys_err_list:
                    # there are 3 ** i possible combinations of X, Y, Z error on i physical qubits
                    for k in range(3 ** i):
                        phys_err = np.zeros([2 * self.n, 1], dtype=np.int)
                        errstr = base_b(k, 3)
                        errstr = '0' * (i - len(errstr)) + errstr
                        for l in range(i):
                            if errstr[l] == '0':
                                phys_err[j[l]] = 1
                            elif errstr[l] == '1':
                                phys_err[j[l] + self.n] = 1
                            elif errstr[l] == '2':
                                phys_err[j[l]] = 1
                                phys_err[j[l] + self.n] = 1
                        synd_key = np.array2string(self.phys_err2syndrome(phys_err))
                        if synd_key not in self.syndrome_table:
                            self.syndrome_table[synd_key] = phys_err
                            if len(self.syndrome_table) == 2 ** (self.n - self.k):
                                break
                    if len(self.syndrome_table) == 2 ** (self.n - self.k):
                        break
                if len(self.syndrome_table) == 2 ** (self.n - self.k):
                    break

        else:
            assert 0, "Rule error. "

    def __generate_syndrome_basis(self):
        """
        Generate a set of syndrome basis for syndrome elimination according to minimum weight principle. The number of
        the basis is n - k.

        Notice the linear combination of the minimal weight syndrome basis may not be the minimal weight syndrome, but
        this method is pretty efficient.

        Returns:
            self.syndrome_basis: a 2n * (n - k) numpy matrix.
        """
        synd_tab = dict()
        for i in range(1, 2 * self.n):
            # search for weight-i error
            phys_err_list = binomial_list(2 * self.n, i)
            for j in phys_err_list:
                phys_err = np.zeros([2 * self.n, 1], dtype=np.int)
                for k in range(i):
                    phys_err[j[k]] = 1
                synd = self.phys_err2syndrome(phys_err)
                nonzero, tag = 0, 0
                for x in range(self.n - self.k):
                    if synd[x, 0] != 0:
                        nonzero += 1
                        tag = x
                    if nonzero > 1:
                        continue
                if (nonzero == 1) and (tag not in synd_tab):
                    synd_tab[tag] = phys_err
                    if len(synd_tab) == self.n - self.k:
                        break
            if len(synd_tab) == self.n - self.k:
                break

        self.syndrome_basis = np.zeros([2 * self.n, self.n - self.k], dtype=np.int)
        for i in range(self.n - self.k):
            synd = np.zeros([self.n - self.k, 1], dtype=np.int)
            synd[i] = 1
            self.syndrome_basis[:, i] = synd_tab[i][:, 0]

    def uniform_random_err_history(self, l: int, px: float, py: float, pz: float) -> np.ndarray:
        """
        Generate physical error history randomly. There is px, py, pz probability on each physical qubit at each time
        step.

        Args:
            l: Number of total time steps.
            px: Probability of X error.
            py: Probability of Y error.
            pz: Probability of Z error.

        Return:
            phy_error_his: A 2n x l array of physical errors.
        """
        phy_error = np.zeros([2 * self.n], dtype=np.int)
        phy_error_his = np.zeros([2 * self.n, l], dtype=np.int)

        for i in range(l):
            for j in range(self.n):
                if random.random() < px:
                    phy_error[j] += 1
                if random.random() < py:
                    phy_error[j] += 1
                    phy_error[j + self.n] += 1
                if random.random() < pz:
                    phy_error[j + self.n] += 1
                phy_error_his[:, i] = phy_error[:]

        return phy_error_his
