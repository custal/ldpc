import numpy as np
import scipy.sparse
from typing import Optional, List, Union, Tuple
import warnings
import ldpc.helpers.scipy_helpers
from ldpc.bp_decoder._bp_decoder import (


def available_precisions() -> List[int]:
    """
    The message-passing precisions, in mantissa bits, that the C++ extension was
    compiled with, in ascending order.

    Precision is a compile-time property of the C++ template, so the decoder offers
    a menu of tiers rather than a continuum. A requested precision is rounded *up*
    to the smallest tier that is at least as precise; 24 is IEEE single, 53 is IEEE
    double, and everything above that is a Boost ``cpp_bin_float`` of the stated
    mantissa width.

    Returns:
        List[int]: The available precisions in mantissa bits.
    """


def resolve_precision(requested) -> int:
    """
    Returns the precision tier a request of `requested` mantissa bits will actually
    run at, without having to construct a decoder.

    Args:
        requested (int): The desired number of mantissa bits.

    Returns:
        int: The smallest available tier that is at least as precise.

    Raises:
        ValueError: If `requested` is not a positive integer or exceeds the largest
            compiled-in tier.
    """


class RelayBpDecoderBase:

    """
    Relay Bp Decoder class. This class is identical to BpDecoderBase except it initialises RelayBpDecoderCpp instead of
    BpDecoderCpp. Ideally the code should be restructed so that BpDecoderBase can accept the bp implementation as an
    argument but I don't want to edit the original files. We should also consider refactoring the C++ class structure
    itself to better implement RelayBp as explained in relay_bp.cpp
    """

    def __cinit__(self,pcm, **kwargs): ...


    def __dealloc__(self): ...

    @property
    def error_rate(self) -> np.ndarray:
        """
        Returns the current error rate vector.

        Returns:
            np.ndarray: A numpy array containing the current error rate vector.
        """

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        """
        Sets the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set. Must be a single float value.
        """

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None: ...

    @property
    def channel_probs(self) -> np.ndarray: ...


    @property
    def input_vector_type(self)-> str:
        """
        Returns the current input vector type.

        Returns:
            str: The current input vector type.
        """


    @input_vector_type.setter
    def input_vector_type(self, input_type: str):
        """
        Sets the input vector type.

        Args:
            input_type (str): The input vector type to be set. Must be either 'syndrome' or 'received_vector'.
        """


    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """

    @property
    def max_iter(self) -> int:
        """
        Returns the maximum number of iterations allowed by the decoder.

        Returns:
            int: The maximum number of iterations allowed by the decoder.
        """

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        """
        Sets the maximum number of iterations allowed by the decoder.

        Args:
            value (int): The maximum number of iterations allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def bp_method(self) -> str:
        """
        Returns the belief propagation method used.

        Returns:
            str: The belief propagation method used. Possible values are 'product_sum' or 'minimum_sum'.
        """

    @bp_method.setter
    def bp_method(self, value: Union[str,int]) -> None:
        """
        Sets the belief propagation method used.

        Args:
            value (str): The belief propagation method to use. Possible values are 'product_sum' or 'minimum_sum'.

        Raises:
            ValueError: If value is not a valid option.
        """

    @property
    def schedule(self) -> str:
        """
        Returns the scheduling method used.

        Returns:
            str: The scheduling method used. Possible values are 'parallel' or 'serial'.
        """

    @schedule.setter
    def schedule(self, value: Union[str,int]) -> None:
        """
        Sets the scheduling method used.

        Args:
            value (str): The scheduling method to use. Possible values are 'parallel' or 'serial'.

        Raises:
            ValueError: If value is not a valid option.
        """

    @property
    def serial_schedule_order(self) -> Union[None, np.ndarray]:
        """
        Returns the serial schedule order.

        Returns:
            Union[None, np.ndarray]: The serial schedule order as a numpy array, or None if no schedule has been set.
        """

    @serial_schedule_order.setter
    def serial_schedule_order(self, value: Union[None, List[int], np.ndarray]) -> None:
        """
        Sets the serial schedule order.

        Args:
            value (Union[None, List[int]]): The serial schedule order to set. Must have length equal to the block
            length of the code `self.n`.

        Raises:
            Exception: If value does not have the correct length.
            ValueError: If value contains an invalid integer value.
        """

    @property
    def ms_scaling_factor(self) -> float:
        """Get the scaling factor for minimum sum method.

        Returns:
            float: The current scaling factor.
        """

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value: float) -> None:
        """Set the scaling factor for minimum sum method.

        Args:
            value (float): The new scaling factor.

        Raises:
            TypeError: If the input value is not a float.
        """

    @property
    def omp_thread_count(self) -> int:
        """Get the number of OpenMP threads.

        Returns:
            int: The number of threads used.
        """

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        """Set the number of OpenMP threads.

        Args:
            value (int): The number of threads to use.

        Raises:
            TypeError: If the input value is not an integer or is less than 1.
        """

    @property
    def random_schedule_seed(self) -> int:
        """Get the value of random_schedule_seed.

        Returns:
            int: The current value of random_schedule_seed.
        """

    @random_schedule_seed.setter
    def random_schedule_seed(self, value: int) -> None:
        """Set the value of random_schedule_seed.

        Args:
            value (int): The new value of random_schedule_seed.

        Raises:
            ValueError: If the input value is not a postive integer.
        """

    @property
    def random_serial_schedule(self) -> bool:
        """
        Returns whether the random serial schedule is enabled.

        Returns:
            bool: True if random serial schedule is enabled, False otherwise.
        """

    @random_serial_schedule.setter
    def random_serial_schedule(self, value: bool) -> None:
        """
        Sets whether the random serial schedule is enabled.

        Args:
            value (int): True to enable random serial schedule, False to disable it.

        Raises:
            ValueError: If random serial schedule is enabled while a fixed serial schedule is set.
        """

    @property
    def maximum_legs(self) -> int: ...

    @property
    def maximum_solutions(self) -> int: ...

    @maximum_solutions.setter
    def maximum_solutions(self, value: int) -> int: ...

    @property
    def iterations0(self) -> int: ...

    @iterations0.setter
    def iterations0(self, value: Optional[Union[np.ndarray, List, Tuple]]) -> None: ...

    @property
    def memory_strengths_per_leg(self) -> np.ndarray: ...

    @memory_strengths_per_leg.setter
    def memory_strengths_per_leg(self, value): ...

    @property
    def gamma0(self) -> float: ...

    @gamma0.setter
    def gamma0(self, value: Optional[Union[np.ndarray, List, Tuple]]) -> None: ...

    @property
    def gamma_dist_interval(self) -> np.ndarray: ...

    @gamma_dist_interval.setter
    def gamma_dist_interval(self, value): ...


    @property
    def memory_seed(self) -> int: ...

    @memory_seed.setter
    def memory_seed(self, value: int) -> None: ...

    @property
    def precision(self) -> int:
        """
        The number of mantissa bits the message passing is actually carried out in.

        This is `precision_request` rounded up to the nearest tier the extension
        was compiled with; see `available_precisions()`. Assigning to it changes
        the precision used from the next call to `decode` onwards.
        """

    @precision.setter
    def precision(self, value) -> None: ...

    @property
    def precision_request(self) -> int:
        """
        The precision that was asked for, in mantissa bits, before it was rounded
        up to an available tier. Equal to `precision` when the request landed
        exactly on a tier.
        """

    @property
    def available_precisions(self) -> List[int]:
        """
        The precisions this decoder can be switched between, in mantissa bits.
        """


class RelayBpDecoder(RelayBpDecoderBase):
    """
    Relay belief propagation decoder for binary linear codes.

    This class provides an implementation of relay belief propagation decoding for binary linear codes. The decoder uses a sparse
    parity check matrix to decode received codewords. The decoding algorithm can be configured using various parameters,
    such as the belief propagation method used, the scheduling method used, and the maximum number of iterations.

    This is an extension of the standard belief propagation class.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix of the binary linear code, represented as a NumPy array or a SciPy sparse matrix.
    error_rate : Optional[float], optional
        The initial error rate for the decoder, by default None.
    error_channel : Optional[List[float]], optional
        The initial error channel probabilities for the decoder, by default None.
    maximum_legs: Optional[int] optional,
        The maximum number of legs to run. Will finish early if maximum number of solutions is found.
    maximum_solutions: Optional[int] optional,
        The maximum number of solutions to find. The simulation will finish early if this number of solutions is found.
    iterations0: Optional[int]
        Number of BP iterations run on leg 0 (paired with gamma0)
    gamma0: Optional[float]
        Memory strengths for leg 0. Ignored if memory_strengths_per_leg is provided explicitly
    gamma_dist_interval: Optional[Union[List[float], Tuple]]
        Interval to sample memory strengths from for leg>0. Ignored if memory_strengths_per_leg is provided explicitly
    memory_strengths_per_leg: Optional[np.array] optional,
        The memory strengths to use on each leg. This must be a list of length 'maximum_legs' where each element is a
        list of floats representing the memory on each qubit for that leg.
    max_iter : Optional[int], optional
        The maximum number of iterations allowed for decoding, by default 0 (adaptive).
    bp_method : Optional[str], optional
        The belief propagation method to use: 'product_sum' or 'minimum_sum', by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor for the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method for belief propagation: 'parallel', 'serial', or 'serial_relative'. By default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads to use, by default 1.
    random_schedule_seed : Optional[int], optional
        The seed for the random serial schedule, by default 0. If set to 0, the seed is set according to the system clock.
    serial_schedule_order : Optional[List[int]], optional
        The custom order for serial scheduling, by default None.
    random_serial_schedule : bool, optional
        Whether to enable random serial scheduling. If True, the serial schedule order is randomized in each iteration.
        By default False.
    input_vector_type: str, optional
        Use this parameter to specify the input type. Choose either: 1) 'syndrome' or 2) 'received_vector' or 3) 'auto'.
        Note, it is only necessary to specify this value when the parity check matrix is square. When the
        parity matrix is non-square, the input vector type is inferred automatically from its length.
    memory_seed: int, optional
        seed for the per-leg memory strength RNG; -1 -> seed non-deterministically
    precision : Optional[int], optional
        The size, in mantissa bits, of the floating point numbers used to carry the
        messages in the message passing algorithm. By default 53, which is IEEE
        double precision and reproduces the decoder's previous behaviour exactly.
        24 gives IEEE single precision, and anything above 53 is run in Boost
        ``cpp_bin_float`` software floating point of the requested width, so the
        precision can be pushed arbitrarily high at a cost in runtime.

        Because the working type is a C++ template parameter, the extension is
        compiled with a menu of tiers rather than a continuum: a request is rounded
        *up* to the smallest available tier that is at least as precise, and the
        value actually in use is reported by the `precision` attribute.
        `available_precisions()` lists the tiers, and tiers can be added or removed
        by defining ``RELAY_BP_PRECISION_TIERS`` before ``relay_bp.hpp`` is included.

        Only the message passing is affected. The error channel, the memory
        strengths and the reported `log_prob_ratios` remain float64, so every
        existing attribute keeps the type and meaning it always had.

        Note that the emulated types above 53 bits widen the mantissa but not the
        exponent range, so the 11 bit tier is not a faithful model of IEEE half
        precision (and, being software floating point, it is slower than 24 or 53,
        not faster).
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, maximum_legs: Optional[int] = 1,
                 maximum_solutions: Optional[int] = 1, iterations0: Optional[int] = None, gamma0: Optional[int] = None,
                 gamma_dist_interval: Optional[Union[List[float], Tuple]] = None,
                 memory_strengths_per_leg: Optional[Union[np.ndarray, List, Tuple]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, input_vector_type: str = "auto", random_serial_schedule: bool = False,
                 memory_seed: Optional[int] = -1, precision: Optional[int] = None, **kwargs): ...

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, maximum_legs: Optional[int] = 1,
                                 maximum_solutions: Optional[int] = 1, iterations0: Optional[int] = None, gamma0: Optional[int] = None,
                                 gamma_dist_interval: Optional[Union[List[float], Tuple]] = None,
                                 memory_strengths_per_leg: Optional[Union[np.ndarray, List, Tuple]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, input_vector_type: str = "auto", random_serial_schedule: bool = False,
                                 memory_seed: Optional[int] = -1, precision: Optional[int] = None, **kwargs): ...

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input input_vector using belief propagation decoding algorithm.

        Parameters
        ----------
        input_vector : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input input_vector does not match the number of rows in the parity check matrix.
        """


    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """

    @property
    def solution_number(self) -> int: ...

    @property
    def iterations(self) -> int: ...
