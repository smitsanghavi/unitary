# Copyright 2022 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import itertools

import numpy as np

from typing import Tuple


def num_bits(num: int) -> int:
    """Returns the minimum number of bits needed to represent the input."""
    result = 1
    while num > 2**result:
        result += 1
    return result


def _nearest_power_of_two_ceiling(qudit_dim: int) -> int:
    """Returns the smallest power of two greater than or equal to qudit_dim."""
    if qudit_dim == 0:
        return 0
    result = 1
    while result < qudit_dim:
        result = result << 1
    return result


def qudit_to_qubit_state(
    qudit_state_vector: np.ndarray,
    qid_shape: Tuple[int, ...],
    _pad_value: np.complex_ = 0,
) -> np.ndarray:
    """Converts a qudit-space quantum state vector to m-qubit-per-qudit column vector.

    Each qudit is replaced by a set of qubits. Since the set of qubits can represent a larger state
    space than the qudit, the state vector needs to be padded with 0s for those extra elements.
    Based on https://drive.google.com/file/d/1n1Ym7JdnM44NnvNQDUXSk5rBLTQZzJ9t and
    https://colab.research.google.com/drive/1MC6D4FyOXG0RjvzyV9IFcdRsLbUwwAcx

    Args:
        qudit_state_vector: A numpy array representing the state vector in qudit form.
            Expected shape: `(qudit_dimension ^ num_qudits,)`.
        qid_shape: Dimensions of the input qudits.
        _pad_value: The value to set for the elements in the extra state space created in the
            conversion. This field is mostly private, mostly for use by other methods. Do not set
            this unless you really need to.

    Returns:
        A flat numpy array representing the input state vector using m-qubits-per-qudit.
            Expected shape: `((2 ^ m) ^ num_qudits,)`.
    """
    # Reshape the state vector to a `num_qudits` rank tensor.
    state_tensor = qudit_state_vector.reshape(qid_shape)
    # Number of extra elements needed in each dimension if represented using qubits.
    padding_amount = tuple([(0, _nearest_power_of_two_ceiling(qid_dim) - qid_dim) for qid_dim in qid_shape]) or 0
    # Expand the number of elements in each dimension by the padding_amount. Fill
    # the new elements with the _pad_value.
    padded_state_tensor = np.pad(
        state_tensor, pad_width=padding_amount, constant_values=_pad_value
    )
    # Return a flattened state vector view of the final tensor.
    return np.ravel(padded_state_tensor)


def qubit_to_qudit_state(
    qubit_state_vector: np.ndarray,
    qid_shape: Tuple[int, ...],
) -> np.ndarray:
    """Converts a m-qubit-per-qudit column vector to a qudit-space quantum state vector.

    Each qudit was replaced by a set of qubits. Since the set of qubits could represent a larger
    state space than the qudit, the state vector needs to be sliced up to the qudit length in each
    dimension.
    Based on https://drive.google.com/file/d/1n1Ym7JdnM44NnvNQDUXSk5rBLTQZzJ9t and
    https://colab.research.google.com/drive/1MC6D4FyOXG0RjvzyV9IFcdRsLbUwwAcx

    Args:
        qubit_state_vector: A numpy array representing the state vector in an
            m-qubit-per-qudit form. Expected shape: `((2 ^ m) ^ num_qudits,)`.
        qid_shape: Dimensions of the output qudits.

    Returns:
        A flat numpy array representing the input state vector using qudits.
            Expected shape: `(qudit_dimension ^ num_qudits,)`.
    """
    mbit_shape = [_nearest_power_of_two_ceiling(qid_dim) for qid_dim in qid_shape]
    # Reshape the state vector to a `num_qudits` rank tensor.
    state_tensor = qubit_state_vector.reshape(mbit_shape)
    # Shrink the number of elements in each dimension up to the qudit_dimension, ignoring the rest.
    slices = tuple([slice(qid_dim) for qid_dim in qid_shape])
    trimmed_state_tensor = state_tensor[slices]
    # Return a flattened state vector view of the final tensor.
    return np.ravel(trimmed_state_tensor)


def qudit_to_qubit_unitary(
    qudit_unitary: np.ndarray,
    qid_shape: Tuple[int, ...],
    memoize: bool = True,
) -> np.ndarray:
    """Converts a qudit-space quantum unitary to m-qubit-per-qudit unitary.

    Each qudit is replaced by a set of qubits. Since the set of qubits can represent a larger state
    space than the qudit, the unitary needs to be padded with 0s for those extra elements. A
    unitary is treated similar to a 2*num_qudits system's state vector and padded using the state
    vector protocol. The resulting unitary is updated to have the extra dimensions map to
    themselves (identity) to preserve unitarity.
    Based on https://drive.google.com/file/d/1n1Ym7JdnM44NnvNQDUXSk5rBLTQZzJ9t and
    https://colab.research.google.com/drive/1MC6D4FyOXG0RjvzyV9IFcdRsLbUwwAcx

    Args:
        qudit_dimension: The dimension of a single qudit i.e. the number of states it can
            represent.
        num_qudits: The number of qudits in the given unitary.
        qudit_unitary: A 2-D numpy array representing the unitary in qudit form.
            Expected shape: `(qudit_dimension ^ num_qudits, qudit_dimension ^ num_qudits)`.
        memoize: Currently, this method has two independent implementations. If memoize is True, an
            alternate implementation than above is used. A special state vector is passed to the
            state vector protocol to get a mapping from qudit state indices to qubit state indices.
            This mapping is then iteratively applied to the input unitary's elements.

    Returns:
        A numpy array representing the input unitary using m-qubits-per-qudit.
            Expected shape: `((2 ^ m) ^ num_qudits, (2 ^ m) ^ num_qudits)`.
    """
    qubit_shape = [_nearest_power_of_two_ceiling(qid_dim) for qid_dim in qid_shape]
    dim_qubit_space = int(np.prod(qubit_shape))
    dim_qudit_space = int(np.prod(qid_shape))

    if memoize:
        # Perform the transform of the below array from qubit to qudit space so that the indices
        # represent the position in qudit space and the values represent the position in the qubit
        # space.
        d_to_b_index_map = qubit_to_qudit_state(
            # An array of ints from 0 to dim_qubit_space. Each element represents the original
            # index.
            np.arange(dim_qubit_space),
            qid_shape=qid_shape,
        )
        # Initialize the result to the identity unitary in the qubit space.
        result = np.identity(dim_qubit_space, dtype=qudit_unitary.dtype)
        # Iterate over each element in the qudit space dimension x qudit space dimension.
        for i, j in itertools.product(range(dim_qudit_space), range(dim_qudit_space)):
            # Use the index map to populate the appropriate element in the qubit representation.
            result[d_to_b_index_map[i]][d_to_b_index_map[j]] = qudit_unitary[i][j]
        return result

    # # Treat the unitary as a num_qudits^2 system's state vector and represent it using qubits (pad
    # # with 0s).
    # padded_unitary = qudit_to_qubit_state(
    #     qudit_dimension, num_qudits * 2, np.ravel(qudit_unitary)
    # )
    # # A qubit-based state vector with the extra padding bits having 1s and rest having 0s. This
    # # vector marks only the bits that are padded.
    # pad_qubits_vector = qudit_to_qubit_state(
    #     qudit_dimension,
    #     num_qudits,
    #     np.zeros(qudit_dimension**num_qudits),
    #     _pad_value=1,
    # )
    # # Reshape the padded unitary to the final shape and add a diagonal matrix corresponding to the
    # # pad_qubits_vector. This addition ensures that the invalid states with the "padding" bits map
    # # to identity, preserving unitarity.
    # return padded_unitary.reshape(dim_qubit_space, dim_qubit_space) + np.diag(
    #     pad_qubits_vector
    # )


def qubit_to_qudit_unitary(
    qubit_unitary: np.ndarray,
    qid_shape: Tuple[int, ...],
):
    """Converts a m-qubit-per-qudit unitary to a qudit-space quantum unitary.

    Each qudit was replaced by a set of qubits. Since the set of qubits could represent a larger
    state space than the qudit, the unitary needs to be sliced up to the qudit length in each
    dimension. A unitary is treated similar to a 2*num_qudits system's state vector.
    Based on https://drive.google.com/file/d/1n1Ym7JdnM44NnvNQDUXSk5rBLTQZzJ9t and
    https://colab.research.google.com/drive/1MC6D4FyOXG0RjvzyV9IFcdRsLbUwwAcx

    Args:
        qudit_dimension: The dimension of a single qudit i.e. the number of states it can
            represent.
        num_qudits: The number of qudits in the given/output unitary.
        qubit_unitary: A 2-D numpy array representing the unitary in m-qubit-per-qudit form.
            Expected shape: `((2 ^ m) ^ num_qudits, (2 ^ m) ^ num_qudits)`.

    Returns:
        A numpy array representing the input unitary using qudits.
            Expected shape: `(qudit_dimension ^ num_qudits, qudit_dimension ^ num_qudits)`.
    """
    qubit_shape = [_nearest_power_of_two_ceiling(qid_dim) for qid_dim in qid_shape]
    # Treat unitary as a `num_qudits*2` qudit system state vector.
    effective_qubit_shape = qubit_shape * 2
    # Reshape the state vector to a `num_qudits*2` rank tensor.
    unitary_tensor = qubit_unitary.reshape(effective_qubit_shape)
    # Shrink the number of elements in each dimension up to the qudit_dimension, ignoring the rest.
    slices = tuple([slice(qid_dim) for qid_dim in qid_shape])*2
    trimmed_unitary_tensor = unitary_tensor[slices]
    dim_qudit_space = int(np.prod(qid_shape))
    # Return a flat unitary view of the final tensor.
    return trimmed_unitary_tensor.reshape(
        dim_qudit_space, dim_qudit_space
    )
