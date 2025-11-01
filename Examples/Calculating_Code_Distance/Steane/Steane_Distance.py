import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'LEGO_HQEC'))

from LEGO_HQEC.OperatorPush.Presets.Holographic_Steane_code import setup_heptagon_zero_rate_steane
from LEGO_HQEC.OperatorPush.PushingToolbox import batch_push, batch_push_multiprocessing
from LEGO_HQEC.QuDec.InputProcessor import extract_stabilizers_from_result_dict, extract_logicals_from_result_dict
from LEGO_HQEC.DIstanceFind.OperatorProcessor import (pauli_to_binary_vector, batch_convert_to_binary_vectors,
                                                      apply_mod2_sum, binary_vector_to_pauli)
from LEGO_HQEC.DIstanceFind.DistanceFInder import minimize_logical_operator_weight, calculate_pauli_weight

if __name__ == '__main__':
    # Initialize parameters
    R = 1  # Radius

    # Set up the Holographic Steane code tensor network
    tensor_list = setup_heptagon_zero_rate_steane(R=R)

    # Perform parallel operator pushing operations on the tensor network
    results_dict = batch_push_multiprocessing(tensor_list)

    # Extract stabilizers and logical operators from the results (in string form)
    stabilizers = extract_stabilizers_from_result_dict(results_dict)
    logical_zs, logical_xs = extract_logicals_from_result_dict(results_dict)

    # Convert stabilizers to binary/symplectic representation
    symplectic_stabilizers = batch_convert_to_binary_vectors(stabilizers)

    # Convert the first Z-type logical operator to binary vector (to calculate the distance of this operator)
    L = pauli_to_binary_vector(logical_zs[0])

    # Find the minimal weight equivalent logical operator by solving a MIP problem
    lambda_results = minimize_logical_operator_weight(L=L, stabilizers=symplectic_stabilizers,
                                                      mip_focus=3, heuristics=0.2)

    # Apply the found coefficients to get the minimal weight operator in binary form
    binary_result = apply_mod2_sum(L=L, stabilizers=symplectic_stabilizers,
                                   lambda_values=lambda_results)

    # Convert the binary result back to Pauli operator form
    result = binary_vector_to_pauli(binary_result)

    # Calculate and print the weight (number of non-identity terms) of the minimal operator
    min_wt = calculate_pauli_weight(result)
    print(f"Distance of the chosen operator = {min_wt}")  # This is the distance for the Z logical operator
