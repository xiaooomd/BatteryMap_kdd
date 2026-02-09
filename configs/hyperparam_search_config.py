"""Hyperparameter search space configuration.

Defines search spaces for different models and tasks.
Supports two formats: discrete value lists and continuous intervals.

Format description:
    - Discrete space: List, e.g., [0.001, 0.0001, 0.00001]
    - Continuous space: Tuple, e.g., (0.00001, 0.001) represents a continuous interval [0.00001, 0.001]
"""

from typing import Dict, List, Any, Union, Tuple


# Common hyperparameter search space (applicable to most models)
COMMON_SEARCH_SPACE = {
    'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
    'batch_size': [16, 32, 64],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'wd': [0.0, 0.0001, 0.001],  # weight decay
}


# MLP model search space
MLP_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'd_ff': [32, 64, 128, 256],
    'e_layers': [1, 2, 3, 4],
    'd_layers': [1, 2, 3],
}


# CPMLP model search space (Charge-Profile MLP)
CPMLP_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'd_ff': [32, 64, 128, 256],
    'e_layers': [1, 2, 3],
    'd_layers': [1, 2],
    'n_heads': [2, 4, 8],  # If using attention
}


# Transformer series model search space
TRANSFORMER_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'n_heads': [2, 4, 8],
    'e_layers': [1, 2, 3],
    'd_layers': [1, 2],
    'd_ff': [32, 64, 128, 256],
    'factor': [1, 3, 5],
}


# LSTM/GRU series model search space
RNN_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'lstm_layers': [1, 2, 3],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
}


# CNN model search space
CNN_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'e_layers': [2, 3, 4, 5],
    'd_ff': [32, 64, 128],
}


# PatchTST model search space
PATCHTST_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [16, 32, 64, 128],
    'n_heads': [2, 4, 8],
    'e_layers': [1, 2, 3],
    'd_ff': [64, 128, 256],
    'patch_len': [5, 10, 16, 20],
    'stride': [5, 10, 16, 20],
}


# iTransformer model search space
ITRANSFORMER_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [32, 64, 128],
    'n_heads': [4, 8],
    'e_layers': [2, 3, 4],
    'd_ff': [128, 256, 512],
}


# DLinear model search space
DLINEAR_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'moving_avg': [5, 10, 25, 50],
}


# Autoformer model search space
AUTOFORMER_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [32, 64, 128],
    'n_heads': [4, 8],
    'e_layers': [2, 3],
    'd_layers': [1, 2],
    'd_ff': [128, 256],
    'moving_avg': [25, 50, 75],
    'factor': [1, 3, 5],
}


# MICN model search space
MICN_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [32, 64, 128],
    'e_layers': [2, 3, 4],
    'd_ff': [64, 128, 256],
}


# Search space for feature extraction mode (for extracted_features)
FEATURE_MODE_SEARCH_SPACE = {
    **COMMON_SEARCH_SPACE,
    'd_model': [32, 64, 128, 256],
    'd_ff': [64, 128, 256, 512],
    'e_layers': [1, 2, 3, 4],  # Modified to be consistent with MLP: [1, 2, 3, 4]
    'd_layers': [1, 2, 3],
    'lstm_layers': [1, 2, 3],  # Added: ensure number of layers for RNN-like models can be optimized
    'dropout': [0.1, 0.2, 0.3, 0.4],
}


# PSO continuous search space example (for particle swarm optimization)
# Note: Continuous space is represented by a tuple (min, max)
PSO_CONTINUOUS_SPACE = {
    'learning_rate': (0.00001, 0.001),  # Continuous interval
    'd_model': [16, 32, 64, 128],  # Discrete options
    'dropout': (0.0, 0.5),  # Continuous interval
    'batch_size': [16, 32, 64],  # Discrete options
    'd_ff': (32, 256),  # Continuous interval
    'wd': (0.0, 0.01),  # Continuous interval
}


def get_search_space(model_name: str, feature_type: str = 'curve') -> Dict[str, Union[List[Any], Tuple[float, float]]]:
    """Get the search space based on model name and feature type.

    Args:
        model_name: Model name, e.g., 'MLP', 'CPMLP', 'Transformer'
        feature_type: Feature type, 'curve' or 'extracted_features'

    Returns:
        Search space dictionary

    Raises:
        ValueError: If the model name is not supported
    """
    # Standardize model name
    model_name = model_name.upper()

    # If it is feature extraction mode, use a dedicated search space
    if feature_type == 'extracted_features':
        return FEATURE_MODE_SEARCH_SPACE.copy()

    # Select search space based on model type
    if model_name == 'MLP':
        return MLP_SEARCH_SPACE.copy()
    elif model_name == 'CPMLP':
        return CPMLP_SEARCH_SPACE.copy()
    elif model_name in ['TRANSFORMER', 'CPTRANSFORMER']:
        return TRANSFORMER_SEARCH_SPACE.copy()
    elif model_name in ['LSTM', 'CPLSTM', 'BILSTM', 'CPBILSTM', 'GRU', 'CPGRU', 'BIGRU', 'CPBIGRU']:
        return RNN_SEARCH_SPACE.copy()
    elif model_name == 'CNN':
        return CNN_SEARCH_SPACE.copy()
    elif model_name == 'PATCHTST':
        return PATCHTST_SEARCH_SPACE.copy()
    elif model_name == 'ITRANSFORMER':
        return ITRANSFORMER_SEARCH_SPACE.copy()
    elif model_name == 'DLINEAR':
        return DLINEAR_SEARCH_SPACE.copy()
    elif model_name == 'AUTOFORMER':
        return AUTOFORMER_SEARCH_SPACE.copy()
    elif model_name == 'MICN':
        return MICN_SEARCH_SPACE.copy()
    else:
        # Default to common search space
        print(f"Warning: No dedicated search space configuration for model {model_name}, using common configuration")
        return COMMON_SEARCH_SPACE.copy()


def get_pso_search_space(model_name: str) -> Dict[str, Union[List[Any], Tuple[float, float]]]:
    """Get the continuous search space for PSO optimization.

    For PSO, some hyperparameters can be defined as continuous intervals to improve search efficiency.

    Args:
        model_name: Model name

    Returns:
        PSO search space dictionary
    """
    # Get the base search space
    base_space = get_search_space(model_name)

    # Convert some parameters to continuous space (optional)
    pso_space = base_space.copy()

    # Use continuous space for learning rate
    if 'learning_rate' in pso_space:
        pso_space['learning_rate'] = (0.00001, 0.001)

    # Use continuous space for dropout
    if 'dropout' in pso_space:
        pso_space['dropout'] = (0.0, 0.5)

    # Use continuous space for weight decay
    if 'wd' in pso_space:
        pso_space['wd'] = (0.0, 0.01)

    return pso_space


def customize_search_space(base_space: Dict[str, Union[List[Any], Tuple[float, float]]],
                          custom_params: Dict[str, Union[List[Any], Tuple[float, float]]]) -> Dict[str, Union[List[Any], Tuple[float, float]]]:
    """Customize search space.

    Allows users to override or add specific hyperparameter search ranges.

    Args:
        base_space: Base search space
        custom_params: Custom parameter configuration

    Returns:
        Merged search space

    Example:
        >>> base = get_search_space('MLP')
        >>> custom = {'learning_rate': [0.0001, 0.00001], 'new_param': [1, 2, 3]}
        >>> final_space = customize_search_space(base, custom)
    """
    result = base_space.copy()
    result.update(custom_params)
    return result


# Predefined search configuration examples
QUICK_SEARCH = {
    """Quick search configuration (small number of parameter combinations for fast validation)"""
    'learning_rate': [0.0001, 0.00005],
    'd_model': [32, 64],
    'batch_size': [32],
    'dropout': [0.1, 0.2],
}


THOROUGH_SEARCH = {
    """Thorough search configuration (large number of parameter combinations for final optimization)"""
    'learning_rate': [0.001, 0.0005, 0.0001, 0.00005, 0.00001],
    'd_model': [16, 32, 64, 128, 256],
    'batch_size': [8, 16, 32, 64],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'd_ff': [32, 64, 128, 256, 512],
    'e_layers': [1, 2, 3, 4, 5],
}
