"""Hyperparameter optimization algorithm implementation.

Includes Grid Search and Particle Swarm Optimization (PSO) algorithms.
"""

import itertools
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np


class GridSearchOptimizer:
    """Grid Search Optimizer.

    Performs an exhaustive search on all parameter combinations in the search space.
    Suitable for scenarios with small search spaces.

    Attributes:
        search_space: Search space dictionary, key is parameter name, value is list of candidate values
    """
    
    def __init__(self, search_space: Dict[str, List[Any]]) -> None:
        """Initialize the Grid Search optimizer.

        Args:
            search_space: Search space, e.g., {'lr': [0.001, 0.0001], 'd_model': [16, 32]}
        """
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        self.param_values = [search_space[name] for name in self.param_names]
    
    def get_total_combinations(self) -> int:
        """Get the total number of parameter combinations."""
        total = 1
        for values in self.param_values:
            total *= len(values)
        return total
    
    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """Execute grid search.

        Args:
            evaluate_fn: Evaluation function, takes parameter dictionary, returns performance score (lower is better)

        Returns:
            Best parameter dictionary and corresponding performance score
        """
        best_params = None
        best_score = float('inf')
        
        # Generate all parameter combinations
        for values in itertools.product(*self.param_values):
            params = dict(zip(self.param_names, values))

            # Evaluate current parameter combination
            score = evaluate_fn(params)

            # Update best result
            if score < best_score:
                best_score = score
                best_params = params.copy()
        
        return best_params, best_score


class PSOOptimizer:
    """Particle Swarm Optimizer.

    Uses particle swarm optimization algorithm to search for the best hyperparameters.
    Suitable for continuous or large-scale discrete search spaces.

    Attributes:
        search_space: Search space dictionary
        n_particles: Number of particles
        n_iterations: Number of iterations
        w: Inertia weight
        c1: Individual learning factor
        c2: Social learning factor
    """
    
    def __init__(self,
                 search_space: Dict[str, Union[List[Any], Tuple[float, float]]],
                 n_particles: int = 20,
                 n_iterations: int = 50,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5) -> None:
        """Initialize the Particle Swarm Optimizer.

        Args:
            search_space: Search space, supports two formats:
                - Discrete: {'lr': [0.001, 0.0001, 0.00001]}
                - Continuous: {'lr': (0.00001, 0.001)}
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight, controls the extent to which the particle maintains its current velocity
            c1: Individual learning factor, controls the extent to which the particle moves towards its individual best position
            c2: Social learning factor, controls the extent to which the particle moves towards the global best position
        """
        self.search_space = search_space
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parse search space
        self.param_names: List[str] = []
        self.param_types: List[str] = []  # 'continuous' or 'discrete'
        self.param_bounds: List[Tuple[float, float]] = []
        self.param_choices: List[List[Any]] = []

        for name, space in search_space.items():
            self.param_names.append(name)

            if isinstance(space, (list, tuple)) and len(space) == 2 and \
               isinstance(space[0], (int, float)) and isinstance(space[1], (int, float)) and \
               space[0] < space[1]:
                # Continuous space
                self.param_types.append('continuous')
                self.param_bounds.append((float(space[0]), float(space[1])))
                self.param_choices.append([])
            else:
                # Discrete space
                self.param_types.append('discrete')
                choices = list(space) if not isinstance(space, list) else space
                self.param_choices.append(choices)
                self.param_bounds.append((0, len(choices) - 1))
        
        self.n_dims = len(self.param_names)
    
    def _initialize_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle positions and velocities.

        Returns:
            positions: Particle position matrix (n_particles, n_dims)
            velocities: Particle velocity matrix (n_particles, n_dims)
        """
        positions = np.zeros((self.n_particles, self.n_dims))
        velocities = np.zeros((self.n_particles, self.n_dims))
        
        for i in range(self.n_dims):
            lower, upper = self.param_bounds[i]
            positions[:, i] = np.random.uniform(lower, upper, self.n_particles)
            velocities[:, i] = np.random.uniform(-1, 1, self.n_particles) * (upper - lower) * 0.1
        
        return positions, velocities
    
    def _position_to_params(self, position: np.ndarray) -> Dict[str, Any]:
        """Convert particle position to parameter dictionary.

        Args:
            position: Particle position vector

        Returns:
            Parameter dictionary
        """
        params = {}
        
        for i, name in enumerate(self.param_names):
            if self.param_types[i] == 'continuous':
                params[name] = float(position[i])
            else:  # discrete
                idx = int(np.clip(np.round(position[i]), 0, len(self.param_choices[i]) - 1))
                params[name] = self.param_choices[i][idx]
        
        return params
    
    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip particle position within the search space.

        Args:
            position: Particle position

        Returns:
            Clipped position
        """
        clipped = position.copy()
        for i in range(self.n_dims):
            lower, upper = self.param_bounds[i]
            clipped[i] = np.clip(clipped[i], lower, upper)
        return clipped
    
    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """Execute particle swarm optimization.

        Args:
            evaluate_fn: Evaluation function, takes parameter dictionary, returns performance score (lower is better)

        Returns:
            Best parameter dictionary and corresponding performance score
        """
        # Initialize particles
        positions, velocities = self._initialize_particles()

        # Initialize individual best and global best
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, float('inf'))
        gbest_position = None
        gbest_score = float('inf')

        # Evaluate initial positions
        for i in range(self.n_particles):
            params = self._position_to_params(positions[i])
            score = evaluate_fn(params)
            pbest_scores[i] = score

            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

        # Check if there is a valid initial solution
        if gbest_position is None:
            raise RuntimeError(
                f"PSO optimization failed: Evaluation failed for all {self.n_particles} initial particles."
                "Please check: 1) if the dataset path is correct, 2) if run_main.py runs normally, 3) if the output parsing logic is correct"
            )

        # Iterative optimization
        for iteration in range(self.n_iterations):
            print(f"\nPSO Iteration {iteration + 1}/{self.n_iterations}")
            print(f"Current global best score: {gbest_score:.4f}")

            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                    self.c2 * r2 * (gbest_position - positions[i])
                )

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = self._clip_position(positions[i])

                # Evaluate new position
                params = self._position_to_params(positions[i])
                score = evaluate_fn(params)

                # Update individual best
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()

                    # Update global best
                    if score < gbest_score:
                        gbest_score = score
                        gbest_position = positions[i].copy()

        # Return best parameters
        best_params = self._position_to_params(gbest_position)
        return best_params, gbest_score


class RandomSearchOptimizer:
    """Random Search Optimizer.

    Randomly sample parameter combinations in the search space for evaluation.
    Computational cost is between grid search and PSO.

    Attributes:
        search_space: Search space dictionary
        n_trials: Number of random trials
    """
    
    def __init__(self,
                 search_space: Dict[str, Union[List[Any], Tuple[float, float]]],
                 n_trials: int = 50) -> None:
        """Initialize the Random Search optimizer.

        Args:
            search_space: Search space
            n_trials: Number of random trials
        """
        self.search_space = search_space
        self.n_trials = n_trials
    
    def _sample_params(self) -> Dict[str, Any]:
        """Randomly sample a set of parameters from the search space.

        Returns:
            Parameter dictionary
        """
        params = {}
        
        for name, space in self.search_space.items():
            if isinstance(space, (list, tuple)) and len(space) == 2 and \
               isinstance(space[0], (int, float)) and isinstance(space[1], (int, float)) and \
               space[0] < space[1]:
                # Continuous space: uniform sampling
                params[name] = np.random.uniform(space[0], space[1])
            else:
                # Discrete space: random selection
                choices = list(space) if not isinstance(space, list) else space
                params[name] = np.random.choice(choices)
        
        return params
    
    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """Execute random search.

        Args:
            evaluate_fn: Evaluation function

        Returns:
            Best parameter dictionary and corresponding performance score
        """
        best_params = None
        best_score = float('inf')
        
        for trial in range(self.n_trials):
            # Randomly sample parameters
            params = self._sample_params()

            # Evaluate
            score = evaluate_fn(params)

            # Update best result
            if score < best_score:
                best_score = score
                best_params = params.copy()

            print(f"Random search trial {trial + 1}/{self.n_trials}, current best: {best_score:.4f}")
        
        return best_params, best_score
