
from typing import Dict, List
import os
import glob
import fcntl
import contextlib
import numpy as np
from numpy.typing import NDArray

from lerobothackathonenv.structs import MujocoState


class TrajectorySet():
    def __init__(self, data_dir: str = "trajectories"):
        self.data_dir = data_dir
        self.cache = {}  # In-memory cache for trajectories
        self._assure_file_dir()
        self.index = self._reconstruct_index()

    def _assure_file_dir(self,):
        """
        Makes the file dir if it doesn't exist
        """
        os.makedirs(self.data_dir, exist_ok=True)

    @contextlib.contextmanager
    def _file_lock(self, filepath, mode='r'):
        """
        Context manager for file locking operations
        """
        lock_filepath = filepath + '.lock'
        lock_file = None
        
        try:
            # Create/open lock file
            lock_file = open(lock_filepath, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            yield filepath
            
        finally:
            if lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Clean up lock file
                try:
                    os.remove(lock_filepath)
                except OSError:
                    pass

    def _reconstruct_index(self,):
        """
        reconstruct what index should be, based on trajectories in the file dir
        """
        existing_files = glob.glob(os.path.join(self.data_dir, "trajectory_*.npz"))
        if not existing_files:
            return 0
        
        indices = []
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            try:
                # Parse trajectory_XXXXXX.npz format
                if not filename.startswith('trajectory_') or not filename.endswith('.npz'):
                    continue
                index_str = filename[len('trajectory_'):-len('.npz')]
                index = int(index_str)
                indices.append(index)
            except (IndexError, ValueError):
                continue
        
        return max(indices) + 1 if indices else 0

    def record_trajectory(
        self,
        observations: List[NDArray],
        rewards: List[float],
        terminations: List[bool],
        trunctuations: List[bool],
        infos: List[Dict[str, NDArray | float]],
        simulator_states: List = None
    ):
        """
        Saves the trajectory in file dir with unique index that is maintained
        even between script runs.
        """
        if not observations:
            return
        
        filename = f"trajectory_{self.index:06d}.npz"
        filepath = os.path.join(self.data_dir, filename)
        
        with self._file_lock(filepath, 'w'):
            # Convert dict observations to a format that can be saved
            obs_keys = list(observations[0].keys()) if observations else []
            obs_arrays = {}
            for key in obs_keys:
                obs_arrays[f'obs_{key}'] = np.array([obs[key] for obs in observations])
            
            # Convert simulator states to arrays
            sim_state_arrays = {}
            if simulator_states:
                sim_state_arrays['sim_time'] = np.array([state.time for state in simulator_states])
                sim_state_arrays['sim_qpos'] = np.array([state.qpos for state in simulator_states])
                sim_state_arrays['sim_qvel'] = np.array([state.qvel for state in simulator_states])
                sim_state_arrays['sim_xpos'] = np.array([state.xpos for state in simulator_states])
                sim_state_arrays['sim_xquat'] = np.array([state.xquat for state in simulator_states])
                sim_state_arrays['sim_mocap_pos'] = np.array([state.mocap_pos for state in simulator_states])
                sim_state_arrays['sim_mocap_quat'] = np.array([state.mocap_quat for state in simulator_states])
                sim_state_arrays['sim_metadata'] = [state.sim_metadata for state in simulator_states]
            
            np.savez_compressed(
                filepath,
                **obs_arrays,
                **sim_state_arrays,
                rewards=np.array(rewards),
                terminations=np.array(terminations),
                trunctuations=np.array(trunctuations),
                infos=infos,
                obs_keys=obs_keys,
                has_sim_states=simulator_states is not None
            )
        
        print(f"Saved trajectory {self.index} to {filepath}")
        self.index += 1

    def get_trajectory(
        self,
        index: int
    ):
        """
        Gets a specific trajectory from cache or file
        """
        # Check cache first
        if index in self.cache:
            return self.cache[index]
        
        # Load from file if not in cache
        filename = f"trajectory_{index:06d}.npz"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory {index} not found at {filepath}")
        
        with self._file_lock(filepath, 'r'):
            data = np.load(filepath, allow_pickle=True)
            
            # Reconstruct dict observations
            obs_keys = data['obs_keys'] if 'obs_keys' in data else []
            # Handle case where obs_keys might be a scalar array
            if hasattr(obs_keys, 'item'):
                try:
                    obs_keys = obs_keys.item()
                except ValueError:
                    # If it's not a scalar, use as-is
                    pass
            # Ensure it's a list
            if not isinstance(obs_keys, list):
                obs_keys = list(obs_keys) if hasattr(obs_keys, '__iter__') else []
            observations = []
            if obs_keys:
                num_obs = len(data[f'obs_{obs_keys[0]}'])
                for i in range(num_obs):
                    obs_dict = {}
                    for key in obs_keys:
                        obs_dict[key] = data[f'obs_{key}'][i]
                    observations.append(obs_dict)
            
            # Reconstruct simulator states if they exist
            simulator_states = []
            has_sim_states = data.get('has_sim_states', False)
            if has_sim_states:
                num_states = len(data['sim_time'])
                for i in range(num_states):
                    state = MujocoState(
                        time=data['sim_time'][i],
                        qpos=data['sim_qpos'][i],
                        qvel=data['sim_qvel'][i],
                        xpos=data['sim_xpos'][i],
                        xquat=data['sim_xquat'][i],
                        mocap_pos=data['sim_mocap_pos'][i],
                        mocap_quat=data['sim_mocap_quat'][i],
                        sim_metadata=data['sim_metadata'][i]
                    )
                    simulator_states.append(state)
            
            trajectory_data = {
                'observations': observations,
                'rewards': data['rewards'],
                'terminations': data['terminations'],
                'trunctuations': data['trunctuations'],
                'infos': data['infos'],
                'simulator_states': simulator_states
            }
        
        # Cache the loaded trajectory
        self.cache[index] = trajectory_data
        return trajectory_data

    def load_into_cache(self):
        """
        Load all trajectories with unknown indices into the cache
        """
        existing_files = glob.glob(os.path.join(self.data_dir, "trajectory_*.npz"))
        
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            try:
                # Parse trajectory_XXXXXX.npz format
                if not filename.startswith('trajectory_') or not filename.endswith('.npz'):
                    continue
                index_str = filename[len('trajectory_'):-len('.npz')]
                index = int(index_str)
                if index not in self.cache:
                    with self._file_lock(file_path, 'r'):
                        data = np.load(file_path, allow_pickle=True)
                        
                        # Reconstruct dict observations
                        obs_keys = data['obs_keys'] if 'obs_keys' in data else []
                        # Handle case where obs_keys might be a scalar array
                        if hasattr(obs_keys, 'item'):
                            try:
                                obs_keys = obs_keys.item()
                            except ValueError:
                                # If it's not a scalar, use as-is
                                pass
                        # Ensure it's a list
                        if not isinstance(obs_keys, list):
                            obs_keys = list(obs_keys) if hasattr(obs_keys, '__iter__') else []
                        observations = []
                        if obs_keys:
                            num_obs = len(data[f'obs_{obs_keys[0]}'])
                            for i in range(num_obs):
                                obs_dict = {}
                                for key in obs_keys:
                                    obs_dict[key] = data[f'obs_{key}'][i]
                                observations.append(obs_dict)
                        
                        # Reconstruct simulator states if they exist
                        simulator_states = []
                        has_sim_states = data.get('has_sim_states', False)
                        if has_sim_states:
                            num_states = len(data['sim_time'])
                            for i in range(num_states):
                                state = MujocoState(
                                    time=data['sim_time'][i],
                                    qpos=data['sim_qpos'][i],
                                    qvel=data['sim_qvel'][i],
                                    xpos=data['sim_xpos'][i],
                                    xquat=data['sim_xquat'][i],
                                    mocap_pos=data['sim_mocap_pos'][i],
                                    mocap_quat=data['sim_mocap_quat'][i],
                                    sim_metadata=data['sim_metadata'][i]
                                )
                                simulator_states.append(state)
                        
                        self.cache[index] = {
                            'observations': observations,
                            'rewards': data['rewards'],
                            'terminations': data['terminations'],
                            'trunctuations': data['trunctuations'],
                            'infos': data['infos'],
                            'simulator_states': simulator_states
                        }
                    print(f"Loaded trajectory {index} into cache")
            except (IndexError, ValueError):
                print(f"Warning: Could not parse trajectory index from {filename}")
                continue

    def sample_state(self, n: int):
        """
        Returns n random states (observations) from the in-memory cache
        """
        if not self.cache:
            raise ValueError("Cache is empty. Call load_into_cache() first.")
        
        all_observations = []
        for trajectory_data in self.cache.values():
            observations = trajectory_data['observations']
            all_observations.extend(observations)
        
        if len(all_observations) < n:
            raise ValueError(f"Not enough states in cache. Requested {n}, available {len(all_observations)}")
        
        indices = np.random.choice(len(all_observations), size=n, replace=False)
        return [all_observations[i] for i in indices]

    def flatten_observations(self, observations):
        """
        Flatten dict observations as they would be returned by FlattenObservation wrapper
        Uses the same approach as gymnasium.spaces.utils.flatten()
        """
        if not observations:
            return []
        
        if not isinstance(observations[0], dict):
            return observations
        
        flattened = []
        for obs_dict in observations:
            flat_components = []
            # Process keys in sorted order for consistent flattening
            for key in sorted(obs_dict.keys()):
                value = obs_dict[key]
                if isinstance(value, np.ndarray):
                    flat_components.append(value.flatten())
                else:
                    flat_components.append(np.array([value]).flatten())
            
            # Concatenate all components into single flat array
            if flat_components:
                flattened.append(np.concatenate(flat_components))
            else:
                flattened.append(np.array([]))
        
        return flattened

    def get_simulator_state(self, trajectory_index: int, frame_index: int):
        """
        Get the simulator state for a specific frame in a trajectory
        
        Args:
            trajectory_index: Index of the trajectory
            frame_index: Index of the frame within the trajectory
            
        Returns:
            MujocoState: The simulator state at the specified frame
        """
        trajectory_data = self.get_trajectory(trajectory_index)
        simulator_states = trajectory_data.get('simulator_states', [])
        
        if not simulator_states:
            raise ValueError(f"No simulator states found in trajectory {trajectory_index}")
        
        if frame_index < 0 or frame_index >= len(simulator_states):
            raise IndexError(f"Frame index {frame_index} out of range for trajectory {trajectory_index} (0-{len(simulator_states)-1})")
        
        return simulator_states[frame_index]
