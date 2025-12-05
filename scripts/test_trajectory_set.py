
import unittest
import tempfile
import shutil
import os
import numpy as np
from trajectory_set import TrajectorySet
from lerobothackathonenv.structs import MujocoState


class TestTrajectorySet(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.ts = TrajectorySet(data_dir=self.test_dir)
        
        # Create sample data
        self.sample_observations = [
            {'robot_state': np.array([1.0, 2.0, 3.0]), 'image': np.random.rand(64, 64, 3)},
            {'robot_state': np.array([1.1, 2.1, 3.1]), 'image': np.random.rand(64, 64, 3)},
            {'robot_state': np.array([1.2, 2.2, 3.2]), 'image': np.random.rand(64, 64, 3)}
        ]
        self.sample_rewards = [-0.1, -0.05, 0.0]
        self.sample_terminations = [False, False, True]
        self.sample_trunctuations = [False, False, False]
        self.sample_infos = [
            {'step': 0, 'success': False},
            {'step': 1, 'success': False},
            {'step': 2, 'success': True}
        ]
        self.sample_simulator_states = [
            MujocoState(
                time=0.1,
                qpos=np.array([0.1, 0.2, 0.3]),
                qvel=np.array([0.01, 0.02, 0.03]),
                xpos=np.array([1.0, 2.0, 3.0]),
                xquat=np.array([0.0, 0.0, 0.0, 1.0]),
                mocap_pos=np.array([0.0, 0.0, 1.0]),
                mocap_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                sim_metadata={'controller': 'pid'}
            ),
            MujocoState(
                time=0.2,
                qpos=np.array([0.2, 0.3, 0.4]),
                qvel=np.array([0.02, 0.03, 0.04]),
                xpos=np.array([1.1, 2.1, 3.1]),
                xquat=np.array([0.0, 0.0, 0.1, 1.0]),
                mocap_pos=np.array([0.1, 0.0, 1.0]),
                mocap_quat=np.array([0.0, 0.0, 0.1, 1.0]),
                sim_metadata={'controller': 'pid'}
            ),
            MujocoState(
                time=0.3,
                qpos=np.array([0.3, 0.4, 0.5]),
                qvel=np.array([0.03, 0.04, 0.05]),
                xpos=np.array([1.2, 2.2, 3.2]),
                xquat=np.array([0.0, 0.0, 0.2, 1.0]),
                mocap_pos=np.array([0.2, 0.0, 1.0]),
                mocap_quat=np.array([0.0, 0.0, 0.2, 1.0]),
                sim_metadata={'controller': 'pid'}
            )
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init(self):
        """Test TrajectorySet initialization."""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(self.ts.data_dir, self.test_dir)
        self.assertEqual(self.ts.index, 0)
        self.assertEqual(len(self.ts.cache), 0)
    
    def test_get_trajectory(self):
        """Test trajectory retrieval from pre-recorded data."""
        # Record a trajectory first
        self.ts.record_trajectory(
            self.sample_observations,
            self.sample_rewards,
            self.sample_terminations,
            self.sample_trunctuations,
            self.sample_infos,
            self.sample_simulator_states
        )
        
        # Retrieve trajectory
        trajectory_data = self.ts.get_trajectory(0)
        
        # Verify structure
        self.assertIn('observations', trajectory_data)
        self.assertIn('rewards', trajectory_data)
        self.assertIn('terminations', trajectory_data)
        self.assertIn('trunctuations', trajectory_data)
        self.assertIn('infos', trajectory_data)
        self.assertIn('simulator_states', trajectory_data)
        
        # Verify data integrity
        self.assertEqual(len(trajectory_data['observations']), 3)
        np.testing.assert_array_equal(trajectory_data['rewards'], self.sample_rewards)
        
        # Verify simulator states
        self.assertEqual(len(trajectory_data['simulator_states']), 3)
        self.assertIsInstance(trajectory_data['simulator_states'][0], MujocoState)
        self.assertEqual(trajectory_data['simulator_states'][0].time, 0.1)
    
    def test_get_nonexistent_trajectory(self):
        """Test retrieving non-existent trajectory."""
        with self.assertRaises(FileNotFoundError):
            self.ts.get_trajectory(999)
    
    def test_caching(self):
        """Test trajectory caching functionality."""
        # Record trajectory
        self.ts.record_trajectory(
            self.sample_observations,
            self.sample_rewards,
            self.sample_terminations,
            self.sample_trunctuations,
            self.sample_infos
        )
        
        # First retrieval should load from file
        trajectory_data1 = self.ts.get_trajectory(0)
        self.assertIn(0, self.ts.cache)
        
        # Second retrieval should use cache
        trajectory_data2 = self.ts.get_trajectory(0)
        
        # Should be the same data
        np.testing.assert_array_equal(trajectory_data1['rewards'], trajectory_data2['rewards'])
    
    def test_load_into_cache(self):
        """Test loading multiple trajectories into cache."""
        # Record multiple trajectories
        for i in range(3):
            modified_rewards = [r + i * 0.1 for r in self.sample_rewards]
            self.ts.record_trajectory(
                self.sample_observations,
                modified_rewards,
                self.sample_terminations,
                self.sample_trunctuations,
                self.sample_infos
            )
        
        # Clear cache
        self.ts.cache.clear()
        
        # Load all into cache
        self.ts.load_into_cache()
        
        # Verify all trajectories are cached
        self.assertEqual(len(self.ts.cache), 3)
        self.assertIn(0, self.ts.cache)
        self.assertIn(1, self.ts.cache)
        self.assertIn(2, self.ts.cache)
    
    def test_sample_state(self):
        """Test random state sampling."""
        # Record multiple trajectories
        for i in range(2):
            self.ts.record_trajectory(
                self.sample_observations,
                self.sample_rewards,
                self.sample_terminations,
                self.sample_trunctuations,
                self.sample_infos
            )
        
        # Clear cache to ensure we test loading
        self.ts.cache.clear()
        
        # Load into cache  
        self.ts.load_into_cache()
        
        # Verify cache was loaded before sampling
        self.assertGreater(len(self.ts.cache), 0, "Cache should be populated after load_into_cache")
        
        # Sample states (we have 2 trajectories * 3 observations = 6 total)
        sampled_states = self.ts.sample_state(3)
        
        # Verify we got the right number
        self.assertEqual(len(sampled_states), 3)
        
        # Verify they are observations (dicts)
        for state in sampled_states:
            self.assertIsInstance(state, dict)
            self.assertIn('robot_state', state)
            self.assertIn('image', state)
    
    def test_sample_state_empty_cache(self):
        """Test sampling from empty cache."""
        with self.assertRaises(ValueError):
            self.ts.sample_state(1)
    
    def test_sample_state_insufficient_data(self):
        """Test sampling more states than available."""
        # Record one small trajectory
        self.ts.record_trajectory(
            self.sample_observations[:1],  # Only 1 observation
            self.sample_rewards[:1],
            self.sample_terminations[:1],
            self.sample_trunctuations[:1],
            self.sample_infos[:1]
        )
        
        self.ts.load_into_cache()
        
        # Try to sample more than available
        with self.assertRaises(ValueError):
            self.ts.sample_state(5)
    
    def test_flatten_observations(self):
        """Test observation flattening."""
        flattened = self.ts.flatten_observations(self.sample_observations)
        
        # Should have same number of observations
        self.assertEqual(len(flattened), len(self.sample_observations))
        
        # Each should be a 1D numpy array
        for flat_obs in flattened:
            self.assertIsInstance(flat_obs, np.ndarray)
            self.assertEqual(len(flat_obs.shape), 1)
    
    def test_flatten_empty_observations(self):
        """Test flattening empty observations."""
        result = self.ts.flatten_observations([])
        self.assertEqual(result, [])
    
    def test_get_simulator_state(self):
        """Test getting specific simulator state."""
        # Record trajectory with simulator states
        self.ts.record_trajectory(
            self.sample_observations,
            self.sample_rewards,
            self.sample_terminations,
            self.sample_trunctuations,
            self.sample_infos,
            self.sample_simulator_states
        )
        
        # Get specific simulator state
        sim_state = self.ts.get_simulator_state(0, 1)
        
        # Verify it's the right state
        self.assertIsInstance(sim_state, MujocoState)
        self.assertEqual(sim_state.time, 0.2)
        np.testing.assert_array_equal(sim_state.qpos, np.array([0.2, 0.3, 0.4]))
    
    def test_get_simulator_state_no_states(self):
        """Test getting simulator state when none exist."""
        # Record trajectory without simulator states
        self.ts.record_trajectory(
            self.sample_observations,
            self.sample_rewards,
            self.sample_terminations,
            self.sample_trunctuations,
            self.sample_infos
        )
        
        with self.assertRaises(ValueError):
            self.ts.get_simulator_state(0, 0)
    
    def test_get_simulator_state_invalid_index(self):
        """Test getting simulator state with invalid indices."""
        # Record trajectory with simulator states
        self.ts.record_trajectory(
            self.sample_observations,
            self.sample_rewards,
            self.sample_terminations,
            self.sample_trunctuations,
            self.sample_infos,
            self.sample_simulator_states
        )
        
        # Test invalid frame index
        with self.assertRaises(IndexError):
            self.ts.get_simulator_state(0, 999)
        
        # Test negative frame index
        with self.assertRaises(IndexError):
            self.ts.get_simulator_state(0, -1)
    
    def test_reconstruct_index(self):
        """Test index reconstruction from existing files."""
        # Create a new TrajectorySet in a different temp dir
        another_dir = tempfile.mkdtemp()
        try:
            ts1 = TrajectorySet(data_dir=another_dir)
            
            # Record some trajectories
            for i in range(3):
                ts1.record_trajectory(
                    self.sample_observations,
                    self.sample_rewards,
                    self.sample_terminations,
                    self.sample_trunctuations,
                    self.sample_infos
                )
            
            # Create new TrajectorySet pointing to same directory
            ts2 = TrajectorySet(data_dir=another_dir)
            
            # Should start with index 3 (next available)
            self.assertEqual(ts2.index, 3)
            
        finally:
            shutil.rmtree(another_dir, ignore_errors=True)
    
    def test_file_locking_context_manager(self):
        """Test file locking context manager works correctly."""
        # This is a basic test - in practice you'd need concurrent processes to fully test locking
        filepath = os.path.join(self.test_dir, "test_lock.txt")
        
        # Test that context manager works without exceptions
        try:
            with self.ts._file_lock(filepath, 'w'):
                pass  # Just test that context manager doesn't raise exceptions
            # Test completed successfully if we reach here
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"File locking context manager raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
