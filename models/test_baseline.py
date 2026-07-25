import unittest
import pandas as pd
from datetime import datetime
from baseline import StatisticalBaselineProfiler

class TestBaselineProfiler(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset of 20 normal sessions for entity 'user1'
        data = []
        for i in range(20):
            data.append({
                'entity_id': 'user1',
                'entity_type': 'user',
                'timestamp': datetime(2023, 1, i+1, 9, 30), # 9:30 AM every day
                'geo_location': 'United States',
                'resource_accessed': 'web_app',
                'auth_method': 'password',
                'session_duration': 3600
            })
        self.df_normal = pd.DataFrame(data)
        
        self.profiler = StatisticalBaselineProfiler(min_sessions_for_individual_profile=10)
        self.profiler.fit(self.df_normal)

    def test_normal_session(self):
        normal_session = {
            'entity_id': 'user1',
            'entity_type': 'user',
            'timestamp': datetime(2023, 2, 1, 9, 45), # Still 9 AM hour
            'geo_location': 'United States',
            'resource_accessed': 'web_app',
            'auth_method': 'password',
            'session_duration': 3650
        }
        
        score, is_high_conf = self.profiler.score_session(normal_session)
        self.assertTrue(is_high_conf)
        self.assertLess(score, 0.3) # Should be low anomaly score

    def test_anomalous_session(self):
        # Anomalous: different hour, different geo, different resource
        anomalous_session = {
            'entity_id': 'user1',
            'entity_type': 'user',
            'timestamp': datetime(2023, 2, 1, 3, 15), # 3 AM
            'geo_location': 'China',
            'resource_accessed': 'admin_panel',
            'auth_method': 'token',
            'session_duration': 120
        }
        
        score, is_high_conf = self.profiler.score_session(anomalous_session)
        self.assertTrue(is_high_conf)
        self.assertGreater(score, 0.5) # Should be high anomaly score
        
    def test_cold_start(self):
        cold_session = {
            'entity_id': 'new_user',
            'entity_type': 'user',
            'timestamp': datetime(2023, 2, 1, 9, 30),
            'geo_location': 'United States',
            'resource_accessed': 'web_app',
            'auth_method': 'password',
            'session_duration': 3600
        }
        
        score, is_high_conf = self.profiler.score_session(cold_session)
        self.assertFalse(is_high_conf) # Should fall back to peer group

if __name__ == '__main__':
    unittest.main()
