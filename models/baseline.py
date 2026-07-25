import pandas as pd
import numpy as np

class StatisticalBaselineProfiler:
    def __init__(self, min_sessions_for_individual_profile=10):
        self.min_sessions = min_sessions_for_individual_profile
        self.profiles = {}
        self.peer_group_profiles = {}
    
    def _compute_profile(self, df_subset):
        if len(df_subset) == 0:
            return None
            
        profile = {
            'hour_dist': df_subset['timestamp'].dt.hour.value_counts(normalize=True).to_dict(),
            'geo_dist': df_subset['geo_location'].value_counts(normalize=True).to_dict(),
            'resource_dist': df_subset['resource_accessed'].value_counts(normalize=True).to_dict(),
            'auth_dist': df_subset['auth_method'].value_counts(normalize=True).to_dict(),
            'duration_mean': df_subset['session_duration'].mean(),
            'duration_std': df_subset['session_duration'].std() if len(df_subset) > 1 else 0
        }
        return profile
        
    def fit(self, df_normal):
        """Train the baseline profiler on normal sessions."""
        # Ensure timestamp is datetime
        df_normal['timestamp'] = pd.to_datetime(df_normal['timestamp'])
        
        # 1. Compute peer group profiles (by entity_type)
        for e_type in df_normal['entity_type'].unique():
            df_type = df_normal[df_normal['entity_type'] == e_type]
            self.peer_group_profiles[e_type] = self._compute_profile(df_type)
            
        # 2. Compute individual profiles
        entity_counts = df_normal['entity_id'].value_counts()
        
        for entity_id in df_normal['entity_id'].unique():
            count = entity_counts.get(entity_id, 0)
            if count >= self.min_sessions:
                df_entity = df_normal[df_normal['entity_id'] == entity_id]
                self.profiles[entity_id] = self._compute_profile(df_entity)
                
    def get_profile(self, entity_id, entity_type):
        """Retrieve profile and confidence level."""
        if entity_id in self.profiles:
            return self.profiles[entity_id], True # True = High confidence
        else:
            return self.peer_group_profiles.get(entity_type, None), False # False = Low confidence (cold start)
            
    def score_session(self, session):
        """
        Score a new session against its baseline. 
        Returns (anomaly_score, is_high_confidence).
        Higher score = more anomalous.
        """
        entity_id = session['entity_id']
        entity_type = session['entity_type']
        
        profile, is_high_confidence = self.get_profile(entity_id, entity_type)
        if profile is None:
            return 1.0, False # Unknown entity type and id
            
        score = 0.0
        
        # 1. Hour score
        hour = pd.to_datetime(session['timestamp']).hour
        hour_prob = profile['hour_dist'].get(hour, 0)
        if hour_prob < 0.05:
            score += 0.2
            
        # 2. Geo score
        geo = session['geo_location']
        geo_prob = profile['geo_dist'].get(geo, 0)
        if geo_prob < 0.05:
            score += 0.3 # Geolocation is a stronger signal
            
        # 3. Resource score
        res = session['resource_accessed']
        res_prob = profile['resource_dist'].get(res, 0)
        if res_prob < 0.05:
            score += 0.2
            
        # 4. Auth score
        auth = session['auth_method']
        auth_prob = profile['auth_dist'].get(auth, 0)
        if auth_prob < 0.05:
            score += 0.1
            
        # 5. Duration score
        dur = session['session_duration']
        mean = profile['duration_mean']
        std = profile['duration_std']
        
        if std > 0:
            z_score = abs(dur - mean) / std
            if z_score > 3:
                score += 0.2
        elif abs(dur - mean) > mean * 0.5: # If std is 0 but duration varies by 50%
            score += 0.2
            
        return min(score, 1.0), is_high_confidence
