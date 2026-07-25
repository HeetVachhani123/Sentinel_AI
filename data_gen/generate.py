import pandas as pd
import numpy as np
from faker import Faker
import random
import datetime
import os
import argparse

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_baseline_entities(n=500):
    """
    Generate n entities with habitual baseline patterns.
    """
    entities = []
    entity_types = ['user', 'service_account', 'edge_device']
    
    for _ in range(n):
        e_type = random.choices(entity_types, weights=[0.7, 0.2, 0.1])[0]
        
        # Determine baseline characteristics based on type
        if e_type == 'user':
            typical_hours = random.choice([range(8, 18), range(9, 17), range(18, 24)]) # Day shift or night shift
            geo = fake.country()
            typical_resources = random.sample(['web_app', 'jira', 'github', 'confluence', 'office365', 'salesforce'], k=3)
            auth_methods = ['password', 'token', 'biometric']
            typical_auth = random.choices(auth_methods, weights=[0.4, 0.5, 0.1])[0]
        elif e_type == 'service_account':
            typical_hours = range(0, 24) # Usually 24/7
            geo = 'Datacenter'
            typical_resources = random.sample(['db_server', 'api_gateway', 'storage_bucket', 'kafka_cluster'], k=2)
            typical_auth = 'certificate'
        else: # edge_device
            typical_hours = range(0, 24)
            geo = fake.country()
            typical_resources = random.sample(['iot_hub', 'telemetry_endpoint', 'ota_server'], k=1)
            typical_auth = 'token'

        entity = {
            'entity_id': fake.uuid4(),
            'entity_type': e_type,
            'typical_hours': list(typical_hours),
            'typical_geo': geo,
            'typical_ips': [fake.ipv4() for _ in range(random.randint(1, 3))], # 1-3 typical IPs
            'typical_resources': typical_resources,
            'typical_auth': typical_auth,
            'typical_device_fingerprints': [fake.md5()[:12] for _ in range(random.randint(1, 2))]
        }
        entities.append(entity)
        
    return entities

def generate_normal_session(entity, current_time):
    """Generate a single normal session based on entity's baseline."""
    # Add some noise to time (maybe log in slightly outside typical hours)
    if random.random() < 0.9:
        # Normal hour
        hour = random.choice(entity['typical_hours'])
    else:
        # Noise hour
        hour = random.randint(0, 23)
        
    session_time = current_time.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
    
    # Select attributes based on typicals with slight noise
    ip = random.choice(entity['typical_ips']) if random.random() < 0.9 else fake.ipv4()
    geo = entity['typical_geo'] if random.random() < 0.95 else fake.country()
    resource = random.choice(entity['typical_resources']) if random.random() < 0.9 else random.choice(['web_app', 'db_server', 'admin_panel', 'vpn'])
    auth = entity['typical_auth'] if random.random() < 0.95 else 'password'
    device = random.choice(entity['typical_device_fingerprints']) if random.random() < 0.9 else fake.md5()[:12]
    
    duration = random.randint(30, 3600) if entity['entity_type'] == 'user' else random.randint(1, 600)
    
    status = 'success'
    if random.random() < 0.05: # 5% normal failure rate
        status = 'failed'
        
    return {
        'entity_id': entity['entity_id'],
        'entity_type': entity['entity_type'],
        'timestamp': session_time,
        'source_ip': ip,
        'geo_location': geo,
        'resource_accessed': resource,
        'auth_method': auth,
        'auth_status': status,
        'session_duration': duration,
        'command_sequence': "cmd_" + fake.word() if entity['entity_type'] in ['user', 'service_account'] else "ping",
        'device_fingerprint': device,
        'label': 'normal'
    }

def inject_anomalies(sessions_df, entities):
    """
    Inject anomaly patterns at 0.5-3% of sessions.
    - Brute force: rapid repeated failed-auth from one source
    - Impossible travel: same entity from geographically distant locations in implausible time gap
    - Credential stuffing: many entity_ids, few source_ips, high failure rate
    - Lateral movement: entity accessing unusual sequence/breadth of resources
    - Device spoofing: device_id reappearing with mismatched fingerprint
    - Low-and-slow exfiltration: gradual off-hours resource access building over days
    - Insider drift (edge case, ambiguous): entity slowly expanding privilege/resource footprint
    """
    anomalous_sessions = []
    
    # Convert sessions to a list for easier manipulation and adding new ones
    
    # 1. Brute Force (rapid repeated failed auth)
    # Target 0.5% of dataset
    target_bf = int(len(sessions_df) * 0.005) // 10  # 10 attempts per attack
    for _ in range(target_bf):
        target_entity = random.choice(entities)
        attack_time = fake.date_time_between(start_date='-30d', end_date='now')
        attacker_ip = fake.ipv4()
        for i in range(10): # 10 rapid failures
            sess = generate_normal_session(target_entity, attack_time)
            sess['timestamp'] = attack_time + datetime.timedelta(seconds=i*2)
            sess['source_ip'] = attacker_ip
            sess['auth_status'] = 'failed'
            sess['label'] = 'brute_force'
            anomalous_sessions.append(sess)
            
    # 2. Impossible travel
    target_it = int(len(sessions_df) * 0.005) // 2 # 2 sessions per attack
    for _ in range(target_it):
        target_entity = random.choice(entities)
        time1 = fake.date_time_between(start_date='-30d', end_date='now')
        time2 = time1 + datetime.timedelta(minutes=10) # 10 mins later
        
        sess1 = generate_normal_session(target_entity, time1)
        sess1['timestamp'] = time1
        sess1['geo_location'] = 'United States'
        sess1['label'] = 'impossible_travel'
        
        sess2 = generate_normal_session(target_entity, time2)
        sess2['timestamp'] = time2
        sess2['geo_location'] = 'China' # Geographically distant
        sess2['label'] = 'impossible_travel'
        
        anomalous_sessions.append(sess1)
        anomalous_sessions.append(sess2)
        
    # 3. Credential stuffing (many entity_ids, few source_ips, high failure rate)
    target_cs = int(len(sessions_df) * 0.005) // 20
    for _ in range(target_cs):
        attacker_ip = fake.ipv4()
        attack_time = fake.date_time_between(start_date='-30d', end_date='now')
        targets = random.sample(entities, 20)
        for i, target in enumerate(targets):
            sess = generate_normal_session(target, attack_time)
            sess['timestamp'] = attack_time + datetime.timedelta(seconds=i*5)
            sess['source_ip'] = attacker_ip
            sess['auth_status'] = 'failed' if random.random() < 0.9 else 'success'
            sess['label'] = 'credential_stuffing'
            anomalous_sessions.append(sess)

    # 4. Lateral movement (entity accessing unusual sequence/breadth of resources)
    target_lm = int(len(sessions_df) * 0.005) // 5
    for _ in range(target_lm):
        target_entity = random.choice(entities)
        attack_time = fake.date_time_between(start_date='-30d', end_date='now')
        unusual_resources = ['admin_panel', 'production_db', 'secrets_manager', 'domain_controller', 'backup_server']
        for i in range(5):
            sess = generate_normal_session(target_entity, attack_time)
            sess['timestamp'] = attack_time + datetime.timedelta(minutes=i*2)
            sess['resource_accessed'] = unusual_resources[i]
            sess['auth_status'] = 'success'
            sess['label'] = 'lateral_movement'
            anomalous_sessions.append(sess)

    # 5. Device spoofing (device_id reappearing with mismatched fingerprint)
    target_ds = int(len(sessions_df) * 0.005)
    for _ in range(target_ds):
        target_entity = random.choice(entities)
        sess = generate_normal_session(target_entity, fake.date_time_between(start_date='-30d', end_date='now'))
        # use a valid entity but a completely new fingerprint that is very different
        sess['device_fingerprint'] = "spoofed_" + fake.md5()[:5]
        sess['label'] = 'device_spoofing'
        anomalous_sessions.append(sess)

    # 6. Low-and-slow exfiltration
    target_lse = int(len(sessions_df) * 0.005) // 7
    for _ in range(target_lse):
        target_entity = random.choice(entities)
        start_time = fake.date_time_between(start_date='-30d', end_date='-7d')
        for i in range(7): # Over 7 days
            sess = generate_normal_session(target_entity, start_time)
            # strictly off-hours
            sess['timestamp'] = start_time + datetime.timedelta(days=i, hours=random.choice([2, 3, 4]))
            sess['resource_accessed'] = 'customer_data_bucket'
            sess['session_duration'] = 7200 + i * 600 # slowly increasing duration
            sess['label'] = 'low_and_slow_exfil'
            anomalous_sessions.append(sess)

    # 7. Insider drift
    target_id = int(len(sessions_df) * 0.005) // 10
    for _ in range(target_id):
        target_entity = random.choice(entities)
        start_time = fake.date_time_between(start_date='-30d', end_date='-10d')
        new_resource = 'financial_records'
        for i in range(10):
            sess = generate_normal_session(target_entity, start_time)
            sess['timestamp'] = start_time + datetime.timedelta(days=i*2) # every 2 days
            sess['resource_accessed'] = new_resource
            sess['label'] = 'insider_drift'
            anomalous_sessions.append(sess)

    # Add anomalous sessions to the dataframe
    anomalous_df = pd.DataFrame(anomalous_sessions)
    combined_df = pd.concat([sessions_df, anomalous_df], ignore_index=True)
    return combined_df

def generate_dataset(num_records=50000):
    print("Generating baseline entities...")
    entities = generate_baseline_entities(n=500)
    
    print("Generating normal sessions...")
    normal_sessions = []
    current_time = datetime.datetime.now() - datetime.timedelta(days=30)
    
    # Generate ~48500 normal sessions
    for _ in range(int(num_records * 0.97)):
        target_entity = random.choice(entities)
        sess = generate_normal_session(target_entity, current_time)
        # Randomize timestamp over the last 30 days
        sess['timestamp'] = fake.date_time_between(start_date='-30d', end_date='now')
        normal_sessions.append(sess)
        
    df_normal = pd.DataFrame(normal_sessions)
    
    print("Injecting anomalies...")
    df_final = inject_anomalies(df_normal, entities)
    
    # Sort by timestamp
    df_final = df_final.sort_values(by='timestamp').reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--num_records', type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = generate_dataset(args.num_records)
    
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    print("\nAnomaly Percentage: {:.2f}%".format(
        (len(df[df['label'] != 'normal']) / len(df)) * 100
    ))
    
    # Separate labels
    labels_df = df[['label']]
    production_df = df.drop(columns=['label'])
    
    prod_path = os.path.join(args.output_dir, 'production_logs.csv')
    labels_path = os.path.join(args.output_dir, 'labels_holdout.csv')
    
    production_df.to_csv(prod_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    
    print(f"\nSaved production data (no labels) to {prod_path}")
    print(f"Saved holdout labels to {labels_path}")
