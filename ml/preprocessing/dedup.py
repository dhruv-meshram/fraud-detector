"""Deduplication of rapid-fire logins."""

import pandas as pd

def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merges consecutive rapid-fire logins from the same user/device within 10 seconds."""
    # Ensure dataframe is sorted by user_id and timestamp ascending
    df = df.sort_values(by=['user_id', 'timestamp']).copy()
    
    # Calculate time difference and device matches between consecutive rows
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
    df['same_device'] = (df['device_hash'] == df['device_hash'].shift()) & (df['user_id'] == df['user_id'].shift())
    
    # Identify rows to drop: those that are within 10 seconds of the previous row AND have the same device hash
    # Note: To merge multiple rapid attempts (e.g. 5 attempts in 10 seconds), we track the time relative to the 
    # start of the window. An iterative/cumulative threshold or group shift handles this cleanly.
    initial_len = len(df)
    
    # Let's perform a clean chronological scan per user to avoid cascade drops that span beyond 10s
    to_drop = []
    
    # Group by user_id
    grouped = df.groupby('user_id')
    
    for user_id, group in grouped:
        last_kept_time = None
        last_device = None
        
        for idx, row in group.iterrows():
            curr_time = row['timestamp']
            curr_device = row['device_hash']
            
            if last_kept_time is not None and curr_device == last_device:
                if (curr_time - last_kept_time) <= 10.0:
                    to_drop.append(idx)
                    continue
            
            # Update reference kept login
            last_kept_time = curr_time
            last_device = curr_device
            
    df_deduped = df.drop(to_drop).drop(columns=['time_diff', 'same_device'])
    
    print(f"Deduplicated {len(to_drop)} rapid-fire login records.")
    return df_deduped
