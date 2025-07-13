# Sample BDEW H0 Profile Generator
# This creates a realistic synthetic German household load profile
# Place this in data/generate_bdew_profile.py and run to create the CSV

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_bdew_h0_profile():
    """
    Generate a synthetic BDEW H0 household load profile
    Based on typical German household consumption patterns
    """
    
    # Create 8760 hours for a full year
    hours = 8760
    
    # Base consumption patterns
    # Typical German household daily pattern (normalized)
    daily_base = np.array([
        0.6, 0.55, 0.5, 0.45, 0.45, 0.5,   # 00-05: Night minimum
        0.7, 0.9, 1.0, 0.9, 0.8, 0.9,      # 06-11: Morning peak
        1.0, 0.95, 0.85, 0.8, 0.9, 1.2,    # 12-17: Afternoon
        1.4, 1.3, 1.1, 0.9, 0.8, 0.7       # 18-23: Evening peak
    ])
    
    # Weekly pattern (weekday vs weekend)
    weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.85])  # Mon-Sun
    
    # Monthly/seasonal pattern
    monthly_pattern = np.array([
        1.2, 1.15, 1.1, 1.0, 0.9, 0.85,    # Jan-Jun: Winter higher
        0.8, 0.8, 0.85, 0.95, 1.1, 1.2     # Jul-Dec: Summer lower
    ])
    
    # Generate the profile
    profile = np.zeros(hours)
    
    start_date = datetime(2024, 1, 1)
    
    for h in range(hours):
        current_time = start_date + timedelta(hours=h)
        
        # Hour of day (0-23)
        hour_of_day = current_time.hour
        
        # Day of week (0=Monday, 6=Sunday)
        day_of_week = current_time.weekday()
        
        # Month (1-12)
        month = current_time.month
        
        # Base daily pattern
        daily_factor = daily_base[hour_of_day]
        
        # Weekly adjustment
        weekly_factor = weekly_pattern[day_of_week]
        
        # Monthly/seasonal adjustment
        monthly_factor = monthly_pattern[month - 1]
        
        # Combine all factors
        profile[h] = daily_factor * weekly_factor * monthly_factor
    
    # Add some random variation (±10%)
    noise = np.random.normal(1.0, 0.05, hours)
    profile = profile * noise
    
    # Ensure no negative values
    profile = np.maximum(profile, 0.1)
    
    # Normalize to sum = 1 (will be scaled by annual consumption)
    profile = profile / profile.sum()
    
    return profile

def create_bdew_csv():
    """Create the BDEW H0 profile CSV file"""
    
    # Generate the profile
    profile = generate_bdew_h0_profile()
    
    # Create timestamps
    timestamps = pd.date_range('2024-01-01', periods=8760, freq='H')
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load_factor': profile
    })
    
    # Save to CSV
    df.to_csv('data/bdew_h0.csv', index=False)
    
    print(f"Created BDEW H0 profile with {len(df)} hours")
    print(f"Profile sum: {profile.sum():.6f} (should be 1.0)")
    print(f"Min value: {profile.min():.6f}")
    print(f"Max value: {profile.max():.6f}")
    print(f"Mean value: {profile.mean():.6f}")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate the profile
    df = create_bdew_csv()
    
    # Show sample data
    print("\nSample data (first 24 hours):")
    print(df.head(24))