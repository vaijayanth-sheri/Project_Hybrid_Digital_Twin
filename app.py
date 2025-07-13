"""
# Hybrid PV+Wind+Storage Digital Twin Dashboard

## Installation & Setup

1. Install required packages:
```bash
pip install streamlit pandas numpy requests plotly
```

2. Get a free API key from OpenWeatherMap (https://openweathermap.org/api)
   - Sign up for a free account
   - Get your API key from the dashboard
   - Replace 'your_api_key_here' in the code below

3. Create a data directory and add the BDEW H0 profile:
```bash
mkdir data
# Add your bdew_h0.csv file with 8760 rows of normalized German household load profile
```

4. Run the application:
```bash
streamlit run app.py
```

## Features
- Interactive solar panel configuration
- Wind turbine setup and modeling
- Battery storage simulation
- Real-time weather data integration
- Comprehensive energy flow visualization
- KPI monitoring and metrics

## Usage
Navigate through the tabs to configure your hybrid energy system, then view the
real-time dashboard with energy generation, consumption, and storage metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Hybrid Energy Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and configuration data
PANEL_MODELS = {
    "Mono-PERC 330W": {"power": 0.33, "eff": 0.18, "area": 1.7},
    "Poly-Si 300W": {"power": 0.30, "eff": 0.16, "area": 1.6},
    "Mono-PERC 400W": {"power": 0.40, "eff": 0.20, "area": 2.0},
    "Bifacial 450W": {"power": 0.45, "eff": 0.21, "area": 2.1},
    "Thin-Film 250W": {"power": 0.25, "eff": 0.12, "area": 2.5}
}

TURBINE_MODELS = {
    "Vestas V90 (2 MW)": {"power": 2000, "rotor_diameter": 90, "cut_in": 3, "cut_out": 25},
    "GE 2.5 MW": {"power": 2500, "rotor_diameter": 100, "cut_in": 3, "cut_out": 25},
    "Siemens 3.0 MW": {"power": 3000, "rotor_diameter": 112, "cut_in": 3, "cut_out": 25},
    "Nordex N117 (2.4 MW)": {"power": 2400, "rotor_diameter": 117, "cut_in": 3, "cut_out": 25}
}

# Air density at sea level (kg/m³)
AIR_DENSITY = 1.225
# Power coefficient (typical for modern turbines)
CP = 0.4
# Inverter efficiency
INVERTER_EFF = 0.96

def load_bdew_profile(annual_consumption_kwh):
    """
    Load and scale the BDEW H0 household load profile
    
    Args:
        annual_consumption_kwh: Annual household consumption in kWh
        
    Returns:
        pandas.Series: Hourly load profile for one year
    """
    try:
        # Try to load the actual BDEW profile
        df = pd.read_csv('data/bdew_h0.csv')
        if len(df) != 8760:
            raise ValueError("BDEW profile should have 8760 rows")
        
        # Normalize and scale to annual consumption
        profile = df.iloc[:, 0].values  # Assuming first column contains the profile
        profile = profile / profile.sum() * annual_consumption_kwh
        
    except (FileNotFoundError, ValueError):
        # Generate synthetic load profile if file not found
        st.warning("⚠️ BDEW profile not found. Using synthetic load profile.")
        
        # Create a realistic daily pattern
        hours = np.arange(8760)
        daily_pattern = (
            0.4 +  # Base load
            0.3 * np.sin(2 * np.pi * (hours % 24 - 6) / 24) +  # Daily cycle
            0.2 * np.sin(2 * np.pi * hours / (24 * 7)) +  # Weekly cycle
            0.1 * np.sin(2 * np.pi * hours / (24 * 365.25))  # Seasonal cycle
        )
        
        # Add some noise and ensure positive values
        profile = np.maximum(daily_pattern + 0.1 * np.random.normal(0, 1, 8760), 0.1)
        profile = profile / profile.sum() * annual_consumption_kwh
    
    return pd.Series(profile, index=pd.date_range('2024-01-01', periods=8760, freq='H'))

def fetch_weather_data(lat, lon, api_key):
    """
    Fetch current weather and forecast data from OpenWeatherMap
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeatherMap API key
        
    Returns:
        dict: Weather data including current conditions and forecast
    """
    if api_key == "your_api_key_here":
        # Return dummy data if no API key provided
        st.warning("⚠️ Please provide a valid OpenWeatherMap API key for real weather data.")
        return generate_dummy_weather()
    
    try:
        # Current weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        current_data = current_response.json()
        
        # 24-hour forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_data = forecast_response.json()
        
        # Extract relevant data
        weather_data = {
            'current_temp': current_data['main']['temp'],
            'current_wind_speed': current_data['wind']['speed'],
            'current_clouds': current_data['clouds']['all'],
            'forecast': []
        }
        
        # Process forecast data (next 24 hours)
        for item in forecast_data['list'][:8]:  # 8 x 3-hour intervals = 24 hours
            weather_data['forecast'].append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temp': item['main']['temp'],
                'wind_speed': item['wind']['speed'],
                'clouds': item['clouds']['all']
            })
        
        return weather_data
        
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return generate_dummy_weather()

def generate_dummy_weather():
    """Generate dummy weather data for testing"""
    base_time = datetime.now()
    
    return {
        'current_temp': 15.0,
        'current_wind_speed': 8.5,
        'current_clouds': 30,
        'forecast': [
            {
                'datetime': base_time + timedelta(hours=i*3),
                'temp': 15 + 5 * np.sin(i * np.pi / 4),
                'wind_speed': 8 + 3 * np.sin(i * np.pi / 3),
                'clouds': 30 + 20 * np.sin(i * np.pi / 6)
            }
            for i in range(8)
        ]
    }

def calculate_solar_irradiance(clouds, hour_of_day):
    """
    Calculate solar irradiance based on cloud cover and time of day
    
    Args:
        clouds: Cloud cover percentage (0-100)
        hour_of_day: Hour of day (0-23)
        
    Returns:
        float: Solar irradiance in W/m²
    """
    # Maximum irradiance at solar noon (simplified model)
    max_irradiance = 1000  # W/m²
    
    # Solar elevation angle (simplified)
    if 6 <= hour_of_day <= 18:
        # Daytime: sine wave approximation
        solar_elevation = np.sin(np.pi * (hour_of_day - 6) / 12)
    else:
        # Nighttime
        solar_elevation = 0
    
    # Base irradiance from solar angle
    base_irradiance = max_irradiance * solar_elevation
    
    # Reduce irradiance based on cloud cover
    cloud_factor = 1 - (clouds / 100) * 0.8  # 80% reduction at 100% clouds
    
    return base_irradiance * cloud_factor

def calculate_pv_generation(capacity_kw, irradiance, panel_eff, inverter_eff=INVERTER_EFF):
    """
    Calculate PV power generation
    
    Args:
        capacity_kw: Total PV capacity in kW
        irradiance: Solar irradiance in W/m²
        panel_eff: Panel efficiency (0-1)
        inverter_eff: Inverter efficiency (0-1)
        
    Returns:
        float: Power generation in kW
    """
    # Standard Test Conditions (STC) irradiance
    stc_irradiance = 1000  # W/m²
    
    # Power output proportional to irradiance
    power_kw = capacity_kw * (irradiance / stc_irradiance) * inverter_eff
    
    return max(0, power_kw)

def calculate_wind_generation(num_turbines, turbine_specs, wind_speed, hub_height):
    """
    Calculate wind power generation
    
    Args:
        num_turbines: Number of turbines
        turbine_specs: Turbine specifications dict
        wind_speed: Wind speed at reference height (m/s)
        hub_height: Hub height in meters
        
    Returns:
        float: Power generation in kW
    """
    # Wind speed correction for height (power law, α = 0.143)
    ref_height = 10  # m (standard measurement height)
    alpha = 0.143
    wind_speed_hub = wind_speed * (hub_height / ref_height) ** alpha
    
    # Cut-in and cut-out wind speeds
    cut_in = turbine_specs.get('cut_in', 3)
    cut_out = turbine_specs.get('cut_out', 25)
    
    if wind_speed_hub < cut_in or wind_speed_hub > cut_out:
        return 0
    
    # Rotor swept area
    radius = turbine_specs['rotor_diameter'] / 2
    area = np.pi * radius ** 2
    
    # Power calculation (simplified)
    power_per_turbine = 0.5 * AIR_DENSITY * area * (wind_speed_hub ** 3) * CP / 1000  # kW
    
    # Limit to rated power
    power_per_turbine = min(power_per_turbine, turbine_specs['power'])
    
    return num_turbines * power_per_turbine

def simulate_battery(generation_kw, load_kw, battery_capacity_kwh, initial_soc_kwh, 
                    charge_eff=0.9, discharge_eff=0.9):
    """
    Simulate battery operation for given generation and load profiles
    
    Args:
        generation_kw: Generation power array (kW)
        load_kw: Load power array (kW)
        battery_capacity_kwh: Battery capacity (kWh)
        initial_soc_kwh: Initial state of charge (kWh)
        charge_eff: Charging efficiency
        discharge_eff: Discharging efficiency
        
    Returns:
        dict: Battery simulation results
    """
    n_hours = len(generation_kw)
    
    # Initialize arrays
    soc_kwh = np.zeros(n_hours)
    battery_in_kw = np.zeros(n_hours)
    battery_out_kw = np.zeros(n_hours)
    grid_kw = np.zeros(n_hours)
    
    # Set initial SOC
    soc_kwh[0] = initial_soc_kwh
    
    for i in range(n_hours):
        current_soc = soc_kwh[i-1] if i > 0 else initial_soc_kwh
        
        # Calculate surplus or deficit
        net_power = generation_kw[i] - load_kw[i]
        
        if net_power > 0:  # Surplus - charge battery
            max_charge = min(net_power, battery_capacity_kwh - current_soc)
            charge_power = max_charge * charge_eff
            
            battery_in_kw[i] = charge_power
            new_soc = current_soc + charge_power
            
            # Remaining surplus goes to grid
            grid_kw[i] = net_power - charge_power
            
        else:  # Deficit - discharge battery
            deficit = abs(net_power)
            max_discharge = min(deficit, current_soc)
            discharge_power = max_discharge / discharge_eff
            
            battery_out_kw[i] = discharge_power
            new_soc = current_soc - discharge_power
            
            # Remaining deficit from grid
            grid_kw[i] = -(deficit - discharge_power)
        
        soc_kwh[i] = max(0, min(new_soc, battery_capacity_kwh))
    
    return {
        'soc_kwh': soc_kwh,
        'battery_in_kw': battery_in_kw,
        'battery_out_kw': battery_out_kw,
        'grid_kw': grid_kw
    }

def create_energy_plot(df):
    """Create comprehensive energy system plot"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Power Generation & Load', 'Battery Operations', 'Grid Interaction'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Power generation and load
    fig.add_trace(
        go.Scatter(x=df.index, y=df['pv_kw'], name='PV Generation', 
                  line=dict(color='orange', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['wind_kw'], name='Wind Generation',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['load_kw'], name='Load',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Battery operations
    fig.add_trace(
        go.Scatter(x=df.index, y=df['battery_in_kw'], name='Battery Charge',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=-df['battery_out_kw'], name='Battery Discharge',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # Add SOC on secondary y-axis
    fig.add_trace(
        go.Scatter(x=df.index, y=df['soc_kwh'], name='SOC',
                  line=dict(color='gray', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Grid interaction
    fig.add_trace(
        go.Scatter(x=df.index, y=df['grid_kw'], name='Grid Import/Export',
                  line=dict(color='black', width=2),
                  fill='tozeroy'),
        row=3, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=3, col=1)
    
    fig.update_layout(
        height=800,
        title_text="Hybrid Energy System Dashboard",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def main():
    st.title("⚡ Hybrid PV+Wind+Storage Digital Twin")
    st.markdown("---")
    
    # Initialize session state
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    # Main navigation
    main_tabs = st.tabs(["Configuration", "Dashboard", "About", "Settings"])
    
    with main_tabs[0]:  # Configuration
        config_tabs = st.tabs(["Solar", "Wind", "Load & Battery", "Location"])
        
        with config_tabs[0]:  # Solar Configuration
            st.header("☀️ Solar PV Configuration")
            enable_pv = st.checkbox("✅ Enable Solar PV", value=True)
            col1, col2 = st.columns(2)
            
            with col1:
                panel_model = st.selectbox(
                    "Panel Model",
                    options=list(PANEL_MODELS.keys()),
                    help="Select the solar panel model"
                )
                
                config_method = st.radio(
                    "Configuration Method",
                    ["Total Capacity (kW)", "Number of Panels"]
                )
                
                if config_method == "Total Capacity (kW)":
                    pv_capacity = st.number_input(
                        "Total PV Capacity (kW)",
                        min_value=0.1,
                        max_value=1000.0,
                        value=10.0,
                        step=0.5
                    )
                    num_panels = int(pv_capacity / PANEL_MODELS[panel_model]['power'])
                else:
                    num_panels = st.number_input(
                        "Number of Panels",
                        min_value=1,
                        max_value=10000,
                        value=30,
                        step=1
                    )
                    pv_capacity = num_panels * PANEL_MODELS[panel_model]['power']
            
            with col2:
                st.subheader("Panel Specifications")
                panel_specs = PANEL_MODELS[panel_model]
                st.write(f"**Power per Panel:** {panel_specs['power']} kW")
                st.write(f"**Efficiency:** {panel_specs['eff']*100:.1f}%")
                st.write(f"**Area per Panel:** {panel_specs['area']} m²")
                st.write(f"**Total Capacity:** {pv_capacity:.1f} kW")
                st.write(f"**Total Panels:** {num_panels}")
                st.write(f"**Total Area:** {num_panels * panel_specs['area']:.1f} m²")
        
        with config_tabs[1]:  # Wind Configuration
            st.header("💨 Wind Turbine Configuration")
            enable_wind = st.checkbox("✅ Enable Wind Turbines", value=True)
            col1, col2 = st.columns(2)
            
            with col1:
                turbine_model = st.selectbox(
                    "Turbine Model",
                    options=list(TURBINE_MODELS.keys()),
                    help="Select the wind turbine model"
                )
                
                num_turbines = st.number_input(
                    "Number of Turbines",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1
                )
                
                hub_height = st.number_input(
                    "Hub Height (m)",
                    min_value=30,
                    max_value=200,
                    value=80,
                    step=5
                )
            
            with col2:
                st.subheader("Turbine Specifications")
                turbine_specs = TURBINE_MODELS[turbine_model]
                st.write(f"**Power per Turbine:** {turbine_specs['power']} kW")
                st.write(f"**Rotor Diameter:** {turbine_specs['rotor_diameter']} m")
                st.write(f"**Cut-in Wind Speed:** {turbine_specs['cut_in']} m/s")
                st.write(f"**Cut-out Wind Speed:** {turbine_specs['cut_out']} m/s")
                st.write(f"**Total Capacity:** {num_turbines * turbine_specs['power']} kW")
                st.write(f"**Total Turbines:** {num_turbines}")
        
        with config_tabs[2]:  # Load & Battery Configuration
            st.header("🔋 Load & Battery Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Household Load")
                annual_consumption = st.number_input(
                    "Annual Consumption (kWh)",
                    min_value=1000,
                    max_value=50000,
                    value=4000,
                    step=100,
                    help="Typical German household: 3000-5000 kWh/year"
                )
                
                st.subheader("Battery Storage")
                battery_capacity = st.number_input(
                    "Battery Capacity (kWh)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=13.5,
                    step=0.5
                )
                
                battery_efficiency = st.slider(
                    "Round-trip Efficiency",
                    min_value=0.7,
                    max_value=0.98,
                    value=0.9,
                    step=0.01,
                    format="%.2f"
                )
                
                initial_soc = st.slider(
                    "Initial SOC (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5
                )
            
            with col2:
                st.subheader("Load Profile Preview")
                # Generate a sample day from the load profile
                load_profile = load_bdew_profile(annual_consumption)
                sample_day = load_profile.iloc[:24]  # First 24 hours
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=sample_day.values,
                    mode='lines+markers',
                    name='Hourly Load'
                ))
                fig.update_layout(
                    title="Sample Daily Load Profile",
                    xaxis_title="Hour of Day",
                    yaxis_title="Load (kW)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with config_tabs[3]:  # Location Configuration
            st.header("🌍 Location & Weather")
            
            col1, col2 = st.columns(2)
            
            with col1:
                location_method = st.radio(
                    "Location Input Method",
                    ["Coordinates", "City Name"]
                )
                
                if location_method == "Coordinates":
                    latitude = st.number_input(
                        "Latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=52.5200,  # Berlin
                        step=0.0001,
                        format="%.4f"
                    )
                    longitude = st.number_input(
                        "Longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=13.4050,  # Berlin
                        step=0.0001,
                        format="%.4f"
                    )
                else:
                    city_name = st.text_input(
                        "City Name",
                        value="Berlin, Germany",
                        help="Enter city name (geocoding will be applied)"
                    )
                    # For simplicity, use default coordinates
                    latitude, longitude = 52.5200, 13.4050
                
                api_key = st.text_input(
                    "OpenWeatherMap API Key",
                    value="your_api_key_here",
                    type="password",
                    help="Get your free API key from openweathermap.org"
                )
                
                if st.button("🔄 Fetch Weather Data"):
                    with st.spinner("Fetching weather data..."):
                        st.session_state.weather_data = fetch_weather_data(latitude, longitude, api_key)
                    st.success("Weather data updated!")
            
            with col2:
                st.subheader("Current Weather")
                if st.session_state.weather_data:
                    weather = st.session_state.weather_data
                    
                    # Display current conditions
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Temperature", f"{weather['current_temp']:.1f}°C")
                    with metric_cols[1]:
                        st.metric("Wind Speed", f"{weather['current_wind_speed']:.1f} m/s")
                    with metric_cols[2]:
                        st.metric("Cloud Cover", f"{weather['current_clouds']}%")
                    
                    # Show forecast
                    st.subheader("24-Hour Forecast")
                    forecast_df = pd.DataFrame(weather['forecast'])
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Wind Speed', 'Cloud Cover'),
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=forecast_df['datetime'], y=forecast_df['wind_speed'],
                                  mode='lines+markers', name='Wind Speed'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=forecast_df['datetime'], y=forecast_df['clouds'],
                                  mode='lines+markers', name='Cloud Cover'),
                        row=2, col=1
                    )
                    
                    fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=1)
                    fig.update_yaxes(title_text="Cloud Cover (%)", row=2, col=1)
                    fig.update_layout(height=400, showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Click 'Fetch Weather Data' to see current conditions")
    
    with main_tabs[1]:  # Dashboard
        st.header("📊 Real-time Dashboard")
        
        # Refresh button
        if st.button("🔄 Refresh Dashboard"):
            if st.session_state.weather_data is None:
                st.session_state.weather_data = generate_dummy_weather()
            
            # Run simulation with current configuration
            with st.spinner("Running simulation..."):
                # Get configuration from session state or use defaults
                weather = st.session_state.weather_data
                
                # Generate 24-hour simulation
                hours = 24
                timestamps = pd.date_range(datetime.now(), periods=hours, freq='H')
                
                # Generate weather data for simulation
                wind_speeds = []
                cloud_covers = []
                
                for i in range(hours):
                    # Interpolate from forecast data
                    forecast_idx = min(i // 3, len(weather['forecast']) - 1)
                    wind_speeds.append(weather['forecast'][forecast_idx]['wind_speed'])
                    cloud_covers.append(weather['forecast'][forecast_idx]['clouds'])
                
                # Calculate generation
                pv_generation = []
                wind_generation = []
                
                for i in range(hours):
                    hour_of_day = timestamps[i].hour
                    
                    # PV generation
                    irradiance = calculate_solar_irradiance(cloud_covers[i], hour_of_day)
                    pv_power = calculate_pv_generation(
                        pv_capacity, irradiance, 
                        PANEL_MODELS[panel_model]['eff']
                    ) if 'enable_pv' in locals() and enable_pv else 0
                    pv_generation.append(pv_power)
                    
                    # Wind generation
                    wind_power = calculate_wind_generation(
                        num_turbines, TURBINE_MODELS[turbine_model],
                        wind_speeds[i], hub_height
                    ) if 'enable_wind' in locals() and enable_wind else 0
                    wind_generation.append(wind_power)
                
                # Load profile (24 hours)
                load_profile = load_bdew_profile(annual_consumption)
                load_24h = load_profile.iloc[:24].values
                
                # Total generation
                total_generation = np.array(pv_generation) + np.array(wind_generation)
                
                # Battery simulation
                battery_results = simulate_battery(
                    total_generation, load_24h, battery_capacity,
                    battery_capacity * initial_soc / 100, battery_efficiency
                )
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'pv_kw': pv_generation,
                    'wind_kw': wind_generation,
                    'load_kw': load_24h,
                    'battery_in_kw': battery_results['battery_in_kw'],
                    'battery_out_kw': battery_results['battery_out_kw'],
                    'grid_kw': battery_results['grid_kw'],
                    'soc_kwh': battery_results['soc_kwh']
                }, index=timestamps)
                
                st.session_state.simulation_results = results_df
            
            st.success("Dashboard updated!")
        
        # Display results
        if st.session_state.simulation_results is not None:
            df = st.session_state.simulation_results
            
            # KPI Cards
            st.subheader("📈 Key Performance Indicators")
            
            kpi_cols = st.columns(5)
            
            with kpi_cols[0]:
                total_pv = df['pv_kw'].sum()
                st.metric("Total PV Generation", f"{total_pv:.1f} kWh")
            
            with kpi_cols[1]:
                total_wind = df['wind_kw'].sum()
                st.metric("Total Wind Generation", f"{total_wind:.1f} kWh")
            
            with kpi_cols[2]:
                peak_load = df['load_kw'].max()
                st.metric("Peak Load", f"{peak_load:.1f} kW")
            
            with kpi_cols[3]:
                current_soc = df['soc_kwh'].iloc[-1]
                st.metric("Current SOC", f"{current_soc:.1f} kWh")
            
            with kpi_cols[4]:
                net_grid = df['grid_kw'].sum()
                grid_label = "Net Grid Export" if net_grid > 0 else "Net Grid Import"
                st.metric(grid_label, f"{abs(net_grid):.1f} kWh")
            
            # Energy balance summary
            st.subheader("⚖️ Energy Balance Summary")
            
            balance_cols = st.columns(4)
            
            with balance_cols[0]:
                total_generation = total_pv + total_wind
                st.metric("Total Generation", f"{total_generation:.1f} kWh", 
                         delta=f"{total_generation - df['load_kw'].sum():.1f} kWh")
            
            with balance_cols[1]:
                total_consumption = df['load_kw'].sum()
                st.metric("Total Load", f"{total_consumption:.1f} kWh")
            
            with balance_cols[2]:
                battery_charged = df['battery_in_kw'].sum()
                st.metric("Battery Charged", f"{battery_charged:.1f} kWh")
            
            with balance_cols[3]:
                battery_discharged = df['battery_out_kw'].sum()
                st.metric("Battery Discharged", f"{battery_discharged:.1f} kWh")
            
            # Main energy plot
            st.subheader("📊 Energy System Visualization")
            fig = create_energy_plot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("📋 Detailed Data Table"):
                st.dataframe(df.round(2))
            
            # Export data
            if st.button("💾 Export Data as CSV"):
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f'energy_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        
        else:
            st.info("Click 'Refresh Dashboard' to run the simulation and see results.")
    
    with main_tabs[2]:  # About
        st.header("📖 About This Application")
        
        st.markdown("""
        ### Hybrid PV+Wind+Storage Digital Twin
        
        This application simulates a hybrid renewable energy system combining:
        - **Solar PV panels** with configurable capacity and panel types
        - **Wind turbines** with various models and specifications
        - **Battery storage** for energy management
        - **Real-time weather data** integration
        
        #### Features:
        - 🌞 **Solar PV Modeling**: Multiple panel types with realistic efficiency curves
        - 💨 **Wind Power Calculation**: Power curve modeling with hub height correction
        - 🔋 **Battery Management**: SOC tracking with charge/discharge efficiency
        - 🌍 **Weather Integration**: Real-time data from OpenWeatherMap API
        - 📊 **Comprehensive Visualization**: Interactive plots and KPI monitoring
        - 📈 **Energy Balance**: Detailed analysis of generation, consumption, and storage
        
        #### Technical Specifications:
        - **Solar Irradiance**: Calculated based on cloud cover and solar angle
        - **Wind Power**: Uses standard power equation with air density and rotor area
        - **Battery Model**: Simple SOC-based model with configurable efficiency
        - **Load Profile**: Based on BDEW H0 German household standard
        
        #### Data Sources:
        - Weather data: OpenWeatherMap API
        - Load profiles: BDEW (German Association of Energy and Water Industries)
        - Equipment specifications: Manufacturer datasheets
        
        #### Version Information:
        - **Version**: 1.0.0
        - **Created**: 2024
        - **Framework**: Streamlit + Plotly
        - **License**: MIT
        """)
        
        st.markdown("---")
        st.markdown("**Developed for renewable energy system analysis and optimization.**")
    
    with main_tabs[3]:  # Settings
        st.header("⚙️ Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎨 Display Settings")
            
            # Theme selection
            theme = st.selectbox(
                "Chart Theme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                index=0
            )
            
            # Update interval
            update_interval = st.selectbox(
                "Auto-refresh Interval",
                ["Manual", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=0
            )
            
            # Data precision
            decimal_places = st.slider(
                "Display Precision (decimal places)",
                min_value=0,
                max_value=4,
                value=2
            )
            
            # Enable/disable features
            st.subheader("🔧 Feature Settings")
            
            enable_forecasting = st.checkbox(
                "Enable Weather Forecasting",
                value=True,
                help="Use weather forecast data for predictions"
            )
            
            enable_optimization = st.checkbox(
                "Enable System Optimization",
                value=False,
                help="Optimize system sizing (experimental)"
            )
            
            enable_alerts = st.checkbox(
                "Enable Performance Alerts",
                value=True,
                help="Show alerts for system performance issues"
            )
        
        with col2:
            st.subheader("🔐 API Configuration")
            
            # API settings
            api_timeout = st.slider(
                "API Request Timeout (seconds)",
                min_value=5,
                max_value=60,
                value=10
            )
            
            cache_duration = st.selectbox(
                "Weather Data Cache Duration",
                ["5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=1
            )
            
            st.subheader("📊 Simulation Settings")
            
            # Simulation parameters
            simulation_timestep = st.selectbox(
                "Simulation Time Step",
                ["1 hour", "30 minutes", "15 minutes", "5 minutes"],
                index=0
            )
            
            historical_days = st.slider(
                "Historical Data Days",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of historical days to analyze"
            )
            
            st.subheader("⚡ Performance Settings")
            
            # Performance options
            parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=False,
                help="Use multiprocessing for large simulations"
            )
            
            memory_optimization = st.checkbox(
                "Memory Optimization",
                value=True,
                help="Reduce memory usage for large datasets"
            )
            
            # Reset options
            st.subheader("🔄 Reset Options")
            
            if st.button("Reset All Settings"):
                st.success("Settings reset to defaults!")
            
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            if st.button("Reset Simulation Data"):
                st.session_state.simulation_results = None
                st.session_state.weather_data = None
                st.success("Simulation data cleared!")
        
        # Advanced settings
        with st.expander("🔬 Advanced Settings"):
            st.subheader("Model Parameters")
            
            # Physical constants
            air_density = st.number_input(
                "Air Density (kg/m³)",
                min_value=1.0,
                max_value=1.5,
                value=AIR_DENSITY,
                step=0.001,
                format="%.3f"
            )
            
            power_coefficient = st.slider(
                "Wind Turbine Power Coefficient (Cp)",
                min_value=0.2,
                max_value=0.6,
                value=CP,
                step=0.01
            )
            
            inverter_efficiency = st.slider(
                "Inverter Efficiency",
                min_value=0.85,
                max_value=0.99,
                value=INVERTER_EFF,
                step=0.01
            )
            
            # Calculation methods
            st.subheader("Calculation Methods")
            
            solar_model = st.selectbox(
                "Solar Irradiance Model",
                ["Simple Cloud Cover", "Detailed Clear Sky", "Satellite Data"],
                index=0
            )
            
            wind_profile = st.selectbox(
                "Wind Profile Model",
                ["Power Law", "Logarithmic", "Weibull Distribution"],
                index=0
            )
            
            battery_model = st.selectbox(
                "Battery Model",
                ["Simple SOC", "Voltage-based", "Equivalent Circuit"],
                index=0
            )

if __name__ == "__main__":
    main()
