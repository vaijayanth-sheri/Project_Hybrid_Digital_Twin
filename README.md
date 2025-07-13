# 🌞 Hybrid Digital Twin Dashboard

An interactive Streamlit app to simulate a hybrid renewable energy plant (solar + wind + battery) with live weather integration via OpenWeatherMap API.

## 🚀 Features
- Configure solar panel model, wind turbine type, battery capacity, and household load
- Live weather fetch (irradiance + wind speed)
- Simulated 24h system performance (PV, wind, load, battery SOC, grid interaction)
- Visual charts and KPIs

## 📦 Requirements
- Python 3.8+
- `pip install -r requirements.txt`
- Set your OpenWeatherMap API key at: https://openweathermap.org/

## ▶️ Run the App
```bash
streamlit run app.py
