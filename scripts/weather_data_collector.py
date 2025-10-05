"""
Sistema de coleta de dados meteorológicos para previsão de floração
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np
from datetime import datetime, timedelta
import os
import json

class WeatherDataCollector:
    def __init__(self, cache_dir='.cache'):
        """Inicializa o coletor de dados meteorológicos"""
        self.cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        
    def collect_weather_data(self, latitude, longitude, start_date, end_date):
        """
        Coleta dados meteorológicos para um período específico
        
        Args:
            latitude (float): Latitude da localização
            longitude (float): Longitude da localização
            start_date (str): Data de início no formato YYYY-MM-DD
            end_date (str): Data de fim no formato YYYY-MM-DD
            
        Returns:
            pd.DataFrame: DataFrame com dados meteorológicos
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "precipitation", "rain", "relative_humidity_2m", 
                "dew_point_2m", "apparent_temperature", "snowfall", "snow_depth", 
                "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", 
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", 
                "vapour_pressure_deficit", "et0_fao_evapotranspiration", 
                "wind_gusts_10m", "wind_direction_100m", "wind_speed_100m", 
                "wind_direction_10m", "wind_speed_10m"
            ],
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            print(f"Coordenadas: {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevação: {response.Elevation()} m asl")
            print(f"Diferença de fuso horário para GMT+0: {response.UtcOffsetSeconds()}s")
            
            # Processa dados horários
            hourly = response.Hourly()
            
            # Extrai todas as variáveis
            variables = [
                "temperature_2m", "precipitation", "rain", "relative_humidity_2m",
                "dew_point_2m", "apparent_temperature", "snowfall", "snow_depth",
                "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                "vapour_pressure_deficit", "et0_fao_evapotranspiration",
                "wind_gusts_10m", "wind_direction_100m", "wind_speed_100m",
                "wind_direction_10m", "wind_speed_10m"
            ]
            
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            # Adiciona todas as variáveis ao DataFrame
            for i, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
            
            df = pd.DataFrame(data=hourly_data)
            return df
            
        except Exception as e:
            print(f"Erro ao coletar dados meteorológicos: {e}")
            return None
    
    def collect_forecast_data(self, latitude, longitude, days=14):
        """
        Coleta dados de previsão meteorológica para os próximos dias
        
        Args:
            latitude (float): Latitude da localização
            longitude (float): Longitude da localização
            days (int): Número de dias para previsão (padrão: 14)
            
        Returns:
            pd.DataFrame: DataFrame com dados de previsão
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m", "precipitation", "rain", "relative_humidity_2m",
                "dew_point_2m", "apparent_temperature", "snowfall", "snow_depth",
                "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                "vapour_pressure_deficit", "et0_fao_evapotranspiration",
                "wind_gusts_10m", "wind_direction_100m", "wind_speed_100m",
                "wind_direction_10m", "wind_speed_10m"
            ],
            "forecast_days": days
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            hourly = response.Hourly()
            
            variables = [
                "temperature_2m", "precipitation", "rain", "relative_humidity_2m",
                "dew_point_2m", "apparent_temperature", "snowfall", "snow_depth",
                "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                "vapour_pressure_deficit", "et0_fao_evapotranspiration",
                "wind_gusts_10m", "wind_direction_100m", "wind_speed_100m",
                "wind_direction_10m", "wind_speed_10m"
            ]
            
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            for i, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
            
            df = pd.DataFrame(data=hourly_data)
            return df
            
        except Exception as e:
            print(f"Erro ao coletar dados de previsão: {e}")
            return None
    
    def save_weather_data(self, df, filename):
        """Salva dados meteorológicos em arquivo CSV"""
        if df is not None:
            filepath = os.path.join('data', filename)
            df.to_csv(filepath, index=False)
            print(f"Dados salvos em: {filepath}")
            return filepath
        return None

def main():
    """Função principal para testar o coletor"""
    collector = WeatherDataCollector()
    
    # Coordenadas de exemplo (Jefferson City, MO)
    latitude = 38.6275
    longitude = -92.5666
    
    # Coleta dados históricos dos últimos 30 dias
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print("Coletando dados históricos...")
    historical_data = collector.collect_weather_data(latitude, longitude, start_date, end_date)
    if historical_data is not None:
        collector.save_weather_data(historical_data, 'historical_weather.csv')
    
    # Coleta dados de previsão para os próximos 14 dias
    print("Coletando dados de previsão...")
    forecast_data = collector.collect_forecast_data(latitude, longitude, days=14)
    if forecast_data is not None:
        collector.save_weather_data(forecast_data, 'forecast_weather.csv')

if __name__ == "__main__":
    main()
