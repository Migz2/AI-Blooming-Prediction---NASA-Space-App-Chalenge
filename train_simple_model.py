"""
Script to train a simplified model using only basic and reliable features
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adiciona o diretório scripts ao path
sys.path.append('scripts')

from scripts.weather_data_collector import WeatherDataCollector
from scripts.feature_engineering import FeatureEngineer
from scripts.blooming_predictor import BloomingPredictor

def create_simple_features(df):
    """Creates only basic and reliable features"""
    df = df.copy()
    
    # Convert date to datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Basic temporal features
    df['hour'] = df['date'].dt.hour
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Features de temperatura básicas
    if 'temperature_2m' in df.columns:
        df['temp_avg_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
        df['temp_std_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).std()
        df['temp_trend'] = df['temperature_2m'].diff()
    
    # Features de umidade básicas
    if 'relative_humidity_2m' in df.columns:
        df['humidity_avg_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).mean()
        df['humidity_std_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).std()
    
    # Features de precipitação básicas
    if 'precipitation' in df.columns:
        df['precip_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
        df['precip_7d'] = df['precipitation'].rolling(window=168, min_periods=1).sum()
    
    # Features de vento básicas
    if 'wind_speed_10m' in df.columns:
        df['wind_speed_avg'] = df['wind_speed_10m'].rolling(window=24, min_periods=1).mean()
    
    # Features de pressão básicas
    if 'pressure_msl' in df.columns:
        df['pressure_trend'] = df['pressure_msl'].diff()
        df['pressure_24h_avg'] = df['pressure_msl'].rolling(window=24, min_periods=1).mean()
    
    # Features de cobertura de nuvens básicas
    if 'cloud_cover' in df.columns:
        df['cloud_cover_avg'] = df['cloud_cover'].rolling(window=24, min_periods=1).mean()
    
    # Features de floração básicas
    # Temperatura para floração
    if 'temperature_2m' in df.columns:
        df['temp_optimal_bloom'] = ((df['temperature_2m'] >= 15) & (df['temperature_2m'] <= 25)).astype(int)
        df['temp_bloom_score'] = np.where(
            df['temperature_2m'] < 15, 
            (df['temperature_2m'] - 5) / 10,
            np.where(
                df['temperature_2m'] > 25,
                (35 - df['temperature_2m']) / 10,
                1
            )
        )
    else:
        df['temp_optimal_bloom'] = 0
        df['temp_bloom_score'] = 0.5
    
    # Umidade para floração
    if 'relative_humidity_2m' in df.columns:
        df['humidity_optimal_bloom'] = ((df['relative_humidity_2m'] >= 40) & 
                                       (df['relative_humidity_2m'] <= 70)).astype(int)
        df['humidity_bloom_score'] = np.where(
            df['relative_humidity_2m'] < 40,
            df['relative_humidity_2m'] / 40,
            np.where(
                df['relative_humidity_2m'] > 70,
                (100 - df['relative_humidity_2m']) / 30,
                1
            )
        )
    else:
        df['humidity_optimal_bloom'] = 0
        df['humidity_bloom_score'] = 0.5
    
    # Precipitação para floração
    if 'precipitation' in df.columns:
        df['precip_optimal'] = ((df['precipitation'] > 0) & (df['precipitation'] < 5)).astype(int)
        df['precip_bloom_score'] = np.where(
            df['precipitation'] == 0,
            0.5,
            np.where(
                df['precipitation'] <= 5,
                1,
                np.maximum(0, 1 - (df['precipitation'] - 5) / 10)
            )
        )
    else:
        df['precip_optimal'] = 0
        df['precip_bloom_score'] = 0.5
    
    # Luz solar
    if 'cloud_cover' in df.columns:
        df['sunlight_score'] = 1 - (df['cloud_cover'] / 100)
        df['sunlight_optimal'] = (df['sunlight_score'] > 0.5).astype(int)
    else:
        df['sunlight_score'] = 0.5
        df['sunlight_optimal'] = 0
    
    # Score combinado de floração
    score_components = []
    weights = []
    
    if 'temp_bloom_score' in df.columns:
        score_components.append(df['temp_bloom_score'])
        weights.append(0.4)
    
    if 'humidity_bloom_score' in df.columns:
        score_components.append(df['humidity_bloom_score'])
        weights.append(0.3)
    
    if 'precip_bloom_score' in df.columns:
        score_components.append(df['precip_bloom_score'])
        weights.append(0.2)
    
    if 'sunlight_score' in df.columns:
        score_components.append(df['sunlight_score'])
        weights.append(0.1)
    
    if score_components:
        # Normaliza os pesos
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calcula score combinado
        df['bloom_score'] = sum(comp * weight for comp, weight in zip(score_components, weights))
    else:
        df['bloom_score'] = 0.5
    
    # Features de tendência
    df['bloom_score_24h_avg'] = df['bloom_score'].rolling(window=24, min_periods=1).mean()
    df['bloom_score_7d_avg'] = df['bloom_score'].rolling(window=168, min_periods=1).mean()
    df['bloom_score_trend'] = df['bloom_score'].diff()
    
    # Features sazonais básicas
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_spring_peak'] = df['month'].isin([4, 5]).astype(int)
    
    # Growing Degree Days
    if 'temperature_2m' in df.columns:
        base_temp = 10
        df['gdd'] = np.maximum(0, df['temperature_2m'] - base_temp)
        df['gdd_7d'] = df['gdd'].rolling(window=168, min_periods=1).sum()
        df['gdd_30d'] = df['gdd'].rolling(window=720, min_periods=1).sum()
    else:
        df['gdd'] = 0
        df['gdd_7d'] = 0
        df['gdd_30d'] = 0
    
    # Features de lag básicas
    if 'bloom_score' in df.columns:
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'bloom_score_lag_{lag}h'] = df['bloom_score'].shift(lag)
    
    # Features de janela deslizante básicas
    basic_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'bloom_score']
    available_cols = [col for col in basic_cols if col in df.columns]
    
    for col in available_cols:
        for window in [6, 12, 24, 48]:
            df[f'{col}_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
    
    # Remove colunas com muitos valores nulos
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Preenche valores nulos restantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def main():
    """Pipeline de treinamento simplificado"""
    print("=" * 60)
    print("SIMPLIFIED MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    latitude = 38.6275  # Jefferson City, MO
    longitude = -92.5666
    
    # 1. Weather data collection
    print("\n1. COLLECTING WEATHER DATA...")
    print("-" * 40)
    
    weather_collector = WeatherDataCollector()
    
    # Collect data from the last 30 days for training
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Collecting data from {start_date} to {end_date}")
    historical_data = weather_collector.collect_weather_data(
        latitude, longitude, start_date, end_date
    )
    
    if historical_data is None:
        print("❌ Error: Could not collect weather data")
        return False
    
    print(f"✅ Data collected: {len(historical_data)} records")
    
    # 2. Simplified processing
    print("\n2. PROCESSING DATA WITH SIMPLIFIED FEATURES...")
    print("-" * 40)
    
    processed_data = create_simple_features(historical_data)
    
    print(f"✅ Features processed: {processed_data.shape}")
    print(f"✅ Total features: {len(processed_data.columns) - 1}")  # -1 to exclude 'date'
    
    # Save processed data
    processed_data.to_csv('data/simple_processed_features.csv', index=False)
    print("✅ Processed data saved")
    
    # 3. Model training
    print("\n3. TRAINING SIMPLIFIED MODEL...")
    print("-" * 40)
    
    predictor = BloomingPredictor()
    X, y = predictor.prepare_training_data(processed_data)
    
    print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train models
    scores = predictor.train_models(X, y)
    
    print(f"✅ Best model: {predictor.best_model_name}")
    
    # Save model
    predictor.save_model('models/simple_blooming_model.pkl')
    
    # 4. Model testing
    print("\n4. TESTING MODEL...")
    print("-" * 40)
    
    # Collect forecast data
    forecast_data = weather_collector.collect_forecast_data(latitude, longitude, days=7)
    
    if forecast_data is not None:
        # Process forecast data
        forecast_processed = create_simple_features(forecast_data)
        
        # Make prediction
        probabilities = predictor.predict_blooming_probability(forecast_processed)
        
        # Find peak
        peak_idx = np.argmax(probabilities)
        peak_date = forecast_processed.iloc[peak_idx]['date']
        peak_probability = probabilities[peak_idx]
        
        print(f"✅ Prediction generated for 7 days")
        print(f"✅ Peak flowering: {peak_date} (probability: {peak_probability:.1%})")
        print(f"✅ Average probability: {np.mean(probabilities):.1%}")
        print(f"✅ Maximum probability: {np.max(probabilities):.1%}")
        
        # Save prediction
        forecast_result = forecast_processed.copy()
        forecast_result['bloom_probability'] = probabilities
        forecast_result.to_csv('outputs/simple_forecast_result.csv', index=False)
        print("✅ Prediction result saved")
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run 'python app.py' to start the API")
    print("2. Access http://localhost:5000 to use the web interface")
    print("3. Use the API to make custom predictions")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
        print("\n🎉 Simplified system ready for use!")
    else:
        print("\n❌ Training error")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
