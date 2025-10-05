"""
Script principal para treinar o modelo de previsão de floração
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

def main():
    """Pipeline completo de treinamento do modelo"""
    print("=" * 60)
    print("SISTEMA DE PREVISÃO DE FLORAÇÃO - TREINAMENTO")
    print("=" * 60)
    
    # Configurações
    latitude = 38.6275  # Jefferson City, MO
    longitude = -92.5666
    
    # 1. Coleta de dados meteorológicos
    print("\n1. COLETANDO DADOS METEOROLÓGICOS...")
    print("-" * 40)
    
    weather_collector = WeatherDataCollector()
    
    # Coleta dados dos últimos 60 dias para treinamento
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"Coletando dados de {start_date} até {end_date}")
    historical_data = weather_collector.collect_weather_data(
        latitude, longitude, start_date, end_date
    )
    
    if historical_data is None:
        print("❌ Erro: Não foi possível coletar dados meteorológicos")
        return False
    
    print(f"✅ Dados coletados: {len(historical_data)} registros")
    weather_collector.save_weather_data(historical_data, 'historical_weather.csv')
    
    # 2. Processamento e feature engineering
    print("\n2. PROCESSANDO DADOS E CRIANDO FEATURES...")
    print("-" * 40)
    
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.process_weather_data(historical_data)
    
    print(f"✅ Features processadas: {processed_data.shape}")
    print(f"✅ Total de features: {len(feature_engineer.feature_columns)}")
    
    # Salva dados processados
    processed_data.to_csv('data/processed_weather_features.csv', index=False)
    print("✅ Dados processados salvos")
    
    # 3. Treinamento do modelo
    print("\n3. TREINANDO MODELO DE MACHINE LEARNING...")
    print("-" * 40)
    
    predictor = BloomingPredictor()
    X, y = predictor.prepare_training_data(processed_data)
    
    print(f"✅ Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Treina modelos
    scores = predictor.train_models(X, y)
    
    print(f"✅ Melhor modelo: {predictor.best_model_name}")
    
    # Salva modelo
    predictor.save_model()
    
    # 4. Análise de importância das features
    print("\n4. ANÁLISE DE IMPORTÂNCIA DAS FEATURES...")
    print("-" * 40)
    
    importance = predictor.get_feature_importance()
    if importance is not None:
        print("Top 10 features mais importantes:")
        for i, (_, row) in enumerate(importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # 5. Teste do modelo com dados de previsão
    print("\n5. TESTANDO MODELO COM DADOS DE PREVISÃO...")
    print("-" * 40)
    
    # Coleta dados de previsão para os próximos 14 dias
    forecast_data = weather_collector.collect_forecast_data(latitude, longitude, days=14)
    
    if forecast_data is not None:
        # Processa dados de previsão
        forecast_processed = feature_engineer.process_weather_data(forecast_data)
        
        # Faz predição
        probabilities = predictor.predict_blooming_probability(forecast_processed)
        
        # Encontra pico
        peak_idx = np.argmax(probabilities)
        peak_date = forecast_processed.iloc[peak_idx]['date']
        peak_probability = probabilities[peak_idx]
        
        print(f"✅ Previsão gerada para 14 dias")
        print(f"✅ Pico de floração: {peak_date} (probabilidade: {peak_probability:.1%})")
        print(f"✅ Probabilidade média: {np.mean(probabilities):.1%}")
        print(f"✅ Probabilidade máxima: {np.max(probabilities):.1%}")
        
        # Salva previsão
        forecast_result = forecast_processed.copy()
        forecast_result['bloom_probability'] = probabilities
        forecast_result.to_csv('outputs/forecast_result.csv', index=False)
        print("✅ Resultado da previsão salvo em outputs/forecast_result.csv")
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("Próximos passos:")
    print("1. Execute 'python app.py' para iniciar a API")
    print("2. Acesse http://localhost:5000 para usar a interface web")
    print("3. Use a API para fazer previsões personalizadas")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Sistema pronto para uso!")
        else:
            print("\n❌ Erro no treinamento")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)
