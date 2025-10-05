"""
Test script to validate the complete flowering prediction system
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# Adiciona o diretório scripts ao path
sys.path.append('scripts')

def test_weather_collector():
    """Testa o coletor de dados meteorológicos"""
    print("🧪 Testando coletor de dados meteorológicos...")
    
    try:
        from scripts.weather_data_collector import WeatherDataCollector
        
        collector = WeatherDataCollector()
        
        # Testa coleta de dados históricos
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        data = collector.collect_weather_data(38.6275, -92.5666, start_date, end_date)
        
        if data is None:
            print("❌ Falha na coleta de dados históricos")
            return False
        
        print(f"✅ Dados históricos coletados: {len(data)} registros")
        
        # Testa coleta de dados de previsão
        forecast_data = collector.collect_forecast_data(38.6275, -92.5666, days=7)
        
        if forecast_data is None:
            print("❌ Falha na coleta de dados de previsão")
            return False
        
        print(f"✅ Dados de previsão coletados: {len(forecast_data)} registros")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do coletor: {e}")
        return False

def test_feature_engineering():
    """Testa o sistema de feature engineering"""
    print("\n🧪 Testando feature engineering...")
    
    try:
        from scripts.feature_engineering import FeatureEngineer
        
        # Cria dados de exemplo
        dates = pd.date_range(start='2025-01-01', periods=168, freq='H')
        sample_data = pd.DataFrame({
            'date': dates,
            'temperature_2m': np.random.normal(20, 5, 168),
            'relative_humidity_2m': np.random.normal(60, 15, 168),
            'precipitation': np.random.exponential(2, 168),
            # Soil features removed
            'cloud_cover': np.random.uniform(0, 100, 168)
        })
        
        engineer = FeatureEngineer()
        processed_data = engineer.process_weather_data(sample_data)
        
        if processed_data is None or len(processed_data) == 0:
            print("❌ Falha no processamento de features")
            return False
        
        print(f"✅ Features processadas: {processed_data.shape}")
        print(f"✅ Total de features: {len(engineer.feature_columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de feature engineering: {e}")
        return False

def test_model_training():
    """Testa o treinamento do modelo"""
    print("\n🧪 Testando treinamento do modelo...")
    
    try:
        from scripts.blooming_predictor import BloomingPredictor
        
        # Cria dados de exemplo para treinamento
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.uniform(0, 1, n_samples))
        
        predictor = BloomingPredictor()
        predictor.feature_columns = X.columns.tolist()
        
        # Testa treinamento
        scores = predictor.train_models(X, y)
        
        if predictor.best_model is None:
            print("❌ Falha no treinamento do modelo")
            return False
        
        print(f"✅ Modelo treinado: {predictor.best_model_name}")
        print(f"✅ R² Score: {scores[predictor.best_model_name]['r2']:.4f}")
        
        # Testa predição
        test_X = X.iloc[:10]
        predictions = predictor.predict_blooming_probability(test_X)
        
        if len(predictions) != 10:
            print("❌ Falha na predição")
            return False
        
        print(f"✅ Predições geradas: {len(predictions)}")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do modelo: {e}")
        return False

def test_api():
    """Testa a API Flask"""
    print("\n🧪 Testando API...")
    
    try:
        # Testa health check
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code != 200:
            print("❌ API não está respondendo")
            return False
        
        print("✅ Health check passou")
        
        # Testa endpoint de previsão
        forecast_data = {
            'latitude': 38.6275,
            'longitude': -92.5666,
            'days_ahead': 7
        }
        
        response = requests.post('http://localhost:5000/api/forecast', 
                               json=forecast_data, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Erro na previsão: {response.status_code}")
            print(f"Resposta: {response.text}")
            return False
        
        result = response.json()
        if 'error' in result:
            print(f"❌ Erro na previsão: {result['error']}")
            return False
        
        print("✅ Previsão gerada com sucesso")
        print(f"✅ Pico de floração: {result['peak_analysis']['peak_date']}")
        print(f"✅ Probabilidade máxima: {result['peak_analysis']['peak_probability']:.1%}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API não está rodando. Execute 'python app.py' primeiro")
        return False
    except Exception as e:
        print(f"❌ Erro no teste da API: {e}")
        return False

def test_forecast_system():
    """Testa o sistema completo de previsão"""
    print("\n🧪 Testando sistema de previsão...")
    
    try:
        from scripts.blooming_forecast import BloomingForecast
        
        forecast_system = BloomingForecast()
        
        # Testa geração de previsão
        result = forecast_system.generate_forecast(38.6275, -92.5666, days_ahead=7)
        
        if 'error' in result:
            print(f"❌ Erro na previsão: {result['error']}")
            return False
        
        print("✅ Previsão gerada com sucesso")
        print(f"✅ Pico de floração: {result['peak_analysis']['peak_date']}")
        print(f"✅ Probabilidade máxima: {result['peak_analysis']['peak_probability']:.1%}")
        print(f"✅ Confiança: {result['peak_analysis']['confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do sistema de previsão: {e}")
        return False

def test_file_structure():
    """Testa a estrutura de arquivos"""
    print("\n🧪 Testando estrutura de arquivos...")
    
    required_files = [
        'app.py',
        'train_model.py',
        'requirements.txt',
        'README.md',
        'scripts/weather_data_collector.py',
        'scripts/feature_engineering.py',
        'scripts/blooming_predictor.py',
        'scripts/blooming_forecast.py',
        'templates/index.html'
    ]
    
    required_dirs = [
        'data',
        'models',
        'outputs',
        'scripts',
        'templates',
        'static'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    if missing_files:
        print(f"❌ Arquivos faltando: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Diretórios faltando: {missing_dirs}")
        return False
    
    print("✅ Estrutura de arquivos OK")
    return True

def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("COMPLETE FLOWERING PREDICTION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Weather Data Collector", test_weather_collector),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Forecast System", test_forecast_system),
        ("Flask API", test_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        print("\nPróximos passos:")
        print("1. Execute 'python train_model.py' para treinar o modelo")
        print("2. Execute 'python app.py' para iniciar a API")
        print("3. Acesse http://localhost:5000 para usar a interface web")
    else:
        print(f"\n⚠️ {total - passed} teste(s) falharam. Verifique os erros acima.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)
