"""
Teste simples do sistema de previsão de floração
"""

import sys
sys.path.append('scripts')

from scripts.weather_data_collector import WeatherDataCollector
import pandas as pd
from datetime import datetime, timedelta

def test_basic_functionality():
    """Testa funcionalidade básica"""
    print("🧪 Testando funcionalidade básica...")
    
    try:
        # Testa coletor de dados
        collector = WeatherDataCollector()
        print("✅ Coletor de dados inicializado")
        
        # Testa coleta de dados
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        print(f"Coletando dados de {start_date} até {end_date}")
        data = collector.collect_weather_data(38.6275, -92.5666, start_date, end_date)
        
        if data is not None:
            print(f"✅ Dados coletados: {len(data)} registros")
            print(f"✅ Colunas: {list(data.columns)}")
            
            # Testa processamento básico
            print("\n🧪 Testando processamento básico...")
            
            # Features básicas
            data['hour'] = data['date'].dt.hour
            data['day_of_year'] = data['date'].dt.dayofyear
            data['month'] = data['date'].dt.month
            
            # Features de temperatura
            if 'temperature_2m' in data.columns:
                data['temp_avg_24h'] = data['temperature_2m'].rolling(window=24, min_periods=1).mean()
                print("✅ Features de temperatura criadas")
            
            # Features de umidade
            if 'relative_humidity_2m' in data.columns:
                data['humidity_avg_24h'] = data['relative_humidity_2m'].rolling(window=24, min_periods=1).mean()
                print("✅ Features de umidade criadas")
            
            # Features de precipitação
            if 'precipitation' in data.columns:
                data['precip_24h'] = data['precipitation'].rolling(window=24, min_periods=1).sum()
                print("✅ Features de precipitação criadas")
            
            # Score básico de floração
            if 'temperature_2m' in data.columns and 'relative_humidity_2m' in data.columns:
                # Temperatura ideal: 15-25°C
                temp_score = ((data['temperature_2m'] >= 15) & (data['temperature_2m'] <= 25)).astype(float)
                
                # Umidade ideal: 40-70%
                humidity_score = ((data['relative_humidity_2m'] >= 40) & 
                                (data['relative_humidity_2m'] <= 70)).astype(float)
                
                # Score combinado
                data['bloom_score'] = (temp_score * 0.6 + humidity_score * 0.4)
                print("✅ Score de floração criado")
                
                # Estatísticas
                avg_score = data['bloom_score'].mean()
                max_score = data['bloom_score'].max()
                print(f"✅ Score médio: {avg_score:.2f}")
                print(f"✅ Score máximo: {max_score:.2f}")
            
            print(f"\n✅ Processamento concluído: {data.shape}")
            print(f"✅ Features criadas: {len(data.columns)}")
            
            return True
        else:
            print("❌ Falha na coleta de dados")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Função principal"""
    print("=" * 50)
    print("TESTE SIMPLES DO SISTEMA")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 Sistema básico funcionando!")
        print("\nPróximos passos:")
        print("1. Execute 'python train_simple_model.py' para treinar modelo")
        print("2. Execute 'python app.py' para iniciar a API")
    else:
        print("\n❌ Sistema básico com problemas")
    
    return success

if __name__ == "__main__":
    main()
