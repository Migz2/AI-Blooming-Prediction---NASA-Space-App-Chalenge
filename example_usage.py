"""
Exemplo de uso do sistema de previsão de floração
"""

import requests
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def example_api_usage():
    """Exemplo de uso da API"""
    print("🌸 EXEMPLO DE USO DA API DE PREVISÃO DE FLORAÇÃO")
    print("=" * 60)
    
    # Configurações
    base_url = "http://localhost:5000"
    
    # 1. Health Check
    print("\n1. Verificando status da API...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ API está funcionando")
            print(f"   Status: {response.json()['status']}")
        else:
            print("❌ API não está funcionando")
            return
    except requests.exceptions.ConnectionError:
        print("❌ API não está rodando. Execute 'python app.py' primeiro")
        return
    
    # 2. Informações do Modelo
    print("\n2. Verificando informações do modelo...")
    try:
        response = requests.get(f"{base_url}/api/model/info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Modelo carregado: {model_info['model_loaded']}")
            if model_info['model_loaded']:
                print(f"   Modelo: {model_info['model_name']}")
                print(f"   Features: {model_info['features_count']}")
        else:
            print("❌ Modelo não encontrado")
            print("   Execute 'python train_model.py' primeiro")
    except Exception as e:
        print(f"❌ Erro ao verificar modelo: {e}")
    
    # 3. Previsão de Floração
    print("\n3. Fazendo previsão de floração...")
    
    # Coordenadas de exemplo (Jefferson City, MO)
    forecast_data = {
        "latitude": 38.6275,
        "longitude": -92.5666,
        "days_ahead": 14
    }
    
    try:
        response = requests.post(f"{base_url}/api/forecast", json=forecast_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Previsão gerada com sucesso!")
            print(f"   Pico de floração: {result['peak_analysis']['peak_date']}")
            print(f"   Probabilidade máxima: {result['peak_analysis']['peak_probability']:.1%}")
            print(f"   Tendência: {result['peak_analysis']['trend']}")
            print(f"   Confiança: {result['peak_analysis']['confidence']}")
            
            # Mostra insights
            print("\n   Insights:")
            for insight in result['insights']:
                print(f"   - {insight}")
            
            # Mostra recomendações
            print("\n   Recomendações:")
            for action in result['summary']['action_items']:
                print(f"   - {action}")
            
            # Salva resultado
            save_forecast_result(result)
            
        else:
            print(f"❌ Erro na previsão: {response.status_code}")
            print(f"   Resposta: {response.text}")
            
    except Exception as e:
        print(f"❌ Erro ao fazer previsão: {e}")
    
    # 4. Dados Meteorológicos
    print("\n4. Coletando dados meteorológicos...")
    
    weather_data = {
        "latitude": 38.6275,
        "longitude": -92.5666,
        "start_date": "2025-09-01",
        "end_date": "2025-09-15"
    }
    
    try:
        response = requests.post(f"{base_url}/api/weather", json=weather_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Dados meteorológicos coletados: {result['summary']['total_records']} registros")
            print(f"   Período: {result['summary']['date_range']}")
            print(f"   Localização: {result['summary']['location']}")
        else:
            print(f"❌ Erro ao coletar dados meteorológicos: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erro ao coletar dados meteorológicos: {e}")

def save_forecast_result(result):
    """Salva resultado da previsão em arquivo"""
    try:
        # Cria DataFrame com dados da previsão
        forecast_df = pd.DataFrame(result['forecast_data'])
        forecast_df['bloom_probability'] = result['probabilities']
        
        # Salva em CSV
        forecast_df.to_csv('outputs/example_forecast.csv', index=False)
        print("   📁 Resultado salvo em: outputs/example_forecast.csv")
        
        # Cria visualização
        create_visualization(forecast_df, result)
        
    except Exception as e:
        print(f"   ⚠️ Erro ao salvar resultado: {e}")

def create_visualization(df, result):
    """Cria visualização dos resultados"""
    try:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Previsão de Floração - Análise Completa', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Probabilidades ao longo do tempo
        axes[0, 0].plot(df['date'], df['bloom_probability'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Limiar Alto')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Limiar Médio')
        axes[0, 0].set_title('Probabilidades de Floração')
        axes[0, 0].set_ylabel('Probabilidade')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Temperatura vs Probabilidade
        axes[0, 1].scatter(df['temperature_2m'], df['bloom_probability'], 
                          alpha=0.6, c=df['bloom_probability'], cmap='RdYlBu')
        axes[0, 1].set_title('Temperatura vs Probabilidade')
        axes[0, 1].set_xlabel('Temperatura (°C)')
        axes[0, 1].set_ylabel('Probabilidade')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Umidade vs Probabilidade
        axes[1, 0].scatter(df['relative_humidity_2m'], df['bloom_probability'], 
                          alpha=0.6, c=df['bloom_probability'], cmap='RdYlBu')
        axes[1, 0].set_title('Umidade vs Probabilidade')
        axes[1, 0].set_xlabel('Umidade Relativa (%)')
        axes[1, 0].set_ylabel('Probabilidade')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Distribuição das probabilidades
        axes[1, 1].hist(df['bloom_probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=df['bloom_probability'].mean(), color='red', linestyle='--', 
                          label=f'Média: {df["bloom_probability"].mean():.2f}')
        axes[1, 1].set_title('Distribuição das Probabilidades')
        axes[1, 1].set_xlabel('Probabilidade')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/example_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 Visualização salva em: outputs/example_visualization.png")
        
    except Exception as e:
        print(f"   ⚠️ Erro ao criar visualização: {e}")

def example_multiple_locations():
    """Exemplo com múltiplas localizações"""
    print("\n🌍 EXEMPLO COM MÚLTIPLAS LOCALIZAÇÕES")
    print("=" * 60)
    
    locations = [
        {"name": "Jefferson City, MO", "lat": 38.6275, "lon": -92.5666},
        {"name": "Springfield, MO", "lat": 37.2089, "lon": -93.2923},
        {"name": "Kansas City, MO", "lat": 39.0997, "lon": -94.5786}
    ]
    
    base_url = "http://localhost:5000"
    
    for location in locations:
        print(f"\n📍 {location['name']}")
        print("-" * 40)
        
        try:
            response = requests.post(f"{base_url}/api/forecast", json={
                "latitude": location["lat"],
                "longitude": location["lon"],
                "days_ahead": 7
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Pico: {result['peak_analysis']['peak_date']}")
                print(f"   Probabilidade: {result['peak_analysis']['peak_probability']:.1%}")
                print(f"   Tendência: {result['peak_analysis']['trend']}")
            else:
                print(f"   ❌ Erro: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")

def main():
    """Função principal"""
    print("🚀 INICIANDO EXEMPLOS DE USO")
    print("=" * 60)
    print("Certifique-se de que a API está rodando:")
    print("1. Execute 'python train_model.py' para treinar o modelo")
    print("2. Execute 'python app.py' para iniciar a API")
    print("3. Execute este script para testar")
    
    input("\nPressione Enter para continuar...")
    
    # Exemplo principal
    example_api_usage()
    
    # Exemplo com múltiplas localizações
    example_multiple_locations()
    
    print("\n" + "=" * 60)
    print("✅ EXEMPLOS CONCLUÍDOS!")
    print("=" * 60)
    print("Arquivos gerados:")
    print("- outputs/example_forecast.csv")
    print("- outputs/example_visualization.png")
    print("\nPara usar a interface web, acesse: http://localhost:5000")

if __name__ == "__main__":
    main()
