"""
Sistema de previsão de pico de floração e análise de tendências
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.blooming_predictor import BloomingPredictor
from scripts.feature_engineering import FeatureEngineer
from scripts.weather_data_collector import WeatherDataCollector
import warnings
warnings.filterwarnings('ignore')

class BloomingForecast:
    def __init__(self):
        """Inicializa o sistema de previsão de floração"""
        self.predictor = BloomingPredictor()
        self.feature_engineer = FeatureEngineer()
        self.weather_collector = WeatherDataCollector()
        
    def generate_forecast(self, latitude, longitude, days_ahead=14):
        """
        Gera previsão completa de floração para os próximos dias
        
        Args:
            latitude (float): Latitude da localização
            longitude (float): Longitude da localização
            days_ahead (int): Número de dias para prever
            
        Returns:
            dict: Previsão completa de floração
        """
        print(f"Gerando previsão de floração para {days_ahead} dias...")
        
        # Coleta dados de previsão meteorológica
        forecast_data = self.weather_collector.collect_forecast_data(
            latitude, longitude, days=days_ahead
        )
        
        if forecast_data is None:
            return {"error": "Não foi possível coletar dados meteorológicos"}
        
        # Processa features
        processed_data = self.feature_engineer.process_weather_data(forecast_data)
        
        # Carrega modelo treinado
        try:
            self.predictor.load_model()
        except FileNotFoundError:
            return {"error": "Modelo não encontrado. Execute o treinamento primeiro."}
        
        # Prediz probabilidades de floração
        probabilities = self.predictor.predict_blooming_probability(processed_data)
        
        # Analisa pico de floração
        peak_analysis = self.analyze_peak_blooming(processed_data, probabilities)
        
        # Gera insights
        insights = self.generate_insights(processed_data, probabilities)
        
        # Cria visualizações
        self.create_visualizations(processed_data, probabilities, peak_analysis)
        
        return {
            "forecast_data": processed_data.to_dict('records'),
            "probabilities": probabilities.tolist(),
            "peak_analysis": peak_analysis,
            "insights": insights,
            "summary": self.generate_summary(peak_analysis, insights)
        }
    
    def analyze_peak_blooming(self, df, probabilities):
        """
        Analisa o pico de floração
        
        Args:
            df (pd.DataFrame): Dados processados
            probabilities (np.array): Probabilidades de floração
            
        Returns:
            dict: Análise do pico de floração
        """
        # Encontra o pico
        peak_idx = np.argmax(probabilities)
        peak_date = df.iloc[peak_idx]['date']
        # Converte para string ISO se for datetime
        if hasattr(peak_date, 'isoformat'):
            peak_date = peak_date.isoformat()
        elif hasattr(peak_date, 'strftime'):
            peak_date = peak_date.strftime('%Y-%m-%d')
        peak_probability = float(probabilities[peak_idx])
        
        # Encontra períodos de alta probabilidade
        high_prob_threshold = 0.7
        high_prob_indices = np.where(probabilities >= high_prob_threshold)[0]
        
        # Calcula estatísticas
        avg_probability = np.mean(probabilities)
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        
        # Identifica tendências
        trend = self.calculate_trend(probabilities)
        
        # Encontra janelas de melhor floração
        best_windows = self.find_best_blooming_windows(probabilities, window_size=3)
        
        return {
            "peak_date": peak_date,
            "peak_probability": peak_probability,
            "peak_day": int(peak_idx + 1),
            "avg_probability": float(avg_probability),
            "max_probability": float(max_probability),
            "min_probability": float(min_probability),
            "high_probability_days": int(len(high_prob_indices)),
            "high_probability_dates": [df.iloc[i]['date'].isoformat() if hasattr(df.iloc[i]['date'], 'isoformat') else str(df.iloc[i]['date']) for i in high_prob_indices],
            "trend": trend,
            "best_windows": best_windows,
            "confidence": self.calculate_confidence(probabilities)
        }
    
    def calculate_trend(self, probabilities):
        """Calcula tendência das probabilidades"""
        if len(probabilities) < 2:
            return "estável"
        
        # Regressão linear simples
        x = np.arange(len(probabilities))
        slope = np.polyfit(x, probabilities, 1)[0]
        
        if slope > 0.01:
            return "crescendo"
        elif slope < -0.01:
            return "decrescendo"
        else:
            return "estável"
    
    def find_best_blooming_windows(self, probabilities, window_size=3):
        """Encontra as melhores janelas de tempo para floração"""
        if len(probabilities) < window_size:
            return []
        
        window_scores = []
        for i in range(len(probabilities) - window_size + 1):
            window_avg = np.mean(probabilities[i:i+window_size])
            window_scores.append((i, window_avg))
        
        # Ordena por score e retorna top 3
        window_scores.sort(key=lambda x: x[1], reverse=True)
        return window_scores[:3]
    
    def calculate_confidence(self, probabilities):
        """Calcula nível de confiança da previsão"""
        # Baseado na consistência das probabilidades
        std_prob = np.std(probabilities)
        avg_prob = np.mean(probabilities)
        
        if std_prob < 0.1 and avg_prob > 0.5:
            return "alta"
        elif std_prob < 0.2:
            return "média"
        else:
            return "baixa"
    
    def generate_insights(self, df, probabilities):
        """Gera insights sobre as condições de floração"""
        insights = []
        
        # Análise de temperatura
        avg_temp = df['temperature_2m'].mean()
        if 15 <= avg_temp <= 25:
            insights.append("Temperatura ideal para floração")
        elif avg_temp < 15:
            insights.append("Temperatura baixa pode atrasar a floração")
        else:
            insights.append("Temperatura alta pode acelerar a floração")
        
        # Análise de umidade
        avg_humidity = df['relative_humidity_2m'].mean()
        if 40 <= avg_humidity <= 70:
            insights.append("Umidade ideal para floração")
        elif avg_humidity < 40:
            insights.append("Umidade baixa pode prejudicar a floração")
        else:
            insights.append("Umidade alta pode favorecer doenças")
        
        # Análise de precipitação
        total_precip = df['precipitation'].sum()
        if 0 < total_precip < 50:
            insights.append("Precipitação moderada favorece a floração")
        elif total_precip == 0:
            insights.append("Falta de chuva pode prejudicar a floração")
        else:
            insights.append("Chuva excessiva pode danificar as flores")
        
        # Soil analysis removed - using only weather data
        
        # Análise de luz solar
        avg_sunlight = (100 - df['cloud_cover'].mean()) / 100
        if avg_sunlight > 0.6:
            insights.append("Boa disponibilidade de luz solar")
        else:
            insights.append("Pouca luz solar pode atrasar a floração")
        
        return insights
    
    def generate_summary(self, peak_analysis, insights):
        """Gera resumo da previsão"""
        summary = {
            "recommendation": self.get_recommendation(peak_analysis),
            "key_findings": [
                f"Pico de floração previsto para: {peak_analysis['peak_date']}",
                f"Probabilidade máxima: {peak_analysis['peak_probability']:.1%}",
                f"Tendência: {peak_analysis['trend']}",
                f"Confiança: {peak_analysis['confidence']}"
            ],
            "action_items": self.get_action_items(peak_analysis, insights)
        }
        return summary
    
    def get_recommendation(self, peak_analysis):
        """Gera recomendação baseada na análise"""
        if peak_analysis['peak_probability'] > 0.8:
            return "Excelente período para floração! Prepare-se para o pico."
        elif peak_analysis['peak_probability'] > 0.6:
            return "Bom período para floração. Monitore as condições."
        elif peak_analysis['peak_probability'] > 0.4:
            return "Período moderado para floração. Algumas flores podem florescer."
        else:
            return "Período desfavorável para floração. Aguarde condições melhores."
    
    def get_action_items(self, peak_analysis, insights):
        """Gera itens de ação baseados na análise"""
        actions = []
        
        if peak_analysis['confidence'] == 'baixa':
            actions.append("Monitore as condições meteorológicas diariamente")
        
        if any("Temperatura baixa" in insight for insight in insights):
            actions.append("Considere proteger as plantas do frio")
        
        if any("Umidade baixa" in insight for insight in insights):
            actions.append("Aumente a irrigação se necessário")
        
        if any("Pouca luz solar" in insight for insight in insights):
            actions.append("Remova obstáculos que bloqueiem a luz solar")
        
        if peak_analysis['trend'] == 'crescendo':
            actions.append("Prepare-se para o aumento da floração")
        
        return actions
    
    def create_visualizations(self, df, probabilities, peak_analysis):
        """Cria visualizações da previsão"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Previsão de Floração - Análise Completa', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Probabilidades de floração ao longo do tempo
        axes[0, 0].plot(df['date'], probabilities, 'b-', linewidth=2, label='Probabilidade de Floração')
        axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Limiar Alto')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Limiar Médio')
        axes[0, 0].scatter([peak_analysis['peak_date']], [peak_analysis['peak_probability']], 
                           color='red', s=100, zorder=5, label='Pico')
        axes[0, 0].set_title('Probabilidades de Floração')
        axes[0, 0].set_ylabel('Probabilidade')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Temperatura vs Probabilidade
        axes[0, 1].scatter(df['temperature_2m'], probabilities, alpha=0.6, c=probabilities, cmap='RdYlBu')
        axes[0, 1].set_title('Temperatura vs Probabilidade de Floração')
        axes[0, 1].set_xlabel('Temperatura (°C)')
        axes[0, 1].set_ylabel('Probabilidade')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Umidade vs Probabilidade
        axes[1, 0].scatter(df['relative_humidity_2m'], probabilities, alpha=0.6, c=probabilities, cmap='RdYlBu')
        axes[1, 0].set_title('Umidade vs Probabilidade de Floração')
        axes[1, 0].set_xlabel('Umidade Relativa (%)')
        axes[1, 0].set_ylabel('Probabilidade')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Distribuição das probabilidades
        axes[1, 1].hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=np.mean(probabilities), color='red', linestyle='--', 
                          label=f'Média: {np.mean(probabilities):.2f}')
        axes[1, 1].set_title('Distribuição das Probabilidades')
        axes[1, 1].set_xlabel('Probabilidade')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/blooming_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizações salvas em: outputs/blooming_forecast.png")

def main():
    """Função principal para testar o sistema de previsão"""
    print("Sistema de Previsão de Floração")
    print("=" * 40)
    
    # Coordenadas de exemplo
    latitude = 38.6275
    longitude = -92.5666
    
    forecast_system = BloomingForecast()
    
    # Gera previsão
    result = forecast_system.generate_forecast(latitude, longitude, days_ahead=14)
    
    if "error" in result:
        print(f"Erro: {result['error']}")
    else:
        print("Previsão gerada com sucesso!")
        print(f"Pico de floração: {result['peak_analysis']['peak_date']}")
        print(f"Probabilidade máxima: {result['peak_analysis']['peak_probability']:.1%}")
        print(f"Confiança: {result['peak_analysis']['confidence']}")
        
        print("\nInsights:")
        for insight in result['insights']:
            print(f"- {insight}")
        
        print("\nResumo:")
        for finding in result['summary']['key_findings']:
            print(f"- {finding}")

if __name__ == "__main__":
    main()
