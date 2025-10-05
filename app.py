"""
API Flask para previsão de floração (blooming) das flores
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Adiciona o diretório scripts ao path
sys.path.append('scripts')

from scripts.blooming_forecast import BloomingForecast
from scripts.weather_data_collector import WeatherDataCollector
from scripts.feature_engineering import FeatureEngineer

app = Flask(__name__)

# Inicializa o sistema de previsão
forecast_system = BloomingForecast()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """
    Endpoint para obter previsão de floração
    
    Body JSON:
    {
        "latitude": float,
        "longitude": float,
        "days_ahead": int (opcional, padrão: 14)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Dados JSON não fornecidos"}), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        days_ahead = data.get('days_ahead', 14)
        
        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude e longitude são obrigatórios"}), 400
        
        if not (-90 <= latitude <= 90):
            return jsonify({"error": "Latitude deve estar entre -90 e 90"}), 400
        
        if not (-180 <= longitude <= 180):
            return jsonify({"error": "Longitude deve estar entre -180 e 180"}), 400
        
        if not (1 <= days_ahead <= 30):
            return jsonify({"error": "days_ahead deve estar entre 1 e 30"}), 400
        
        # Gera previsão
        result = forecast_system.generate_forecast(latitude, longitude, days_ahead)
        
        if "error" in result:
            return jsonify(convert_numpy_types(result)), 500
        
        return jsonify(convert_numpy_types(result))
        
    except Exception as e:
        # Converte exceção para string de forma segura
        error_msg = str(e).replace("'", '"').replace('\\', '/')
        return jsonify({"error": f"Erro interno: {error_msg}"}), 500

@app.route('/api/weather', methods=['POST'])
def get_weather_data():
    """
    Endpoint para obter dados meteorológicos
    
    Body JSON:
    {
        "latitude": float,
        "longitude": float,
        "start_date": string (YYYY-MM-DD),
        "end_date": string (YYYY-MM-DD)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Dados JSON não fornecidos"}), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([latitude, longitude, start_date, end_date]):
            return jsonify({"error": "Todos os parâmetros são obrigatórios"}), 400
        
        # Valida datas
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                return jsonify({"error": "Data de início deve ser anterior à data de fim"}), 400
        except ValueError:
            return jsonify({"error": "Formato de data inválido. Use YYYY-MM-DD"}), 400
        
        # Coleta dados meteorológicos
        weather_collector = WeatherDataCollector()
        weather_data = weather_collector.collect_weather_data(
            latitude, longitude, start_date, end_date
        )
        
        if weather_data is None:
            return jsonify({"error": "Não foi possível coletar dados meteorológicos"}), 500
        
        # Converte para formato JSON
        result = {
            "data": weather_data.to_dict('records'),
            "summary": {
                "total_records": len(weather_data),
                "date_range": f"{start_date} to {end_date}",
                "location": f"{latitude}, {longitude}"
            }
        }
        
        return jsonify(convert_numpy_types(result))
        
    except Exception as e:
        # Converte exceção para string de forma segura
        error_msg = str(e).replace("'", '"').replace('\\', '/')
        return jsonify({"error": f"Erro interno: {error_msg}"}), 500

@app.route('/api/features', methods=['POST'])
def get_processed_features():
    """
    Endpoint para obter features processadas
    
    Body JSON:
    {
        "latitude": float,
        "longitude": float,
        "start_date": string (YYYY-MM-DD),
        "end_date": string (YYYY-MM-DD)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Dados JSON não fornecidos"}), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([latitude, longitude, start_date, end_date]):
            return jsonify({"error": "Todos os parâmetros são obrigatórios"}), 400
        
        # Coleta dados meteorológicos
        weather_collector = WeatherDataCollector()
        weather_data = weather_collector.collect_weather_data(
            latitude, longitude, start_date, end_date
        )
        
        if weather_data is None:
            return jsonify({"error": "Não foi possível coletar dados meteorológicos"}), 500
        
        # Processa features
        feature_engineer = FeatureEngineer()
        processed_data = feature_engineer.process_weather_data(weather_data)
        
        # Converte para formato JSON
        result = {
            "data": processed_data.to_dict('records'),
            "features": feature_engineer.feature_columns,
            "summary": {
                "total_records": len(processed_data),
                "total_features": len(feature_engineer.feature_columns),
                "date_range": f"{start_date} to {end_date}",
                "location": f"{latitude}, {longitude}"
            }
        }
        
        return jsonify(convert_numpy_types(result))
        
    except Exception as e:
        # Converte exceção para string de forma segura
        error_msg = str(e).replace("'", '"').replace('\\', '/')
        return jsonify({"error": f"Erro interno: {error_msg}"}), 500

@app.route('/api/health')
def health_check():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/model/info')
def model_info():
    """Informações sobre o modelo"""
    try:
        # Tenta carregar o modelo para verificar se existe
        forecast_system.predictor.load_model()
        
        return jsonify({
            "model_loaded": True,
            "model_name": forecast_system.predictor.best_model_name,
            "features_count": len(forecast_system.predictor.feature_columns),
            "status": "ready"
        })
    except FileNotFoundError:
        return jsonify({
            "model_loaded": False,
            "status": "model_not_found",
            "message": "Modelo não encontrado. Execute o treinamento primeiro."
        }), 404
    except Exception as e:
        return jsonify({
            "model_loaded": False,
            "status": "error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handler para 404"""
    return jsonify({"error": "Endpoint não encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para 500"""
    return jsonify({"error": "Erro interno do servidor"}), 500

if __name__ == '__main__':
    # Cria diretórios necessários
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("Iniciando API de Previsão de Floração...")
    print("Endpoints disponíveis:")
    print("- POST /api/forecast - Previsão de floração")
    print("- POST /api/weather - Dados meteorológicos")
    print("- POST /api/features - Features processadas")
    print("- GET /api/health - Health check")
    print("- GET /api/model/info - Informações do modelo")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
