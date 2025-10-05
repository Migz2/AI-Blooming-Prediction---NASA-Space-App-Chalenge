"""
Feature engineering system for weather data and Blooming prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineering system"""
        self.feature_columns = []
        
    def create_weather_features(self, df):
        """
        Creates features derived from weather data
        
        Args:
            df (pd.DataFrame): DataFrame with weather data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        df = df.copy()
        
        # Converte data para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Features temporais
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Features de temperatura
        df['temp_range'] = df['temperature_2m'].max() - df['temperature_2m'].min()
        df['temp_avg_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
        df['temp_std_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).std()
        df['temp_trend'] = df['temperature_2m'].diff()
        
        # Features de umidade
        df['humidity_avg_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).mean()
        df['humidity_std_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).std()
        
        # Features de precipitação
        df['precip_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
        df['precip_7d'] = df['precipitation'].rolling(window=168, min_periods=1).sum()  # 7 dias
        # Features de intensidade de chuva (verifica se as colunas existem)
        if 'rain' in df.columns and 'precipitation' in df.columns:
            df['rain_intensity'] = df['rain'] / (df['precipitation'] + 1e-6)  # Evita divisão por zero
        else:
            df['rain_intensity'] = 0  # Valor padrão se não houver dados
        
        # Features de vento (verifica se as colunas existem)
        if 'wind_speed_10m' in df.columns:
            df['wind_speed_avg'] = df['wind_speed_10m'].rolling(window=24, min_periods=1).mean()
        
        if 'wind_gusts_10m' in df.columns and 'wind_speed_10m' in df.columns:
            df['wind_gust_ratio'] = df['wind_gusts_10m'] / (df['wind_speed_10m'] + 1e-6)
        else:
            df['wind_gust_ratio'] = 1  # Valor padrão
        
        # Features de solo completamente removidas
        # Não cria nenhuma feature de solo
        
        # Features de pressão (verifica se a coluna existe)
        if 'pressure_msl' in df.columns:
            df['pressure_trend'] = df['pressure_msl'].diff()
            df['pressure_24h_avg'] = df['pressure_msl'].rolling(window=24, min_periods=1).mean()
        else:
            df['pressure_trend'] = 0
            df['pressure_24h_avg'] = 1013.25  # Pressão padrão ao nível do mar
        
        # Features de evapotranspiração (verifica se a coluna existe)
        if 'et0_fao_evapotranspiration' in df.columns:
            df['et_24h_sum'] = df['et0_fao_evapotranspiration'].rolling(window=24, min_periods=1).sum()
            df['et_7d_sum'] = df['et0_fao_evapotranspiration'].rolling(window=168, min_periods=1).sum()
        else:
            df['et_24h_sum'] = 0
            df['et_7d_sum'] = 0
        
        # Features de cobertura de nuvens
        if 'cloud_cover' in df.columns:
            df['cloud_cover_avg'] = df['cloud_cover'].rolling(window=24, min_periods=1).mean()
        
        if all(col in df.columns for col in ['cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']):
            df['cloud_layers'] = df['cloud_cover_low'] + df['cloud_cover_mid'] + df['cloud_cover_high']
        
        # Features de déficit de pressão de vapor (verifica se a coluna existe)
        if 'vapour_pressure_deficit' in df.columns:
            df['vpd_avg'] = df['vapour_pressure_deficit'].rolling(window=24, min_periods=1).mean()
        else:
            df['vpd_avg'] = 0
        
        return df
    
    def create_basic_weather_features(self, df):
        """
        Cria features meteorológicas básicas e confiáveis
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos
            
        Returns:
            pd.DataFrame: DataFrame com features básicas
        """
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
        
        # Basic temperature features (no data leakage)
        if 'temperature_2m' in df.columns:
            # Use only past data
            df['temp_avg_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean().shift(1)
            df['temp_std_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).std().shift(1)
            df['temp_trend'] = df['temperature_2m'].diff().shift(1)
        
        # Features de umidade básicas (sem data leakage)
        if 'relative_humidity_2m' in df.columns:
            df['humidity_avg_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).mean().shift(1)
            df['humidity_std_24h'] = df['relative_humidity_2m'].rolling(window=24, min_periods=1).std().shift(1)
        
        # Features de precipitação básicas (sem data leakage)
        if 'precipitation' in df.columns:
            df['precip_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum().shift(1)
            df['precip_7d'] = df['precipitation'].rolling(window=168, min_periods=1).sum().shift(1)
        
        # Features de vento básicas (sem data leakage)
        if 'wind_speed_10m' in df.columns:
            df['wind_speed_avg'] = df['wind_speed_10m'].rolling(window=24, min_periods=1).mean().shift(1)
        
        # Features de pressão básicas (sem data leakage)
        if 'pressure_msl' in df.columns:
            df['pressure_trend'] = df['pressure_msl'].diff().shift(1)
            df['pressure_24h_avg'] = df['pressure_msl'].rolling(window=24, min_periods=1).mean().shift(1)
        
        # Features de cobertura de nuvens básicas (sem data leakage)
        if 'cloud_cover' in df.columns:
            df['cloud_cover_avg'] = df['cloud_cover'].rolling(window=24, min_periods=1).mean().shift(1)
        
        return df
    
    def create_basic_blooming_features(self, df):
        """
        Cria features básicas de floração
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos
            
        Returns:
            pd.DataFrame: DataFrame com features de floração básicas
        """
        df = df.copy()
        
        # Features de temperatura para floração
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
        
        # Features de umidade para floração
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
        
        # Features de precipitação para floração
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
        
        # Features de luz solar
        if 'cloud_cover' in df.columns:
            df['sunlight_score'] = 1 - (df['cloud_cover'] / 100)
            df['sunlight_optimal'] = (df['sunlight_score'] > 0.5).astype(int)
        else:
            df['sunlight_score'] = 0.5
            df['sunlight_optimal'] = 0
        
        # Score combinado de floração (apenas com features básicas)
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
        
        # Features de tendência de floração (sem data leakage)
        # Usa apenas dados passados para evitar vazamento
        df['bloom_score_24h_avg'] = df['bloom_score'].rolling(window=24, min_periods=1).mean().shift(1)
        df['bloom_score_7d_avg'] = df['bloom_score'].rolling(window=168, min_periods=1).mean().shift(1)
        df['bloom_score_trend'] = df['bloom_score'].diff().shift(1)
        
        return df
    
    def create_blooming_features(self, df):
        """
        Cria features específicas para previsão de floração
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos
            
        Returns:
            pd.DataFrame: DataFrame com features de floração
        """
        df = df.copy()
        
        # Features de temperatura para floração
        # Temperatura ideal para floração: 15-25°C
        df['temp_optimal_bloom'] = ((df['temperature_2m'] >= 15) & (df['temperature_2m'] <= 25)).astype(int)
        df['temp_bloom_score'] = np.where(
            df['temperature_2m'] < 15, 
            (df['temperature_2m'] - 5) / 10,  # Normaliza entre 0-1 para temperaturas baixas
            np.where(
                df['temperature_2m'] > 25,
                (35 - df['temperature_2m']) / 10,  # Normaliza entre 0-1 para temperaturas altas
                1  # Temperatura ideal
            )
        )
        
        # Features de umidade para floração
        # Umidade ideal: 40-70%
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
        
        # Features de precipitação para floração
        # Precipitação moderada é boa para floração
        df['precip_optimal'] = ((df['precipitation'] > 0) & (df['precipitation'] < 5)).astype(int)
        df['precip_bloom_score'] = np.where(
            df['precipitation'] == 0,
            0.5,  # Sem chuva não é ideal
            np.where(
                df['precipitation'] <= 5,
                1,  # Chuva moderada é ideal
                np.maximum(0, 1 - (df['precipitation'] - 5) / 10)  # Chuva excessiva reduz score
            )
        )
        
        # Features de solo completamente removidas
        # Não cria nenhuma feature de solo
        
        # Features de luz solar (baseado na cobertura de nuvens)
        df['sunlight_score'] = 1 - (df['cloud_cover'] / 100)
        df['sunlight_optimal'] = (df['sunlight_score'] > 0.5).astype(int)
        
        # Score combinado de floração (verifica se todas as colunas existem)
        score_components = []
        weights = []
        
        if 'temp_bloom_score' in df.columns:
            score_components.append(df['temp_bloom_score'])
            weights.append(0.3)
        
        if 'humidity_bloom_score' in df.columns:
            score_components.append(df['humidity_bloom_score'])
            weights.append(0.2)
        
        if 'precip_bloom_score' in df.columns:
            score_components.append(df['precip_bloom_score'])
            weights.append(0.2)
        
        # Soil features removed
        
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
            # Se não houver componentes, usa valor neutro
            df['bloom_score'] = 0.5
        
        # Features de tendência de floração (sem data leakage)
        # Usa apenas dados passados para evitar vazamento
        df['bloom_score_24h_avg'] = df['bloom_score'].rolling(window=24, min_periods=1).mean().shift(1)
        df['bloom_score_7d_avg'] = df['bloom_score'].rolling(window=168, min_periods=1).mean().shift(1)
        df['bloom_score_trend'] = df['bloom_score'].diff().shift(1)
        
        return df
    
    def create_seasonal_features(self, df):
        """
        Cria features sazonais para previsão de floração
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos
            
        Returns:
            pd.DataFrame: DataFrame com features sazonais
        """
        df = df.copy()
        
        # Features sazonais
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Features de primavera (período de floração)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_spring_peak'] = df['month'].isin([4, 5]).astype(int)
        
        # Features de acúmulo de calor (Growing Degree Days)
        base_temp = 10  # Temperatura base para crescimento
        df['gdd'] = np.maximum(0, df['temperature_2m'] - base_temp)
        df['gdd_7d'] = df['gdd'].rolling(window=168, min_periods=1).sum()
        df['gdd_30d'] = df['gdd'].rolling(window=720, min_periods=1).sum()
        
        # Features de resfriamento (Chilling Hours)
        df['chilling_hours'] = (df['temperature_2m'] <= 7).astype(int)
        df['chilling_hours_7d'] = df['chilling_hours'].rolling(window=168, min_periods=1).sum()
        df['chilling_hours_30d'] = df['chilling_hours'].rolling(window=720, min_periods=1).sum()
        
        return df
    
    def create_lag_features(self, df, target_col='bloom_score', lags=[1, 2, 3, 6, 12, 24, 48, 72]):
        """
        Cria features de lag para capturar dependências temporais
        
        Args:
            df (pd.DataFrame): DataFrame com dados
            target_col (str): Coluna alvo para criar lags
            lags (list): Lista de lags em horas
            
        Returns:
            pd.DataFrame: DataFrame com features de lag
        """
        df = df.copy()
        
        if target_col not in df.columns:
            print(f"⚠️ Coluna alvo {target_col} não encontrada, pulando features de lag")
            return df
        
        for lag in lags:
            try:
                df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
            except Exception as e:
                print(f"⚠️ Erro ao criar lag {lag}h para {target_col}: {e}")
                continue
        
        return df
    
    def create_rolling_features(self, df, columns, windows=[6, 12, 24, 48, 72]):
        """
        Cria features de janela deslizante
        
        Args:
            df (pd.DataFrame): DataFrame com dados
            columns (list): Colunas para criar features de janela
            windows (list): Janelas em horas
            
        Returns:
            pd.DataFrame: DataFrame com features de janela
        """
        df = df.copy()
        
        print(f"Processando {len(columns)} colunas para features de janela deslizante...")
        
        for col in columns:
            if col not in df.columns:
                print(f"⚠️ Coluna {col} não encontrada, pulando...")
                continue
            
            print(f"  Processando coluna: {col}")
            
            for window in windows:
                try:
                    df[f'{col}_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()
                    df[f'{col}_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
                except Exception as e:
                    print(f"⚠️ Erro ao criar features de janela para {col} (window={window}h): {e}")
                    continue
        
        print(f"✅ Features de janela deslizante criadas com sucesso")
        return df
    
    def process_weather_data(self, df):
        """
        Processa dados meteorológicos completos criando todas as features
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos brutos
            
        Returns:
            pd.DataFrame: DataFrame com todas as features processadas
        """
        print("Creating basic weather features...")
        df = self.create_basic_weather_features(df)
        
        print("Creating basic Blooming features...")
        df = self.create_basic_blooming_features(df)
        
        print("Creating seasonal features...")
        df = self.create_seasonal_features(df)
        
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        print("Creating rolling window features...")
        # Usa apenas colunas básicas e confiáveis (SEM bloom_score para evitar data leakage)
        basic_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation']
        available_cols = [col for col in basic_cols if col in df.columns]
        
        print(f"Basic columns for rolling window: {available_cols}")
        
        if available_cols:
            df = self.create_rolling_features(df, available_cols)
        else:
            print("⚠️ No basic columns found for rolling window features")
        
        # CRITICAL: Remove bloom_score to avoid data leakage
        if 'bloom_score' in df.columns:
            df = df.drop('bloom_score', axis=1)
            print("⚠️ Removed bloom_score to avoid data leakage")
        
        # Remove colunas com muitos valores nulos
        df = df.dropna(thresh=len(df) * 0.5, axis=1)
        
        # Preenche valores nulos restantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        self.feature_columns = [col for col in df.columns if col not in ['date']]
        
        return df
    
    def get_feature_importance(self, df):
        """
        Calcula importância das features baseada em correlação com bloom_score
        
        Args:
            df (pd.DataFrame): DataFrame com features
            
        Returns:
            pd.Series: Série com importância das features
        """
        if 'bloom_score' not in df.columns:
            return None
        
        correlations = df.corr()['bloom_score'].abs().sort_values(ascending=False)
        return correlations.drop('bloom_score', errors='ignore')

def main():
    """Função principal para testar o feature engineering"""
    # Exemplo de uso
    print("Sistema de Feature Engineering para Previsão de Floração")
    print("=" * 60)
    
    # Carrega dados de exemplo (se existirem)
    try:
        df = pd.read_csv('data/historical_weather.csv')
        print(f"Dados carregados: {df.shape}")
        
        engineer = FeatureEngineer()
        processed_df = engineer.process_weather_data(df)
        
        print(f"Dados processados: {processed_df.shape}")
        print(f"Features criadas: {len(engineer.feature_columns)}")
        
        # Salva dados processados
        processed_df.to_csv('data/processed_weather_features.csv', index=False)
        print("Dados processados salvos em: data/processed_weather_features.csv")
        
        # Mostra importância das features
        importance = engineer.get_feature_importance(processed_df)
        if importance is not None:
            print("\nTop 10 features mais importantes:")
            print(importance.head(10))
            
    except FileNotFoundError:
        print("Arquivo de dados não encontrado. Execute primeiro o weather_data_collector.py")

if __name__ == "__main__":
    main()
