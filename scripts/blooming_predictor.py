"""
Modelo de Machine Learning para previsão de floração (blooming) das flores
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BloomingPredictor:
    def __init__(self):
        """Inicializa o preditor de floração"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = None
        
    def prepare_training_data(self, df, target_col='bloom_score'):
        """
        Prepara dados para treinamento
        
        Args:
            df (pd.DataFrame): DataFrame com features
            target_col (str): Coluna alvo
            
        Returns:
            tuple: X, y para treinamento
        """
        # Remove colunas não numéricas e a coluna alvo
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col and col != 'date']
        
        # Remove colunas com muitos valores nulos
        feature_cols = [col for col in feature_cols if df[col].isnull().sum() < len(df) * 0.5]
        
        print(f"Features selecionadas para treinamento: {len(feature_cols)}")
        print(f"Colunas removidas por muitos nulos: {len(numeric_cols) - len(feature_cols)}")
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        self.feature_columns = feature_cols
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Treina múltiplos modelos e seleciona o melhor
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proporção para teste
            random_state (int): Seed para reprodutibilidade
        """
        # Divide dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Normaliza features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Define modelos para testar
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                verbosity=0
            ),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        # Treina e avalia cada modelo
        model_scores = {}
        
        for name, model in models.items():
            print(f"Treinando {name}...")
            
            if name in ['Ridge', 'Lasso']:
                # Modelos lineares usam dados normalizados
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Modelos de árvore usam dados originais
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calcula métricas
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Validação cruzada
            if name in ['Ridge', 'Lasso']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            model_scores[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.models[name] = model
            
            print(f"{name} - R²: {r2:.4f}, MAE: {mae:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Seleciona o melhor modelo
        best_score = -np.inf
        for name, scores in model_scores.items():
            # Usa R² como critério principal
            if scores['r2'] > best_score:
                best_score = scores['r2']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\nMelhor modelo: {self.best_model_name} (R²: {best_score:.4f})")
        
        return model_scores
    
    def predict_blooming_probability(self, X):
        """
        Prediz probabilidade de floração
        
        Args:
            X (pd.DataFrame): Features para predição
            
        Returns:
            np.array: Probabilidades de floração
        """
        if self.best_model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Prepara features - verifica quais colunas existem
        available_features = [col for col in self.feature_columns if col in X.columns]
        missing_features = [col for col in self.feature_columns if col not in X.columns]
        
        if missing_features:
            print(f"⚠️ Features ausentes: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            print(f"   Usando {len(available_features)} de {len(self.feature_columns)} features")
        
        if not available_features:
            raise ValueError("Nenhuma feature disponível para predição")
        
        # Cria DataFrame com todas as features na ordem correta
        X_processed = pd.DataFrame(index=X.index)
        
        # Adiciona features disponíveis
        for col in self.feature_columns:
            if col in X.columns:
                X_processed[col] = X[col].fillna(X[col].mean())
            else:
                X_processed[col] = 0  # Valor padrão para features ausentes
        
        # Garante que as colunas estão na ordem correta
        X_processed = X_processed[self.feature_columns]
        
        # Faz predição
        if self.best_model_name in ['Ridge', 'Lasso']:
            X_scaled = self.scalers['standard'].transform(X_processed)
            predictions = self.best_model.predict(X_scaled)
        else:
            predictions = self.best_model.predict(X_processed)
        
        # Converte para probabilidade (0-1)
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def predict_peak_blooming(self, df, days_ahead=14):
        """
        Prediz quando será o pico de floração nos próximos dias
        
        Args:
            df (pd.DataFrame): DataFrame com dados meteorológicos
            days_ahead (int): Número de dias para prever
            
        Returns:
            dict: Informações sobre o pico de floração
        """
        # Prediz probabilidades para os próximos dias
        probabilities = self.predict_blooming_probability(df)
        
        # Encontra o pico
        peak_idx = np.argmax(probabilities)
        peak_probability = probabilities[peak_idx]
        peak_date = df.iloc[peak_idx]['date'] if 'date' in df.columns else None
        
        # Calcula estatísticas
        avg_probability = np.mean(probabilities)
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        
        # Identifica períodos de alta probabilidade (>0.7)
        high_prob_periods = np.where(probabilities > 0.7)[0]
        
        result = {
            'peak_date': peak_date,
            'peak_probability': peak_probability,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'min_probability': min_probability,
            'high_probability_days': len(high_prob_periods),
            'probabilities': probabilities.tolist(),
            'dates': df['date'].tolist() if 'date' in df.columns else None
        }
        
        return result
    
    def save_model(self, filepath='models/blooming_model.pkl'):
        """Salva o modelo treinado"""
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'scalers': self.scalers
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath='models/blooming_model.pkl'):
        """Carrega modelo treinado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_columns = model_data['feature_columns']
        self.scalers = model_data['scalers']
        
        print(f"Modelo carregado: {self.best_model_name}")
    
    def get_feature_importance(self):
        """Retorna importância das features"""
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None

def main():
    """Função principal para treinar o modelo"""
    print("Sistema de Previsão de Floração")
    print("=" * 40)
    
    try:
        # Carrega dados processados
        df = pd.read_csv('data/processed_weather_features.csv')
        print(f"Dados carregados: {df.shape}")
        
        # Inicializa preditor
        predictor = BloomingPredictor()
        
        # Prepara dados
        X, y = predictor.prepare_training_data(df)
        print(f"Features: {X.shape[1]}, Amostras: {X.shape[0]}")
        
        # Treina modelos
        scores = predictor.train_models(X, y)
        
        # Salva modelo
        predictor.save_model()
        
        # Mostra importância das features
        importance = predictor.get_feature_importance()
        if importance is not None:
            print("\nTop 10 features mais importantes:")
            print(importance.head(10))
        
        print("\nTreinamento concluído!")
        
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Execute primeiro o feature_engineering.py para processar os dados")

if __name__ == "__main__":
    main()
