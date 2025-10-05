"""
Flowering prediction system and trend analysis
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
        """Initialize the flowering prediction system"""
        self.predictor = BloomingPredictor()
        self.feature_engineer = FeatureEngineer()
        self.weather_collector = WeatherDataCollector()
        
    def generate_forecast(self, latitude, longitude, days_ahead=14):
        """
        Generate complete flowering forecast for the next days
        
        Args:
            latitude (float): Latitude of the location
            longitude (float): Longitude of the location
            days_ahead (int): Number of days to predict
            
        Returns:
            dict: Complete flowering forecast
        """
        print(f"Generating flowering forecast for {days_ahead} days...")
        
        # Collect forecast weather data
        forecast_data = self.weather_collector.collect_forecast_data(
            latitude, longitude, days=days_ahead
        )
        
        if forecast_data is None:
            return {"error": "Unable to collect weather data"}
        
        # Process features
        processed_data = self.feature_engineer.process_weather_data(forecast_data)
        
        # Load trained model
        try:
            self.predictor.load_model()
        except FileNotFoundError:
            return {"error": "Model not found. Execute the training first."}
        
        # Predict flowering probabilities
        probabilities = self.predictor.predict_blooming_probability(processed_data)
        
        # Analyze the peak of flowering
        peak_analysis = self.analyze_peak_blooming(processed_data, probabilities)
        
        # Generate insights
        insights = self.generate_insights(processed_data, probabilities)
        
        # Create visualizations
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
        Analyze the peak of flowering
        
        Args:
            df (pd.DataFrame): Processed data
            probabilities (np.array): Flowering probabilities
            
        Returns:
            dict: Analysis of the peak of flowering
        """
        # Find the peak
        peak_idx = np.argmax(probabilities)
        peak_date = df.iloc[peak_idx]['date']
        # Convert to ISO string if datetime
        if hasattr(peak_date, 'isoformat'):
            peak_date = peak_date.isoformat()
        elif hasattr(peak_date, 'strftime'):
            peak_date = peak_date.strftime('%Y-%m-%d')
        peak_probability = float(probabilities[peak_idx])
        
        # Find periods of high probability
        high_prob_threshold = 0.7
        high_prob_indices = np.where(probabilities >= high_prob_threshold)[0]
        
        # Calculate statistics
        avg_probability = np.mean(probabilities)
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        
        # Identify trends
        trend = self.calculate_trend(probabilities)
        
        # Find the best time windows for flowering
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
        """Calculate the trend of the probabilities"""
        if len(probabilities) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(probabilities))
        slope = np.polyfit(x, probabilities, 1)[0]
        
        if slope > 0.01:
            return "growing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def find_best_blooming_windows(self, probabilities, window_size=3):
        """Find the best time windows for flowering"""
        if len(probabilities) < window_size:
            return []
        
        window_scores = []
        for i in range(len(probabilities) - window_size + 1):
            window_avg = np.mean(probabilities[i:i+window_size])
            window_scores.append((i, window_avg))
        
        # Sort by score and return top 3
        window_scores.sort(key=lambda x: x[1], reverse=True)
        return window_scores[:3]
    
    def calculate_confidence(self, probabilities):
        """Calculate the level of confidence of the forecast"""
        # Based on the consistency of the probabilities
        std_prob = np.std(probabilities)
        avg_prob = np.mean(probabilities)
        
        if std_prob < 0.1 and avg_prob > 0.5:
            return "high"
        elif std_prob < 0.2:
            return "medium"
        else:
            return "low"
    
    def generate_insights(self, df, probabilities):
        """Generate insights about the conditions of flowering"""
        insights = []
        
        # Temperature analysis
        avg_temp = df['temperature_2m'].mean()
        if 15 <= avg_temp <= 25:
            insights.append("Ideal temperature for flowering")
        elif avg_temp < 15:
            insights.append("Low temperature can delay flowering")
        else:
            insights.append("High temperature can accelerate flowering")
        
        # Humidity analysis
        avg_humidity = df['relative_humidity_2m'].mean()
        if 40 <= avg_humidity <= 70:
            insights.append("Ideal humidity for flowering")
        elif avg_humidity < 40:
            insights.append("Low humidity can damage flowering")
        else:
            insights.append("High humidity can promote diseases")
        
        # Precipitation analysis
        total_precip = df['precipitation'].sum()
        if 0 < total_precip < 50:
            insights.append("Moderate precipitation promotes flowering")
        elif total_precip == 0:
            insights.append("Low precipitation can damage flowering")
        else:
            insights.append("High precipitation can damage flowers")
        
        # Soil analysis removed - using only weather data
        
        # Sunlight analysis
        avg_sunlight = (100 - df['cloud_cover'].mean()) / 100
        if avg_sunlight > 0.6:
            insights.append("Good sunlight availability")
        else:
            insights.append("Low sunlight can delay flowering")
        
        return insights
    
    def generate_summary(self, peak_analysis, insights):
        """Generate summary of the forecast"""
        summary = {
            "recommendation": self.get_recommendation(peak_analysis),
            "key_findings": [
                f"Peak flowering predicted for: {peak_analysis['peak_date']}",
                f"Maximum probability: {peak_analysis['peak_probability']:.1%}",
                f"Trend: {peak_analysis['trend']}",
                f"Confidence: {peak_analysis['confidence']}"
            ],
            "action_items": self.get_action_items(peak_analysis, insights)
        }
        return summary
    
    def get_recommendation(self, peak_analysis):
        """Generate recommendation based on the analysis"""
        if peak_analysis['peak_probability'] > 0.8:
            return "Excellent period for flowering! Prepare for the peak."
        elif peak_analysis['peak_probability'] > 0.6:
            return "Good period for flowering. Monitor conditions."
        elif peak_analysis['peak_probability'] > 0.4:
            return "Moderate period for flowering. Some flowers may bloom."
        else:
            return "Unfavorable period for flowering. Wait for better conditions."
    
    def get_action_items(self, peak_analysis, insights):
        """Generate action items based on the analysis"""
        actions = []
        
        if peak_analysis['confidence'] == 'low':
            actions.append("Monitor weather conditions daily")
        
        if any("Low temperature" in insight for insight in insights):
            actions.append("Consider protecting plants from cold")
        
        if any("Low humidity" in insight for insight in insights):
            actions.append("Increase irrigation if necessary")
        
        if any("Low sunlight" in insight for insight in insights):
            actions.append("Remove obstacles that block sunlight")
        
        if peak_analysis['trend'] == 'growing':
            actions.append("Prepare for the increase in flowering")
        
        return actions
    
    def create_visualizations(self, df, probabilities, peak_analysis):
        """Create visualizations of the forecast"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flowering Forecast - Complete Analysis', fontsize=16, fontweight='bold')
        
        # Graph 1: Flowering probabilities over time
        # Graph 1: Flowering probabilities over time
        # Graph 1: Flowering probabilities over time
        axes[0, 0].plot(df['date'], probabilities, 'b-', linewidth=2, label='Flowering Probability')
        axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='High Threshold')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Threshold')
        axes[0, 0].scatter([peak_analysis['peak_date']], [peak_analysis['peak_probability']], 
                           color='red', s=100, zorder=5, label='Peak')
        axes[0, 0].set_title('Flowering Probabilities')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Graph 2: Temperature vs Probability
        # Graph 2: Temperature vs Probability
        # Graph 2: Temperature vs Probability
        axes[0, 1].scatter(df['temperature_2m'], probabilities, alpha=0.6, c=probabilities, cmap='RdYlBu')
        axes[0, 1].set_title('Temperature vs Flowering Probability')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Graph 3: Humidity vs Probability
        # Graph 3: Humidity vs Probability
        axes[1, 0].scatter(df['relative_humidity_2m'], probabilities, alpha=0.6, c=probabilities, cmap='RdYlBu')
        axes[1, 0].set_title('Humidity vs Flowering Probability')
        axes[1, 0].set_xlabel('Relative Humidity (%)')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Graph 4: Distribution of probabilities
        # Graph 4: Distribution of probabilities
        axes[1, 1].hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=np.mean(probabilities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(probabilities):.2f}')
        axes[1, 1].set_title('Distribution of Probabilities')
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/blooming_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved in: outputs/blooming_forecast.png")

def main():
    """Function principal to test the prediction system"""
    print("Flowering Prediction System")
    print("=" * 40)
    
    # Example coordinates
    latitude = 38.6275
    longitude = -92.5666
    
    forecast_system = BloomingForecast()
    
    # Generate forecast
    result = forecast_system.generate_forecast(latitude, longitude, days_ahead=14)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Forecast generated successfully!")
        print(f"Peak of flowering: {result['peak_analysis']['peak_date']}")
        print(f"Maximum probability: {result['peak_analysis']['peak_probability']:.1%}")
        print(f"Confidence: {result['peak_analysis']['confidence']}")
        
        print("\nInsights:")
        for insight in result['insights']:
            print(f"- {insight}")
        
        print("\nSummary:")
        for finding in result['summary']['key_findings']:
            print(f"- {finding}")

if __name__ == "__main__":
    main()
