# Sistema de Previsão de Floração com IA

Sistema inteligente que prevê a probabilidade de floração (blooming) das flores nos próximos 14 dias e identifica quando será o pico dessa floração usando machine learning e dados meteorológicos em tempo real.

## 🌸 Funcionalidades

- **Previsão de Floração**: Calcula a probabilidade de floração para os próximos 14 dias
- **Identificação de Pico**: Determina quando será o pico de floração
- **Análise Meteorológica**: Integra dados de temperatura, umidade, precipitação, solo e mais
- **Interface Web**: Interface amigável para visualização e interação
- **API REST**: API completa para integração com outros sistemas
- **Machine Learning**: Múltiplos algoritmos (Random Forest, XGBoost, LightGBM, etc.)

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <repository-url>
cd ML
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o treinamento do modelo
```bash
python train_model.py
```

### 4. Inicie a aplicação
```bash
python app.py
```

### 5. Acesse a interface web
Abra seu navegador em: `http://localhost:5000`

## 📁 Estrutura do Projeto

```
ML/
├── app.py                          # API Flask principal
├── train_model.py                  # Script de treinamento
├── requirements.txt                # Dependências Python
├── README.md                      # Este arquivo
├── data/                          # Dados meteorológicos
│   ├── historical_weather.csv
│   └── processed_weather_features.csv
├── models/                        # Modelos treinados
│   └── blooming_model.pkl
├── outputs/                       # Resultados e visualizações
│   └── forecast_result.csv
├── scripts/                       # Módulos do sistema
│   ├── weather_data_collector.py  # Coleta dados meteorológicos
│   ├── feature_engineering.py     # Processamento de features
│   ├── blooming_predictor.py      # Modelo de ML
│   └── blooming_forecast.py       # Sistema de previsão
├── templates/                     # Interface web
│   └── index.html
└── static/                        # Arquivos estáticos
```

## 🔧 Uso da API

### Endpoint: Previsão de Floração
```bash
POST /api/forecast
Content-Type: application/json

{
    "latitude": 38.6275,
    "longitude": -92.5666,
    "days_ahead": 14
}
```

**Resposta:**
```json
{
    "forecast_data": [...],
    "probabilities": [0.65, 0.72, 0.81, ...],
    "peak_analysis": {
        "peak_date": "2025-10-15",
        "peak_probability": 0.85,
        "trend": "crescendo",
        "confidence": "alta"
    },
    "insights": [
        "Temperatura ideal para floração",
        "Umidade ideal para floração"
    ],
    "summary": {
        "recommendation": "Excelente período para floração!",
        "key_findings": [...],
        "action_items": [...]
    }
}
```

### Endpoint: Dados Meteorológicos
```bash
POST /api/weather
Content-Type: application/json

{
    "latitude": 38.6275,
    "longitude": -92.5666,
    "start_date": "2025-09-01",
    "end_date": "2025-09-15"
}
```

### Endpoint: Health Check
```bash
GET /api/health
```

### Endpoint: Informações do Modelo
```bash
GET /api/model/info
```

## 🧠 Algoritmos de Machine Learning

O sistema utiliza múltiplos algoritmos e seleciona automaticamente o melhor:

- **Random Forest**: Robusto para dados meteorológicos
- **XGBoost**: Excelente performance em features complexas
- **LightGBM**: Rápido e eficiente
- **Gradient Boosting**: Boa generalização
- **Ridge/Lasso**: Modelos lineares para baseline

## 📊 Features Utilizadas

### Dados Meteorológicos
- Temperatura (2m, solo em múltiplas profundidades)
- Umidade relativa e ponto de orvalho
- Precipitação e chuva
- Pressão atmosférica
- Cobertura de nuvens
- Vento (velocidade e direção)
- Evapotranspiração
- Umidade do solo

### Features Derivadas
- **Temporais**: Hora, dia do ano, mês, estação
- **Tendências**: Médias móveis, diferenças, tendências
- **Sazonais**: Growing Degree Days, Chilling Hours
- **Específicas de Floração**: Scores baseados em condições ideais
- **Lags**: Dependências temporais
- **Janelas Deslizantes**: Médias, desvios, máximos, mínimos

## 🎯 Condições Ideais para Floração

O sistema considera as seguintes condições ideais:

- **Temperatura**: 15-25°C
- **Umidade**: 40-70%
- **Precipitação**: Moderada (0-5mm/dia)
- **Umidade do Solo**: 20-40%
- **Luz Solar**: >60% (baseado na cobertura de nuvens)

## 📈 Métricas de Avaliação

- **R² Score**: Correlação entre predições e valores reais
- **MAE**: Erro absoluto médio
- **MSE**: Erro quadrático médio
- **Validação Cruzada**: 5-fold para robustez

## 🔄 Pipeline Completo

1. **Coleta de Dados**: API Open-Meteo para dados meteorológicos
2. **Feature Engineering**: Criação de features derivadas
3. **Treinamento**: Múltiplos algoritmos com validação cruzada
4. **Seleção**: Melhor modelo baseado em R²
5. **Previsão**: Probabilidades e análise de pico
6. **Visualização**: Gráficos e insights
7. **API**: Servir previsões via REST

## 🌐 Interface Web

A interface web oferece:

- **Formulário Intuitivo**: Entrada de coordenadas e parâmetros
- **Visualizações Interativas**: Gráficos de probabilidades
- **Análise de Pico**: Data e probabilidade máxima
- **Insights Automáticos**: Recomendações baseadas nos dados
- **Design Responsivo**: Funciona em desktop e mobile

## 🛠️ Desenvolvimento

### Estrutura Modular
- `weather_data_collector.py`: Coleta dados meteorológicos
- `feature_engineering.py`: Processamento e criação de features
- `blooming_predictor.py`: Modelos de machine learning
- `blooming_forecast.py`: Sistema de previsão completo
- `app.py`: API Flask e interface web

### Extensibilidade
- Fácil adição de novos algoritmos
- Configurável para diferentes tipos de plantas
- Suporte a múltiplas localizações
- Integração com outras APIs meteorológicas

## 📝 Exemplos de Uso

### Python
```python
import requests

# Fazer previsão
response = requests.post('http://localhost:5000/api/forecast', json={
    'latitude': 38.6275,
    'longitude': -92.5666,
    'days_ahead': 14
})

result = response.json()
print(f"Pico de floração: {result['peak_analysis']['peak_date']}")
print(f"Probabilidade: {result['peak_analysis']['peak_probability']:.1%}")
```

### JavaScript
```javascript
// Fazer previsão
const response = await fetch('/api/forecast', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        latitude: 38.6275,
        longitude: -92.5666,
        days_ahead: 14
    })
});

const result = await response.json();
console.log('Pico:', result.peak_analysis.peak_date);
```

## 🚨 Troubleshooting

### Erro: "Modelo não encontrado"
```bash
python train_model.py
```

### Erro: "Dados meteorológicos não disponíveis"
- Verifique sua conexão com a internet
- Tente coordenadas diferentes
- Verifique se as datas estão corretas

### Erro: "Features insuficientes"
- Execute o feature engineering primeiro
- Verifique se os dados meteorológicos foram coletados

## 📄 Licença

Este projeto foi desenvolvido para o NASA Space App Challenge 2025.

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📞 Suporte

Para dúvidas ou problemas:
- Abra uma issue no repositório
- Consulte a documentação da API
- Verifique os logs de erro

---

**Desenvolvido com ❤️ para o NASA Space App Challenge 2025**
