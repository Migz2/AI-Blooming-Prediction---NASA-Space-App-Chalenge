"""
Script de inicialização rápida do sistema de previsão de floração
"""

import os
import sys
import subprocess
import time
import requests
from datetime import datetime

def print_banner():
    """Imprime banner do sistema"""
    print("=" * 80)
    print("🌸 SISTEMA DE PREVISÃO DE FLORAÇÃO COM IA")
    print("   NASA Space App Challenge 2025")
    print("=" * 80)
    print()

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    try:
        import pandas
        import numpy
        import sklearn
        import lightgbm
        import xgboost
        import flask
        import requests
        print("✅ Todas as dependências estão instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("Execute: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Instala dependências se necessário"""
    print("📦 Instalando dependências...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def train_model():
    """Treina o modelo de machine learning"""
    print("\n🧠 Treinando modelo de machine learning...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Modelo treinado com sucesso")
            return True
        else:
            print(f"❌ Erro no treinamento: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout no treinamento (5 minutos)")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def start_api():
    """Inicia a API Flask"""
    print("\n🚀 Iniciando API Flask...")
    print("-" * 50)
    
    try:
        # Inicia a API em background
        process = subprocess.Popen([sys.executable, "app.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Aguarda a API inicializar
        print("⏳ Aguardando API inicializar...")
        time.sleep(10)
        
        # Testa se a API está funcionando
        try:
            response = requests.get("http://localhost:5000/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ API iniciada com sucesso")
                print("🌐 Interface web disponível em: http://localhost:5000")
                return process
            else:
                print("❌ API não está respondendo corretamente")
                process.terminate()
                return None
        except requests.exceptions.ConnectionError:
            print("❌ API não está acessível")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Erro ao iniciar API: {e}")
        return None

def test_system():
    """Testa o sistema completo"""
    print("\n🧪 Testando sistema...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Todos os testes passaram")
            return True
        else:
            print(f"⚠️ Alguns testes falharam: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout nos testes (2 minutos)")
        return False
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        return False

def show_usage_instructions():
    """Mostra instruções de uso"""
    print("\n📖 INSTRUÇÕES DE USO")
    print("=" * 50)
    print("1. Interface Web:")
    print("   - Acesse: http://localhost:5000")
    print("   - Digite as coordenadas da localização")
    print("   - Clique em 'Gerar Previsão'")
    print()
    print("2. API REST:")
    print("   - POST /api/forecast - Previsão de floração")
    print("   - POST /api/weather - Dados meteorológicos")
    print("   - GET /api/health - Status da API")
    print()
    print("3. Exemplos de uso:")
    print("   - python example_usage.py")
    print("   - python test_system.py")
    print()
    print("4. Coordenadas de exemplo:")
    print("   - Jefferson City, MO: 38.6275, -92.5666")
    print("   - Springfield, MO: 37.2089, -93.2923")
    print("   - Kansas City, MO: 39.0997, -94.5786")

def main():
    """Função principal"""
    print_banner()
    
    # Verifica dependências
    if not check_dependencies():
        print("\n🔧 Tentando instalar dependências automaticamente...")
        if not install_dependencies():
            print("\n❌ Não foi possível instalar dependências automaticamente")
            print("Execute manualmente: pip install -r requirements.txt")
            return False
    
    # Pergunta se deve treinar o modelo
    print("\n🤔 Deseja treinar o modelo agora? (s/n)")
    train_choice = input("Resposta: ").lower().strip()
    
    if train_choice in ['s', 'sim', 'y', 'yes']:
        if not train_model():
            print("\n❌ Falha no treinamento do modelo")
            return False
    else:
        print("⚠️ Pulando treinamento do modelo")
        print("   Certifique-se de que o modelo já foi treinado")
    
    # Pergunta se deve iniciar a API
    print("\n🤔 Deseja iniciar a API agora? (s/n)")
    api_choice = input("Resposta: ").lower().strip()
    
    if api_choice in ['s', 'sim', 'y', 'yes']:
        api_process = start_api()
        if api_process is None:
            print("\n❌ Falha ao iniciar a API")
            return False
        
        # Mostra instruções
        show_usage_instructions()
        
        # Pergunta se deve testar o sistema
        print("\n🤔 Deseja testar o sistema agora? (s/n)")
        test_choice = input("Resposta: ").lower().strip()
        
        if test_choice in ['s', 'sim', 'y', 'yes']:
            test_system()
        
        # Mantém a API rodando
        print("\n🔄 API rodando... Pressione Ctrl+C para parar")
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Parando API...")
            api_process.terminate()
            print("✅ API parada")
    else:
        print("\n📝 Para iniciar a API manualmente:")
        print("   python app.py")
        print("\n📝 Para testar o sistema:")
        print("   python test_system.py")
        print("\n📝 Para ver exemplos de uso:")
        print("   python example_usage.py")
    
    print("\n🎉 Sistema configurado com sucesso!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Falha na configuração do sistema")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Configuração interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)
