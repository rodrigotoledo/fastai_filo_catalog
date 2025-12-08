#!/usr/bin/env python3
"""
Script para limpar tabelas de clients e fazer upload de dados de exemplo
para testar busca por similaridade vetorial.
"""

import os
import sys
from pathlib import Path
import requests
import time

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from app.db.database import SessionLocal
from app.models.client import Client, ClientAddress

BASE_URL = "http://localhost:8000"

def clear_tables():
    """Limpa as tabelas de clients e client_addresses"""
    print("🧹 Limpando tabelas de clients e client_addresses...")

    db = SessionLocal()
    try:
        # Contar registros antes
        clients_count = db.query(Client).count()
        addresses_count = db.query(ClientAddress).count()

        print(f"📊 Antes: {clients_count} clients, {addresses_count} addresses")

        # Limpar tabelas (ordem importa por FK)
        db.query(ClientAddress).delete()
        db.query(Client).delete()
        db.commit()

        print("✅ Tabelas limpas com sucesso!")

    except Exception as e:
        print(f"❌ Erro ao limpar tabelas: {e}")
        db.rollback()
    finally:
        db.close()

def upload_example_files():
    """Faz upload dos arquivos de exemplo"""
    print("\n📤 Fazendo upload dos arquivos de exemplo...")

    example_files = [
        "test_client.txt",
        "test_client2.txt"
    ]

    for filename in example_files:
        filepath = root_dir / filename
        if not filepath.exists():
            print(f"⚠️  Arquivo {filename} não encontrado, pulando...")
            continue

        print(f"📄 Fazendo upload de {filename}...")

        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'text/plain')}
                data = {
                    'create_client': 'true',
                    'extraction_prompt': 'Extraia nome, email, telefone, CPF e endereço do documento'
                }

                response = requests.post(
                    f"{BASE_URL}/api/v1/clients/upload-document",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Upload de {filename}: {result.get('processing_status', 'unknown')}")
                    print(f"   📄 Texto extraído: {result.get('extracted_data', {}).get('raw_text', 'N/A')[:100]}...")
                    if result.get('client_created'):
                        print(f"   👤 Cliente criado: {result['created_client']['name']}")
                    else:
                        print(f"   ⚠️  Cliente não criado: {result.get('validation_errors', [])}")
                        print(f"   📋 Dados extraídos: {result.get('extracted_data', {})}")
                else:
                    print(f"❌ Erro no upload de {filename}: {response.status_code}")
                    print(f"   Resposta: {response.text[:200]}...")

        except Exception as e:
            print(f"❌ Erro ao fazer upload de {filename}: {e}")

        # Pequena pausa entre uploads
        time.sleep(1)

def test_similarity_search():
    """Testa a busca por similaridade"""
    print("\n🔍 Testando busca por similaridade...")

    test_queries = [
        "João Silva",
        "cliente de São Paulo",
        "pessoa com email gmail",
        "Maria Santos",
        "cliente do Rio de Janeiro"
    ]

    for query in test_queries:
        print(f"\n🔎 Buscando: '{query}'")

        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/clients/search-similar",
                params={'q': query, 'limit': 5},
                timeout=10
            )

            if response.status_code == 200:
                results = response.json()
                if results:
                    print(f"✅ Encontrados {len(results)} resultados:")
                    for i, result in enumerate(results[:3], 1):  # Mostrar apenas top 3
                        client_data = result.get('client_data', {})
                        similarity = result.get('similarity', 0)
                        print(f"   {i}. {client_data.get('name', 'N/A')} (similaridade: {similarity:.3f})")
                else:
                    print("❌ Nenhum resultado encontrado")
            else:
                print(f"❌ Erro na busca: {response.status_code}")
                print(f"   Resposta: {response.text[:200]}...")

        except Exception as e:
            print(f"❌ Erro ao testar busca: {e}")

        time.sleep(0.5)

def main():
    """Função principal"""
    print("🚀 Iniciando teste de busca por similaridade vetorial")
    print("=" * 60)

    # Aguardar serviços ficarem prontos
    print("⏳ Aguardando serviços ficarem prontos...")
    time.sleep(5)

    # Limpar tabelas
    clear_tables()

    # Aguardar um pouco
    time.sleep(2)

    # Fazer upload dos exemplos
    upload_example_files()

    # Aguardar processamento
    print("\n⏳ Aguardando processamento dos embeddings...")
    time.sleep(3)

    # Testar busca
    test_similarity_search()

    print("\n🎉 Teste concluído!")

if __name__ == "__main__":
    main()
