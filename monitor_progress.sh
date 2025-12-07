#!/bin/bash
# Script para acompanhar o progresso do reprocessamento das fotos

echo "🔍 Acompanhando o reprocessamento das fotos..."
echo "=============================================="

while true; do
    # Faz a requisição e formata a saída
    response=$(curl -s http://localhost:8000/api/v1/photos/processing/stats)

    if [ $? -eq 0 ] && [ ! -z "$response" ]; then
        # Extrai as informações usando jq se disponível, senão usa grep
        if command -v jq &> /dev/null; then
            status=$(echo "$response" | jq -r '.status')
            total=$(echo "$response" | jq -r '.total_photos')
            processed=$(echo "$response" | jq -r '.processed_photos')
            percentage=$(echo "$response" | jq -r '.processing_percentage')
            remaining=$(echo "$response" | jq -r '.estimated_remaining_time')
        else
            # Fallback usando grep e sed
            status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            total=$(echo "$response" | grep -o '"total_photos":[0-9]*' | cut -d':' -f2)
            processed=$(echo "$response" | grep -o '"processed_photos":[0-9]*' | cut -d':' -f2)
            percentage=$(echo "$response" | grep -o '"processing_percentage":[0-9.]*' | cut -d':' -f2)
            remaining=$(echo "$response" | grep -o '"estimated_remaining_time":"[^"]*"' | cut -d'"' -f4)
        fi

        # Limpa a tela e mostra o status
        clear
        echo "🔄 Status do Reprocessamento: $status"
        echo "📊 Progresso: $processed / $total fotos ($percentage%)"
        echo "⏱️  Tempo restante estimado: $remaining"
        echo ""
        echo "✅ Últimas fotos processadas:"
        echo "$response" | grep -A 10 '"recent_processed_photos"' | head -20
        echo ""
        echo "🔄 Atualizando a cada 30 segundos... (Ctrl+C para sair)"
    else
        echo "❌ Erro ao conectar com a API. Verifique se o servidor está rodando."
    fi

    sleep 30
done
