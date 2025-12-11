# Sistema de Aprendizado Contínuo com Feedback Humano

## 🎯 **Visão Geral**

Este documento explica como implementar um sistema de **aprendizado ativo (Active Learning)** onde a IA aprende continuamente com as correções e feedback dos usuários, melhorando a precisão das buscas ao longo do tempo.

## 🧠 **Problema Atual**

Quando você pesquisa por "pássaro":
- ✅ Sistema retorna 2 fotos corretas de pássaros
- ❌ Sistema retorna 8 fotos incorretas (gatos, carros, etc.)
- 🤔 Sistema não "aprende" que essas fotos não são pássaros

## 🚀 **Solução: Active Learning + Fine-tuning**

### **1. Sistema de Feedback/Correção**

#### **API de Correção**
```python
# Novo endpoint para correções
@app.post("/photos/{photo_id}/correct")
def correct_photo_classification(
    photo_id: int,
    correct_label: str,  # "pássaro", "gato", "cachorro", etc.
    incorrect_search: str = None  # termo que deu resultado errado
):
    # Salva correção no banco
    # Re-treina modelo incrementalmente
    pass
```

#### **Interface de Correção**
```python
# Após busca, mostrar botão "❌ Esta foto não é relevante"
# Usuário clica e informa o que a foto realmente é
# Sistema aprende com a correção
```

### **2. Armazenamento de Correções**

#### **Tabela de Correções**
```sql
CREATE TABLE photo_corrections (
    id SERIAL PRIMARY KEY,
    photo_id INTEGER REFERENCES photos(id),
    search_term VARCHAR(255),           -- termo que deu errado ("pássaro")
    correct_label VARCHAR(255),         -- o que foto realmente é ("gato")
    incorrect_label VARCHAR(255),       -- o que sistema pensou que era
    confidence_before FLOAT,            -- confiança do modelo antes
    user_id INTEGER,                    -- quem fez a correção
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### **Tabela de Labels Aprendidos**
```sql
CREATE TABLE learned_labels (
    id SERIAL PRIMARY KEY,
    label VARCHAR(255) UNIQUE,          -- "pássaro", "gato", etc.
    embedding VECTOR(512),              -- embedding médio das fotos desta classe
    sample_count INTEGER DEFAULT 0,     -- quantas fotos desta classe
    last_updated TIMESTAMP DEFAULT NOW()
);
```

### **3. Re-treinamento Incremental**

#### **Fine-tuning do CLIP**
```python
def fine_tune_with_corrections():
    """
    Re-treina o modelo com as correções dos usuários
    """
    # 1. Busca correções recentes
    corrections = get_recent_corrections()

    # 2. Cria pares (imagem, texto correto)
    training_pairs = []
    for correction in corrections:
        image = load_image(correction.photo.file_path)
        text = correction.correct_label
        training_pairs.append((image, text))

    # 3. Fine-tuning incremental do CLIP
    # Usa LoRA ou adapters para não re-treinar tudo
    fine_tune_clip_model(training_pairs)

    # 4. Atualiza embeddings de todas as fotos
    reprocess_all_embeddings()
```

#### **Aprendizado por Similaridade**
```python
def learn_from_similarity_feedback(photo_id, similar_photo_ids, dissimilar_photo_ids):
    """
    Aprende que certas fotos são similares/dissimilares
    """
    # Ajusta embeddings baseado no feedback
    # Fotos marcadas como similares ficam mais próximas no espaço vetorial
    # Fotos marcadas como diferentes ficam mais distantes
    adjust_embeddings_with_feedback(photo_id, similar_photo_ids, dissimilar_photo_ids)
```

### **4. Algoritmos de Active Learning**

#### **Uncertainty Sampling**
```python
def find_uncertain_predictions():
    """
    Encontra fotos onde o modelo tem baixa confiança
    """
    # Busca fotos com baixa similaridade máxima
    uncertain_photos = db.query(Photo).filter(
        Photo.processed == True,
        Photo.max_similarity_score < 0.3  # baixa confiança
    ).all()

    # Pede feedback do usuário para essas fotos
    return uncertain_photos
```

#### **Query by Committee**
```python
def query_by_committee():
    """
    Usa múltiplas versões do modelo para encontrar discordâncias
    """
    # Treina 3 versões diferentes do modelo
    # Para cada foto, vê se há discordância entre os modelos
    # Fotos com discordância alta precisam de feedback humano
```

### **5. Pipeline de Aprendizado Contínuo**

#### **Fluxo Completo**
```
1. Usuário faz busca → Sistema retorna resultados
2. Usuário marca incorretos → Correções salvas no banco
3. Scheduler detecta correções → Aciona re-treinamento
4. Modelo é fine-tunado → Embeddings atualizados
5. Buscas futuras são mais precisas
```

#### **Scheduler de Re-treinamento**
```python
# Novo serviço no docker-compose
scheduler-retrain:
    command: python retrain_scheduler.py
    environment:
        RETRAIN_INTERVAL_HOURS: 24  # re-treina a cada 24h
        MIN_CORRECTIONS_FOR_RETRAIN: 10  # precisa de 10 correções
```

### **6. Métricas e Monitoramento**

#### **Dashboard de Precisão**
```python
@app.get("/analytics/precision")
def get_precision_metrics():
    """
    Mostra evolução da precisão ao longo do tempo
    """
    return {
        "overall_precision": calculate_overall_precision(),
        "precision_by_label": get_precision_by_label(),
        "corrections_over_time": get_corrections_timeline(),
        "model_versions": get_model_versions()
    }
```

#### **A/B Testing**
```python
def ab_test_models():
    """
    Testa novo modelo vs modelo antigo
    """
    # 10% das buscas usam novo modelo
    # Compara precisão entre versões
    # Promove modelo melhor automaticamente
```

### **7. Estratégias Avançadas**

#### **Personalização por Usuário**
```python
# Diferentes usuários têm preferências diferentes
# Sistema aprende o que cada usuário considera "correto"
user_profiles = {
    "user_123": {
        "preferred_labels": ["pássaro", "ave"],
        "disliked_labels": ["avião", "drone"]
    }
}
```

#### **Aprendizado por Exemplos Positivos/Negativos**
```python
# Usuário mostra exemplos do que quer
# Sistema aprende padrões visuais específicos
positive_examples = ["esta foto é perfeita para 'pássaro'"]
negative_examples = ["esta NÃO é um pássaro"]
```

#### **Ensemble Learning**
```python
# Combina múltiplos modelos
# CLIP + classificadores customizados
# Votação para melhor precisão
def ensemble_predict(embedding, text_embedding):
    clip_score = clip_similarity(embedding, text_embedding)
    custom_score = custom_classifier.predict(embedding)
    return (clip_score + custom_score) / 2
```

## 🎯 **Benefícios Esperados**

### **Curto Prazo (Semanas)**
- ✅ Precisão aumenta 20-30% após primeiras correções
- ✅ Sistema aprende termos específicos do usuário
- ✅ Redução de resultados irrelevantes

### **Médio Prazo (Meses)**
- ✅ Precisão >80% para termos frequentes
- ✅ Personalização por usuário
- ✅ Detecção automática de ambiguidades

### **Longo Prazo (Anos)**
- ✅ Sistema "entende" o contexto específico das suas fotos
- ✅ Buscas semânticas avançadas
- ✅ Sugestões proativas de organização

## 🚀 **Implementação Faseada**

### **Fase 1: Correção Básica** ⭐ (1-2 semanas)
- Interface para marcar fotos incorretas
- Armazenamento de correções
- Re-treinamento batch semanal

### **Fase 2: Active Learning** ⭐⭐ (1 mês)
- Sistema pede feedback para fotos incertas
- Fine-tuning incremental
- Métricas de precisão

### **Fase 3: Personalização** ⭐⭐⭐ (2-3 meses)
- Perfis por usuário
- Ensemble learning
- A/B testing automático

## 💡 **Por que isso funciona?**

1. **Feedback Humano é Ouro** - Correções humanas são dados de treinamento perfeitos
2. **Aprendizado Incremental** - Não precisa re-treinar tudo do zero
3. **Personalização** - Cada usuário tem contexto único
4. **Escalabilidade** - Sistema melhora quanto mais usado

## 🔧 **Próximos Passos**

Quer implementar a **Fase 1** primeiro? Podemos começar com:
1. Endpoint de correção `/photos/{id}/correct`
2. Interface simples para feedback
3. Re-treinamento semanal

Isso já melhoraria significativamente a precisão das suas buscas!</content>
<parameter name="filePath">/home/rtoledo/www/fullstack/photo-finder/backend/APRENDIZADO_CONTINUO.md
