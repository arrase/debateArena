# Debate Arena

CLI autónomo para hacer debatir a dos LLM sobre una tesis dada por el usuario. La reescritura actual usa **LangChain**, **LangGraph** y **Ollama** con una arquitectura modular, prompts externos y compactación preventiva de contexto.

## Flujo central

1. `debater_a` toma la postura **a favor**.
2. `debater_b` toma la postura **en contra**.
3. `referee` revisa cada ronda, detecta bucles, prohíbe líneas agotadas y decide cuándo cerrar.
4. `compactor` resume historial antes de saturar `num_ctx`.
5. `referee` emite un veredicto final con ganador o empate.

## Arquitectura

```text
src/debate_arena/
├── config/        # Carga y validación tipada de settings.yaml
├── domain/        # Modelos de dominio y resultados
├── graph/         # Orquestación LangGraph
├── llm/           # Factoría ChatOllama y contratos de modelos
├── prompts/       # Repositorio Jinja para prompts externos
└── services/      # Presupuesto de contexto, parsing y presentación
```

## Configuración

La configuración principal vive en `config/settings.yaml` y separa:

- `runtime`: conexión con Ollama
- `debate`: idioma, rondas y límites de respuesta
- `prompt_repository`: directorio de prompts y prompt de apertura
- `context_policy`: umbrales de compactación y buffer de contexto
- `models`: configuración por rol, incluyendo `context_window`

Todos los prompts viven en `config/prompts/`:

- `debater.j2`
- `opening_instruction.j2`
- `referee_review.j2`
- `referee_final.j2`
- `compactor.j2`

## Uso

```bash
python3 -m pip install -e .
debate-cli --config config/settings.yaml -p "La regulación de la IA debe ser global"
debate-cli -p "La educación universitaria debe ser gratuita" -f salida.txt
```

## Gestión de contexto

Cada llamada a Ollama registra `prompt_eval_count` y `eval_count` a través de `ChatOllama`. Esa telemetría, junto con una estimación conservadora del siguiente prompt, activa la compactación antes de acercarse al límite configurado de `context_window`.

## Tests

```bash
python3 -m unittest discover -s tests
```
