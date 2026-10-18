# Debate Arena

Sistema CLI autónomo que enfrenta a dos agentes de debate (`PRO` y `CON`) sobre cualquier tesis utilizando el SDK nativo de **Ollama** (`ollama.Client`), separando la generación argumentativa de la toma de decisiones probabilística.

## Modelos y Roles

- **`gemma4:12b-it-qat` (`think=False`)**:
  - **Debatientes (`PRO` / `CON`)**: Cada debatiente recibe el historial desde su propia perspectiva en primera persona (`assistant` para sus intervenciones previas y `user` para las de su oponente), evitando la confusión de identidad y el intercambio accidental de posturas.
  - **Compactador de contexto**: Resume periódicamente las rondas antiguas en un resumen operativo conciso y extrae la lista acumulada de argumentos agotados mediante salidas estructuradas JSON (`format`).
  - **Redactor del fallo**: Sintetiza en lenguaje natural la explicación final a partir del fallo cuantitativo e inamovible emitido por `d1:latest`.
- **`d1:latest` (`client.systemone`)**:
  - **Guardia de turno (`validate_turn`)**: Evalúa probabilísticamente (`noul`) si la intervención mantiene la postura asignada en primera persona y si repite argumentos agotados.
  - **Árbitro de ronda (`evaluate_round`)**: Puntúa el rigor de `PRO` y `CON` (`score`), detecta concesiones explícitas (`choice`) y evalúa estancamiento o cierre temprano (`noul`).
  - **Juez del veredicto (`issue_verdict`)**: Determina el ganador (`pro`, `con` o `draw`) y el factor decisivo mediante probabilidades calibradas (`choice`).

## Estructura del Proyecto

```text
config/
├── settings.yaml                  # Configuración de Ollama, modelos, contexto y umbrales d1
└── prompts/
    ├── debater_system.j2          # Identidad en primera persona, resumen y argumentos prohibidos
    ├── opening_user.j2            # Mensaje user inicial o de continuación tras compactación
    ├── compactor.j2               # Compactación estructurada de rondas previas
    └── verdict_summary.j2         # Síntesis narrativa subordinada al fallo de d1
src/debate_arena/
├── models.py                      # Dataclasses inmutables de dominio
├── config.py                      # Carga tipada de settings.yaml y entorno Jinja2
├── context.py                     # Perspectiva assistant/user por debatiente y compactación
├── judge.py                       # Evaluaciones probabilísticas con d1:latest y compactación
├── arena.py                       # Bucle principal del debate
├── presenter.py                   # Salida enriquecida con Rich y exportación a Markdown
└── main.py                        # Punto de entrada CLI (debate-cli)
```

## Instalación y Uso

```bash
.venv/bin/pip install -e .
.venv/bin/debate-cli -p "La energía nuclear es imprescindible para la transición ecológica"
.venv/bin/debate-cli -p "El teletrabajo debe ser un derecho laboral por defecto" -f debate.md
```
