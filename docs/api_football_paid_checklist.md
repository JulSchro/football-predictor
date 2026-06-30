# API-Football Paid Checklist

## Objetivo

Automatizar el flujo que hoy ya existe manualmente:

1. detectar partidos del dia
2. guardar snapshot prepartido
3. actualizar resultado final
4. evaluar metricas
5. calibrar mercados

## Endpoints principales

- `leagues`: validar competiciones activas.
- `fixtures`: fixtures del dia, resultados finales y sedes.
- `fixtures/statistics`: corners, tiros, tiros a puerta, posesion, tarjetas y xG cuando este disponible.
- `injuries`: bajas por fixture.
- `players` y `teams`: enriquecer plantillas cuando el plan lo permita.

## Jobs diarios

### Manana

```powershell
python -m football_predictor.cli today-fixtures
python -m football_predictor.cli predict-today
```

### Despues de partidos

```powershell
python -m football_predictor.cli sync-api-football-fixtures --league ID --season YEAR
python -m football_predictor.cli update-results-from-fixtures
python -m football_predictor.cli prediction-backtest-report
```

## Ligas iniciales

Prioridad 1:

- World Cup
- Champions League
- Premier League
- LaLiga
- Serie A
- Bundesliga
- Liga MX
- Copa Libertadores
- Brasileirao
- Primera Division Argentina

Prioridad 2:

- Ligue 1
- MLS
- Europa League
- Copa Sudamericana
- CONCACAF Champions Cup
- Copa America
- Euro
- Gold Cup

## Control de requests

- 1 request por competicion para fixtures del dia.
- 1 request por fixture terminado para statistics.
- 1 request por fixture importante para injuries.
- Evitar repetir details si ya estan guardados.
- Guardar todo en SQLite antes de recalcular.

## Reglas anti data leakage

- El snapshot se guarda antes del partido.
- El resultado real solo actualiza columnas `actual_*`.
- La calibracion usa partidos ya terminados.
- El entrenamiento cronologico nunca usa partidos posteriores a la fecha evaluada.
