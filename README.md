# FootballPredictor

Base modular para crear una herramienta de machine learning que predice partidos de futbol. Carga historiales, ranking FIFA, datos enriquecidos, fixtures reales y estadisticas avanzadas cuando hay fuentes conectadas.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## CSV esperado

```csv
date,home_team,away_team,home_goals,away_goals,competition,season
2024-08-10,Team A,Team B,2,1,League,2024-2025
2024-08-17,Team C,Team A,0,0,League,2024-2025
```

## Comandos

```bash
football-predictor init-db
football-predictor import-worldcup-2026
football-predictor import-worldcups --years 2014,2018,2022,2026
football-predictor import-international-history --start-year 2018
football-predictor import-wc2026-enriched
football-predictor load-csv data/raw/matches.csv
football-predictor train-baseline
football-predictor train-ml
football-predictor backtest
football-predictor import-prediction-backtest data/backtests/worldcup_2026_2026-06-27.csv
football-predictor prediction-backtest-report
football-predictor sync-competition-catalog --season 2026
football-predictor today-fixtures
football-predictor predict-today
football-predictor update-results-from-fixtures
football-predictor prepare-data-quality
football-predictor data-quality-report
football-predictor data-quality
football-predictor sync-fifa-ranking
football-predictor sync-api-football-fixtures --league 1 --season 2026
football-predictor sync-api-football-fixture-details --fixture-id 123456
football-predictor show-teams
football-predictor predict --home "Team A" --away "Team B" --save
football-predictor simulate --home "Team A" --away "Team B" --simulations 20000 --mode poker
football-predictor serve
```

Tambien puedes usar:

```bash
python -m football_predictor.cli predict --home "Team A" --away "Team B"
```

## Que incluye

- Esquema SQLite local para equipos, partidos, estadisticas, predicciones, sedes, jugadores y datos externos.
- Validadores Pydantic para equipos, partidos, estadisticas y predicciones.
- Loader CSV validado.
- Importadores para Mundial 2026, mundiales historicos, historial internacional y dataset enriquecido.
- Sincronizador API-Football para fixtures, resultados, estadisticas por partido y lesiones.
- Sincronizador del ranking FIFA masculino oficial desde `api.fifa.com`.
- Metricas externas opcionales: ranking FIFA, puntos FIFA, valor de plantilla, tamano y edad media.
- Features de forma, goles, diferencial, win rate, fuerza simple, localia y tendencia.
- Baselines Poisson y Elo.
- Modelo ML basico con LogisticRegression o RandomForest.
- Backtesting cronologico sin usar partidos futuros.
- Backtesting formal de predicciones guardadas contra resultados reales.
- Metricas automaticas: Accuracy, Log Loss, Brier, MAE, RMSE y sesgos por mercado.
- Seguimiento manual desde la interfaz: snapshot prepartido y carga de resultado real.
- Catalogo multi-liga preparado para Europa, America y selecciones.
- Comandos base para automatizacion diaria local.
- Calibracion simple de mercados usando sesgos observados.
- Mapeo de nombres de equipos para reducir duplicados entre fuentes.
- Readiness de API y cobertura por competicion.
- Simulaciones Monte Carlo clasicas, hibridas y tipo poker por escenarios.
- Mercados secundarios estimados: over, ambos anotan, tiros, corners, tarjetas y xG.
- Interfaz web local con perfiles, factores, backtest, partidos, predicciones y simulaciones.
- CLI con Typer y tests minimos.

## Fuentes externas

Variables opcionales en `.env`:

```bash
API_FOOTBALL_KEY=
FOOTBALL_DATA_ORG_TOKEN=
THESPORTSDB_API_KEY=3
```

OpenFootball no requiere API key. API-Football y football-data.org requieren token para datos actualizados segun el plan disponible. TheSportsDB ofrece API JSON gratuita con limites y opciones premium.

La interfaz muestra muchas variables avanzadas desde el principio. Cuando no hay fuente conectada, usa proxies del historial o valores neutrales visibles; no inventa lesiones, odds, clima ni datos de jugadores.

## Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/JulSchro/football-predictor)

La app puede subirse como servicio web con FastAPI/Uvicorn.

Render:

1. Sube el repositorio a GitHub.
2. En Render crea un Blueprint usando `render.yaml`.
3. Configura `API_FOOTBALL_KEY` como variable secreta.
4. Mantén `FOOTBALL_PREDICTOR_DB_PATH=/var/data/football_predictor.sqlite`.
5. Usa disco persistente para no perder fixtures, predicciones ni calibracion.

Comando de arranque:

```bash
uvicorn football_predictor.web.asgi:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
/health
```

SQLite funciona bien para uso personal y una primera version publica. Si luego hay varios usuarios o jobs automaticos pesados, el siguiente paso natural es PostgreSQL.

## Proximos pasos

- Mejorar el mapeo de nombres entre fuentes externas y selecciones.
- Conectar clima, arbitro, odds y closing line.
- Separar pesos del modelo en un archivo de configuracion entrenable.
