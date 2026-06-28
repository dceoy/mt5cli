# Grafana Integration for mt5cli

This directory contains example configuration and dashboard files for visualising
mt5cli SQLite data in [Grafana](https://grafana.com/) using the
[Grafana SQLite datasource plugin](https://grafana.com/grafana/plugins/frser-sqlite-datasource/).

## Prerequisites

- mt5cli installed and able to connect to MetaTrader 5
- Grafana 10+ with the `frser-sqlite-datasource` plugin installed
- (Optional) Docker and Docker Compose for the containerised setup

## Generating the SQLite database

Collect historical data and snapshot current account state:

```sh
# Collect OHLCV history
mt5cli -o history.db collect-history --symbol EURUSD --date-from 2024-01-01 --date-to 2024-12-31

# Create Grafana-ready views and indexes
mt5cli -o history.db grafana-schema

# Snapshot current account, positions, and orders
mt5cli -o history.db snapshot --with-grafana-schema
```

## Publishing a Grafana-readable copy

Grafana reads the SQLite file directly. To avoid read/write conflicts, publish
a consistent copy after each update:

```sh
mt5cli -o history.db grafana-schema --publish-copy history.mt5cli.db
mt5cli -o history.db snapshot --publish-copy history.mt5cli.db
```

The `--publish-copy` option uses the SQLite online backup API, which is safe
even when the source database uses WAL journal mode.

## Configuring the datasource path

Edit `provisioning/datasources/mt5cli-sqlite.yml` and set the `path` field
to the absolute path of your published `.db` file:

```yaml
jsonData:
  path: /absolute/path/to/history.mt5cli.db
```

## Running Grafana on Windows (native)

1. Download and install Grafana from <https://grafana.com/grafana/download/>.
2. Install the SQLite plugin: `grafana-cli plugins install frser-sqlite-datasource`.
3. Copy `provisioning/datasources/mt5cli-sqlite.yml` into
   `%ProgramFiles%\GrafanaLabs\grafana\conf\provisioning\datasources\`.
   Do not copy `provisioning/dashboards/mt5cli.yml` — it contains a
   Docker-specific dashboard path that is not valid on Windows.
4. Import the dashboards from `dashboards/` via the Grafana UI
   (Dashboards → Import → Upload JSON file).

## Running with Docker Compose

Set `MT5CLI_DB_PATH` to the absolute path of your published `.db` file, then
start the stack:

```sh
# From the examples/grafana directory
MT5CLI_DB_PATH=/absolute/path/to/history.mt5cli.db docker compose up -d
```

Alternatively, create a `.env` file in `examples/grafana/` containing
`MT5CLI_DB_PATH=/absolute/path/to/history.mt5cli.db` and run
`docker compose up -d`. Compose refuses to start if the variable is unset or
empty.

Then open <http://localhost:3000> (default credentials: admin / admin).

## Dashboard overview

| Dashboard              | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `mt5cli-overview.json` | Account balance, equity, margin, and snapshot freshness |
| `mt5cli-trades.json`   | Trade P/L, win rate, symbol breakdown                   |
| `mt5cli-market.json`   | OHLCV rates, spreads, and tick volume                   |

All panel queries use the `grafana_*` views; they do not read internal storage
tables directly.

## Importing dashboards

1. Open Grafana and navigate to **Dashboards → Import**.
2. Click **Upload JSON file** and select one of the files in `dashboards/`.
3. Select the `mt5cli-SQLite` datasource when prompted.
4. Click **Import**.
