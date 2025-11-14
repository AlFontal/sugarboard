# Sugarboard

[![Docker Ready](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](DOCKER.md)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![NiceGUI 2](https://img.shields.io/badge/NiceGUI-2.0-9333EA)](https://nicegui.io/)

Modern [NiceGUI](https://nicegui.io/) dashboard for personal Continuous Glucose Monitoring data powered by the [Nightscout](http://www.nightscout.info/) API. The project runs exclusively as a NiceGUI app and is designed to be deployed via Docker (compose or standalone).

<img width="1800" height="1100" alt="sugarboard_screenshot" src="https://github.com/user-attachments/assets/3a30d0f9-8bb5-4a53-bd23-46223c129ee2" />



## Quick Start (Docker Compose)

```bash
git clone https://github.com/AlFontal/sugarboard.git
cd sugarboard
docker-compose up -d --build
```

The NiceGUI server will be exposed on [http://localhost:8080](http://localhost:8080). Logs can be tailed with `docker-compose logs -f sugarboard`, and `docker-compose down` stops the stack. The compose file also mounts a named volume to persist cached CGM data between restarts.

### Nightscout Credentials

When the dashboard loads, fill in the **Nightscout Connection** card with your base URL plus either:

- **Read token (recommended):** create a read-only API token inside your Nightscout instance (`Settings → API → Add Token`). This grants GET access without exposing the master secret.
- **API secret:** the classic admin secret string (we hash it and send it via the `api-secret` header). Only use this if you have tokens disabled.

Credentials stay on the server and are never rendered back to the browser. If you set the optional `CGM_SITE` environment variable, it simply pre-fills the base URL field for convenience.

### Configuration

- Runtime credentials: provided through the UI card described above.
- `STORAGE_SECRET`: required for NiceGUI's secure server-side storage (set to any long random string).
- `RECENT_REQUEST_TIMEOUT`: Nightscout API timeout in seconds (defaults to 60). Increase if your server responds slowly.
- `TZ`: optional timezone for the container.
- Cache persistence: by default a Docker volume named `sugarboard-cache` stores `.cache/`.
- Credential persistence: another volume `sugarboard-storage` stores `.nicegui/` so saved Nightscout credentials survive rebuilds.

Set the non-interactive settings in `docker-compose.yml` (see `DOCKER.md` for other deployment targets). The Nightscout token/secret should always be entered via the UI.

To keep secrets out of version control, place them in a local `.env` file (already git-ignored) and let Docker Compose load it:

```bash
python - <<'PY' > .env
from secrets import token_hex
print(f"STORAGE_SECRET={token_hex(32)}")
PY
```

## Local Development (optional, without Docker)

If you still want to run it locally without containers, use the lightweight requirements file:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.dev.txt
make run
```

The NiceGUI app binds to `0.0.0.0:8080`. To use a different port, set `PORT=9090 make run`. Hot reload is enabled by default via the `NICEGUI_RELOAD` env var in the `Makefile`.

### Developer tools

```bash
make lint             # black, isort, flake8
./venv/bin/pytest     # unit + integration tests (fast path)
RUN_E2E=1 ./venv/bin/pytest -m e2e  # spins up NiceGUI + Selenium smoke tests
```

The Selenium suite now starts SugarBoard automatically using deterministic fixture data under `tests/data/`, so there’s no manual “nicegui run …” step. Set `RUN_E2E=1` (optionally override `E2E_APP_PORT`) and the fixture will: (1) launch the app, (2) wait for `/health`, (3) run SeleniumBase against it, and (4) tear everything down.

The codebase no longer ships the historical Streamlit implementation. Check out an older branch/tag if you need it.
