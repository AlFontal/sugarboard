# Sugarboard

Modern [NiceGUI](https://nicegui.io/) dashboard for personal Continuous Glucose Monitoring data powered by the [Nightscout](http://www.nightscout.info/) API. The project runs exclusively as a NiceGUI app and is designed to be deployed via Docker (compose or standalone).

## Quick Start (Docker Compose)

```bash
git clone https://github.com/AlFontal/sugarboard.git
cd sugarboard
docker-compose up -d --build
```

The NiceGUI server will be exposed on [http://localhost:8080](http://localhost:8080). Logs can be tailed with `docker-compose logs -f sugarboard`, and `docker-compose down` stops the stack. The compose file also mounts a named volume to persist cached CGM data between restarts.

### Configuration

- `CGM_SITE`: override the Nightscout base hostname (defaults to `cgm-monitor-alfontal.herokuapp.com` in code).
- `TZ`: optional timezone for the container.
- Cache persistence: by default a Docker volume named `sugarboard-cache` stores `.cache/`.

Set these in `docker-compose.yml` (see `DOCKER.md` for other deployment targets).

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
make lint        # black, isort, flake8
make test        # pytest + coverage (non-e2e)
make test-e2e    # e2e suite (requires running app)
make coverage    # serve coverage HTML report
```

The codebase no longer ships the historical Streamlit implementation. Check out an older branch/tag if you need it.
