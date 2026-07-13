# FULL build: named IDs and the admin reveal path are present. Run this
# image only behind an access-controlled host (private network, VPN, or an
# authenticating reverse proxy) -- never expose it directly to the public
# internet. See DEPLOY.md for the deploy-gate rationale.
#
# The same image can also run the PUBLIC build by setting
# PGI_PUBLIC_BUILD=1 at `docker run` time (the two builds are one codebase
# behind one environment-variable gate, not a fork) -- but for a public
# deployment, a hosted platform is the documented path (DEPLOY.md, path 1).

FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY .streamlit/config.toml .streamlit/config.toml

# Only the read-only artifacts the dashboard actually loads -- never
# data/raw/ or data/curated/ (the 12M-row source parquet never belongs in
# a deploy image).
COPY data/features/ data/features/
COPY data/results/ data/results/

# .streamlit/secrets.toml is intentionally NOT copied into the image: mount
# it at runtime instead, e.g.
#   docker run -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro ...
# or inject anon_salt via your host's secret manager as an env-backed file.
# Never bake secrets.toml into an image layer.

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
