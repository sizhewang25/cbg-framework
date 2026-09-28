#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load CLICKHOUSE_PASSWORD from .env
if [ -f "${SCRIPT_DIR}/.env" ]; then
    export $(grep CLICKHOUSE_PASSWORD "${SCRIPT_DIR}/.env" | grep -v 'SHA256' | xargs)
fi

# Host ports to publish. Overridable because the defaults are popular: a
# Jupyter kernel or another service holding 9000 makes `docker run` fail to
# bind, and re-mapping is cheaper than evicting whatever got there first.
# Keep these in sync with CLICKHOUSE_PORT / CLICKHOUSE_HTTP_PORT in .env.
CH_NATIVE_PORT="${CH_NATIVE_PORT:-9000}"
CH_HTTP_PORT="${CH_HTTP_PORT:-8123}"

# Stop any existing geoloc ClickHouse container
echo 'Stopping existing ClickHouse containers...'
sudo docker stop geoloc_clickhouse 2>/dev/null || true
sudo docker rm   geoloc_clickhouse 2>/dev/null || true

# Ensure logs dir exists and is world-writable (CH runs as uid 101 inside container)
mkdir -p "${SCRIPT_DIR}/clickhouse_files/logs"
chmod 777 "${SCRIPT_DIR}/clickhouse_files/logs"

echo 'Starting ClickHouse...'
sudo docker run -d     --name geoloc_clickhouse     -v "${SCRIPT_DIR}/clickhouse_files/data:/var/lib/clickhouse/"     -v "${SCRIPT_DIR}/clickhouse_files/logs:/var/log/clickhouse-server/"     -v "${SCRIPT_DIR}/clickhouse_files/users.d:/etc/clickhouse-server/users.d:ro"     -p "${CH_HTTP_PORT}":8123     -p "${CH_NATIVE_PORT}":9000     --ulimit nofile=262144:262144     clickhouse/clickhouse-server:22.6

echo ''
echo 'ClickHouse started successfully!'
echo '  Host:     ubuntu.nuwins.lab'
echo "  Port:     ${CH_NATIVE_PORT} (native), ${CH_HTTP_PORT} (HTTP)"
echo '  User:     default'
echo '  Password: ${CLICKHOUSE_PASSWORD:-<from .env>}'
echo '  Database: geolocation_replication'
echo ''
echo 'Waiting for ClickHouse to be ready...'
sleep 5
curl -s "http://localhost:${CH_HTTP_PORT}/ping" && echo ' — ready!'
