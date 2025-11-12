#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load CLICKHOUSE_PASSWORD from .env
if [ -f .env ]; then
    export $(grep CLICKHOUSE_PASSWORD .env | grep -v "SHA256" | xargs)
fi

# Stop any existing ClickHouse containers
echo "Stopping existing ClickHouse containers..."
docker ps -q --filter ancestor=clickhouse/clickhouse-server:22.6 | xargs docker stop 2>/dev/null || true

# Start ClickHouse (password is in users.d/password_local.xml, which is git-ignored)
echo "Starting ClickHouse..."
docker run --rm -d \
    -v "${SCRIPT_DIR}/clickhouse_files/data:/var/lib/clickhouse/" \
    -v "${SCRIPT_DIR}/clickhouse_files/logs:/var/log/clickhouse-server/" \
    -v "${SCRIPT_DIR}/clickhouse_files/users.d:/etc/clickhouse-server/users.d:ro" \
    -p 8123:8123 \
    -p 9000:9000 \
    --ulimit nofile=262144:262144 \
    clickhouse/clickhouse-server:22.6

echo ""
echo "ClickHouse started successfully!"
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 9000 (native), 8123 (HTTP)"
echo "  User: default"
echo "  Password: ${CLICKHOUSE_PASSWORD:-<from .env>}"
echo "  Database: geolocation_replication"
echo ""
echo "Waiting for ClickHouse to be ready..."
sleep 5
