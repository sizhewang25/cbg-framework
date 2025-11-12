# pull the docker image
docker pull clickhouse/clickhouse-server:22.6


# start the server using docker
./start_clickhouse.sh

# download clickhouse client binary
curl https://clickhouse.com/ | sh
mv clickhouse ./clickhouse_files/

# install source files
poetry lock 
poetry install

# run clickhouse db installer for table init
poetry run python scripts/utils/clickhouse_installer.py
