#!/bin/sh

set -e

until pg_isready -h postgres -U airflow; do
  echo "Waiting for PostgreSQL..."
  sleep 1
done

if ! psql -h postgres -U airflow -tc "SELECT 1 FROM pg_database WHERE datname = 'optuna'" | grep -q 1; then
  echo "Creating 'optuna' database..."
  psql -h postgres -U airflow -c 'CREATE DATABASE optuna'
else
  echo "'optuna' database already exists."
fi
