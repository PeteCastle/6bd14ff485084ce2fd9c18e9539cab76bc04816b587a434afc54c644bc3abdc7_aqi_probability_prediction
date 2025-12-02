#!/bin/sh

set -e

until pg_isready -h postgres -U airflow; do
  echo "Waiting for PostgreSQL..."
  sleep 1
done

if ! psql -h postgres -U airflow -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1; then
  echo "Creating 'mlflow' database..."
  psql -h postgres -U airflow -c 'CREATE DATABASE mlflow'
else
  echo "'mlflow' database already exists."
fi
