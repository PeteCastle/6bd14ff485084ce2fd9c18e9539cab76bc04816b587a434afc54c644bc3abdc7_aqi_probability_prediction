## Data Dictionary
This project uses air quality index (AQI) data sourced from the **OpenWeatherMap API**, covering 138 cities globally from 2023 to 2025. For consistency and regional focus, we limit the scope to cities within Metro Manila, Philippines. The dataset contains hourly measurements of seven key air pollutants: **sulfur dioxide (SO₂), nitrogen dioxide (NO₂), particulate matter (PM10 and PM2.5), ozone (O₃), and carbon monoxide (CO)**. These pollutants are commonly monitored in environmental health studies and serve as the prediction targets for our probabilistic forecasting models.

Sources:
- [2023 to 2024 Data (Kaggle)](https://www.kaggle.com/datasets/bwandowando/philippine-major-cities-air-quality-data)
- [2025 Data (Kaggle)](https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2025/data)

| Feature Name        | Data Type   | Description                                                         | Expected Values |
|---------------------|------------|---------------------------------------------------------------------|-----------------|
| `datetime`          | timestamp  | Measurement timestamp (UTC). Rounded to hour during preprocessing.  | ISO 8601 |
| `city_name`         | categorical| City label provided by source.                                      | One of configured cities |
| `components.co`     | numerical  | Carbon monoxide concentration (µg/m³).                              | ≥ 0 |
| `components.no`     | numerical  | Nitric oxide concentration (µg/m³).                                 | ≥ 0 |
| `components.no2`    | numerical  | Nitrogen dioxide concentration (µg/m³).                             | ≥ 0 |
| `components.o3`     | numerical  | Ozone concentration (µg/m³).                                        | ≥ 0 |
| `components.so2`    | numerical  | Sulfur dioxide concentration (µg/m³).                               | ≥ 0 |
| `components.pm2_5`  | numerical  | PM2.5 mass concentration (µg/m³).                                   | ≥ 0 |
| `components.pm10`   | numerical  | PM10 mass concentration (µg/m³).                                    | ≥ 0 |
| `components.nh3`    | numerical  | Ammonia concentration (µg/m³).                                      | ≥ 0 |

**Notes**
- Negative pollutant values are clamped to `0` in preprocessing.
- Duplicate hourly records are averaged per hour per city.
- Time gaps are linearly interpolated per city to build a complete hourly grid.
