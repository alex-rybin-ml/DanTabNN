# Real-World Large Datasets for Testing (100-500 MB)

Cache location: `%TEMP%/dantabnn_test_datasets/`

First run downloads datasets (OpenML + sklearn). Subsequent runs load from cached Parquet files instantly.

## Binary (3 datasets)

| Dataset | Samples | Features | Parquet Size | Description |
|---------|---------|----------|-------------|-------------|
| **Adult Census** | 48,842 | 14 | ~6 MB | Predict income >$50K. Mixed types (age, education, occupation). ~75/25 class balance |
| **Electricity** | 45,312 | 8 | ~3 MB | Time-series electricity price change prediction. Real market data |
| **Bank Marketing** | 45,211 | 16 | ~7 MB | Telemarketing term deposit prediction. Heavy class imbalance (~90/10) |

## Regression (3 datasets)

| Dataset | Samples | Features | Parquet Size | Description |
|---------|---------|----------|-------------|-------------|
| **California Housing** | 20,640 | 8 | ~1 MB | Median house value prediction. sklearn built-in. Skewed target |
| **Year Prediction MSD** | 515,345 | 90 | ~350 MB | Audio features → music release year. **Largest regression** in test suite |
| **Online News Popularity** | 39,644 | 60 | ~20 MB | Article features → share count. Real media data |

## Multiclass (3 datasets)

| Dataset | Samples | Features | Parquet Size | Description |
|---------|---------|----------|-------------|-------------|
| **Forest Covertype** | 100,000 | 54 | ~40 MB | Tree cover type classification (7 classes). sklearn built-in. Real ecological data |
| **Letter Recognition** | 20,000 | 16 | ~2 MB | OCR uppercase letter classification (26 classes). Real image features |
| **Optdigits** | 5,620 | 64 | ~3 MB | Handwritten digit recognition (10 classes). Pixels → class |

## Generation

All datasets sourced from `sklearn.datasets.fetch_california_housing`, `fetch_covtype`, and `fetch_openml`. Cached as Parquet files.

## Test Coverage

| Test Class | What It Validates |
|-----------|-------------------|
| `TestRealBinary` | Fit/predict on real census, market, banking data |
| `TestRealRegression` | Fit/predict on housing, audio, news data |
| `TestRealMulticlass` | Fit/predict on forest, OCR, digit data |
| `TestMemoryOptimizations` | gc.collect, minimal mode, fit_from_parquet |
| `TestReproducibility` | Two runs with same seed produce identical predictions |