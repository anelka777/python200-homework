import numpy as np
import pandas as pd

from prefect import task, flow

@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    arr = np.array([
        12.0,
        15.0,
        np.nan,
        14.0,
        10.0,
        np.nan,
        18.0,
        14.0,
        16.0,
        22.0,
        np.nan,
        13.0
    ])

    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)

    return summary

if __name__ == "__main__":
    result = pipeline_flow()

    print("Pipeline summary:")

    for key, value in result.items():
        print(f"{key}: {value}")



# Why Prefect may be unnecessary here:
# This pipeline is very small and processes only a few values.
# Using Prefect adds extra setup and complexity compared to simply
# running regular Python functions.

# When Prefect is useful:
# Prefect is helpful for larger workflows with many steps, large datasets,
# scheduled jobs, monitoring, retries, logging, and managing dependencies
# between tasks.
