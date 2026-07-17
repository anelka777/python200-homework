from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from prefect import flow, task, get_run_logger


DATA_DIR = (
        Path(__file__).resolve().parent.parent
        / "assignments"
        / "resources"
        / "happiness_project"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@task(
    retries=3,
    retry_delay_seconds=2
)
def load_data():
    logger = get_run_logger()

    all_years = []

    OUTPUT_DIR.mkdir(exist_ok=True)

    expected_columns = {
        "Country",
        "region",
        "happiness_score",
        "year"
    }

    # Every standardized column now goes through the same detect-and-log
    # process (previously only happiness_score got this treatment), so we
    # can't silently end up with an inconsistent schema across years. Each
    # target column lists every raw name it might appear under across the
    # 2015-2024 files.
    column_aliases = {
        "happiness_score": ["Happiness score", "Ladder score", "happiness_score"],
        "region": ["Regional indicator", "region"],
        "Country": ["Country", "Country name", "country"],
    }

    # Load data from 2015 to 2024
    for year in range(2015, 2025):
        file_path = DATA_DIR / f"world_happiness_{year}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=","
        )

        rename_map = {}

        for target_col, aliases in column_aliases.items():
            matched = [col for col in aliases if col in df.columns]

            if not matched:
                raise ValueError(
                    f"{year}: could not find a source column for "
                    f"'{target_col}'. Expected one of {aliases}, "
                    f"got columns: {df.columns.tolist()}"
                )

            source_col = matched[0]

            logger.info(
                f"{year}: using '{source_col}' as the source for "
                f"'{target_col}'"
            )

            if source_col != target_col:
                rename_map[source_col] = target_col

        # Normalize column names so every year ends up with the exact same
        # schema before concatenation.
        df = df.rename(columns=rename_map)

        # Add year information
        df["year"] = year

        logger.info(
            f"Loaded {year}: {len(df)} rows"
        )

        missing_columns = expected_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{year} is missing columns: {missing_columns}"
            )

        all_years.append(df)

    # Merge all yearly datasets
    merged_df = pd.concat(
        all_years,
        ignore_index=True
    )

    duplicates = merged_df.duplicated().sum()

    logger.info(
        f"Number of duplicate rows: {duplicates}"
    )

    logger.info(
        f"Missing values:\n{merged_df.isnull().sum()}"
    )

    output_file = OUTPUT_DIR / "merged_happiness.csv"

    merged_df.to_csv(
        output_file,
        index=False
    )

    logger.info(f"Merged dataset saved to {output_file}")
    logger.info(f"Merged dataframe shape: {merged_df.shape}")
    logger.info(f"Columns: {merged_df.columns.tolist()}")

    return merged_df


@task
def descriptive_statistics(df):
    logger = get_run_logger()

    # Overall statistics
    mean_score = df["happiness_score"].mean()
    median_score = df["happiness_score"].median()
    std_score = df["happiness_score"].std()

    logger.info(f"Mean happiness score: {mean_score:.2f}")
    logger.info(f"Median happiness score: {median_score:.2f}")
    logger.info(f"Standard deviation: {std_score:.2f}")

    # Mean happiness score by year
    yearly_mean = (
        df.groupby("year")["happiness_score"]
        .mean()
    )

    logger.info("Mean happiness score by year:")

    for year, score in yearly_mean.items():
        logger.info(f"{year}: {score:.2f}")

    # Mean happiness score by region
    region_mean = (
        df.groupby("region")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )

    logger.info("Mean happiness score by region:")

    for region, score in region_mean.items():
        logger.info(f"{region}: {score:.2f}")



@task
def create_visualizations(df):
    logger = get_run_logger()

    # Histogram of happiness scores
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="happiness_score",
        bins=30
    )

    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")

    histogram_path = OUTPUT_DIR / "happiness_histogram.png"

    plt.savefig(histogram_path)
    plt.close()

    logger.info(f"Histogram saved to {histogram_path}")


    # Boxplot happiness score by year
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="year",
        y="happiness_score"
    )

    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    boxplot_path = OUTPUT_DIR / "happiness_by_year.png"

    plt.savefig(boxplot_path)
    plt.close()

    logger.info(f"Boxplot saved to {boxplot_path}")


    # GDP vs Happiness scatter plot
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="GDP per capita",
        y="happiness_score"
    )

    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")

    scatter_path = OUTPUT_DIR / "gdp_vs_happiness.png"

    plt.savefig(scatter_path)
    plt.close()

    logger.info(f"Scatter plot saved to {scatter_path}")


    # Correlation heatmap
    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True
    )

    plt.title("Correlation Heatmap")

    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"

    plt.savefig(heatmap_path)
    plt.close()

    logger.info(f"Correlation heatmap saved to {heatmap_path}")


@task
def hypothesis_testing(df):
    logger = get_run_logger()

    # Compare happiness scores between 2019 and 2020
    happiness_2019 = df[df["year"] == 2019]["happiness_score"]
    happiness_2020 = df[df["year"] == 2020]["happiness_score"]

    t_stat, p_value = ttest_ind(
        happiness_2019,
        happiness_2020
    )

    mean_2019 = happiness_2019.mean()
    mean_2020 = happiness_2020.mean()

    logger.info(f"Mean happiness score in 2019: {mean_2019:.2f}")
    logger.info(f"Mean happiness score in 2020: {mean_2020:.2f}")

    logger.info(f"T-statistic: {t_stat:.4f}")
    logger.info(f"P-value: {p_value:.4f}")

    alpha = 0.05

    # The Interpretation now names the exact means, the direction of the
    # change, and the size of the gap, so the wording maps directly onto
    # this specific 2019 vs 2020 comparison rather than a generic template.
    direction = "higher" if mean_2020 > mean_2019 else "lower"
    gap = abs(mean_2020 - mean_2019)

    if p_value < alpha:
        interpretation = (
            f"Mean happiness score went from {mean_2019:.2f} in 2019 to "
            f"{mean_2020:.2f} in 2020 — a {gap:.2f}-point {direction} shift. "
            f"With p={p_value:.4f} (below alpha={alpha}), this change is "
            "statistically significant and unlikely to be due to random "
            "chance."
        )
    else:
        interpretation = (
            f"Mean happiness score went from {mean_2019:.2f} in 2019 to "
            f"{mean_2020:.2f} in 2020 — a {gap:.2f}-point {direction} shift. "
            f"With p={p_value:.4f} (above alpha={alpha}), this change is "
            "not statistically significant, so it could reasonably be "
            "explained by random variation rather than a real year-over-year "
            "effect."
        )

    logger.info(f"Interpretation: {interpretation}")


    # Second test: compare two regions
    western_europe = df[
        df["region"] == "Western Europe"
        ]["happiness_score"]

    sub_saharan_africa = df[
        df["region"] == "Sub-Saharan Africa"
        ]["happiness_score"]


    region_t_stat, region_p_value = ttest_ind(
        western_europe,
        sub_saharan_africa
    )

    logger.info(
        f"Western Europe mean happiness: {western_europe.mean():.2f}"
    )

    logger.info(
        f"Sub-Saharan Africa mean happiness: {sub_saharan_africa.mean():.2f}"
    )

    logger.info(
        f"Regional comparison t-statistic: {region_t_stat:.4f}"
    )

    logger.info(
        f"Regional comparison p-value: {region_p_value:.4f}"
    )

    if region_p_value < alpha:
        logger.info(
            "The happiness difference between these regions "
            "is statistically significant."
        )
    else:
        logger.info(
            "The happiness difference between these regions "
            "is not statistically significant."
        )

    return {
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "p_value": p_value,
        "significant": p_value < alpha,
        "region_p_value": region_p_value,
        "region_significant": region_p_value < alpha,
    }

@task
def correlation_analysis(df):
    logger = get_run_logger()

    numeric_variables = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]

    alpha = 0.05
    results = []

    for variable in numeric_variables:

        clean_data = df[[variable, "happiness_score"]].dropna()

        if len(clean_data) < 2:
            logger.warning(
                f"Not enough data for {variable}"
            )
            continue

        correlation, p_value = pearsonr(
            clean_data[variable],
            clean_data["happiness_score"]
        )

        results.append({
            "variable": variable,
            "correlation": correlation,
            "p_value": p_value
        })

        logger.info(
            f"{variable}: correlation={correlation:.4f}, "
            f"p-value={p_value:.4f}"
        )


    # Bonferroni correction
    number_of_tests = len(results)

    if number_of_tests == 0:
        logger.warning(
            "No correlation tests were performed"
        )
        return

    adjusted_alpha = alpha / number_of_tests

    logger.info(
        f"Bonferroni adjusted alpha: {adjusted_alpha:.4f}"
    )


    logger.info("Significance results:")

    significant_variables = []

    for result in results:

        original_significance = result["p_value"] < alpha
        corrected_significance = result["p_value"] < adjusted_alpha

        logger.info(
            f"{result['variable']}: "
            f"original alpha={original_significance}, "
            f"after Bonferroni={corrected_significance}"
        )

        if corrected_significance:
            significant_variables.append(result)


    # Find strongest correlation after correction
    if significant_variables:
        strongest = max(
            significant_variables,
            key=lambda x: abs(x["correlation"])
        )

        logger.info(
            f"Strongest correlation after Bonferroni correction: "
            f"{strongest['variable']} "
            f"(correlation={strongest['correlation']:.4f})"
        )
    else:
        strongest = None

    # Save correlation results
    correlation_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "correlation_results.csv"

    correlation_df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Correlation results saved to {output_file}"
    )
    return strongest

@task
def summary_report(df, hypothesis_result, strongest_correlation):
    logger = get_run_logger()

    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()

    region_mean = (
        df.groupby("region")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_three = region_mean.head(3)
    bottom_three = region_mean.tail(3)

    logger.info(
        f"Dataset contains {total_countries} countries across "
        f"{total_years} years."
    )

    logger.info("Top 3 happiest regions:")

    for region, score in top_three.items():
        logger.info(f"{region}: {score:.2f}")

    logger.info("Bottom 3 happiest regions:")

    for region, score in bottom_three.items():
        logger.info(f"{region}: {score:.2f}")

    mean_2019 = hypothesis_result["mean_2019"]
    mean_2020 = hypothesis_result["mean_2020"]
    direction = "higher" if mean_2020 > mean_2019 else "lower"
    gap = abs(mean_2020 - mean_2019)

    if hypothesis_result["significant"]:
        logger.info(
            f"Average happiness was {direction} in 2020 ({mean_2020:.2f}) "
            f"than in 2019 ({mean_2019:.2f}), a {gap:.2f}-point difference "
            f"that is statistically significant (p={hypothesis_result['p_value']:.4f}) "
            "and unlikely to be due to chance."
        )
    else:
        logger.info(
            f"Average happiness was {direction} in 2020 ({mean_2020:.2f}) "
            f"than in 2019 ({mean_2019:.2f}), a {gap:.2f}-point difference, "
            f"but this was not statistically significant "
            f"(p={hypothesis_result['p_value']:.4f}). The analysis did not "
            "find enough evidence to conclude that average happiness "
            "changed between 2019 and 2020."
        )

    # Make it explicit that this is the *strongest* correlation with
    # happiness_score that survived Bonferroni correction, including its
    # p-value, rather than just naming the variable.
    if strongest_correlation:
        logger.info(
            "Of all variables tested against happiness_score, the "
            f"strongest correlation that remained statistically significant "
            f"after Bonferroni correction is "
            f"'{strongest_correlation['variable']}' "
            f"(correlation={strongest_correlation['correlation']:.4f}, "
            f"p-value={strongest_correlation['p_value']:.4g})."
        )
    else:
        logger.info(
            "No correlations remained statistically significant "
            "after Bonferroni correction."
        )

@flow
def happiness_pipeline():
    logger = get_run_logger()

    df = load_data()

    descriptive_statistics(df)
    create_visualizations(df)
    hypothesis_result = hypothesis_testing(df)
    strongest_correlation = correlation_analysis(df)

    summary_report(
        df,
        hypothesis_result,
        strongest_correlation
    )

    logger.info(
        f"Pipeline completed successfully. Final dataset shape: {df.shape}"
    )

    return df


if __name__ == "__main__":
    happiness_pipeline()