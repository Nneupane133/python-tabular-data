#! /usr/bin/env python3

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

@dataclass(frozen=True)
class RegressionResult:
    """Container for regression results."""
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float


"""
Perform per-species linear regressions on the Iris dataset and generate plots.

This script reproduces the "petal length vs sepal length" regression example,
but does it separately for each of the three species in iris.csv:

- Iris_setosa
- Iris_versicolor
- Iris_virginica

For each species, it:
  1) filters the data
  2) runs a linear regression: sepal_length_cm ~ petal_length_cm
  3) creates a scatter plot + fitted regression line
  4) saves the plot as a PNG

Requirements:
  - pandas
  - matplotlib
  - scipy

Example:
  python3 iris_species_regression.py --csv iris.csv --outdir results
"""

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run per-species linear regressions on the Iris dataset and save plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="iris.csv",
        help="Path to the iris CSV file.",
    )
    parser.add_argument(
        "--outdir",
        default="iris_regression_plots",
        help="Output directory for saved PNG plots.",
    )
    parser.add_argument(
        "--x-col",
        default="petal_length_cm",
        help="Name of the predictor (x-axis) column.",
    )
    parser.add_argument(
        "--y-col",
        default="sepal_length_cm",
        help="Name of the response (y-axis) column.",
    )
    parser.add_argument(
        "--species-col",
        default="species",
        help="Name of the species column.",
    )
    return parser.parse_args()


def load_iris_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load the Iris CSV file into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The iris dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def run_linear_regression(x: pd.Series, y: pd.Series) -> RegressionResult:
    """
    Run a simple linear regression y ~ x.

    Parameters
    ----------
    x : pandas.Series
        Predictor values.
    y : pandas.Series
        Response values.

    Returns
    -------
    RegressionResult
        Regression statistics (slope, intercept, etc.).
    """
    regression = stats.linregress(x, y)
    return RegressionResult(
        slope=regression.slope,
        intercept=regression.intercept,
        rvalue=regression.rvalue,
        pvalue=regression.pvalue,
        stderr=regression.stderr,
    )


def sanitize_species_for_filename(species: str) -> str:
    """Convert a species label into a filesystem-friendly name."""
    return species.strip().replace(" ", "_")


def plot_species_regression(
    dataframe: pd.DataFrame,
    species: str,
    x_col: str,
    y_col: str,
    outdir: str,
) -> str:
    """
    Create and save a regression plot for a single species.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Full iris dataset.
    species : str
        Species name to filter by.
    x_col : str
        Predictor column name.
    y_col : str
        Response column name.
    outdir : str
        Output directory for plot.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    subset = dataframe[dataframe["species"] == species]
    if subset.empty:
        raise ValueError(f"No rows found for species: {species}")

    x = subset[x_col]
    y = subset[y_col]
    result = run_linear_regression(x, y)

    # Create plot
    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, result.slope * x + result.intercept, color="orange", label="Fitted line")

    plt.xlabel("Petal length (cm)" if x_col == "petal_length_cm" else x_col)
    plt.ylabel("Sepal length (cm)" if y_col == "sepal_length_cm" else y_col)
    plt.title(f"{species}: {y_col} vs {x_col}")
    plt.legend()

    os.makedirs(outdir, exist_ok=True)
    safe_species = sanitize_species_for_filename(species)
    out_path = os.path.join(outdir, f"{safe_species}_{y_col}_vs_{x_col}_regress.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def get_species_list(dataframe: pd.DataFrame, species_col: str = "species") -> Tuple[str, ...]:
    """
    Get the sorted unique species names.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Iris dataset.
    species_col : str
        Column containing species labels.

    Returns
    -------
    tuple[str, ...]
        Sorted unique species names.
    """
    if species_col not in dataframe.columns:
        raise KeyError(f"Species column not found: {species_col}")
    return tuple(sorted(dataframe[species_col].unique()))


def main() -> int:
    """Run the script."""
    args = parse_args()
    df = load_iris_dataframe(args.csv)

    # Validate columns early
    required_cols = {args.x_col, args.y_col, args.species_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in CSV: {sorted(missing)}")

    species_list = get_species_list(df, species_col=args.species_col)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Found species: {', '.join(species_list)}")
    print(f"Saving plots to: {args.outdir}")

    for species in species_list:
        out_path = plot_species_regression(
            dataframe=df,
            species=species,
            x_col=args.x_col,
            y_col=args.y_col,
            outdir=args.outdir,
        )
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())