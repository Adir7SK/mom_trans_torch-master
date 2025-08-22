import argparse
import csv
import os
import matplotlib
import yaml

import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

def _plot_ticker(df: pd.DataFrame, output_folder: str, ticker: str) -> None:
    """
    Plot the price series for a given ticker.

    :param df: The DataFrame containing the price series.
    :type df: pd.DataFrame
    :param output_folder: The path to the output folder.
    :type output_folder: str
    :param ticker: The ticker symbol.
    :type ticker: str
    :return: None
    :rtype: None
    """
    fig = (
        df.set_index("date")["price"].plot(title=ticker, figsize=(20, 12)).get_figure()
    )
    ax = fig.gca()
    ax.set_title(ticker, fontsize=18)  # Set the title size to 18
    ax.set_xlabel("date", fontsize=18)  # Set the x-label and its size
    ax.set_ylabel("price", fontsize=18)  # Set the y-label and its size
    ax.tick_params(
        axis="both", which="major", labelsize=16
    )  # Adjust the axis tick label size
    fig.savefig(os.path.join(output_folder, "price-plots", f"{ticker}.png"));
    plt.close(fig)


def prepare_features(name: str, diagnostics: bool = True) -> None:
    # Load settings from YAML file
    yaml_file = os.path.join("nmp", "configs", "data_settings", name + ".yaml")
    with open(yaml_file, "r") as f:
        settings = yaml.safe_load(f)

    # Read CSV file based on settings
    price_series = pd.read_csv(settings["price_series_path"])

    price_series = price_series.rename(columns=settings["col_mappings"])

    # (((((PRCCD/AJEXDI)*TRFD)/((PRCCD(PRIOR)/AJEXDI(PRIOR))*TRFD(PRIOR))) -1 )* 100)
    if "ajexdi" in price_series:
        price_series["price"] = price_series["price"] / price_series["ajexdi"]
    
    if "trfd" in price_series:
        price_series["price"] = price_series["price"] * price_series["trfd"]

    price_series = price_series[["date", "ticker", "price"]].dropna()

    price_series["date"] = pd.to_datetime(price_series["date"])

    output_folder = os.path.join("datasets", name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if diagnostics and not os.path.exists(os.path.join(output_folder, "price-plots")):
        os.makedirs(os.path.join(output_folder, "price-plots"))

    price_series.to_csv(os.path.join(output_folder, "price_series.csv"), index=False)

    if diagnostics:
        # make plots for each price series by ticker
        for ticker in price_series["ticker"].unique():
            _plot_ticker(
                price_series[price_series["ticker"] == ticker], output_folder, ticker
            )

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read CSV file based on settings in YAML file"
    )
    parser.add_argument(
        "--yaml_file", help="Path to the YAML file", default="eurostoxx"
    )

    parser.add_argument(
        "--diagnostics",
        help="Whether to generate diagnostic plots",
        action="store_true",
    )

    args = parser.parse_args()

    prepare_features(args.yaml_file, args.diagnostics)
