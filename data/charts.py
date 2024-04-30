import sys
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("..")

import data.bank as databank


def repackage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["num_of_nodes", "num_of_edges"])
    df["code"] = df["code"].apply(lambda s: s[5:])
    return (
        df.iloc[1:, :]
        .fillna(value=np.nan)
        .melt(
            id_vars=["code"],
            value_vars=[
                # "num_of_nodes",
                # "num_of_edges",
                # "max_deg",
                # "min_deg",
                "avg_deg",
                "median_deg",
                "std_deg",
            ],
            var_name="property",
        )
    )


def create_and_save_plot(key: databank.Dataset, df: pd.DataFrame):
    sns.catplot(data=df, kind="bar", x="code", y="value", hue="property", aspect=2.5)
    plt.xticks(rotation=75)
    plt.savefig("figs/{}.png".format(key))


if __name__ == "__main__":
    for key in databank.get_keys():
        print("Plotting {}...".format(key), end="")
        df = pd.read_csv("{}.csv".format(key))
        df = repackage_dataframe(df)
        create_and_save_plot(key, df)
        print(" done")
