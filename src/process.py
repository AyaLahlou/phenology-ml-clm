import os
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import (
    QuantileTransformer,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
)
import argparse


def sort_loc_time(data_path: str, output_path: str):
    """
    Sort the data by location and time and save to new file.

    Args:
        data_path: Path to the input data file
        output_path: Path to save the sorted data file
    """
    data_df = pd.read_parquet(data_path)

    sorting_order = {
        "location": "asc",
        "time": "asc",
    }

    sorting_values = {
        col: (True if order == "asc" else False) for col, order in sorting_order.items()
    }

    data_df = data_df.sort_values(
        by=list(sorting_order.keys()), ascending=list(sorting_values.values())
    )
    data_df.to_parquet(output_path)


def tft_process(path, hist_len, fut_len, samp_interval, output_path, test_mode=False):
    """
    Processes time series data from NetCDF file for Temporal Fusion Transformer (TFT) model training.

    This function reads a NetCDF file containing time series data with variables:
    ta_min, ta_max, pr, sw, lai, soilm, doy, photo and coordinates latitude, longitude.
    It applies preprocessing steps including feature selection, missing value removal, 
    scaling/encoding, and splits the data into sliding windows for model input.

    Args:
        path (str): Path to the input NetCDF data file.
        hist_len (int): Length of the historical time series window.
        fut_len (int): Length of the future time series window.
        samp_interval (int): Step size for sliding window sampling.
        output_path (str): Directory path to save the processed output file.
        test_mode (bool): Whether to use test mode (40-year test) or regular training split.
    """
    output_filename = path.split("/")[-1].split(".")[0] + ".pkl"
    
    # Load NetCDF data
    ds = xr.open_dataset(path)
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset coordinates: {list(ds.coords)}")
    
    # Convert to DataFrame
    # Flatten spatial dimensions and create location index
    data_df = ds.to_dataframe().reset_index()
    
    # Create location identifier from lat/lon
    data_df['location'] = data_df['latitude'].astype(str) + '_' + data_df['longitude'].astype(str)
    
    # Standardize column names to match expected format
    column_mapping = {
        'ta_min': 'tmin',
        'ta_max': 'tmax', 
        'pr': 'precipitation',
        'sw': 'radiation',
        'photo': 'photoperiod',
        'soilm': 'swvl1',
        'lai': 'sif_clear_inst'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in data_df.columns:
            data_df = data_df.rename(columns={old_name: new_name})
    
    print(f"Columns after renaming: {data_df.columns}")

    # Select required columns
    required_columns = [
        "time",
        "location", 
        "latitude",
        "longitude",
        "tmin",
        "tmax", 
        "precipitation",
        "radiation",
        "photoperiod",
        "swvl1",
        "doy",
        "sif_clear_inst"
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    data_df = data_df[required_columns]

    print(f"Length of sorted parquet file: {len(data_df)}")
    data_df = data_df.dropna()
    data_df["time"] = pd.to_datetime(data_df["time"])

    print(f"Time range: {data_df['time'].min()} to {data_df['time'].max()}")

    # Define feature categories according to the specified feature map
    meta_attrs = ["time", "location", "id"]
    static_attrs = ["latitude", "longitude"]
    historical_attrs = ["tmin", "tmax", "precipitation", "radiation", "photoperiod", "swvl1", "doy", "sif_clear_inst"]
    future_attrs = ["doy"]
    categorical_attrs = []  # No categorical attributes in this case

    unique_locations = data_df["location"].unique()
    print(f"{len(unique_locations)} unique locations.")

    all_cols = list(data_df.columns)
    feature_cols = [col for col in all_cols if col not in meta_attrs]

    # Create feature map matching the specified structure
    feature_map = {
        "static_feats_numeric": ["latitude", "longitude"],
        "static_feats_categorical": [],
        "historical_ts_numeric": ["tmin", "tmax", "precipitation", "radiation", "photoperiod", "swvl1", "doy", "sif_clear_inst"],
        "historical_ts_categorical": [],
        "future_ts_numeric": ["doy"],
        "future_ts_categorical": []
    }

    # Initialize scalers
    scalers = {"numeric": dict()}
    categorical_cardinalities = dict()

    only_train = data_df

    # Fit scalers for numeric features only
    for col in tqdm(feature_cols, desc="Fitting scalers"):
        if col not in categorical_attrs:  # All features are numeric
            if col == "sif_clear_inst":
                scalers["numeric"][col] = StandardScaler().fit(
                    only_train[col].values.astype(float).reshape(-1, 1)
                )
            elif col == "doy":
                scalers["numeric"][col] = MinMaxScaler().fit(
                    only_train[col].values.astype(float).reshape(-1, 1)
                )
            else:
                scalers["numeric"][col] = QuantileTransformer(n_quantiles=256).fit(
                    only_train[col].values.astype(float).reshape(-1, 1)
                )

    # Transform data
    for col in tqdm(feature_cols, desc="Transforming data"):
        if col not in categorical_attrs:  # All features are numeric
            data_df[col] = (
                scalers["numeric"][col]
                .transform(data_df[col].values.reshape(-1, 1))
                .squeeze()
            )
            data_df[col] = data_df[col].astype(np.float32)

    print(
        f"Time range after processing: {data_df['time'].min()} to {data_df['time'].max()}"
    )

    # Create data splits with the requested time periods
    if not test_mode:
        # Training: 1999-2006, Validation: 2006-2007, Test: 2007-2014
        train_subset = data_df[
            (data_df["time"] >= datetime(1999, 1, 1))
            & (data_df["time"] < datetime(2006, 1, 1))
        ]
        validation_subset = data_df[
            (data_df["time"] >= datetime(2006, 1, 1))
            & (data_df["time"] < datetime(2007, 1, 1))
        ]
        test_subset = data_df[
            (data_df["time"] >= datetime(2007, 1, 1))
            & (data_df["time"] < datetime(2014, 1, 1))
        ]

        print(f"{len(train_subset['location'].unique())} train unique locations.")
        print(
            f"{len(validation_subset['location'].unique())} validation unique locations."
        )
    else:
        # 40-year test - keep original logic for compatibility
        train_subset = pd.DataFrame()  # Empty for test mode
        validation_subset = pd.DataFrame()  # Empty for test mode
        test_subset = data_df[
            (data_df["time"] >= datetime(1982, 1, 1))
            & (data_df["time"] < datetime(2022, 1, 1))
        ]

    print(f"{len(test_subset['location'].unique())} test unique locations.")

    subsets_dict = {
        "train": train_subset,
        "validation": validation_subset,
        "test": test_subset,
    }

    data_sets = {"train": dict(), "validation": dict(), "test": dict()}

    # Add ID column
    for subset_name, subset_data in subsets_dict.items():
        if not subset_data.empty:
            subset_data["id"] = (
                subset_data["location"].astype(str)
                + "_"
                + subset_data["time"].astype(str)
            )

    # Process sliding windows
    for subset_key, subset_data in subsets_dict.items():
        if subset_data.empty:
            continue

        print(f"Processing {subset_key} subset...")

        for i in tqdm(
            range(0, len(subset_data) - hist_len - fut_len + 1, samp_interval)
        ):
            slc = subset_data.iloc[i : i + hist_len + fut_len]

            # Validate slice
            if (
                len(slc) < (hist_len + fut_len)
                or slc.iloc[0]["location"] != slc.iloc[-1]["location"]
                or (slc.iloc[-1]["time"] - slc.iloc[0]["time"])
                > timedelta(days=hist_len + fut_len)
            ):
                continue

            # Store time index
            data_sets[subset_key].setdefault("time_index", []).append(
                slc.iloc[hist_len - 1]["location"]
            )

            # Static attributes
            data_sets[subset_key].setdefault("static_feats_numeric", []).append(
                slc.iloc[0][feature_map["static_feats_numeric"]].values.astype(
                    np.float32
                )
            )
            data_sets[subset_key].setdefault("static_feats_categorical", []).append(
                slc.iloc[0][feature_map["static_feats_categorical"]].values.astype(
                    np.int32
                )
            )

            # Historical time series
            data_sets[subset_key].setdefault("historical_ts_numeric", []).append(
                slc.iloc[:hist_len][feature_map["historical_ts_numeric"]]
                .values.astype(np.float32)
                .reshape(hist_len, -1)
            )
            data_sets[subset_key].setdefault("historical_ts_categorical", []).append(
                slc.iloc[:hist_len][feature_map["historical_ts_categorical"]]
                .values.astype(np.int32)
                .reshape(hist_len, -1)
            )

            # Future time series
            data_sets[subset_key].setdefault("future_ts_numeric", []).append(
                slc.iloc[hist_len:][feature_map["future_ts_numeric"]]
                .values.astype(np.float32)
                .reshape(fut_len, -1)
            )
            data_sets[subset_key].setdefault("future_ts_categorical", []).append(
                slc.iloc[hist_len:][feature_map["future_ts_categorical"]]
                .values.astype(np.int32)
                .reshape(fut_len, -1)
            )

            # Target
            data_sets[subset_key].setdefault("target", []).append(
                slc.iloc[hist_len:]["sif_clear_inst"].values.astype(np.float32)
            )
            data_sets[subset_key].setdefault("id", []).append(
                slc.iloc[hist_len:]["id"].values.astype(str)
            )

    # Convert lists to numpy arrays
    print("Converting to arrays...")
    for set_key in data_sets.keys():
        for arr_key in data_sets[set_key].keys():
            if (
                isinstance(data_sets[set_key][arr_key], list)
                and data_sets[set_key][arr_key]
            ):
                if isinstance(data_sets[set_key][arr_key][0], np.ndarray):
                    data_sets[set_key][arr_key] = np.stack(
                        data_sets[set_key][arr_key], axis=0
                    )
                else:
                    data_sets[set_key][arr_key] = np.array(data_sets[set_key][arr_key])

    # Ensure output directory exists
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save processed data
    with open(os.path.join(output_path, output_filename), "wb") as f:
        pickle.dump(
            {
                "data_sets": data_sets,
                "feature_map": feature_map,
                "scalers": scalers,
                "categorical_cardinalities": categorical_cardinalities,
            },
            f,
            pickle.HIGHEST_PROTOCOL,
        )
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Process time series data from NetCDF file for TFT model training."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input NetCDF file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save output"
    )
    parser.add_argument(
        "--hist_len", type=int, default=60, help="Historical window length"
    )
    parser.add_argument("--fut_len", type=int, default=10, help="Future window length")
    parser.add_argument(
        "--samp_interval",
        type=int,
        default=10,
        help="Step size for sliding window sampling"
    )
    parser.add_argument(
        "--test_mode", action="store_true", help="Use test mode (40-year test)"
    )

    args = parser.parse_args()

    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Historical length: {args.hist_len}")
    print(f"Future length: {args.fut_len}")
    print(f"Sampling interval: {args.samp_interval}")
    print(f"Test mode: {args.test_mode}")

    tft_process(
        args.data_path,
        args.hist_len,
        args.fut_len,
        args.samp_interval,
        args.output_path,
        args.test_mode,
    )


if __name__ == "__main__":
    main()
