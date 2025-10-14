import torch
import os
import pickle
import torch
import netCDF4 as nc
from functools import partial
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
"""
feature map is like this:
{'static_feats_numeric': ['latitude', 'longitude'],
 'static_feats_categorical': [],
 'historical_ts_numeric': ['tmin',
  'tmax',
  'precipitation',
  'radiation',
  'photoperiod',
  'swvl1',
  'doy',
  'sif_clear_inst'],
 'historical_ts_categorical': [],
 'future_ts_numeric': ['doy'],
 'future_ts_categorical': []}
"""
def load_nc_variables(nc_path):
    """
    Opens a NetCDF (.nc) file and retrieves 8 variables:
    ta_min, ta_max, pr, sw, lai, soilm, doy, photo

    Returns a dictionary with variable names as keys and numpy arrays as values.
    """
    var_names = ['ta_min', 'ta_max', 'pr', 'sw', 'lai', 'soilm', 'doy', 'photo']
    data = {}
    with nc.Dataset(nc_path, 'r') as ds:
        for var in var_names:
            if var in ds.variables:
                data[var] = ds.variables[var][:]
            else:
                raise KeyError(f"Variable '{var}' not found in {nc_path}")
    return data


#### Forcing Data Path
data_path = "/glade/u/home/linnia/MLphenology/US-MMS_forcing_1999-2014.nc"
raw_data = load_nc_variables(data_path)
print("raw LAI shape:", raw_data['lai'].shape)
print("raw LAI ( first 10 days):", raw_data['lai'][:10])


# Open and inspect the pickle file
pkl_path = "/glade/u/home/ayal/phenology-ml-clm/data/processed/US-MMS_forcing_1999-2014.pkl"
with open(pkl_path, "rb") as f:
    pkl_data = pickle.load(f)



    print("Keys:", list(pkl_data.keys()))
    print("data_sets keys:", list(pkl_data['data_sets'].keys()))
    print("data_sets keys:", list(pkl_data['data_sets']['train'].keys()))
    print("processed lai preview:", pkl_data['data_sets']['train']['historical_ts_numeric'][:, :, 7][0][:10])  # Assuming lai is the 8th variable (index 7)
    print("scalers keys:", list(pkl_data['scalers']['numeric'].keys()))

scaled_back = StandardScaler.inverse_transform(pkl_data['scalers']['numeric']['sif_clear_inst'], pkl_data['data_sets']['train']['historical_ts_numeric'][:, :, 7])
print("StandScaler scaled back lai :", scaled_back[0][:10])
#  scalers["numeric"][col] = StandardScaler().fit(only_train[col].values.astype(float).reshape(-1, 1))
"""
#### DATA 
B, T_hist, T_fut = 1, 60, 10
# static
static_num = torch.randn(B, 2)        #lat,long   
static_cat = torch.empty(B, 0, dtype=torch.long)
# historical
hist_num   = torch.randn(B, T_hist, 8) # tmin, tmax, prcp, srad, swc, photoperiod, doy, lai
hist_cat   = torch.empty(B, 0, dtype=torch.long)
# future
fut_num    = torch.randn(B, T_fut, 1)  
fut_cat    = torch.empty(B, 0, dtype=torch.long)

print("static_num shape:", static_num.shape, "type:", type(static_num))
print("static_cat shape:", static_cat.shape, "type:", type(static_cat))
print("hist_num shape:", hist_num.shape, "type:", type(hist_num))
print("hist_cat shape:", hist_cat.shape, "type:", type(hist_cat))
print("fut_num shape:", fut_num.shape, "type:", type(fut_num))
print("fut_cat shape:", fut_cat.shape, "type:", type(fut_cat))
"""

# Adapt shapes/types from pkl_data['data_sets']['train'] to tensors for model input

train_data = pkl_data['data_sets']['train']

# static numeric: shape (B, 2)
static_num = torch.tensor(train_data['static_feats_numeric'], dtype=torch.float32)
# static categorical: shape (B, 0)
static_cat = torch.empty(static_num.shape[0], 0, dtype=torch.long)

# historical numeric: shape (B, T_hist, 8)
hist_num = torch.tensor(train_data['historical_ts_numeric'], dtype=torch.float32)
# historical categorical: shape (B, 0)
hist_cat = torch.empty(hist_num.shape[0], 0, dtype=torch.long)

# future numeric: shape (B, T_fut, 1)
fut_num = torch.tensor(train_data['future_ts_numeric'], dtype=torch.float32)
# future categorical: shape (B, 0)
fut_cat = torch.empty(fut_num.shape[0], 0, dtype=torch.long)

print("static_num shape:", static_num.shape, "type:", type(static_num))
print("static_cat shape:", static_cat.shape, "type:", type(static_cat))
print("hist_num shape:", hist_num.shape, "type:", type(hist_num))
print("hist_cat shape:", hist_cat.shape, "type:", type(hist_cat))
print("fut_num shape:", fut_num.shape, "type:", type(fut_num))
print("fut_cat shape:", fut_cat.shape, "type:", type(fut_cat))



#### Model Path
checkpoint_path = "/glade/u/home/ayal/phenology-ml-clm/models/tft_scripted.pt"
#### Model Configuration
configuration_test = {
    "task_type": "regression",
    "target_window_start": None,
    "data_props": {'num_historical_numeric': 8, #tmin, tmax, prcp, srad, swc, photoperiod, doy, lai,
  'num_historical_categorical': 0,
  'num_static_numeric': 2, # lat, lon
  'num_static_categorical': 0,
  'num_future_numeric': 1, # doy
  'num_future_categorical': 0,
  'historical_categorical_cardinalities': [],
  'static_categorical_cardinalities': [],
  'future_categorical_cardinalities': []},
    "model": {
        "attention_heads": 4,
        "dropout": 0.2,
        "lstm_layers": 4,
        "state_size": 160,
        "output_quantiles": [0.1, 0.5, 0.9],
    },
}

### Load the model
model = torch.jit.load(checkpoint_path, map_location="cpu")
model.eval()

with torch.no_grad():
    # returns (B, T_fut, 3)
    out = model(static_num,
                   static_cat,
                   hist_num,
                   hist_cat,
                   fut_num,
                   fut_cat)
    
median_prediction = out[:, :, 1]

print("output shape:", median_prediction.shape, "type:", type(median_prediction))
print("PRED ( first 10 days):", median_prediction[0][:10])

pred_scaled = StandardScaler.inverse_transform(pkl_data['scalers']['numeric']['sif_clear_inst'], median_prediction)
print("StandScaler scaled back prediction :", pred_scaled[0][:10])
print("raw LAI = target:", raw_data['lai'][60:70])
print("processed target", pkl_data['data_sets']['train']['target'][0])  # Assuming lai is the 8th variable (index 7)
target_scaled = StandardScaler.inverse_transform(pkl_data['scalers']['numeric']['sif_clear_inst'],  pkl_data['data_sets']['train']['target'])
print("StandScaler scaled back target :", target_scaled[0][:10])


# Flatten predictions and targets for metrics
y_pred = pred_scaled.flatten()
y_true = target_scaled.flatten()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")