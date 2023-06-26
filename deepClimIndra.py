"""
deepClimIndra.py

Sandy Herho <sandy.herho@email.ucr.edu>
23/06/24
"""

# import libs & setups
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras_visualizer import visualizer
import warnings

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams['figure.dpi'] = 800
warnings.filterwarnings("ignore")

# create dir
os.system("mkdir ./data")
os.system("mkdir ./figs")

# plot DEM
gridIndr = pygmt.datasets.load_earth_relief(resolution="01s",
	region=[107.85, 108.6, -6.666667,-6.25])
fig = pygmt.Figure()
fig.grdimage(grid=gridIndr, projection="M15c", frame="a", cmap="geo")
fig.plot(x=108.3258, y=-6.3373, style="c0.5c",fill="red")
fig.colorbar(frame=["a2000", "x+lElevation", "y+lm"])
fig.savefig("./figs/fig1.png")

# EDA
ds = xr.open_dataset("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc").sel(time=slice("1981-01-01", "2022-12-01"))

da_sliced = ds.sel(longitude=slice(107.85, 108.6), 
                   latitude=slice(-6.666667, -6.25)) # Indramayu reg.
df_precip = da_sliced.mean(['longitude', 
                            'latitude'])["precip"].to_dataframe()
df_precip.to_csv("./data/monthlyIndraPr19812022.csv")
aveMonthPr = df_precip.groupby(df_precip.index.month)["precip"].mean()

fig, ax = plt.subplots()
aveMonthPr.plot(kind="bar", color="#0f82d4", rot=0, ax=ax);
ax.set_xlabel("Month", fontsize=25);
ax.set_ylabel("Precipitation (mm/month)", fontsize=25);

labels = ["Jan", "Feb", "Mar", "Apr", "May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

ax.set_xticklabels(labels);
fig.tight_layout();
fig.savefig("./figs/fig2.png")

# RAI calc.
annual_da_sliced = da_sliced.groupby('time.year').sum('time')

def rai(ds, dimension, method="ordinary"):
    """
    rai: calculate the rainfall anomaly index
    - ds: xarray dataset
    - dimension: dataset dim 
    - method: percentile (10th and 90th prctile), 
              ordinary (using mean lowesr & highest)
    """
    ds_ma = ds.mean(dimension)
    ds_anom = ds - ds_ma
    
    if method.lower() == "percentile":
        l_thresh = ds.reduce(np.nanpercentile,q=10,dim=dimension)
        u_thresh = ds.reduce(np.nanpercentile,q=90,dim=dimension)
        ds_low_10 = ds.where(ds<l_thresh).mean(dimension)
        ds_high_10 = ds.where(ds>u_thresh).mean(dimension)
    elif method.lower() == "ordinary":
        thresh = ds.reduce(np.sort,dim=dimension)
        ds_low_10 = thresh[:10].mean(dimension)
        ds_high_10 = thresh[:-10:-1].mean(dimension)
    else:
        print("No method selected")
    
    negatives = -3*((ds_anom.where(ds_anom<0)) / (ds_low_10-ds_ma))
    positives = 3*((ds_anom.where(ds_anom>0)) / (ds_high_10-ds_ma))
    RAI = ds_anom.where(ds_anom>=0, negatives).where(ds_anom<0, positives)
    
    return RAI

da_rai = rai(annual_da_sliced, 'year','percentile')
da_rai = da_rai.rename_vars(name_dict={"precip":"RAI"})
da_rai.to_netcdf("./data/annualRAIIndra19812022.nc")

df_rai = da_rai.mean(['longitude',
                      'latitude'])["RAI"].to_dataframe()
df_rai.to_csv("./data/annualAveSpaRAIIndra.csv")
df_rai.reset_index(inplace=True)

rai = df_rai["RAI"].to_numpy()
xtime = df_rai["year"].to_numpy()

fig = plt.figure(figsize=(15,8));
ax = fig.add_subplot(111);

ax.plot(xtime, rai, 'black', alpha=0.5, linewidth=2);
ax.axhline(0, color='black', lw=0.5);

ax.fill_between(xtime, 0, rai,
                where=rai>0,
                facecolor="#4287f5", interpolate=True)
ax.fill_between(xtime, 0, rai,
                where=rai<0,
                facecolor="#f21624", interpolate=True)

ax.set_xlim(xtime.min(), xtime.max());
ax.set_ylim(-5, 5)

ax.set_ylabel("RAI", fontsize=30);
ax.set_xlabel("Time (year)", fontsize=30);

fig.tight_layout();
fig.savefig("./figs/fig4.png")

print("RAI min (driest): {}, year: {}"\
      .format(round(df_rai["RAI"].min(), 2), 
              int(df_rai["year"][0] + df_rai["RAI"].argmin())))

print("RAI max (wettest): {}, year: {}"\
      .format(round(df_rai["RAI"].max(), 2), 
              int(df_rai["year"][0] + df_rai["RAI"].argmax())))

spa_min = da_rai.sel(year=int(df_rai["year"][0] + df_rai["RAI"].argmin()))
spa_max = da_rai.sel(year=int(df_rai["year"][0] + df_rai["RAI"].argmax()))

fig = plt.figure(figsize=(12, 5));
ax = fig.add_subplot(111, projection=ccrs.PlateCarree());
spa_min["RAI"].plot(ax=ax, transform=ccrs.PlateCarree(),
                    x='longitude', y='latitude', 
                    cmap="RdBu", levels=np.arange(-6, 6, .5))
ax.coastlines();
ax.gridlines();
ax.set_title("(a)");
fig.savefig("./figs/fig5a.png")

fig = plt.figure(figsize=(12, 5));
ax = fig.add_subplot(111, projection=ccrs.PlateCarree());
spa_max["RAI"].plot(ax=ax, transform=ccrs.PlateCarree(),
                    x='longitude', y='latitude', 
                    cmap="RdBu", levels=np.arange(-6, 6, .5))
ax.coastlines();
ax.gridlines();
ax.set_title("(b)");
fig.savefig("./figs/fig5b.png")

# DNN
df_rai = df_rai.rename(columns={"RAI":"IND"})
df_rai = df_rai.set_index("year")
regions=[df_rai.columns[0]]

## add polynomial feature
degree=3
poly = PolynomialFeatures(degree=degree, include_bias=False)
t = df_rai.index.values.reshape(-1, 1) # the feature
X = poly.fit_transform(t)
t_future = np.arange(2023, 2032 + 1).reshape(-1, 1)
X_future = poly.fit_transform(t_future)

## train-test split
train_size = int(.8 * len(X)) 
X_train = X[: train_size,:]
X_test = X[train_size:,:]

print("Train size: ", X_train.shape[0]) # 1981 - 2013
print("Test size: ", X_test.shape[0]) # 2014 - 2022

y_train_indicator = pd.DataFrame(index=df_rai.index[:train_size]) 
for i in regions:
    y_train_indicator[i] = df_rai.loc[:(df_rai.index[0] + train_size), i]

y_test_indicator = pd.DataFrame(index=df_rai.index[train_size:])
for i in regions:
    y_test_indicator[i] = df_rai.loc[(df_rai.index[0] + train_size):, i]

errors_indicator_test = pd.DataFrame(index=regions, columns=["deepnn"])## scaling the features & target
x_scaler = StandardScaler().fit(X_train) # train scaling
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test) # test scaling

x_scaler_all = StandardScaler().fit(X) # scaling all dataset
X_scaled = x_scaler_all.transform(X)

X_future_scaled = x_scaler_all.transform(X_future) # scaling for proj.

y_scaler_indicator = dict() 
for i in regions:
    y_scaler_indicator[i] = StandardScaler().fit(y_train_indicator[i]\
                                                 .values.reshape(-1,1))
y_train_scaled_indicator = pd.DataFrame(index=y_train_indicator.index)
for i in regions:
    y_train_scaled_indicator[i] = y_scaler_indicator[i].\
    transform(y_train_indicator[i]\
              .values.reshape(-1,1))
    
y_test_scaled_indicator = pd.DataFrame(index=y_test_indicator.index)
for i in regions:
    y_test_scaled_indicator[i] = y_scaler_indicator[i]\
    .transform(y_test_indicator[i]\
               .values.reshape(-1,1))
    
y_scaler_all_indicator = dict()
for i in regions:
    y_scaler_all_indicator[i] = StandardScaler()\
    .fit(df_rai[i].values.reshape(-1,1))
    
y_scaled_indicator = pd.DataFrame(index=df_rai.index)
for i in regions:
    y_scaled_indicator[i] = y_scaler_all_indicator[i]\
    .transform(df_rai[i].values.reshape(-1,1))

## model development & fitting to training dataset
deep_model_indicator = dict()
tf.random.set_seed(212) # alumni 212

for i in regions:
    deep_model_indicator[i] = Sequential([
        Dense(units=300, activation="relu"),
        Dense(units=300, activation="relu"),
        Dense(units=300, activation="relu"),
        Dense(units=1)
    ])
    
for i in regions:
    deep_model_indicator[i].compile(optimizer=Adam(learning_rate=.001),
                                    loss='mean_squared_error')

for i in regions:
    history = deep_model_indicator[i].fit(X_train_scaled,
                                y_train_scaled_indicator[i], 
                                epochs=100, batch_size=8, 
                                verbose=0)

epoch = np.array(history.epoch) + 1
mse = history.history["loss"]

plt.figure();
plt.plot(epoch, mse, linestyle="--", marker="o", linewidth=2, color="k");
plt.xlim([0, 100]);
plt.ylabel("Training MSE", fontsize=30);
plt.xlabel("Epochs", fontsize=30);
plt.tight_layout();
plt.savefig("./figs/fig6.png")

print(deep_model_indicator["IND"].summary())

for layer in deep_model_indicator["IND"].layers:
    print("weights: ", np.prod(layer.weights[0].shape), "biases: ", layer.weights[1].shape[0])

visualizer(deep_model_indicator["IND"], file_format='png', view=False)

os.system("mv graph.png fig3.png | mv fig3.png ./figs/")

## test prediction
predictions_indicator_dictionary = dict()

for i in regions:
    predictions_indicator_dictionary[i] = pd.DataFrame(index=df_rai.index, columns=["deepnn"])

for i in regions:
    predictions_indicator_dictionary[i]["deepnn"].iloc[:train_size] = y_scaler_indicator[i]\
    .inverse_transform(deep_model_indicator[i](X_train_scaled)\
                       .numpy()).flatten()
for i in regions:
    predictions_indicator_dictionary[i]["deepnn"].iloc[train_size:] = y_scaler_indicator[i]\
    .inverse_transform(deep_model_indicator[i](X_test_scaled)\
                       .numpy()).flatten()

predictions_indicator_dictionary["IND"].to_csv("./data/predAnnualAveSpaRAIIndra.csv")
pred = predictions_indicator_dictionary["IND"].reset_index()
pred_y = pred["deepnn"]
real = df_rai["IND"]
t = pred["year"]

plt.figure();
plt.plot(t, pred_y, linewidth=2, marker="o", linestyle="--", label="DNN");
plt.plot(t, real, linewidth=2, marker="o", linestyle="--", label="actual data");
plt.xlabel("Time (year)", fontsize=30);
plt.ylabel("RAI", fontsize=30);
plt.tight_layout();
plt.legend();
plt.savefig("./figs/fig7.png")

## calculate RMSE
y_train_true = df_rai["IND"][:train_size].to_numpy()
y_train_pred = predictions_indicator_dictionary["IND"]["deepnn"][:train_size].to_numpy()
rmse_train = round(mean_squared_error(y_true=y_train_true, y_pred=y_train_pred, squared=False)*100, 2)

y_test_true = df_rai["IND"][train_size:].to_numpy()
y_test_pred = predictions_indicator_dictionary["IND"]["deepnn"][train_size:].to_numpy()
rmse_test = round(mean_squared_error(y_true=y_test_true, y_pred=y_test_pred, squared=False)*100, 2)

rmse_diff = round((rmse_test - rmse_train) / rmse_train * 100, 2)

print("RMSE train: {}%, RMSE test: {}%, RMSE difference (test - train): {}%"\
      .format(rmse_train, rmse_test, rmse_diff))

## naive model test-set comparison
df_naive_indicator = df_rai.shift(1)

for i in regions:
    err = mean_absolute_percentage_error(y_true = y_test_indicator[i],
                                         y_pred=df_naive_indicator[i].iloc[train_size:])
    print(f"residency={i}, err={round(err*100, 2)}%") # test
