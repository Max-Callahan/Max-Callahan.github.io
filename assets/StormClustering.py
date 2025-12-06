# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:21:29 2025

@author: mcallahan
"""
# -*- coding: utf-8 -*-
import os
import glob
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# ASOS hourly wind/pressure/temperature
wind_path = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\ACV.csv"

# NDBC waves (yearly folders)
waves_folder = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\waves"

start_year = 2017
end_year  = 2025


#precipitation
station_id = "USW00024213"
url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station_id}.csv"

resp = requests.get(url)
if resp.status_code != 200:
    raise RuntimeError(f"Could not download GHCN data: HTTP {resp.status_code}")

ghcn = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])

# filter years
ghcn = ghcn[(ghcn["DATE"].dt.year >= start_year) &
            (ghcn["DATE"].dt.year <= end_year)]

# rename + convert precip (tenths of mm -> mm)
ghcn = ghcn.rename(columns={
    "DATE": "date",
    "PRCP": "precip_mm",
})
ghcn["precip_mm"] = ghcn["precip_mm"] / 10.0  # tenths mm -> mm

precip_daily = ghcn[['date', 'precip_mm']] #narrowing down the original dataset to just the precipitation column


#hourly wind, temp, and pressure from arcata airport (ACV)
ACV_df = pd.read_csv(wind_path)

# only keep relavent columns
wind_clean = ACV_df[["valid", "tmpf", "drct", "sknt", "alti", "mslp"]].copy()
wind_clean["date"] = pd.to_datetime(wind_clean["valid"], errors="coerce")
wind_clean = wind_clean.dropna(subset=["valid"])

#convert to daily timestamp
wind_clean['days'] = wind_clean['date'].dt.floor("D")
# convert variable to numeric (fixing issue, I think they are stored as strings in the raw data)
wind_clean["sknt"] = pd.to_numeric(wind_clean["sknt"], errors="coerce")
wind_clean["mslp"] = pd.to_numeric(wind_clean["mslp"], errors="coerce")
wind_clean["tmpf"] = pd.to_numeric(wind_clean["tmpf"], errors="coerce")



#converting windspeed into daily means and maximums, and getting daily minimum pressure from hourly datasets
daily_met = (
    wind_clean
    .groupby("days")
    .agg(
        daily_max_sust_knots=("sknt", "max"),
        daily_mean_sust_knots=("sknt", "mean"),
        daily_min_press_hpa=("mslp", "min"),
    )
    .reset_index()
)

#converting to daily temperature
daily_temp = (
    wind_clean
    .groupby("days")["tmpf"]
    .mean()
    .reset_index()
)
daily_temp = daily_temp.rename(columns={"tmpf": "daily_mean_tmpf_F"})

# converting to datetime 
daily_met["date"] = pd.to_datetime(daily_met["days"])
daily_temp["date"] = pd.to_datetime(daily_temp["days"])

#Wave dataset
pattern = os.path.join(waves_folder, "*.txt.gz")
file_list = glob.glob(pattern)


dfs = []
for fpath in file_list:
    dfw = pd.read_csv(
        fpath,
        compression="gzip",
        delim_whitespace=True,
        skiprows=[1],    # skip first row in file (file has multiple columns names)
        na_values=["MM"] #null rows are marked with MM (for missing I guess) and so I had to account for that
    )

    # renaming year column to avoid the #
    if "#YY" in dfw.columns:
        dfw = dfw.rename(columns={"#YY": "YY"})

    # build datetime column
    dfw["time"] = pd.to_datetime(
        dict(
            year=dfw["YY"],
            month=dfw["MM"],
            day=dfw["DD"],
            hour=dfw["hh"],
            minute=dfw["mm"]
        ),
        errors="coerce"
    )
    

    dfs.append(dfw[["time", "WVHT"]].copy())

waves_all = pd.concat(dfs, ignore_index=True)
waves_all = waves_all.set_index("time").sort_index()

# filter by years
waves_all = waves_all[
    (waves_all.index.year >= start_year) &
    (waves_all.index.year <= end_year)
]

# daily mean significant wave height
daily_waves = waves_all["WVHT"].resample("D").mean().to_frame(name="WVHT_mean")
daily_waves = daily_waves.reset_index().rename(columns={"time": "date"})
daily_waves["date"] = pd.to_datetime(daily_waves["date"])

#merge the final weather dataset

weather_df = daily_met.merge(precip_daily, on="date", how="outer")

weather_df = weather_df.merge(daily_temp, on="date", how="outer")

weather_df = weather_df.merge(daily_waves, on="date", how="outer")

# sort by date
weather_df = weather_df.sort_values("date").reset_index(drop=True)




#HERE IS THE CLEANED AND MERGED DATASET SO THAT IT CAN RUN ON OTHER MACHINES
#weather_df = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_forcing_fulltimeseries.csv")

#make sure dat is datetime and the dataframe is in order
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df = weather_df.sort_values('date')
weather_df = weather_df.dropna()

features = ['daily_max_sust_knots', 'daily_mean_sust_knots', 'daily_min_press_hpa', 'precip_mm', 'daily_mean_tmpf_F', 'WVHT_mean']

#feature array
X = weather_df[features]

#standardize variables, I have disproportionately high pressure values vs wave height for example
X_std = StandardScaler().fit_transform(X)

from sklearn.metrics import silhouette_score
print("Silhouette scores: ")
for k in range (2,12):
    kmeans = KMeans(n_clusters = k, random_state = 0)
    labels = kmeans.fit_predict(X_std)
    score = silhouette_score(X_std, labels)
    print(k, score)
    
    
k = 4
kmeans = KMeans(n_clusters = k, random_state = 0)
weather_df['cluster'] = kmeans.fit_predict(X_std)

#finding which cluster respresents storms
clusters= weather_df.groupby('cluster')[features].mean()
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.max_rows', None)      # show all rows
pd.set_option('display.width', None)         # unlimited width
pd.set_option('display.float_format', '{:.5f}'.format)  # optional: full precision

print(clusters)

#0 is my storm clusters



plt.figure(figsize=(15,4))
plt.scatter(weather_df["date"], weather_df["cluster"], c=weather_df["cluster"], cmap="tab10", s=15)
plt.title("Weather Clusters (Automatically Detected Storm Days)")
plt.xlabel("Date")
plt.ylabel("Cluster ID")
plt.show()


#plotting storm days over beach width and weather data in 5 panel plot

beach_widths = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv")
beach_widths['date'] = pd.to_datetime(beach_widths['date'])
average_bw_perdate = (beach_widths.groupby('date')['width_m'].mean().reset_index())

clustered_storms = weather_df.loc[weather_df['cluster']==2, 'date'].sort_values().values



#pltting storms over the beach width time series
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# Beach width (averaged over all transects)
ax = axes[0]
ax.plot(average_bw_perdate["date"], average_bw_perdate["width_m"], lw=1.5, color="c", label="Daily Mean Beach Width (m)")
ax.set_ylabel("Width (m)")
ax.set_title("Samoa Beach: Beach Width and KMeans Derived Storm Days")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

#Max wind speed per day
ax = axes[1]
ax.plot(weather_df["date"], weather_df["daily_max_sust_knots"], lw=1.2, color="tab:orange", label="Mean Wind (kt)")
ax.set_ylabel("Mean Wind (kt)")
ax.set_title("Daily Maximum Sustained Wind")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# minimum Mean Sea Level Pressure (great storm indicator)
ax = axes[2]
ax.plot(weather_df["date"], weather_df["daily_min_press_hpa"], lw=1.2, color="tab:purple", label="Min MSLP (hPa)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title("Daily Minimum Average Sea-Level Pressure")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

#Precipitation (total per day), not super helpful for the kind of erosion-producing storms I am looking for 
#but it is a good indicator of bad weather
ax = axes[3]
ax.plot(weather_df["date"], weather_df["precip_mm"], lw=1.2, color="tab:blue", label="Precipitation (mm)")
ax.set_ylabel("Precip (mm)")
ax.set_title("Daily Precipitation (mm per day)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# average wave height 
ax = axes[4]
ax.plot(weather_df["date"], weather_df["WVHT_mean"],
        lw=1.2, color="tab:red", label="Average Wave Height (m)")
ax.set_ylabel("Mean Wave Height (m)")
ax.set_title("Daily Average Wave Height in Meters")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
ax.set_xlabel("Date")

# adding in storm days, derived from KMeans clustering 

for storm in clustered_storms:
    for ax in axes:
        ax.axvline(x=storm, color="red", alpha=0.15, linewidth=1, label = 'Storm')

plt.tight_layout()
plt.show()
