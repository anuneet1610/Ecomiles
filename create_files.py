import pandas as pd
import pickle
import osmnx as ox
import networkx as nx
import numpy as np
import joblib
from scipy.spatial import cKDTree
from shapely.geometry import LineString

df = pd.read_csv("processed_air_quality_data_gurugram_repeated_ids (2).csv")

lat_lon_list = list(zip(df['latitude'], df['longitude']))

with open("lat_lon_list.pkl","wb") as f:
  pickle.dump(lat_lon_list,f)

G = ox.graph_from_place("Gurugram, Haryana, India")
station_coords = np.array([(lon, lat) for lat, lon in lat_lon_list])
station_tree = cKDTree(station_coords)
joblib.dump(station_tree, "kdtree.pkl")

with open("station_coord.pkl","wb") as f:
  pickle.dump(station_coords,f)

ox.save_graphml(G, "my_graph.graphml")
G=ox.load_graphml("my_graph.graphml")
station_tree=joblib.load("kdtree.pkl")
model=joblib.load("PM25_model.joblib")

data = {
    "G": G,
    "station_tree": joblib.load("kdtree.pkl"),
    "lat_lon_list": joblib.load("lat_lon_list.pkl"),
    "model": joblib.load("PM25_model.joblib")
}
joblib.dump(data, "all_data_ver_2.pkl")
