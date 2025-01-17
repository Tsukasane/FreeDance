import pickle
with open("mean_std.pkl", "rb") as f:
    data = pickle.load(f)
mean_loaded = data["mean"]
std_loaded = data["std"]
print("----------mean_loaded", mean_loaded.shape)  
print("mean_loaded:", mean_loaded)  
print("----------std_loaded", std_loaded.shape)
print("std_loaded:", std_loaded)