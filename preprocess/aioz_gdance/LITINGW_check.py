import pickle
import numpy as np

# load '.pkl'
#file_path = "./train/motions_sliced/_P-JWcq1ewI_02_0_750_slice0.pkl"
file_path = "./train/motions/_P-JWcq1ewI_02_0_750.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

#  item & size
print("Items and their shapes:")
for key in data:
    print(f"Item: {key}")
    if isinstance(data[key], np.ndarray):  
        print(f" - Shape: {data[key].shape}")
    else:
        print(f" - Type: {type(data[key])}")
        print(data[key])

print(f" smpl_poses-example: {data['q'][0][0]}") 
print(f" root_trans-example: {data['pos'][0][0]}")  #  TRANS: [-1.74902737 -0.26801854 -2.56446457]
