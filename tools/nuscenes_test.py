import pickle
import sys
print(sys.path)

with open('/workspace/1B0FF2CE2A57439F/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())