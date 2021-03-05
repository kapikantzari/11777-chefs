import pandas as pd
import numpy as np

# path = "/content/drive/MyDrive/datasets/epic-kitchens-100-annotations-master/"
# # path = "/content/drive/MyDrive/EpicKitchen_Dataset/epic-kitchens-100-annotations-master/"
# train_path = path + "EPIC_100_train.csv"
# val_path = path + "EPIC_100_validation.csv"
# train_data = pd.read_csv(train_path, sep=',')
# val_data = pd.read_csv(val_path, sep=',')

# train_dict = dict()
# val_dict = dict()
# for i in range(97):
#   train_dict[i] = dict()
# for i in range(94):
#   val_dict[i] = dict()

def process(all_data, d):
  video_ids = all_data["video_id"].to_numpy()
  start_frames = all_data["start_frame"].to_numpy()
  stop_frames = all_data["stop_frame"].to_numpy()
  verbs = all_data["verb"].to_numpy()
  verb_classes = all_data["verb_class"].to_numpy()
  for i in range(video_ids.size):
    cur_d = d[verb_classes[i]]
    v = verbs[i]
    cur_d[v] = cur_d.get(v, list())
    cur_d[v].append((video_ids[i], start_frames[i], stop_frames[i]))

# process(train_data, train_dict)
# process(val_data, val_dict)
