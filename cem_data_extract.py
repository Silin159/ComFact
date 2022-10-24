import pickle
import json
import os

with open("CEM/data/ED/dataset_preproc.p", "rb") as f:
    [data_tra, data_val, data_tst, vocab] = pickle.load(f)

print(data_tst["utt_cs"])

new_data_tra = {}
new_data_val = {}
new_data_test = {}
new_data = [new_data_tra, new_data_val, new_data_test]
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]

for sid, data_set in enumerate([data_tra, data_val, data_tst]):
    for did, commonsense in enumerate(data_set["utt_cs"]):
        new_data[sid][did] = {}
        new_data[sid][did]["context"] = []
        for ctx in data_set["context"][did]:
            new_data[sid][did]["context"].append(" ".join(ctx))
        for data_type in ["target", "situation", "emotion_context"]:
            new_data[sid][did][data_type] = " ".join(data_set[data_type][did])
        new_data[sid][did]["emotion"] = data_set["emotion"][did]
        new_data[sid][did]["utt_cs"] = {}
        for rid, r_cs in enumerate(commonsense):
            new_data[sid][did]["utt_cs"][relations[rid]] = []
            for r_cs_single in r_cs:
                new_data[sid][did]["utt_cs"][relations[rid]].append([" ".join(r_cs_single), None])

'''
print(data_tra["context"][0])
print(data_tra["target"][0])
print(data_tra["emotion"][0])
print(data_tra["situation"][0])
print(data_tra["emotion_context"][0])
print(data_tra["utt_cs"][0])
'''

linking_data_path = "data/cem/rel_tail/nlg/test/"
if not os.path.exists(linking_data_path):
    os.makedirs(linking_data_path)

with open(linking_data_path + "train_data.json", "w") as f:
    json.dump(new_data[0], f, indent=2)
with open(linking_data_path + "val_data.json", "w") as f:
    json.dump(new_data[1], f, indent=2)
with open(linking_data_path + "test_data.json", "w") as f:
    json.dump(new_data[2], f, indent=2)
