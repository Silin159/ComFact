import pickle
import json
from copy import deepcopy

new_data = {"train": {}, "val": {}, "test": {}}
for split in new_data.keys():
    with open("data/cem/" + split + "_data.json", "r") as f:
        new_data[split] = json.load(f)

with open("CEM/data/ED/dataset_preproc.p", "rb") as f:
    [_, _, _, vocab] = pickle.load(f)

data_cell = {"context": [], "target": [], "situation": [], "emotion_context": [], "emotion": [], "utt_cs": []}
cem_data = {"train": deepcopy(data_cell), "val": deepcopy(data_cell), "test": deepcopy(data_cell)}
for split, new_d_dict in new_data.items():
    for did, new_d in new_d_dict.items():
        context = []
        for ctx in new_d["context"]:
            context.append(ctx.split(" "))
        cem_data[split]["context"].append(deepcopy(context))
        for data_type in ["target", "situation", "emotion_context"]:
            cem_data[split][data_type].append(new_d[data_type].split(" "))
        cem_data[split]["emotion"].append(new_d["emotion"])
        utt_cs = []
        for rel, tails in new_d["utt_cs"].items():
            utt_cs_single = []
            for tail in tails:
                if tail[1] and tail[0] != "none":
                    utt_cs_single.append(tail[0].split(" "))
            # while len(utt_cs_single) < 5:
            #     utt_cs_single.append(["UNK"])
            utt_cs.append(deepcopy(utt_cs_single))

        cem_data[split]["utt_cs"].append(deepcopy(utt_cs))

with open("CEM/data/ED/dataset_preproc_link.p", "wb") as f:
    pickle.dump([cem_data["train"], cem_data["val"], cem_data["test"], vocab], f)
