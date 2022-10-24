import json

cem_data = {"train": {}, "val": {}, "test": {}}
for split in cem_data.keys():
    with open("data/cem/" + split + "_data.json", "r") as f:
        cem_data[split] = json.load(f)

with open("data/cem/logs.json", "r") as f:
    logs = json.load(f)

with open("pred/cem/predictions.json", "r") as f:
    preds = json.load(f)

for log, pred in zip(logs, preds):
    assert log["cid"] == pred["context_id"] and log["tid"] == pred["turn_id"] and log["fid"] == pred["fact_id"]
    cem_data[log["source"]][log["cid"]]["utt_cs"][log["relation"]][log["fid"]][1] = pred["target"]

for split in cem_data.keys():
    with open("data/cem/" + split + "_data.json", "w") as f:
        json.dump(cem_data[split], f, indent=2)
