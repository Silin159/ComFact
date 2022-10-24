import json
from copy import deepcopy
import os

# import nltk
# nltk.download("punkt")

linking_data_path = "data/cem/rel_tail/nlg/test/"

relation_to_natural = {"xEffect": "as a result, PersonX will",
                       "xIntent": "because PersonX wants",
                       "xNeed": "but before, PersonX needs",
                       "xReact": "as a result, PersonX feels",
                       "xWant": "as a result, PersonX wants"}


def main():
    log_all = []
    label_all = []
    for split in ["train", "val", "test"]:

        print("Preprocessing Data in: " + linking_data_path)

        data_file = linking_data_path + split + "_data.json"

        with open(data_file, 'r') as f:
            cem_data = json.load(f)

        for cid, data_point in cem_data.items():
            log_context = {"source": split, "cid": cid, "tid": 0, "text": []}
            if len(data_point["context"]) > 3:
                log_context["text"].append({"type": "p_context", "utter": data_point["context"][-3].lower()})
                log_context["text"].append({"type": "p_context", "utter": data_point["context"][-2].lower()})
            else:
                for sent in data_point["context"][:-1]:
                    log_context["text"].append({"type": "p_context", "utter": sent.lower()})
            log_context["text"].append({"type": "center", "utter": data_point["context"][-1].lower()})

            for rel, tails in data_point["utt_cs"].items():
                for fid, tail in enumerate(tails):
                    log_single = deepcopy(log_context)
                    log_single["fid"] = fid
                    log_single["relation"] = rel
                    log_single["text"].append({"type": "fact", "utter": relation_to_natural[rel].lower()})
                    log_single["text"].append({"type": "fact", "utter": tail[0].lower()})
                    log_all.append(log_single)
                    label_all.append({"target": True, "linking": None})

    with open(linking_data_path + "logs.json", "w") as f:
        json.dump(log_all, f, indent=2)
    with open(linking_data_path + "labels.json", "w") as f:
        json.dump(label_all, f, indent=2)


if __name__ == "__main__":
    main()
