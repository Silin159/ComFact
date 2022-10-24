import json
from copy import deepcopy
import argparse
import os
from baseline.models import Tokenizer

linking_data_path = "data/"
linking_pred_path = "pred/"

fact_linking_data_file = {"persona": "persona_atomic_final_123.json", "roc": "roc_atomic_final_328.json",
                          "movie": "moviesum_atomic_final_81.json", "mutual": "mutual_atomic_final_237.json"}
fact_linking_id_file = {"persona": {"train": "persona_atomic_did_train_90.json", "val": "persona_atomic_did_val_15.json",
                                    "test": "persona_atomic_did_test_18.json"},
                        "roc": {"train": "done_sid_train_235.json", "val": "done_sid_dev_46.json",
                                "test": "done_sid_test_47.json"},
                        "movie": {"train": "done_mid_train_58.json", "val": "done_mid_dev_11.json",
                                  "test": "done_mid_test_12.json"},
                        "mutual": {"train": "mutual_atomic_did_train_170.json", "val": "mutual_atomic_did_val_33.json",
                                   "test": "mutual_atomic_did_test_34.json"}
                        }

relation_to_natural = {"AtLocation": "located or found at/in/on",
                       "CapableOf": "is/are capable of",
                       "Causes": "causes",
                       "CausesDesire": "makes someone want",
                       "CreatedBy": "is created by",
                       "Desires": "desires",
                       "HasA": "has, possesses or contains",
                       "HasFirstSubevent": "begins with the event/action",
                       "HasLastSubevent": "ends with the event/action",
                       "HasPrerequisite": "to do this, one requires",
                       "HasProperty": "can be characterized by being/having",
                       "HasSubEvent": "includes the event/action",
                       "HinderedBy": "can be hindered by",
                       "InstanceOf": "is an example/instance of",
                       "isAfter": "happens after",
                       "isBefore": "happens before",
                       "isFilledBy": "___ can be filled by",
                       "MadeOf": "is made of",
                       "MadeUpOf": "made (up) of",
                       "MotivatedByGoal": "is a step towards accomplishing the goal",
                       "NotDesires": "do(es) not desire",
                       "ObjectUse": "used for",
                       "UsedFor": "used for",
                       "oEffect": "as a result, PersonY or others will",
                       "oReact": "as a result, PersonY or others feels",
                       "oWant": "as a result, PersonY or others wants",
                       "PartOf": "is a part of",
                       "ReceivesAction": "can receive or be affected by the action",
                       "xAttr": "PersonX is seen as",
                       "xEffect": "as a result, PersonX will",
                       "xIntent": "because PersonX wants",
                       "xNeed": "but before, PersonX needs",
                       "xReact": "as a result, PersonX feels",
                       "xReason": "because",
                       "xWant": "as a result, PersonX wants"}


def prepare_pipeline_test_data(model, window, portion):
    data_file = fact_linking_data_file[portion]
    did_file = fact_linking_id_file[portion]["test"]

    with open(linking_data_path + portion + "/" + data_file, 'r') as f:
        linking_raw_data = json.load(f)
    with open(linking_data_path + portion + "/" + did_file, 'r') as f:
        linking_cid = json.load(f)

    head_rel_pred = {}
    head_log_path = linking_data_path + portion + "/head/" + window + "/test/logs.json"
    head_label_path = linking_pred_path + portion + "-" + model + "-" + window + "-head-test/predictions.json"
    with open(head_log_path, 'r') as f:
        hl_logs = json.load(f)
    with open(head_label_path, 'r') as f:
        hl_labels = json.load(f)
    for log, label in zip(hl_logs, hl_labels):
        head_rel_pred[log["text"][-1]["utter"]] = label["target"]

    log = []
    label = []
    pipe_log_id = []
    head_log_id = []
    hid = 0

    for cid in linking_cid:
        max_turn = len(linking_raw_data[str(cid)]["text"])

        for tid, fact_turn in linking_raw_data[str(cid)]["facts"].items():
            sample = {"cid": str(cid), "tid": int(tid), "text": []}
            left = max(0, int(tid) - 2)
            if window == "nlu":
                right = min(max_turn, int(tid) + 3)
            elif window == "nlg":
                right = int(tid) + 1
            else:
                raise ValueError("window not in ['nlu', 'nlg']")

            for utter in linking_raw_data[str(cid)]["text"][left:right]:
                sample["text"].append({"type": "context", "utter": utter.lower()})

            for head, triples in fact_turn.items():
                head_start_id = deepcopy(hid)
                for fid, rt in enumerate(triples["triples"]):
                    sample_single = deepcopy(sample)
                    sample_single["fid"] = fid
                    sample_single["text"].append({"type": "fact", "utter": head.lower()})
                    sample_single["text"].append({"type": "fact", "utter": relation_to_natural[rt["relation"]].lower()})
                    sample_single["text"].append({"type": "fact", "utter": rt["tail"].lower()})

                    head_relevant_pipe = head_rel_pred[head.lower()]
                    head_relevant_gold = (triples["confidence"] > 0.49)
                    if not head_relevant_gold:
                        if head_relevant_pipe:
                            log.append(sample_single)
                            pipe_log_id.append(hid)
                            label.append({"target": False, "linking": None})
                        hid += 1
                    else:
                        relevance, relation = rt["final"], rt["relationship"]
                        if relevance in ["always", "sometimes", "not"]:
                            if head_relevant_pipe:
                                log.append(sample_single)
                                pipe_log_id.append(hid)
                                if relevance in ["always", "sometimes"]:
                                    label.append({"target": True, "linking": relation})
                                else:
                                    label.append({"target": False, "linking": None})
                            hid += 1

                head_end_id = deepcopy(hid)
                head_log_id.append([head_start_id, head_end_id])
    return log, label, pipe_log_id, head_log_id


def main():
    parser = argparse.ArgumentParser()

    # [bert-base, bert-large, roberta-base, roberta-large, deberta-base, deberta-large, distilbert-base, lstm]
    parser.add_argument("--model", default="roberta-large", type=str, help="Model Type")
    # [nlu, nlg]
    parser.add_argument("--window", default="nlg", type=str, help="Task Type")
    # [persona, roc, mutual, movie, all]
    parser.add_argument("--portion", default="persona", help="Fact Linking Task Type")

    args = parser.parse_args()

    data_path = linking_data_path + args.portion + "/fact_pipe/" + args.window + "/test/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.portion == "all":
        log_all = []
        label_all = []
        pipe_log_id_all = []
        head_log_id_all = []
        for sub_portion in ["persona", "roc", "mutual", "movie"]:
            log, label, pipe_log_id, head_log_id = prepare_pipeline_test_data(args.model, args.window, sub_portion)
            log_all.extend(deepcopy(log))
            label_all.extend(deepcopy(label))
            pipe_log_id_all.extend(deepcopy(pipe_log_id))
            head_log_id_all.extend(deepcopy(head_log_id))

        with open(data_path + "/logs.json", "w") as f:
            json.dump(log_all, f, indent=2)
        with open(data_path + "/labels.json", "w") as f:
            json.dump(label_all, f, indent=2)
        with open(data_path + "/pipe_log_ids.json", "w") as f:
            json.dump(pipe_log_id_all, f, indent=2)
        with open(data_path + "/head_log_ids.json", "w") as f:
            json.dump(head_log_id_all, f, indent=2)

    else:
        log, label, pipe_log_id, head_log_id = prepare_pipeline_test_data(args.model, args.window, args.portion)
        with open(data_path + "/logs.json", "w") as f:
            json.dump(log, f, indent=2)
        with open(data_path + "/labels.json", "w") as f:
            json.dump(label, f, indent=2)
        with open(data_path + "/pipe_log_ids.json", "w") as f:
            json.dump(pipe_log_id, f, indent=2)
        with open(data_path + "/head_log_ids.json", "w") as f:
            json.dump(head_log_id, f, indent=2)


if __name__ == "__main__":
    main()
