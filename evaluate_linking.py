import json
import argparse
import numpy as np

natural_to_relation = {"located or found at/in/on": "AtLocation",
                       "is/are capable of": "CapableOf",
                       "causes": "Causes",
                       "makes someone want": "CausesDesire",
                       "is created by": "CreatedBy",
                       "desires": "Desires",
                       "has, possesses or contains": "HasA",
                       "begins with the event/action": "HasFirstSubevent",
                       "ends with the event/action": "HasLastSubevent",
                       "to do this, one requires": "HasPrerequisite",
                       "can be characterized by being/having": "HasProperty",
                       "includes the event/action": "HasSubEvent",
                       "can be hindered by": "HinderedBy",
                       "is an example/instance of": "InstanceOf",
                       "happens after": "isAfter",
                       "happens before": "isBefore",
                       "___ can be filled by": "isFilledBy",
                       "is made of": "MadeOf",
                       "made (up) of": "MadeUpOf",
                       "is a step towards accomplishing the goal": "MotivatedByGoal",
                       "do(es) not desire": "NotDesires",
                       "used for": "ObjectUse",
                       "as a result, persony or others will": "oEffect",
                       "as a result, persony or others feels": "oReact",
                       "as a result, persony or others wants": "oWant",
                       "is a part of": "PartOf",
                       "can receive or be affected by the action": "ReceivesAction",
                       "personx is seen as": "xAttr",
                       "as a result, personx will": "xEffect",
                       "because personx wants": "xIntent",
                       "but before, personx needs": "xNeed",
                       "as a result, personx feels": "xReact",
                       "because": "xReason",
                       "as a result, personx wants": "xWant"}

physical_list = ["AtLocation", "CapableOf", "ObjectUse", "HasProperty", "MadeOf", "MadeUpOf", "Desires", "NotDesires",
                 "CreatedBy", "HasA", "InstanceOf", "PartOf"]


def main():
    parser = argparse.ArgumentParser()

    # [bert-base, bert-large, roberta-base, roberta-large, deberta-base, deberta-large, distilbert-base, lstm]
    parser.add_argument("--model", default="deberta-large", type=str, help="Model Type")
    # [nlu, nlg]
    parser.add_argument("--window", default="nlg", type=str, help="Task Type")
    # [fact_full, fact_cut]
    parser.add_argument("--linking", default="fact_full", type=str, help="Linking Model Type")
    # [persona, roc, mutual, movie, all]
    parser.add_argument("--portion", default="persona", help="Fact Linking Task Type")

    args = parser.parse_args()

    golden_labels_path = "data/" + args.portion + "/" + args.linking + "/" + args.window + "/test/labels.json"
    logs_path = "data/" + args.portion + "/" + args.linking + "/" + args.window + "/test/logs.json"

    pred_labels_path = "pred/" + args.portion + "-" + args.model + "-" + args.window + "-" \
                       + args.linking + "-test/predictions.json"

    with open(golden_labels_path, 'r') as f:
        golden_labels = json.load(f)
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    with open(pred_labels_path, 'r') as f:
        pred_labels = json.load(f)

    tp_linking = {"rpa": 0, "rpp": 0, "rpf": 0, "all": 0, "rpa_phy": 0, "rpp_phy": 0, "rpf_phy": 0, "all_phy": 0,
                  "rpa_soc": 0, "rpp_soc": 0, "rpf_soc": 0, "all_soc": 0}
    fp_linking = {"all": 0, "phy": 0, "soc": 0}
    fn_linking = {"rpa": 0, "rpp": 0, "rpf": 0, "all": 0, "rpa_phy": 0, "rpp_phy": 0, "rpf_phy": 0, "all_phy": 0,
                  "rpa_soc": 0, "rpp_soc": 0, "rpf_soc": 0, "all_soc": 0}
    tn_linking = {"all": 0, "phy": 0, "soc": 0}
    accuracy = {"rpa": 0.0, "rpp": 0.0, "rpf": 0.0, "all": 0.0, "rpa_phy": 0.0, "rpp_phy": 0.0, "rpf_phy": 0.0,
                "all_phy": 0.0, "rpa_soc": 0.0, "rpp_soc": 0.0, "rpf_soc": 0.0, "all_soc": 0.0}
    precision = {"rpa": 0.0, "rpp": 0.0, "rpf": 0.0, "all": 0.0, "rpa_phy": 0.0, "rpp_phy": 0.0, "rpf_phy": 0.0,
                 "all_phy": 0.0, "rpa_soc": 0.0, "rpp_soc": 0.0, "rpf_soc": 0.0, "all_soc": 0.0}
    recall = {"rpa": 0.0, "rpp": 0.0, "rpf": 0.0, "all": 0.0, "rpa_phy": 0.0, "rpp_phy": 0.0, "rpf_phy": 0.0,
              "all_phy": 0.0, "rpa_soc": 0.0, "rpp_soc": 0.0, "rpf_soc": 0.0, "all_soc": 0.0}
    f1 = {"rpa": 0.0, "rpp": 0.0, "rpf": 0.0, "all": 0.0, "rpa_phy": 0.0, "rpp_phy": 0.0, "rpf_phy": 0.0,
          "all_phy": 0.0, "rpa_soc": 0.0, "rpp_soc": 0.0, "rpf_soc": 0.0, "all_soc": 0.0}

    for idx, label in enumerate(golden_labels):

        if natural_to_relation[logs[idx]["text"][-2]["utter"].lower()] in physical_list:
            f_type = "phy"
        else:
            f_type = "soc"

        if label["target"] and label["linking"] in tp_linking:
            if pred_labels[idx]["target"]:
                tp_linking["all"] += 1
                tp_linking[label["linking"]] += 1
                tp_linking["all_" + f_type] += 1
                tp_linking[label["linking"]+"_"+f_type] += 1
            else:
                fn_linking["all"] += 1
                fn_linking[label["linking"]] += 1
                fn_linking["all_" + f_type] += 1
                fn_linking[label["linking"]+"_"+f_type] += 1
        else:
            if pred_labels[idx]["target"]:
                fp_linking["all"] += 1
                fp_linking[f_type] += 1
            else:
                tn_linking["all"] += 1
                tn_linking[f_type] += 1

    for link_type in precision.keys():
        recall[link_type] = tp_linking[link_type] / (tp_linking[link_type] + fn_linking[link_type])
        if link_type.endswith("_phy"):
            accuracy[link_type] = (tp_linking[link_type] + tn_linking["phy"]) / \
                                (tp_linking[link_type] + fn_linking[link_type] + fp_linking["phy"] + tn_linking["phy"])
            precision[link_type] = tp_linking[link_type] / (tp_linking[link_type] + fp_linking["phy"])
        elif link_type.endswith("_soc"):
            accuracy[link_type] = (tp_linking[link_type] + tn_linking["soc"]) / \
                                (tp_linking[link_type] + fn_linking[link_type] + fp_linking["soc"] + tn_linking["soc"])
            precision[link_type] = tp_linking[link_type] / (tp_linking[link_type] + fp_linking["soc"])
        else:
            accuracy[link_type] = (tp_linking[link_type] + tn_linking["all"]) / \
                                (tp_linking[link_type] + fn_linking[link_type] + fp_linking["all"] + tn_linking["all"])
            precision[link_type] = tp_linking[link_type] / (tp_linking[link_type] + fp_linking["all"])
        f1[link_type] = 2 / (1 / recall[link_type] + 1 / precision[link_type])

    print("Accuracy")
    print(accuracy)
    print("Precision")
    print(precision)
    print("Recall")
    print(recall)
    print("F1")
    print(f1)


if __name__ == "__main__":
    main()
