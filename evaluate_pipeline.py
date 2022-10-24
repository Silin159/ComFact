import json
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score


def main():
    parser = argparse.ArgumentParser()

    # [bert-base, bert-large, roberta-base, roberta-large, deberta-base, deberta-large, distilbert-base, lstm]
    parser.add_argument("--model", default="roberta-large", type=str, help="Model Type")
    # [nlu, nlg]
    parser.add_argument("--window", default="nlg", type=str, help="Task Type")
    # [persona, roc, mutual, movie, all]
    parser.add_argument("--portion", default="persona", help="Fact Linking Task Type")
    # [fact, head]
    parser.add_argument("--linking", default="fact", type=str, help="Linking Model Type")

    args = parser.parse_args()

    full_golden_labels_path = "data/" + args.portion + "/fact_full/" + args.window + "/test/labels.json"
    if args.linking == "fact":
        pred_labels_pipe_path = "pred/" + args.portion + "-" + args.model + "-" + args.window \
                                + "-pipeline-test/predictions.json"
    elif args.linking == "head":
        pred_labels_pipe_path = "pred/" + args.portion + "-" + args.model + "-" + args.window \
                                + "-head-test/predictions.json"
    else:
        raise ValueError("args.linking not in ['fact', 'head']")
    if args.linking == "fact":
        pipe_to_full_log_ids_path = "data/" + args.portion + "/fact_pipe/" + args.window + "/test/pipe_log_ids.json"
    elif args.linking == "head":
        pipe_to_full_log_ids_path = "data/" + args.portion + "/fact_pipe/" + args.window + "/test/head_log_ids.json"
    else:
        raise ValueError("args.linking not in ['fact', 'head']")

    with open(full_golden_labels_path, 'r') as f:
        full_golden_labels = json.load(f)
    with open(pred_labels_pipe_path, 'r') as f:
        pred_labels_pipe = json.load(f)
    with open(pipe_to_full_log_ids_path, 'r') as f:
        pipe_to_full_log_ids = json.load(f)

    full_golden = []
    pred_pipe = []
    pred_labels_pipe_full = []
    pointer = 0
    if args.linking == "head":
        for lid, label_g in enumerate(full_golden_labels):
            full_golden.append(int(label_g["target"]))
        for hid, label_p in enumerate(pred_labels_pipe):
            for _ in range(pipe_to_full_log_ids[hid][1] - pipe_to_full_log_ids[hid][0]):
                pred_pipe.append(int(label_p["target"]))
                pred_labels_pipe_full.append(label_p)
    else:
        for lid, label in enumerate(full_golden_labels):
            full_golden.append(int(label["target"]))
            if pointer < len(pipe_to_full_log_ids) and lid == pipe_to_full_log_ids[pointer]:
                pred_pipe.append(int(pred_labels_pipe[pointer]["target"]))
                pred_labels_pipe_full.append(pred_labels_pipe[pointer])
                pointer += 1
            else:
                pred_pipe.append(False)
                pred_labels_pipe_full.append({"target": False, "linking": None})

    accuracy = np.sum(np.array(pred_pipe) == np.array(full_golden)) / len(full_golden)
    precision = precision_score(np.array(full_golden), np.array(pred_pipe))
    recall = recall_score(np.array(full_golden), np.array(pred_pipe))
    f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))
    result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print(result)

    if args.linking == "fact":
        pred_labels_pipe_full_path = "pred/" + args.portion + "-" + args.model + "-" + args.window \
                                    + "-pipeline-test/full_predictions_fact_linker.json"
    else:
        pred_labels_pipe_full_path = "pred/" + args.portion + "-" + args.model + "-" + args.window \
                                     + "-pipeline-test/full_predictions_head_linker.json"
    with open(pred_labels_pipe_full_path, "w") as f:
        json.dump(pred_labels_pipe_full, f, indent=2)


if __name__ == "__main__":
    main()
