import argparse
import os
import json

import numpy as np


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def evaluate(path):
    b = int(path.split("_")[-2][-1])
    if b:
        ret = {"master": [0, 0], "ensemble": [0, 0]}
        data = load_data(path)
        for comp_dicts in data.values():
            master_logprobs = []
            ensemble_logprobs = []
            for comp_dict in comp_dicts:
                master_comp_logprobs = [0,0,0,0,0,0,0,0]
                ensemble_comp_logprobs = [0,0,0,0,0,0,0,0]
                for branch_name, branch_dict in comp_dict.items():
                    for i in range(8):
                        token_logprobs = branch_dict[i]["token_logprobs"]
                        if branch_name == "master":
                            master_comp_logprobs[i] += (sum(token_logprobs)/len(token_logprobs))
                        else:
                            ensemble_comp_logprobs[i] += (sum(token_logprobs)/len(token_logprobs))
                master_logprobs.append(master_comp_logprobs)
                ensemble_logprobs.append(ensemble_comp_logprobs)
            master_logprobs = np.array(master_logprobs)
            master_logprobs = np.sum(master_logprobs, axis=0)
            if master_logprobs[0] == max(master_logprobs):
                ret["master"][0] += 1
            ret["master"][1] += 1
            ensemble_logprobs = np.array(ensemble_logprobs)
            ensemble_logprobs = np.sum(ensemble_logprobs, axis=0)
            if ensemble_logprobs[0] == max(ensemble_logprobs):
                ret["ensemble"][0] += 1
            ret["ensemble"][1] += 1
        return ret
    else:
        ret = [0,0]
        data = load_data(path)
        for prob_dict in data.values():
            logprobs = []
            for i in range(8):
                token_logprobs = prob_dict[i]["token_logprobs"]
                logprobs.append(sum(token_logprobs)/len(token_logprobs))
            if logprobs[0] == max(logprobs):
                ret[0] += 1
            ret[1] += 1
        return ret


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["RAVEN-10000", "RAVEN-F", "I-RAVEN"], required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--figure_configuration", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--is_branch", type=str, required=True,
        help="Whether or not to branch over components and attributes."
    )
    parser.add_argument(
        "--num_rows", type=int, required=True,
        help="Number of rows to include."
    )
    args = parser.parse_args()
    print(args.__dict__)

    inference_file_path = os.path.join(args.work_dir, "data", args.dataset, f"{args.figure_configuration}_500_{args.model_name}_b{args.is_branch}_n{args.num_rows}.json")
    performance = evaluate(inference_file_path)
    print(performance)

    output_dir = os.path.join(args.work_dir, "outputs", args.dataset, f"{args.model_name}_b{args.is_branch}_n{args.num_rows}", args.figure_configuration)
    output_path = os.path.join(output_dir, "performance.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(performance, f, indent=4)
    return


if __name__ == "__main__":
    main()