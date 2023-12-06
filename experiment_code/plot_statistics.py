import os
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb


def plot_histogram(losses, model_name):
    os.makedirs(f"./figs/{model_name}", exist_ok=True)
    plt.hist(losses, bins=50)
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution")
    os.makedirs(f"./outputs/figs/{model_name}", exist_ok=True)
    plt.savefig(f"./outputs/figs/{model_name}/loss_dist.pdf")


def plot_layer_distance(distances, model_name):
    layers, values = zip(*distances.items())
    plt.figure()
    plt.plot(values)
    print(layers)
    plt.xlabel("Layer")
    plt.ylabel("Distance")
    plt.title("Layer-wise Distance from Initial Weights")
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(f"./outputs/figs/{model_name}", exist_ok=True)
    plt.savefig(f"./outputs/figs/{model_name}/layer_dist.pdf")


def plot_layer_distance_by_functionality(distances, args):
    # Initialize variables
    unique_funcs = set()
    max_layer_index = -1

    # Identify unique functionalities and maximum layer index
    for key in distances.keys():
        parts = key.split('.')
        if parts[1] == 'layers':
            func = parts[-2]
            unique_funcs.add(func)
            layer_index = int(parts[2])
            max_layer_index = max(max_layer_index, layer_index)

    # Create a default dictionary for storing distances
    func_dict = defaultdict(lambda: [0] * (max_layer_index + 1))

    # Populate the dictionary with distance values
    for key, value in distances.items():
        parts = key.split('.')
        if parts[1] == 'layers':
            func = parts[-2]
            layer_index = int(parts[2])
            func_dict[func][layer_index] = value

    # Create and save plots for each functionality
    for func in unique_funcs:
        plt.figure(figsize=(10, 4))
        plt.plot(range(max_layer_index + 1), func_dict[func])
        plt.title(f"Layer-wise Distance for {func} in {args.model_name}")
        plt.ylabel("Distance")
        plt.xlabel("Layer Index")

        # Directory and filename for saving the plot
        fig_dir = f"./outputs/figs/{args.model_name}/{args.checkpoint_index}"
        os.makedirs(fig_dir, exist_ok=True)
        fig_filename = f"{fig_dir}/{func}_dist.png"

        plt.savefig(fig_filename)
        plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics_path",
                        type=str,
                        default="./outputs/statistics")
    parser.add_argument("--model_name", default='neftune', type=str)
    parser.add_argument("--checkpoint_index", default=1217, type=int)
    parser.add_argument("--dataset",
                        type=str,
                        default="alpaca_evol_instruct_70k")
    args = parser.parse_args()

    return args


def upload_weights(distances):
    # Initialize variables
    unique_funcs = set()
    max_layer_index = -1
    other_layers = defaultdict(float)

    # Identify unique functionalities and maximum layer index
    for key, value in distances.items():
        parts = key.split('.')
        if parts[1] == 'layers':
            func = parts[-2]
            unique_funcs.add(func)
            layer_index = int(parts[2])
            max_layer_index = max(max_layer_index, layer_index)
        else:
            other_layers[key] = value

    # Create a default dictionary for storing distances
    func_dict = defaultdict(lambda: [0] * (max_layer_index + 1))

    # Populate the dictionary with distance values
    for key, value in distances.items():
        parts = key.split('.')
        if parts[1] == 'layers':
            func = parts[-2]
            layer_index = int(parts[2])
            func_dict[func][layer_index] = value

    # Upload the weights for each layer
    for layer_index in range(max_layer_index + 1):
        metrics_dict = {
            func: func_dict[func][layer_index]
            for func in unique_funcs
        }
        # Add other layers as horizontal lines
        for other_layer, value in other_layers.items():
            metrics_dict[other_layer] = value
        wandb.log({"layer": layer_index, "metrics": metrics_dict})


if __name__ == "__main__":
    args = parse_arguments()

    statistics_path = args.statistics_path
    model_name = args.model_name
    checkpoint_index = args.checkpoint_index
    dataset = args.dataset

    # Load statistics
    with open(
            f"{statistics_path}/{model_name}_{checkpoint_index}_{dataset}.json",
            "r") as f:
        statistics = json.load(f)
    args.sample_seed = statistics["sample_seed"]
    args.data_fraction = statistics["data_fraction"]

    # Plot loss distribution
    # plot_histogram(statistics["loss"], model_name)

    # Plot layer distance
    # plot_layer_distance(statistics["distance"], model_name)

    # Plot layer distance by functionality
    # plot_layer_distance_by_functionality(statistics["distance"], args)

    wandb.init(project='data_instruct_visulaization',
               name=f"{model_name}_{checkpoint_index}_{dataset}",
               config=args)

    loss = statistics["loss"]
    # remove Nan values
    loss = [s for s in loss if s == s]
    loss = [[s] for s in loss]
    loss = wandb.Table(data=loss, columns=["loss"])
    wandb.log({
        "loss_distribution":
        wandb.plot.histogram(loss, "loss", title="Loss Distribution")
    })
    upload_weights(statistics["distance"])

    wandb.finish()
