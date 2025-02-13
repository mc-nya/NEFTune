import json
import math
import itertools
import numpy as np
from regex import F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import wasserstein_distance
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon
from sympy import div, plot


def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def concat_outputs(json_data):
    return " ".join([entry['output'] for entry in json_data])


def create_ngrams(data, n):
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer="word")
    ngram_matrix = vectorizer.fit_transform(data).toarray()
    vocab = vectorizer.get_feature_names_out()
    ngram_freqs = [{
        vocab[i]: counts[i]
        for i in range(len(counts)) if counts[i] > 0
    } for counts in ngram_matrix]
    return ngram_freqs


def kl_divergence(freqs1, freqs2):
    # Calculate the average count for adaptive epsilon
    all_counts = list(freqs1.values()) + list(freqs2.values())
    avg_count = sum(all_counts) / len(all_counts)
    epsilon = avg_count * 0.01  # 1% of the average count

    # Apply smoothing
    smoothed_freqs1 = {
        ngram: count + epsilon
        for ngram, count in freqs1.items()
    }
    smoothed_freqs2 = {
        ngram: freqs2.get(ngram, 0) + epsilon
        for ngram in freqs1
    }

    # Recalculate totals after smoothing
    total1 = sum(smoothed_freqs1.values())
    total2 = sum(smoothed_freqs2.values())

    divergence = 0.0
    for ngram, count1 in smoothed_freqs1.items():
        count2 = smoothed_freqs2[ngram]

        p1 = count1 / total1
        p2 = count2 / total2
        divergence += p1 * math.log2(p1 / p2)

    return divergence


def compute_matrix(ngram_freqs, metric_function, num_metrics=1):
    # metric_function can generate 1 or more metrics
    # num_metrics is the number of metrics generated by metric_function
    # If num_metrics > 1, then metric_function should return a list of metrics
    num_files = len(ngram_freqs)
    if num_metrics == 1:
        matrix = np.zeros((num_files, num_files))
        for i, j in itertools.product(range(num_files), repeat=2):
            # Calculate the metric for each pair of n-gram frequencies
            # The metric function is now a variable and can be any compatible function
            metric_value = metric_function(ngram_freqs[i], ngram_freqs[j])
            matrix[i, j] = metric_value

        return matrix
    else:
        matrix = np.zeros((num_files, num_files, num_metrics))
        for i, j in itertools.product(range(num_files), repeat=2):
            # Calculate the metric for each pair of n-gram frequencies
            # The metric function is now a variable and can be any compatible function
            metric_values = metric_function(ngram_freqs[i], ngram_freqs[j])
            matrix[i, j, :] = metric_values

        return matrix


def jensen_shannon_divergence(freqs1, freqs2):
    total1 = sum(freqs1.values())
    total2 = sum(freqs2.values())

    all_counts = list(freqs1.values()) + list(freqs2.values())
    avg_count = sum(all_counts) / len(all_counts)
    epsilon = avg_count * 0.01  # 1% of the average count

    all_ngrams = set(freqs1.keys()).union(set(freqs2.keys()))
    smoothed_freqs1 = {
        ngram: freqs1.get(ngram, 0) + epsilon
        for ngram in all_ngrams
    }
    smoothed_freqs2 = {
        ngram: freqs2.get(ngram, 0) + epsilon
        for ngram in all_ngrams
    }

    # Recalculate totals after smoothing
    total1 = sum(smoothed_freqs1.values())
    total2 = sum(smoothed_freqs2.values())

    # Normalize smoothed frequency counts to probabilities
    p1 = [count / total1 for count in smoothed_freqs1.values()]
    p2 = [count / total2 for count in smoothed_freqs2.values()]

    # Calculate Jensen-Shannon Divergence
    js_divergence = jensenshannon(p1, p2, base=2)
    return js_divergence


def chi_squared_test(freqs1, freqs2):
    # Create lists of observed and expected frequencies
    observed = [freqs1.get(ngram, 0) for ngram in freqs2]
    expected = [freqs2.get(ngram, 0) for ngram in freqs2]

    # Normalize observed and expected frequencies so their sums are equal
    sum_observed = sum(observed)
    sum_expected = sum(expected)

    if sum_observed > 0 and sum_expected > 0:
        observed = [o * (sum_expected / sum_observed) for o in observed]

    # Conduct the chi-squared test
    chi2_stat, p_value = chisquare(observed, f_exp=expected)
    return chi2_stat, p_value


def earth_movers_distance(freqs1, freqs2):
    total1 = sum(freqs1.values())
    total2 = sum(freqs2.values())

    # Normalize frequency counts to probabilities
    p1 = [count / total1 for count in freqs1.values()]
    p2 = [freqs2.get(ngram, 0) / total2 for ngram in freqs1]

    # Calculate Earth Mover's Distance
    emd = wasserstein_distance(p1, p2)
    return emd


def plot_heatmap(matrix, labels, title="", cbar_kws=""):
    sns.set()
    ax = sns.heatmap(matrix,
                     annot=True,
                     cmap="YlGnBu",
                     cbar_kws={'label': cbar_kws},
                     xticklabels=labels,
                     yticklabels=labels)
    plt.xlabel("File")
    plt.ylabel("File")
    plt.title(title)
    plt.show()


# List of JSON file names
json_files = [
    "~/research/NEFTune/experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_cutout_0.2_1217.json",
    "~/research/NEFTune/experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_naive_1217.json",
    "~/research/NEFTune/experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_neftune_1217.json",
    "~/research/NEFTune/experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_mixup_simple_0.8_1217.json",
    "~/research/NEFTune/experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_ran_drop_0.3_1217.json",
    "experiment_code/outputs/answers/alpaca_eval_llama7B_Alpaca_sam_0.05_1217.json",
    "~/research/NEFTune/experiment_code/datasets/alpaca-train.jsonl",
]

# Load and concatenate "output" fields in each JSON file
concatenated_outputs = [
    concat_outputs(load_json_file(filename)) for filename in json_files
]
# for n in range(1, 6):
#     # Generate n-grams for each file
#     ngram_freqs = create_ngrams(concatenated_outputs, n)

#     # Compute KL-divergence matrix
#     divergence_matrix = compute_matrix(ngram_freqs, kl_divergence)

#     # Plot divergence matrix as a heatmap
#     file_labels = [f"File {i+1}" for i in range(len(json_files))]
#     file_labels = ["Cutout", "Naive", "NEFTune", "Mixup", "RD", "Train"]
#     plot_heatmap(divergence_matrix, file_labels)
#     print(divergence_matrix)
#     plt.savefig(f"kl_divergence_matrix_{n}.png")
#     plt.clf()
file_labels = ["Cutout", "Naive", "NEFTune", "Mixup", "RD", "SAM", "Train"]
for n in range(1, 6):
    # Generate n-grams for each file
    ngram_freqs = create_ngrams(concatenated_outputs, n)

    # Compute and plot for each measure
    for measure in ['jsd']:  #['kl', 'jsd', 'chi2', 'emd']:
        if measure == 'jsd':
            # Compute Jensen-Shannon Divergence matrix
            divergence_matrix = compute_matrix(ngram_freqs,
                                               jensen_shannon_divergence)
            title = f'Jensen-Shannon Divergence (n={n})'
        elif measure == 'chi2':
            # Compute Chi-Squared Test matrix
            divergence_matrix = compute_matrix(ngram_freqs, chi_squared_test,
                                               2)

            chi2_stat_matrix = divergence_matrix[:, :, 0]

            p_value_matrix = divergence_matrix[:, :, 1]

            # Plot heatmap for p-values
            title = f'Chi-Squared p-Value (n={n})'
            plot_heatmap(p_value_matrix, file_labels, title, "p-value")
            plt.savefig(f"figs/chi2_pvalue_matrix_{n}.png")
            plt.clf()

            divergence_matrix = divergence_matrix[:, :, 0]
            title = f'Chi-Squared Test (n={n})'

        elif measure == 'emd':
            # Compute Earth Mover's Distance matrix
            divergence_matrix = compute_matrix(ngram_freqs,
                                               earth_movers_distance)
            title = f'Earth Mover\'s Distance (n={n})'
        elif measure == 'kl':
            # Compute KL-Divergence matrix
            divergence_matrix = compute_matrix(ngram_freqs, kl_divergence)
            title = f'KL Divergence (n={n})'

        # Plot heatmap
        plot_heatmap(divergence_matrix, file_labels, title, measure)

        # Save plot
        plt.savefig(f"figs/{measure}_{n}.png")
        plt.clf()

        # Reset seaborn to original state (if needed)
        sns.reset_orig()
