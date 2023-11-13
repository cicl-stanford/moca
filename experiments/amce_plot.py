import json
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

BIGGER_SIZE = 22

geo_qualitative = {
    "1": "#FF1F5B",
    "2": "#07C8F9",
    "3": "#08B2F5",
    "4": "#099BF1",
    "5": "#0A85ED",
    "6": "#984BDD",
    "7": "#BC2CD6",
    "8": "#0B6EE9",
    "9": "#0D41E1",
}

name_mapping = {
    "Human": "Human",
    'text-babbage-001': 'GPT-3 text-babbage-001',
    'text-curie-001': 'GPT-3 text-curie-001',
    'text-davinci-002': "GPT-3.5 text-davinci-002",
    'text-davinci-003': "GPT-3.5 text-davinci-003 (RLHF)",
    'alpaca_7b': "Alpaca-7B (Distillation)",
    'claude-v1': "Anthropic Claude-v1 (RLHF)",
    "gpt-4": "GPT-4 (RLHF + Multi-Modal)",
    "gpt-3.5-turbo": "GPT-3.5 ChatGPT (RLHF)"
}

method_name_mapping = {
    "Human": "Human",
    'single shot': 'Zero-Shot',
    'persona - average': 'Social Simulacra (Average)',
    'persona - best': "Social Simulacra (Best)",
    'ape': "Automatic Prompt Engineering",
}


def visualize_causal_acme(file_name='results/acme_causal_result.json', name_mapping=name_mapping,
                          save_fig_name='acme_causal_fig.png', plot_cs_with_normality=False):
    """
    This assumes a JSON file that has the following structure:
    {
        "model_name": {
            "Factor Value 1 - Factor Value 0": (mean, (upper, lower))
        }
    }

    name_mapping is a dictionary that maps "model_name" in result JSON into a more readable name
    """
    acme_causal_result = json.load(open(file_name))
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)

    fig = plt.figure(figsize=(6, 15))

    color_idx = 1
    y_d = 1

    # print(len(acme_causal_result))
    auto_interval = len(acme_causal_result) + 1

    for name, result in acme_causal_result.items():
        y = 0

        labels = []
        for label, (mean, (upper, lower)) in result.items():
            if label == 'Conjunctive - Disjunctive':
                continue
            if plot_cs_with_normality:
                # plot the interaction between causal structure and event normality
                if label == 'Normal - Abnormal':
                    continue
            else:
                # plot normal / abnormal like the figure in the paper
                if label == "Conj Abnormal - Conj Normal" or label == "Disj Abnormal - Disj Normal":
                    continue

            if y == 0:
                plt.plot((lower, mean, upper), (y + y_d, y + y_d, y + y_d), '-', c=geo_qualitative[str(color_idx)],
                         label=name_mapping[name], lw=3)
                plt.plot(mean, y + y_d, c=geo_qualitative[str(color_idx)], marker='D')
            else:
                plt.plot((lower, mean, upper), (y + y_d, y + y_d, y + y_d), '-', c=geo_qualitative[str(color_idx)],
                         lw=3)
                plt.plot(mean, y + y_d, c=geo_qualitative[str(color_idx)], marker='D')

            y += auto_interval  # 10
            labels.append(label)

        y_d += 1
        color_idx += 1

    # if it's positive, it's the first one (leftmost being larger)
    # this line is correct, do not change it
    left_label = [l.split(' - ')[1].strip() for l in labels]
    left_label = [ll if ll != 'Statistical Norm' else "Statistical\nNorm" for ll in left_label]
    left_label = [ll if ll != 'Conj Normal' else "Conjunctive\nNormal" for ll in left_label]
    left_label = [ll if ll != 'Disj Normal' else "Disjunctive\nNormal" for ll in left_label]

    ytick_pos = [i * auto_interval + 5 for i in range(6)]
    plt.yticks(ytick_pos, left_label)

    plt.ylim(0, auto_interval * 6)

    # Initialize variables to store the min and max x-values
    min_x_value = float('inf')
    max_x_value = float('-inf')

    # Iterate through the results to find the min and max x-values
    for name, result in acme_causal_result.items():
        for label, (mean, (upper, lower)) in result.items():
            # Update min and max values
            min_x_value = min(min_x_value, lower)
            max_x_value = max(max_x_value, upper)

    for i in range(1, 6):
        plt.hlines(auto_interval * i, min_x_value, max_x_value, linestyles='dashed', color='gray')

    plt.axvline(0, linestyle='dotted', color='black', lw=1, alpha=0.2)
    plt.xlabel("$\Delta$P")
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=3)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    right_label = [l.split(' - ')[0].strip() for l in labels]
    right_label = [ll if ll != 'Prescriptive Norm' else "Prescriptive\nNorm" for ll in right_label]
    right_label = [ll if ll != 'Conj Abnormal' else "Conjunctive\nAbnormal" for ll in right_label]
    right_label = [ll if ll != 'Disj Abnormal' else "Disjunctive\nAbnormal" for ll in right_label]
    ytick_pos = [i * auto_interval + 5 for i in range(6)]
    plt.yticks(ytick_pos, right_label)

    plt.savefig(save_fig_name, bbox_inches='tight', dpi=300, transparent=False)  #


def visualize_moral_acme(file_name='results/acme_moral_result.json', name_mapping=name_mapping,
                         save_fig_name='acme_moral_fig.png'):
    """
    This assumes a JSON file that has the following structure:
    {
        "model_name": {
            "Factor Value 1 - Factor Value 0": (mean, (upper, lower))
        }
    }

    name_mapping is a dictionary that maps "model_name" in result JSON into a more readable name
    """
    acme_moral_result = json.load(open(file_name))

    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)

    fig = plt.figure(figsize=(6, 15))
    color_idx = 1
    y_d = 1

    label_orders = ['Patient of harm - Agent of harm', 'Impersonal - Personal', 'Side Effect - Means',
                    'Inevitable - Avoidable', 'Self-beneficial - Other-beneficial']
    label_orders.reverse()

    auto_interval = len(acme_moral_result) + 1

    for name, result in acme_moral_result.items():
        y = 0

        labels = []
        for label in label_orders:
            mean, (upper, lower) = result[label]
            if y == 0:
                plt.plot((lower, mean, upper), (y + y_d, y + y_d, y + y_d), '-', c=geo_qualitative[str(color_idx)],
                         label=name_mapping[name], lw=3)
                plt.plot(mean, y + y_d, c=geo_qualitative[str(color_idx)], marker='D')
            else:
                plt.plot((lower, mean, upper), (y + y_d, y + y_d, y + y_d), '-', c=geo_qualitative[str(color_idx)],
                         lw=3)
                plt.plot(mean, y + y_d, c=geo_qualitative[str(color_idx)], marker='D')

            y += auto_interval
            labels.append(label)

        y_d += 1
        color_idx += 1

    left_label = [l.split(' - ')[1].strip() for l in labels]
    left_label = [ll if ll != 'Agent of harm' else "Agent\nof harm" for ll in left_label]
    ytick_pos = [i * auto_interval + 5 for i in range(5)]
    plt.yticks(ytick_pos, left_label)

    plt.ylim(0, auto_interval * 5)

    for i in range(1, 5):
        plt.hlines(auto_interval * i, -0.4, 0.4, linestyles='dashed', color='gray')

    plt.axvline(0, linestyle='dotted', color='black', lw=1, alpha=0.2)
    plt.xlabel("$\Delta$P")
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=3)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    right_label = [l.split(' - ')[0].strip() for l in labels]
    right_label = [ll if ll != 'Patient of harm' else "Patient\nof harm" for ll in right_label]
    ytick_pos = [i * auto_interval + 5 for i in range(5)]
    plt.yticks(ytick_pos, right_label)

    plt.savefig(save_fig_name, bbox_inches='tight', dpi=300, transparent=False)  # , transparent=True


def visualize_acme_radar(file_name="results/acme_causal_result.json",
                         save_file_name='acme_causal_radar_fig.png',
                         plot_cs_with_normality=False):
    acme_causal_result = json.load(open(file_name))
    model_size_comparison = ['text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'Human']
    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#e41a1c']

    BIGGER_SIZE = 20

    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)

    for name, result in acme_causal_result.items():
        if name not in model_size_comparison:
            continue

        labels = []
        for label, (mean, (upper, lower)) in result.items():
            if label == 'Conjunctive - Disjunctive':
                continue
            if plot_cs_with_normality:
                # plot the interaction between causal structure and event normality
                if label == 'Normal - Abnormal':
                    continue
            else:
                # plot normal / abnormal like the figure in the paper
                if label == "Conj Abnormal - Conj Normal" or label == "Disj Abnormal - Disj Normal":
                    continue

            labels.append(label)

    left_label = [l.split(' - ')[1].strip() for l in labels]
    left_label = [ll if ll != 'Statistical Norm' else "Statistical\nNorm" for ll in left_label]
    left_label = [ll if ll != 'Conj Normal' else "Conjunctive\nNormal" for ll in left_label]
    left_label = [ll if ll != 'Disj Normal' else "Disjunctive\nNormal" for ll in left_label]

    right_label = [l.split(' - ')[0].strip() for l in labels]
    right_label = [ll if ll != 'Prescriptive Norm' else "Prescriptive\nNorm" for ll in right_label]
    right_label = [ll if ll != 'Conj Abnormal' else "Conjunctive\nAbnormal" for ll in right_label]
    right_label = [ll if ll != 'Disj Abnormal' else "Disjunctive\nAbnormal" for ll in right_label]

    values_groups = {}
    mean_point_groups = {}

    for name, result in acme_causal_result.items():
        left_values = []
        right_values = []
        mean_points_left, mean_points_right = [], []
        for label, (mean, (upper, lower)) in result.items():
            if label == 'Conjunctive - Disjunctive':
                continue
            if plot_cs_with_normality:
                # plot the interaction between causal structure and event normality
                if label == 'Normal - Abnormal':
                    continue
            else:
                # plot normal / abnormal like the figure in the paper
                if label == "Conj Abnormal - Conj Normal" or label == "Disj Abnormal - Disj Normal":
                    continue

            # put mean, upper, lower in the right value group
            # (lowever -> left, upper -> right)
            if lower > 0 and upper > 0:
                # Left means it's emphasizing on "Unaware"
                left_values.append(0.)
                right_values.append(upper)
            elif lower < 0 and upper > 0:
                left_values.append(abs(lower))
                right_values.append(upper)
            elif lower < 0 and upper < 0:
                right_values.append(0.)
                left_values.append(abs(lower))
            elif lower == 0 and upper == 0:
                right_values.append(0.)
                left_values.append(0.)
            else:
                print(lower, upper)
                raise Exception("not possible")

            if mean < 0:
                mean_points_left.append(mean)
                mean_points_right.append(0.)
            else:
                mean_points_right.append(mean)
                mean_points_left.append(0.)

        values_groups[name] = right_values + left_values
        mean_point_groups[name] = mean_points_right + mean_points_left

    labels = right_label + left_label

    # Create a polar subplot
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # Set the plot properties
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.tick_params(rotation='auto', pad=15)

    # Set the radial ticks and gridlines
    ax.set_rlabel_position(180 / num_vars)
    ax.set_rticks(np.arange(1, num_vars, 1))
    ax.set_yticklabels(map(str, range(1, num_vars)), color='gray', fontsize=10)

    # Plot the data
    angles = np.concatenate((angles, [angles[0]]))
    i = 0
    y_max = 0
    for model_name in model_size_comparison:
        values = values_groups[model_name]
        np_values = np.concatenate((values, [values[0]]))
        mean_points = mean_point_groups[model_name]
        np_mean_points = np.concatenate((mean_points, [mean_points[0]]))
        ax.plot(angles, np_values, linestyle='-', linewidth=2, label=name_mapping[model_name], c=colors[i])
        if model_name == 'Human':
            ax.fill(angles, np_values, colors[i], alpha=0.1)
        else:
            ax.fill(angles, np_values, colors[i], alpha=0.1)
        y_max = max(y_max, max(values))
        i += 1

    # automatically figuring out grid
    rounded_y_max = np.round(y_max * 10, 0)
    grids = [i / 10 for i in range(int(rounded_y_max) + 1)]
    ax.set_rgrids(grids)
    if rounded_y_max < y_max * 10:
        ax.set_ylim(0, rounded_y_max / 10 + 0.05)
    else:
        ax.set_ylim(0, rounded_y_max / 10)

    # Add a legend and display the plot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, facecolor='#f0f0f0')
    plt.savefig(save_file_name, bbox_inches='tight', dpi=300, transparent=False)


if __name__ == '__main__':
    ...
    # visualize_causal_acme()
    # visualize_moral_acme()

    # for method
    # visualize_causal_acme('results/methods_acme_causal_result.json', name_mapping=method_name_mapping,
    #                       save_fig_name="acme_causal_methods.png")
    # visualize_moral_acme('results/methods_acme_moral_result.json', name_mapping=method_name_mapping,
    #                       save_fig_name="acme_moral_methods.png")

    # for radar plot
    # visualize_acme_radar()
    # visualize_acme_radar(plot_cs_with_normality=True, save_file_name='acme_causal_cs_with_normality_radar_fig.png')
    # visualize_acme_radar('results/acme_moral_result.json', save_file_name='acme_moral_radar_fig.png')