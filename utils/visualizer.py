import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import pandas as pd



def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.show()


def plot_images_labels_prediction(images, labels, prediction=[], idx=0, num=10, method_name=""):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(num):
        ax = plt.subplot(5, 5, 1 + i)
        image = images.iloc[idx, np.arange(0, 256)].values.reshape([16, 16])
        ax.imshow(image, cmap='binary')
        title = f"label={labels.iloc[idx, 0]}"
        if len(prediction) > 0:
            title += f", predict={prediction.iloc[idx, 0]}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    os.makedirs("digit_images", exist_ok=True)
    filename = f"digit_images/plot_{method_name}_{np.random.randint(10000)}.png"
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()


def plot_residual_curves(residual_list, color='blue', method_name=""):
    plt.xticks(range(0, 10))
    plt.ylabel('residual')
    plt.xlabel('digital')
    for residual in residual_list:
        plt.plot(residual, color=color)
    os.makedirs("digit_images", exist_ok=True)
    filename = f"digit_images/residual_{np.random.randint(10000)}.png"
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()


def plot_mse_curve(mse_values, label='mse', xlabel='number of basis', method_name="method"):
    plt.figure(figsize=(8, 5))
    plt.plot(mse_values, 's-')
    plt.xticks(range(len(mse_values)))
    plt.ylabel(label)
    plt.xlabel(xlabel)
    plt.title(f"MSE Curve - {method_name}")
    filename = os.path.join("digit_images", f"mse_curve_{method_name}.png")
    plt.savefig(filename)
    plt.close()
    return filename

def plot_confusion(y_true, y_pred, method_name="confusion"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {method_name}")
    filename = os.path.join("digit_images", f"confusion_matrix_{method_name}.png")
    plt.savefig(filename)
    plt.close()
    return filename

def save_prediction_error_analysis_table(y_true, y_pred, method_name="method"):
    df = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    error_df = df[df['actual'] != df['pred']]

    stats = {}
    for digit in range(10):
        group = error_df[error_df['actual'] == digit]['pred']
        counts = group.value_counts()
        first_pred = counts.index[0] if len(counts) > 0 else -1
        first_count = counts.iloc[0] if len(counts) > 0 else 0
        second_pred = counts.index[1] if len(counts) > 1 else -1
        second_count = counts.iloc[1] if len(counts) > 1 else 0
        stats[digit] = {
            '1th': first_pred,
            'count1': first_count,
            '2nd': second_pred,
            'count2': second_count
        }

    result_df = pd.DataFrame(stats).T.rename_axis('actual').reset_index()
    result_df = result_df[['actual', '1th', 'count1', '2nd', 'count2']]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    tbl = pd.plotting.table(ax, result_df.round(1), loc='center', cellLoc='center', colWidths=[0.15]*5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    plt.title("Prediction Error Analysis", fontsize=14, loc='left')
    plt.savefig(f"digit_images/error_analysis_{method_name}.png", bbox_inches='tight')
    plt.close()

def plot_best_worst_residual_curves(residual_list, y_true, y_pred, save_dir="digit_images", method_name=""):
    os.makedirs(save_dir, exist_ok=True)

    residual_list = np.array(residual_list)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for digit in range(10):
        # 抓出預測正確的樣本
        correct_indices = np.where((y_true == digit) & (y_true == y_pred))[0]
        if len(correct_indices) == 0:
            continue

        # residual 是 10 維向量，抓出 residual 最小與最大的 index
        digit_residuals = [residual_list[i][digit] for i in correct_indices]
        min_idx = correct_indices[np.argmin(digit_residuals)]
        max_idx = correct_indices[np.argmax(digit_residuals)]

        # 繪圖
        plt.figure()
        plt.plot(residual_list[min_idx], label='Best (min residual)', color='blue')
        plt.plot(residual_list[max_idx], label='Worst (max residual)', color='red')
        plt.title(f"Residual Curve for Digit {digit} - {method_name}")
        plt.xlabel("Predicted Class")
        plt.ylabel("Residual")
        plt.xticks(np.arange(10))
        plt.legend()
        plt.grid(True)
        filename = os.path.join(save_dir, f"residual_curve_{method_name}_digit{digit}.png")
        plt.savefig(filename)
        plt.close()
        print(f"[Saved] {filename}")

def plot_all_residual_curves_by_digit(residual_list, y_true, method_name=""):
    import matplotlib.pyplot as plt
    import os

    os.makedirs("digit_images", exist_ok=True)

    for digit in range(10):
        indices = np.where(y_true == digit)[0]
        if len(indices) == 0:
            continue

        plt.figure()
        for idx in indices:
            residual = residual_list[idx]
            plt.plot(residual, color='green', alpha=0.3)

        plt.xlabel('Predicted Class')
        plt.ylabel('Residual')
        plt.title(f'Residual Curves for Digit {digit} - {method_name}')
        filename = f"digit_images/residual_curves_{method_name}_digit{digit}.png"
        plt.savefig(filename)
        plt.close()
        print(f"[Saved] {filename}")

def plot_overall_comparison_table(methods_acc, filename="overall_accuracy_table.png"):
    """
    methods_acc: List of dicts like:
    [{"Method": "SVD", "Test": 0.94, "Teacher": 0.9, "Student": 0.9}, ...]
    """
    df = pd.DataFrame(methods_acc)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    colors = [["#f5f5f5", "#ffffff"][(i % 2)] for i in range(len(df))]
    the_table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         loc='center',
                         cellColours=[[c]*len(df.columns) for c in colors],
                         cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.5)

    plt.savefig(f"digit_images/{filename}", bbox_inches='tight')
    plt.close()
    print(f"[Saved] digit_images/{filename}")