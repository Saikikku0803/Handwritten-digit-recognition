import numpy as np
import pandas as pd

def classification_report_by_class(df, actual_col='actual', pred_col='prediction'):
    """回傳每一類的總數、錯誤數、成功數與準確率"""
    total_counts = df[[actual_col]].groupby(actual_col).size().reset_index(name='totalCounts')
    error_df = df[df[actual_col] != df[pred_col]]
    error_counts = error_df[[actual_col]].groupby(actual_col).size().reset_index(name='errorCounts')

    result_df = pd.merge(total_counts, error_counts, how='left', on=actual_col).fillna(0)
    result_df["successCounts"] = result_df["totalCounts"] - result_df["errorCounts"]
    result_df["probability"] = result_df["successCounts"] / result_df["totalCounts"]
    return result_df.sort_values('probability')

def top_misclassifications(df, actual_col='actual', pred_col='prediction', top_n=2):
    """回傳每個類別中最常錯判為哪幾個類別"""
    temp = df[[actual_col, pred_col]].groupby([actual_col, pred_col]).size().reset_index(name='counts')
    error_prediction = np.zeros((top_n * 2, 10))

    for i in range(10):
        group = temp[temp[actual_col] == i].sort_values('counts', ascending=False)
        for n in range(top_n):
            if len(group) > n:
                error_prediction[n * 2][i] = group.iloc[n][pred_col]
                error_prediction[n * 2 + 1][i] = group.iloc[n]['counts']
            else:
                error_prediction[n * 2][i] = -1  # 表示沒有足夠錯誤數據
                error_prediction[n * 2 + 1][i] = 0

    return pd.DataFrame(error_prediction, index=[f"Top{n+1}_class" if i % 2 == 0 else f"Top{n+1}_count"
                                                 for n in range(top_n) for i in range(2)],
                        columns=np.arange(10))

def find_extreme_examples(df, label_col='actual', pred_col='prediction', residual_col='residual'):
    """找出每個類別中預測正確但 residual 最小與最大的圖像 index"""
    correct_df = df[df[label_col] == df[pred_col]]
    min_idx = []
    max_idx = []

    for i in range(10):
        class_df = correct_df[correct_df[label_col] == i]
        if class_df.empty:
            min_idx.append(None)
            max_idx.append(None)
            continue
        min_idx.append(class_df[class_df[residual_col] == class_df[residual_col].min()].index[0])
        max_idx.append(class_df[class_df[residual_col] == class_df[residual_col].max()].index[0])

    return min_idx, max_idx
def compute_extra_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))