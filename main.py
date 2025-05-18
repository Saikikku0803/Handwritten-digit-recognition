import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from utils.data_loader import load_usps_dataset, load_student_images, load_digit_images
from utils.visualizer import plot_images_labels_prediction, plot_residual_curves, plot_mse_curve, plot_confusion, plot_overall_comparison_table
from utils.metrics import compute_extra_accuracy
from methods.simple_mean import compute_mean_images, predict_with_mean
from methods.svd_classification import get_svd_bases, svd_predict
from methods.hosvd_classification import get_s_and_u_list, hosvd_algorithm
from methods.random_forest import train_random_forest, evaluate_model
from methods.mlp_model import train_mlp_model, evaluate_mlp_model
from methods.cnn_model import build_cnn_model, train_cnn_model, evaluate_cnn_model, predict_cnn_model
from methods.rnn_model import build_rnn_model, train_rnn_model, evaluate_rnn_model, predict_rnn_model
from methods.lstm_method import train_and_evaluate_lstm
from methods.gnn_model import train_gcn_model
from config import IMAGE_SIZE

def get_classwise_accuracy_table(y_true, y_pred):
    df = pd.DataFrame({'actual': y_true, 'prediction': y_pred})
    total_counts = df.groupby('actual').size().reset_index(name='totalCounts')
    error_counts = df[df['actual'] != df['prediction']].groupby('actual').size().reset_index(name='errorCounts')
    result_df = pd.merge(total_counts, error_counts, on='actual', how='left').fillna(0)
    result_df['errorCounts'] = result_df['errorCounts'].astype(int)
    result_df['successCounts'] = result_df['totalCounts'] - result_df['errorCounts']
    result_df['probability'] = result_df['successCounts'] / result_df['totalCounts']
    result_df = result_df.sort_values(by='probability', ascending=True).reset_index(drop=True)
    return result_df

def save_accuracy_table_as_image(df, method_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.tight_layout()
    os.makedirs("digit_images", exist_ok=True)
    filename = f'digit_images/{method_name}_classwise_accuracy.png'
    plt.savefig(filename, dpi=300)
    print(f"[Saved] {filename}")

def main():
    # 1. 資料載入
    x_train, y_train, x_test, y_test = load_usps_dataset("./input/usps.h5")
    print(f"[INFO] Training Samples: {x_train.shape[0]}, Testing Samples: {x_test.shape[0]}")

    x_student = load_student_images("./input", prefix="digit_s_")
    x_Ele = load_digit_images("./input", prefix="digit_")
    y_extra = np.arange(10)

    # 2. 平均影像分類
    print("\n[Method] Simple Mean Comparison")
    t0 = time.time()
    train_df = compute_mean_images(x_train, y_train)
    test_df = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test, columns=['label'])], axis=1)
    simple_result, acc_mean, residuals_mean = predict_with_mean(test_df, train_df)
    print(f"Accuracy: {acc_mean:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_mean = simple_result['prediction'].values
    if 'label' in simple_result.columns:
        plot_images_labels_prediction(simple_result.iloc[:, :256], simple_result[['label']], simple_result[['prediction']], idx=0, num=10, method_name="mean")
        plot_residual_curves(residuals_mean, color='gray', method_name="mean")
        plot_confusion(y_test, y_pred_mean, method_name="mean")
        errors = simple_result[simple_result['label'] != simple_result['prediction']]
        plot_images_labels_prediction(errors.iloc[:, :256], errors[['label']], errors[['prediction']], idx=0, num=10,
                                      method_name="mean_errors")

        correct = simple_result[simple_result['label'] == simple_result['prediction']]
        nice = correct.sort_values('residual').head(10)
        ugly = correct.sort_values('residual', ascending=False).head(10)
        plot_images_labels_prediction(nice.iloc[:, :256], nice[['label']], nice[['prediction']], idx=0, num=10,
                                      method_name="mean_nice")
        plot_images_labels_prediction(ugly.iloc[:, :256], ugly[['label']], ugly[['prediction']], idx=0, num=10,
                                      method_name="mean_ugly")

        # 額外驗證
        Ele_df = pd.DataFrame(x_Ele)
        Ele_df['label'] = y_extra
        student_df = pd.DataFrame(x_student)
        student_df['label'] = y_extra
        _, acc_Ele, _ = predict_with_mean(Ele_df, train_df)
        _, acc_student, _ = predict_with_mean(student_df, train_df)
        print(f"[Extra] Ele Accuracy (mean): {acc_Ele:.4f}, Student Accuracy (mean): {acc_student:.4f}")
        y_pred_Ele = \
        predict_with_mean(pd.concat([pd.DataFrame(x_Ele), pd.DataFrame(y_extra, columns=['label'])], axis=1),
                          train_df)[0]['prediction'].values
        y_pred_student = \
        predict_with_mean(pd.concat([pd.DataFrame(x_student), pd.DataFrame(y_extra, columns=['label'])], axis=1),
                          train_df)[0]['prediction'].values
        plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele),
                                      method_name="Ele_mean")
        plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student),
                                      method_name="student_mean")

    # 3. SVD 基底分類
    print("\n[Method] SVD Classification")
    t0 = time.time()
    svd_bases = get_svd_bases(x_train, y_train)
    pred_df_svd, _, acc_svd, residuals_svd, mse_svd = svd_predict(x_test, y_test, svd_bases, max_basis=15)
    print(f"Accuracy: {acc_svd:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_svd = pred_df_svd['prediction'].values
    if 'actual' in pred_df_svd.columns:
        plot_images_labels_prediction(pred_df_svd.iloc[:, :256], pred_df_svd[['actual']], pred_df_svd[['prediction']], idx=0, num=10, method_name="svd")
        plot_residual_curves(residuals_svd, color='blue', method_name="svd")
        plot_mse_curve(mse_svd, method_name="svd")
        plot_confusion(y_test, y_pred_svd, method_name="svd")
        errors = pred_df_svd[pred_df_svd['actual'] != pred_df_svd['prediction']]
        plot_images_labels_prediction(errors.iloc[:, :256], errors[['actual']], errors[['prediction']], idx=0, num=10,
                                      method_name="svd_errors")

        correct = pred_df_svd[pred_df_svd['actual'] == pred_df_svd['prediction']]
        nice = correct.sort_values('residual').head(10)
        ugly = correct.sort_values('residual', ascending=False).head(10)
        plot_images_labels_prediction(nice.iloc[:, :256], nice[['actual']], nice[['prediction']], idx=0, num=10,
                                      method_name="svd_nice")
        plot_images_labels_prediction(ugly.iloc[:, :256], ugly[['actual']], ugly[['prediction']], idx=0, num=10,
                                      method_name="svd_ugly")
        y_pred_student_svd = svd_predict(x_student, y_extra, svd_bases, max_basis=15)[0]['prediction'].values
        plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_svd),
                                      method_name="student_svd")
        y_pred_Ele_svd = svd_predict(x_Ele, y_extra, svd_bases, max_basis=15)[0]['prediction'].values
        plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_svd),
                                      method_name="Ele_svd")

    # 4. HOSVD 方法
    print("\n[Method] HOSVD Classification")
    t0 = time.time()
    S_list, U_list = get_s_and_u_list(x_train, y_train)
    y_pred_hosvd, residuals_hosvd, mse_hosvd = hosvd_algorithm(x_test, y_test, S_list, U_list, basis=15)
    acc_hosvd = np.sum(y_pred_hosvd == y_test) / len(y_test)
    print(f"Accuracy: {acc_hosvd:.4f}, Runtime: {time.time() - t0:.2f}s")
    df_hosvd = pd.DataFrame(x_test)
    df_hosvd['actual'] = y_test
    df_hosvd['prediction'] = y_pred_hosvd
    df_hosvd['residual'] = [r[int(p)] for r, p in zip(residuals_hosvd, y_pred_hosvd)]

    plot_images_labels_prediction(df_hosvd.iloc[:, :256], df_hosvd[['actual']], df_hosvd[['prediction']], idx=0, num=10,
                                  method_name="hosvd")

    errors = df_hosvd[df_hosvd['actual'] != df_hosvd['prediction']]
    plot_images_labels_prediction(errors.iloc[:, :256], errors[['actual']], errors[['prediction']], idx=0, num=10,
                                  method_name="hosvd_errors")

    correct = df_hosvd[df_hosvd['actual'] == df_hosvd['prediction']]
    nice = correct.sort_values('residual').head(10)
    ugly = correct.sort_values('residual', ascending=False).head(10)
    plot_images_labels_prediction(nice.iloc[:, :256], nice[['actual']], nice[['prediction']], idx=0, num=10,
                                  method_name="hosvd_nice")
    plot_images_labels_prediction(ugly.iloc[:, :256], ugly[['actual']], ugly[['prediction']], idx=0, num=10,
                                  method_name="hosvd_ugly")
    plot_residual_curves(residuals_hosvd, color='green', method_name="hosvd")
    plot_mse_curve(mse_hosvd, method_name="hosvd")
    plot_confusion(y_test, y_pred_hosvd, method_name="hosvd")
    y_pred_student_hosvd = hosvd_algorithm(x_student, y_extra, S_list, U_list, basis=15)[0]
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_hosvd),
                                  method_name="student_hosvd")
    y_pred_Ele_hosvd = hosvd_algorithm(x_Ele, y_extra, S_list, U_list, basis=15)[0]
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_hosvd),
                                  method_name="Ele_hosvd")

    # 5. Random Forest
    print("\n[Method] Random Forest")
    t0 = time.time()
    rf_model = train_random_forest(x_train, y_train)
    acc_rf = evaluate_model(rf_model, x_test, y_test)
    if isinstance(acc_rf, tuple):
        acc_rf = acc_rf[0]
    print(f"Accuracy: {acc_rf:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_rf = rf_model.predict(x_test)
    plot_confusion(y_test, y_pred_rf, method_name="rf")
    y_pred_student_rf = rf_model.predict(x_student)
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_rf),
                                  method_name="student_rf")
    y_pred_Ele_rf = rf_model.predict(x_Ele)
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_rf),
                                  method_name="Ele_rf")

    # 6. MLP
    print("\n[Method] MLP")
    t0 = time.time()
    mlp_model, _ = train_mlp_model(x_train, y_train)
    acc_mlp = evaluate_mlp_model(mlp_model, x_test, y_test)
    print(f"Accuracy: {acc_mlp:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_mlp = mlp_model.predict(x_test).argmax(axis=1)
    plot_confusion(y_test, y_pred_mlp, method_name="mlp")
    y_pred_student_mlp = mlp_model.predict(x_student).argmax(axis=1)
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_mlp),
                                  method_name="student_mlp")
    y_pred_Ele_mlp = mlp_model.predict(x_Ele).argmax(axis=1)
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_mlp),
                                  method_name="Ele_mlp")


    # 7. CNN
    print("\n[Method] CNN")
    t0 = time.time()
    cnn_model = build_cnn_model()
    train_cnn_model(cnn_model, x_train.reshape(-1, 16, 16, 1), y_train)
    acc_cnn = evaluate_cnn_model(cnn_model, x_test.reshape(-1, 16, 16, 1), y_test)
    print(f"Accuracy: {acc_cnn:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_cnn = predict_cnn_model(cnn_model, x_test)


    df_cnn = pd.DataFrame(x_test)
    df_cnn['actual'] = y_test
    df_cnn['prediction'] = y_pred_cnn
    plot_confusion(y_test, y_pred_cnn, method_name="cnn")
    errors = df_cnn[df_cnn['actual'] != df_cnn['prediction']]
    plot_images_labels_prediction(errors.iloc[:, :256], errors[['actual']], errors[['prediction']], idx=0, num=10,
                                  method_name="cnn_errors")
    y_pred_student_cnn = predict_cnn_model(cnn_model, x_student)
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_cnn),
                                  method_name="student_cnn")
    y_pred_Ele_cnn = predict_cnn_model(cnn_model, x_Ele)
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_cnn),
                                  method_name="Ele_cnn")

    # 8. RNN
    print("\n[Method] RNN")
    t0 = time.time()
    rnn_model = build_rnn_model()
    train_rnn_model(rnn_model, x_train.reshape(-1, 16, 16).astype('float32'), y_train)
    acc_rnn = evaluate_rnn_model(rnn_model, x_test.reshape(-1, 16, 16).astype('float32'), y_test)
    print(f"Accuracy: {acc_rnn:.4f}, Runtime: {time.time() - t0:.2f}s")
    y_pred_rnn = predict_rnn_model(rnn_model, x_test.reshape(-1, 16, 16).astype('float32'))
    plot_confusion(y_test, y_pred_rnn, method_name="rnn")
    y_pred_student_rnn = predict_rnn_model(rnn_model, x_student.reshape(-1, 16, 16).astype('float32'))
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra), pd.DataFrame(y_pred_student_rnn),
                                  method_name="student_rnn")
    y_pred_Ele_rnn = predict_rnn_model(rnn_model, x_Ele.reshape(-1, 16, 16).astype('float32'))
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra), pd.DataFrame(y_pred_Ele_rnn),
                                  method_name="Ele_rnn")

    # 9. LSTM
    print("\n[Method] LSTM")
    t0 = time.time()
    acc_lstm, y_pred_lstm = train_and_evaluate_lstm(x_train, y_train, x_test, y_test)
    print(f"Accuracy: {acc_lstm:.4f}, Runtime: {time.time() - t0:.2f}s")
    plot_confusion(y_test, y_pred_lstm, method_name="lstm")

    # Extra validation for LSTM
    print("\n[Extra] LSTM")
    x_student_tensor = x_student.reshape(-1, 16, 16).astype('float32')
    y_extra_tensor = np.arange(10)
    from keras.utils import to_categorical
    y_extra_onehot = to_categorical(y_extra_tensor)
    acc_lstm_extra, y_pred_student_lstm = train_and_evaluate_lstm(x_train, y_train, x_student, y_extra)
    plot_images_labels_prediction(pd.DataFrame(x_student), pd.DataFrame(y_extra_tensor),
                                  pd.DataFrame(y_pred_student_lstm), method_name="student_lstm")

    # Extra validation for LSTM
    print("\n[Extra] LSTM Ele")
    x_Ele_tensor = x_Ele.reshape(-1, 16, 16).astype('float32')
    y_extra_tensor = np.arange(10)
    from keras.utils import to_categorical
    y_extra_onehot = to_categorical(y_extra_tensor)
    acc_lstm_Ele_extra, y_pred_Ele_lstm = train_and_evaluate_lstm(x_train, y_train, x_Ele, y_extra)
    plot_images_labels_prediction(pd.DataFrame(x_Ele), pd.DataFrame(y_extra_tensor),
                                  pd.DataFrame(y_pred_Ele_lstm), method_name="Ele_lstm")
    # 10. GCN
    #print("\n[Method] GCN")
    #t0 = time.time()
    #gcn_model, y_pred_gcn, acc_gcn = train_gcn_model(x_train.reshape(-1, 16, 16), y_train, x_test.reshape(-1, 16, 16),
    #                                                 y_test)
    #print(f"Accuracy: {acc_gcn:.4f}, Runtime: {time.time() - t0:.2f}s")
    #plot_confusion(y_test, y_pred_gcn.numpy(), method_name="gcn")

    # === Accuracy Tables ===
    acc_table_mean = get_classwise_accuracy_table(y_test, y_pred_mean)
    save_accuracy_table_as_image(acc_table_mean, method_name="mean")

    acc_table_svd = get_classwise_accuracy_table(y_test, y_pred_svd)
    save_accuracy_table_as_image(acc_table_svd, method_name="svd")

    acc_table_hosvd = get_classwise_accuracy_table(y_test, y_pred_hosvd)
    save_accuracy_table_as_image(acc_table_hosvd, method_name="hosvd")

    acc_table_rf = get_classwise_accuracy_table(y_test, y_pred_rf)
    save_accuracy_table_as_image(acc_table_rf, method_name="rf")

    acc_table_mlp = get_classwise_accuracy_table(y_test, y_pred_mlp)
    save_accuracy_table_as_image(acc_table_mlp, method_name="mlp")

    acc_table_cnn = get_classwise_accuracy_table(y_test, y_pred_cnn)
    save_accuracy_table_as_image(acc_table_cnn, method_name="cnn")

    acc_table_rnn = get_classwise_accuracy_table(y_test, y_pred_rnn)
    save_accuracy_table_as_image(acc_table_rnn, method_name="rnn")

    acc_table_lstm = get_classwise_accuracy_table(y_test, y_pred_lstm)
    save_accuracy_table_as_image(acc_table_lstm, method_name="lstm")

    from utils.visualizer import save_prediction_error_analysis_table

    save_prediction_error_analysis_table(y_test, y_pred_mean, method_name="mean")
    save_prediction_error_analysis_table(y_test, y_pred_svd, method_name="svd")
    save_prediction_error_analysis_table(y_test, y_pred_hosvd, method_name="hosvd")
    save_prediction_error_analysis_table(y_test, y_pred_rf, method_name="rf")
    save_prediction_error_analysis_table(y_test, y_pred_mlp, method_name="mlp")
    save_prediction_error_analysis_table(y_test, y_pred_cnn, method_name="cnn")
    save_prediction_error_analysis_table(y_test, y_pred_rnn, method_name="rnn")
    save_prediction_error_analysis_table(y_test, y_pred_lstm, method_name="lstm")

    from utils.visualizer import plot_all_residual_curves_by_digit
    plot_all_residual_curves_by_digit(
        residuals_mean,
        y_test,
        method_name="mean"
    )
    plot_all_residual_curves_by_digit(
        residuals_svd,
        y_test,
        method_name="svd"
    )
    plot_all_residual_curves_by_digit(
        residuals_hosvd,
        y_test,
        method_name="hosvd"
    )

    student_acc_mean = compute_extra_accuracy(y_extra, y_pred_student)
    student_acc_svd = compute_extra_accuracy(y_extra, y_pred_student_svd)
    student_acc_hosvd = compute_extra_accuracy(y_extra, y_pred_student_hosvd)
    student_acc_rf = compute_extra_accuracy(y_extra, y_pred_student_rf)
    student_acc_mlp = compute_extra_accuracy(y_extra, y_pred_student_mlp)
    student_acc_cnn = compute_extra_accuracy(y_extra, y_pred_student_cnn)
    student_acc_rnn = compute_extra_accuracy(y_extra, y_pred_student_rnn)
    student_acc_lstm = compute_extra_accuracy(y_extra, y_pred_student_lstm)
    Ele_acc_mean = compute_extra_accuracy(y_extra, y_pred_Ele)
    Ele_acc_svd = compute_extra_accuracy(y_extra, y_pred_Ele_svd)
    Ele_acc_hosvd = compute_extra_accuracy(y_extra, y_pred_Ele_hosvd)
    Ele_acc_rf = compute_extra_accuracy(y_extra, y_pred_Ele_rf)
    Ele_acc_mlp = compute_extra_accuracy(y_extra, y_pred_Ele_mlp)
    Ele_acc_cnn = compute_extra_accuracy(y_extra, y_pred_Ele_cnn)
    Ele_acc_rnn = compute_extra_accuracy(y_extra, y_pred_Ele_rnn)
    Ele_acc_lstm = compute_extra_accuracy(y_extra, y_pred_Ele_lstm)


    methods_acc = [
        {"Method": "Simple Mean Comparison", "Test": acc_mean, "Japanese Colonial Period": Ele_acc_mean, "Student": student_acc_mean},
        {"Method": "SVD", "Test": acc_svd, "Japanese Colonial Period": Ele_acc_svd, "Student": student_acc_svd},
        {"Method": "HOSVD", "Test": acc_hosvd, "Japanese Colonial Period": Ele_acc_hosvd, "Student": student_acc_hosvd},
        {"Method": "Random Forest", "Test": acc_rf, "Japanese Colonial Period": Ele_acc_rf, "Student": student_acc_rf},
        {"Method": "MLP", "Test": acc_mlp, "Japanese Colonial Period": Ele_acc_mlp, "Student": student_acc_mlp},
        {"Method": "CNN", "Test": acc_cnn, "Japanese Colonial Period": Ele_acc_cnn, "Student": student_acc_cnn},
        {"Method": "RNN", "Test": acc_rnn, "Japanese Colonial Period": Ele_acc_rnn, "Student": student_acc_rnn},
        {"Method": "LSTM", "Test": acc_lstm, "Japanese Colonial Period": Ele_acc_lstm, "Student": student_acc_lstm},
    ]

    plot_overall_comparison_table(methods_acc)

    def compute_mean_table(x_data, y_data):
        df = pd.concat([pd.DataFrame(x_data), pd.DataFrame(y_data, columns=["label"])], axis=1)
        IMAGE_DIM = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        mean_table = []

        for digit in range(10):
            digit_imgs = df[df["label"] == digit].iloc[:, :IMAGE_DIM]
            mean_vector = digit_imgs.mean().values
            mean_table.append(np.append(mean_vector, digit))  # 加上 label 在最後一欄

        columns = list(range(IMAGE_DIM)) + ["label"]
        return pd.DataFrame(mean_table, columns=columns)

    mean_df = compute_mean_table(x_train, y_train)
    print(mean_df.head())

if __name__ == "__main__":
    main()
