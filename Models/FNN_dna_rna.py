import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle, resample
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, make_scorer

import torch
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import Hyperband
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Metric, F1Score
from tensorflow.keras.losses import Loss
from keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras

data_dir = os.path.join(os.getcwd(), "data")
models_dir = os.path.join(os.getcwd(), "Models")
results_dir = os.path.join(os.getcwd(), "results")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# LOADING DATA
def load_sample_data(var_path, ref_path, target_path):
    var_df = pd.read_csv(var_path, index_col=0)   # variant reads
    ref_df = pd.read_csv(ref_path, index_col=0)   # reference reads
    target_df = pd.read_csv(target_path)          # target labels

    var_df.dropna(axis=1, how="any", inplace=True)
    ref_df.dropna(axis=1, how="any", inplace=True)

    assert var_df.index.equals(ref_df.index) and var_df.columns.equals(ref_df.columns)

    # Filter out invalid SNVs
    snvs_to_remove = target_df[target_df["MutationType"] == -1]["SNV"]
    var_df_filtered = var_df[~var_df.index.isin(snvs_to_remove)]
    ref_df_filtered = ref_df[~ref_df.index.isin(snvs_to_remove)]
    target_df_filtered = target_df[~target_df["SNV"].isin(snvs_to_remove)]

    print("var_filt shape:", var_df_filtered.shape,
          "ref_filt shape:", ref_df_filtered.shape,
          "target_filt shape:", target_df_filtered.shape)

    return var_df_filtered, ref_df_filtered, target_df_filtered["MutationType"].values



# PREPROCESS DATA
def preprocess_data(var_df, ref_df, y_arr):
    X_var = var_df.values[:, 1:]
    X_ref = ref_df.values[:, 1:]
    y_arr = y_arr.astype(int)

    # Convert classes to binary
    y_arr[y_arr == 1] = 0  # DNA class
    y_arr[y_arr == 2] = 1  # RNA class

    # Shuffle data
    combined_data = shuffle(np.hstack((X_var, X_ref, y_arr.reshape(-1, 1))))
    X_var_shuffled = combined_data[:, :X_var.shape[1]]
    X_ref_shuffled = combined_data[:, X_var.shape[1] : X_var.shape[1] + X_ref.shape[1]]
    y_shuffled = combined_data[:, -1].astype(int)

    # Train-validation-test split
    X_var_temp, X_var_test, X_ref_temp, X_ref_test, y_temp, y_test = train_test_split(
        X_var_shuffled, X_ref_shuffled, y_shuffled, test_size=0.2,random_state=1,stratify=y_shuffled
    )

    X_var_train, X_var_val, X_ref_train, X_ref_val, y_train, y_val = train_test_split(
        X_var_temp, X_ref_temp, y_temp,test_size=0.25, random_state=1, stratify=y_temp
    )

    # Scale data using MaxAbsScaler
    scaler_var = MaxAbsScaler()
    X_var_train_scaled = scaler_var.fit_transform(X_var_train)
    X_var_val_scaled = scaler_var.transform(X_var_val)
    X_var_test_scaled = scaler_var.transform(X_var_test)

    scaler_ref = MaxAbsScaler()
    X_ref_train_scaled = scaler_ref.fit_transform(X_ref_train)
    X_ref_val_scaled = scaler_ref.transform(X_ref_val)
    X_ref_test_scaled = scaler_ref.transform(X_ref_test)

    # combining varreads and refreads into 2D tensors (concatenated features' datasets)
    X_train_combined = np.hstack((X_var_train_scaled, X_ref_train_scaled))
    X_val_combined = np.hstack((X_var_val_scaled, X_ref_val_scaled))
    X_test_combined = np.hstack((X_var_test_scaled, X_ref_test_scaled))

    return X_train_combined, X_val_combined, X_test_combined, y_train, y_val, y_test



# FNN HyperModel for Keras Tuner
class FNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(layers.Input(shape=(X_train_combined_final.shape[1:])))

        # fully connected layers
        for i in range(3):
            model.add(layers.Dense(units=hp.Choice(f"units_{i}", 
                                                   values=[100, 200, 400, 800, 1600]), 
                                   activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(hp.Float(f"dropout_{i}", 
                                              min_value=0.2, 
                                              max_value=0.5, step=0.1)))

        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=Adam(hp.Choice("learning_rate",
                                               values=[0.01, 0.001, 0.0001])),
                      loss=focal_loss(alpha=0.25, gamma=2.0),
                      metrics=["accuracy", f1_score, "AUC"])

        return model



# # memory cleanup after trial
# class MemoryCleanupCallback(tf.keras.callbacks.Callback):
#     def on_trial_end(self, trial_id, logs=None):
#         K.clear_session()
#         gc.collect()



# HYPERBAND TUNING
def tune_and_train(X_train, y_train, X_val, y_val, models_dir, sample_id):
    tuner = Hyperband(FNNHyperModel(),
                      objective="val_loss",
                      max_epochs=150,
                      factor=2,
                      directory=models_dir,
                      project_name=f"hyperband_FNN_{sample_id}")

    stop_early = EarlyStopping(monitor="val_loss", 
                               patience=10, 
                               restore_best_weights=False)
    tuner.search(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 callbacks=[stop_early])

    for trial_id, trial in tuner.oracle.trials.items():
        torch.cuda.empty_cache() 
        gc.collect()             
        print(f"Cleared GPU memory after trial {trial_id}")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=150, callbacks=[stop_early], verbose=2)

    best_model_path = os.path.join(models_dir, f"{sample_id}_FNN_model.h5")
    model.save(best_model_path)
    print(f"Best model saved for sample {sample_id} at {best_model_path}")

    return model



# focal loss function
def focal_loss(alpha = 0.25, gamma = 2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = -alpha * (1 - pt) ** gamma * K.log(pt)
        return K.mean(focal_term)
    return loss



# custom f1 score to be used in model.compile
def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(K.round(K.clip(y_pred, 0, 1)), "float32")

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1



# EVALUATE MODEL
def evaluate_model(model, X_test, y_test, sample_id, output_dir):
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    kappa = cohen_kappa_score(y_test, y_pred)

    plt.figure(figsize = (7, 5))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = ['DNA', 'RNA'], yticklabels = ['DNA', 'RNA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {sample_id}')
    image_path = os.path.join(output_dir, f'{sample_id}_FNN_conf_mat.png')
    plt.savefig(image_path, dpi = 300)
    plt.close()

    return f1_macro, kappa, precision_macro, recall_macro, roc_auc, conf_matrix



# MAIN EXECUTION
sample_ids = ["nanopore_SRR21492154", "nanopore_SRR21492155", "nanopore_SRR21492156",
              "nanopore_SRR21492157", "nanopore_SRR21492158", "nanopore_SRR21492159"]
all_results = []

for sample_id in sample_ids:
    print(f"\nProcessing sample: {sample_id}")

    var_path = os.path.join(data_dir, f"{sample_id}_varreads.csv")
    ref_path = os.path.join(data_dir, f"{sample_id}_refreads.csv")
    target_path = os.path.join(data_dir, f"{sample_id}_targets.csv")

    if not all(map(os.path.exists, [var_path, ref_path, target_path])):
        print(f"Missing files for sample {sample_id} in the 'data/' directory.")
        continue

    var_df, ref_df, y_arr = load_sample_data(var_path, ref_path, target_path)
    X_train_combined_final, X_val_combined_final, X_test_combined_final, y_train, y_val, y_test = preprocess_data(var_df, ref_df, y_arr)

    output_dir = os.path.join(results_dir, f"FNN_{sample_id}")
    os.makedirs(output_dir, exist_ok=True)

    # train and evaluate
    model = tune_and_train(X_train_combined_final, y_train, X_val_combined_final, y_val, models_dir, sample_id)
    f1_macro, kappa, precision_macro, recall_macro, roc_auc, conf_matrix = evaluate_model(
        model, X_test_combined_final, y_test, sample_id, output_dir
    )

    all_results.append({"sample_id": sample_id,
                        "f1_macro": f1_macro,
                        "kappa": kappa,
                        "precision_macro": precision_macro,
                        "recall_macro": recall_macro,
                        "roc_auc": roc_auc})

results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(results_dir, "FNN_hyperband_results.csv")
results_df.to_csv(results_csv_path, index=False)
print("All results of FNN saved to 'results/' directory.")
