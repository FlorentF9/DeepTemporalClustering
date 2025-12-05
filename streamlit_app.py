import os
from time import time
from typing import Tuple

import numpy as np
import streamlit as st

from DeepTemporalClustering import DTC
from datasets import all_ucr_datasets, load_data


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and cache a UCR/UEA dataset by name."""
    return load_data(dataset_name)


def run_training(dataset_name: str,
                n_clusters: int,
                n_filters: int,
                kernel_size: int,
                strides: int,
                pool_size: int,
                n_units: Tuple[int, int],
                gamma: float,
                alpha: float,
                dist_metric: str,
                cluster_init: str,
                heatmap: bool,
                pretrain_epochs: int,
                epochs: int,
                eval_epochs: int,
                save_epochs: int,
                batch_size: int,
                tol: float,
                patience: int,
                finetune_heatmap_at_epoch: int,
                initial_heatmap_loss_weight: float,
                final_heatmap_loss_weight: float,
                save_dir: str):
    """Train a DTC model with the provided hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)

    st.info("Loading dataset…")
    X_train, y_train = load_dataset(dataset_name)

    if n_clusters is None:
        n_clusters = len(np.unique(y_train))

    dtc = DTC(
        n_clusters=n_clusters,
        input_dim=X_train.shape[-1],
        timesteps=X_train.shape[1],
        n_filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        pool_size=pool_size,
        n_units=n_units,
        alpha=alpha,
        dist_metric=dist_metric,
        cluster_init=cluster_init,
        heatmap=heatmap,
    )

    dtc.initialize()
    dtc.compile(
        gamma=gamma,
        optimizer="adam",
        initial_heatmap_loss_weight=initial_heatmap_loss_weight,
        final_heatmap_loss_weight=final_heatmap_loss_weight,
    )

    if pretrain_epochs > 0:
        with st.spinner("Pretraining autoencoder…"):
            dtc.pretrain(X=X_train, optimizer="adam", epochs=pretrain_epochs, batch_size=batch_size, save_dir=save_dir)

    dtc.init_cluster_weights(X_train)

    with st.spinner("Training clustering model…"):
        t0 = time()
        dtc.fit(
            X_train,
            y_train,
            None,
            None,
            epochs,
            eval_epochs,
            save_epochs,
            batch_size,
            tol,
            patience,
            finetune_heatmap_at_epoch,
            save_dir,
        )
        training_time = time() - t0

    st.success(f"Training finished in {training_time:.2f} seconds.")

    q = dtc.model.predict(X_train)[1]
    y_pred = q.argmax(axis=1)
    results = {"n_clusters": n_clusters}

    if y_train is not None:
        from metrics import cluster_acc, cluster_purity
        from sklearn import metrics

        results.update(
            acc=cluster_acc(y_train, y_pred),
            pur=cluster_purity(y_train, y_pred),
            nmi=metrics.normalized_mutual_info_score(y_train, y_pred),
            ari=metrics.adjusted_rand_score(y_train, y_pred),
        )

    return results


def main():
    st.set_page_config(page_title="Deep Temporal Clustering", layout="wide")
    st.title("Deep Temporal Clustering")
    st.write(
        "Train and evaluate the DTC model on datasets from the UCR/UEA archive. "
        "Tune hyperparameters in the sidebar, then start a run when you are ready."
    )

    with st.sidebar:
        st.header("Configuration")
        dataset_name = st.selectbox("Dataset", sorted(all_ucr_datasets), index=sorted(all_ucr_datasets).index("CBF"))

        # Determine valid pool sizes for the selected dataset (pool_size must divide timesteps)
        X_sample, _ = load_dataset(dataset_name)
        timesteps = X_sample.shape[1]
        valid_pool_sizes = [size for size in range(1, timesteps + 1) if timesteps % size == 0]
        default_pool_size = 8 if 8 in valid_pool_sizes else max([size for size in valid_pool_sizes if size <= 8], default=min(valid_pool_sizes))

        n_clusters = st.number_input("Clusters (0 = infer from labels)", min_value=0, value=0, step=1)
        n_filters = st.number_input("Conv filters", min_value=1, value=50, step=1)
        kernel_size = st.number_input("Kernel size", min_value=1, value=10, step=1)
        strides = st.number_input("Strides", min_value=1, value=1, step=1)
        pool_size = st.selectbox(
            "Pool size",
            valid_pool_sizes,
            index=valid_pool_sizes.index(default_pool_size),
            help=f"Must divide the sequence length ({timesteps} timesteps).",
        )
        n_units_first = st.number_input("BiLSTM units (first layer)", min_value=1, value=50, step=1)
        n_units_second = st.number_input("BiLSTM units (second layer)", min_value=1, value=1, step=1)
        gamma = st.number_input("Gamma (clustering loss weight)", min_value=0.0, value=1.0, step=0.1)
        alpha = st.number_input("Alpha (Student kernel)", min_value=0.1, value=1.0, step=0.1)
        dist_metric = st.selectbox("Distance metric", ["eucl", "cid", "cor", "acf"], index=0)
        cluster_init = st.selectbox("Cluster initialization", ["kmeans", "hierarchical"], index=0)
        heatmap = st.checkbox("Train heatmap network", value=False)
        pretrain_epochs = st.number_input("Pretrain epochs", min_value=0, value=10, step=1)
        epochs = st.number_input("Training epochs", min_value=1, value=50, step=1)
        eval_epochs = st.number_input("Eval every N epochs", min_value=1, value=1, step=1)
        save_epochs = st.number_input("Save weights every N epochs", min_value=1, value=10, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=64, step=1)
        tol = st.number_input("Tolerance", min_value=0.0, value=0.001, step=0.0001, format="%f")
        patience = st.number_input("Patience", min_value=1, value=5, step=1)
        finetune_heatmap_at_epoch = st.number_input("Heatmap finetune epoch", min_value=1, value=8, step=1)
        initial_heatmap_loss_weight = st.number_input(
            "Initial heatmap loss weight", min_value=0.0, value=0.1, step=0.1
        )
        final_heatmap_loss_weight = st.number_input(
            "Final heatmap loss weight", min_value=0.0, value=0.9, step=0.1
        )
        save_dir = st.text_input("Save directory", value="results/tmp")

        start_training = st.button("Start training", type="primary")

    if start_training:
        if n_clusters == 0:
            n_clusters_value = None
        else:
            n_clusters_value = n_clusters

        results = run_training(
            dataset_name,
            n_clusters_value,
            n_filters,
            kernel_size,
            strides,
            pool_size,
            (n_units_first, n_units_second),
            gamma,
            alpha,
            dist_metric,
            cluster_init,
            heatmap,
            pretrain_epochs,
            epochs,
            eval_epochs,
            save_epochs,
            batch_size,
            tol,
            patience,
            finetune_heatmap_at_epoch,
            initial_heatmap_loss_weight,
            final_heatmap_loss_weight,
            save_dir,
        )

        st.subheader("Training results")
        st.json(results)
    else:
        st.info("Configure the training parameters in the sidebar and click **Start training** to begin.")


if __name__ == "__main__":
    main()
