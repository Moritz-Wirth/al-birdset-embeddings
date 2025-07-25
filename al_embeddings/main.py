import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
os.chdir(root)
import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np
import mlflow
from tqdm import tqdm

from skactiveml.classifier import SklearnMultilabelClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.utils import MISSING_LABEL, call_func

from al_embeddings.model.base import ModelBase
from al_embeddings.utils import FeatureDataset, iterative_train_test_split

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", UndefinedMetricWarning)
warnings.simplefilter("ignore", UserWarning)

print(os.getcwd())
print("imports done")



@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="main.yaml")
def main(cfg):
    #print(OmegaConf.to_yaml(cfg))
    model: ModelBase = hydra.utils.instantiate(cfg.embeddings.network)

    # train dataset initialization
    train_dataset = hydra.utils.instantiate(cfg.dataset["train"].data)
    train_dataset.load_dataset(**cfg.dataset["train"].load_dataset_kwargs)
    train_dataset.prepare_dataset()
    train_dataset = FeatureDataset(train_dataset, model, cfg.paths.dataset_dir, cfg.embeddings.loaders)
    X_train, y_true = train_dataset.build()

    # test dataset initialization or create split
    # TODO could move split-creation to dataset
    if not "split_size" in cfg.dataset["test"]:
        test_dataset = hydra.utils.instantiate(cfg.dataset["test"].data)
        test_dataset.load_dataset(**cfg.dataset["test"].load_dataset_kwargs)
        test_dataset.prepare_dataset()
        test_dataset = FeatureDataset(test_dataset, model, cfg.paths.dataset_dir, cfg.embeddings.loaders)
        X_test, y_test = test_dataset.build()
    else:
        X_train, X_test, y_true, y_test = iterative_train_test_split(X_train,
                                                                     y_true,
                                                                     test_size=cfg.dataset["test"].split_size,
                                                                     random_state=0)

    if cfg.only_create_embeddings:
        print("done creating embeddings")
        return

    print(f"Using {len(X_train)} for training and {len(X_test)} for testing.")

    if cfg.al.n_cycles * cfg.dataset.batch_size > len(X_train):
        cfg.dataset.batch_size = len(X_train) // cfg.al.n_cycles
        print(f"Warning: setting batch size to maximum of {cfg.dataset.batch_size}.")

    # AL cycle
    sklearn_clf = hydra.utils.instantiate(cfg.classifier)
    metrics = hydra.utils.instantiate(cfg.metrics)

    clf = SklearnMultilabelClassifier(sklearn_clf, classes=np.arange(y_true.shape[-1])) # TODO move or abstract to configs? or at least support multiclass
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    # init pool
    init_pool_qs = hydra.utils.instantiate(cfg.al.init_strategy.strategy)
    init_pool_qs_kwargs = hydra.utils.instantiate(cfg.al.init_strategy.query_kwargs)

    init_pool_idx = init_pool_qs.query(X=X_train, y=y, **init_pool_qs_kwargs)
    y[init_pool_idx] = y_true[init_pool_idx]
    clf.fit(X=X_train, Y=y)

    # mlflow setup
    mlflow.set_tracking_uri(uri=cfg.mlflow.uri)
    mlflow.set_experiment(format_mlflow_tracking_names(cfg.mlflow.experiment_name))
    run_name = format_mlflow_tracking_names(cfg.mlflow.run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_cfg(cfg))

        pred_test = clf.predict(X_test)
        proba_test = clf.predict_proba(X_test)

        test_scores = metrics(y_true=y_test, y_pred=pred_test, y_proba=proba_test)
        mlflow.log_metrics(test_scores, step=0)

        qs = hydra.utils.instantiate(cfg.al.query_strategy.strategy)
        qs_kwargs = hydra.utils.instantiate(cfg.al.query_strategy.query_kwargs)

        for step in tqdm(range(1, cfg.al.n_cycles + 1)):
            query_idx = call_func(qs.query, X=X_train, y=y, clf=clf, **qs_kwargs)
            y[query_idx] = y_true[query_idx]
            clf.fit(X_train, y)

            # test metric
            pred_test = clf.predict(X_test)
            proba_test = clf.predict_proba(X_test)
            test_scores = metrics(y_true=y_test, y_pred=pred_test, y_proba=proba_test)

            mlflow.log_metrics(test_scores, step=step)
            mlflow.log_metric("Aquesition Size", (step+1) * qs_kwargs["batch_size"], step=step)


def flatten_cfg(cfg, parent_key='', sep='.'):
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if "._target_" in new_key:
            new_key = new_key.replace("._target_", "")
            v = v.split(".")[-1]

        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_mlflow_tracking_names(name: str) -> str:
    names = name.split("-")
    names = [n.split(".")[-1] for n in names]
    return "-".join(names)


if __name__ == "__main__":
    main()