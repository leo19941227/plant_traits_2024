import random
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import r2_score
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


TARGET_COLUMNS = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X50_mean",
    "X26_mean",
    "X3112_mean",
]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mask(df, labels_describe_df):
    mask = np.empty(shape=df[TARGET_COLUMNS].shape, dtype=bool)
    for idx, t in enumerate(TARGET_COLUMNS):
        labels = df[t].values
        v_min, v_max = (
            labels_describe_df.loc[t][f"{args.filter_low * 100}%"],
            labels_describe_df.loc[t][f"{args.filter_high * 100}%"],
        )
        mask[:, idx] = (labels > v_min) & (labels < v_max)
    return mask.min(axis=1)  # only use the row when all the labels are valid


def get_image_embeddings_dino(
    model, preprocess, batch_size, df, device: str = "cuda:0"
):
    image_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        paths = df["file_path"][i : i + batch_size]
        image_tensor = torch.stack([preprocess(Image.open(path)) for path in paths]).to(
            device
        )
        with torch.no_grad():
            curr_image_embeddings = model(image_tensor)
        image_embeddings.extend(curr_image_embeddings.cpu().numpy())
    return image_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_csv")
    parser.add_argument("--train_csv", default="./data/train.csv")
    parser.add_argument("--test_csv", default="./data/test.csv")
    parser.add_argument("--train_image_dir", default="./data/train_images/")
    parser.add_argument("--test_image_dir", default="./data/test_images/")
    parser.add_argument("--recompute_embedding", action="store_true")
    parser.add_argument("--embedding_dir", default="./embeddings/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--filter_low", type=float, default=0.001)
    parser.add_argument("--filter_high", type=float, default=0.981)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_auxiliary", action="store_true")
    parser.add_argument("--filter_outlier", action="store_true")
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.06)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    seed_all(args.seed)

    train_image_dir = Path(args.train_image_dir)
    test_image_dir = Path(args.test_image_dir)

    # load pickled dataframes from a public dataset; split to train-val
    train_raw = pd.read_csv(args.train_csv)
    train_raw["file_path"] = train_raw["id"].apply(
        lambda s: str(train_image_dir / f"{s}.jpeg")
    )

    test = pd.read_csv(args.test_csv)
    test["file_path"] = test["id"].apply(lambda s: str(test_image_dir / f"{s}.jpeg"))
    feature_columns = test.columns.values[1:-1]

    n_valid = round(len(train_raw) * args.valid_ratio)
    train, val = train_test_split(
        train_raw, test_size=n_valid, shuffle=True, random_state=args.seed
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    logger.info(f"[train/val split] train: {len(train)}; valid: {len(val)}")

    if args.filter_outlier:
        labels_describe_df = (
            train[TARGET_COLUMNS]
            .describe(percentiles=[args.filter_low, args.filter_high])
            .T
        )
        train_masking = get_mask(train, labels_describe_df)
        val_masking = get_mask(val, labels_describe_df)
    else:
        train_masking = np.ones((len(train),), dtype=bool)
        val_masking = np.ones((len(val),), dtype=bool)

    train_masked = train[train_masking].reset_index(drop=True)
    val_masked = val[val_masking].reset_index(drop=True)
    logger.info(
        f"[outlier filtering] train: {len(train_masked)}; valid: {len(val_masked)}"
    )

    if args.use_auxiliary:
        FEATURE_SCALER = StandardScaler()
        train_features_mask = FEATURE_SCALER.fit_transform(
            train_masked[feature_columns].values.astype(np.float32)
        )
        val_features_mask = FEATURE_SCALER.transform(
            val_masked[feature_columns].values.astype(np.float32)
        )
        test_features = FEATURE_SCALER.transform(
            test[feature_columns].values.astype(np.float32)
        )

    emb_dir = Path(args.embedding_dir) / "dinov2_vitg14_reg"
    emb_dir.mkdir(exist_ok=True, parents=True)
    train_emb_path = emb_dir / "train.npy"
    valid_emb_path = emb_dir / "valid.npy"
    test_emb_path = emb_dir / "test.npy"

    if args.recompute_embedding:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg").to(
            args.device
        )
        model.eval()
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        suffix = "image_embs_dinov2_vitg14_reg"
        train_image_embeddings = get_image_embeddings_dino(
            model, preprocess, args.batch_size, train, args.device
        )
        val_image_embeddings = get_image_embeddings_dino(
            model, preprocess, args.batch_size, val, args.device
        )
        test_image_embeddings = get_image_embeddings_dino(
            model, preprocess, args.batch_size, test, args.device
        )

        np.save(train_emb_path, np.array(train_image_embeddings))
        np.save(valid_emb_path, np.array(val_image_embeddings))
        np.save(test_emb_path, np.array(test_image_embeddings))

    train_final_feat = np.load(train_emb_path)[train_masking]
    val_final_feat = np.load(valid_emb_path)[val_masking]
    test_final_feat = np.load(test_emb_path)

    if args.use_auxiliary:
        train_final_feat = np.concatenate(
            (train_features_mask, train_final_feat), axis=1
        )
        val_final_feat = np.concatenate((val_features_mask, val_final_feat), axis=1)
        test_final_feat = np.concatenate((test_features, test_final_feat), axis=1)

    models = {}
    scores = {}
    y_train_masked = train_masked[TARGET_COLUMNS].values
    y_val_masked = val_masked[TARGET_COLUMNS].values

    for i, col in tqdm(enumerate(TARGET_COLUMNS), total=len(TARGET_COLUMNS)):
        y_curr = y_train_masked[:, i]
        y_curr_val = y_val_masked[:, i]
        train_pool = Pool(train_final_feat, y_curr)
        val_pool = Pool(val_final_feat, y_curr_val)

        # tried to tune these parameters but without real success
        model = CatBoostRegressor(
            iterations=args.n_iter,
            learning_rate=args.lr,
            loss_function="RMSE",
            verbose=0,
            random_state=args.seed,
        )
        model.fit(train_pool)
        models[col] = model

        y_curr_val_pred = model.predict(val_pool)

        r2_col = r2_score(y_curr_val, y_curr_val_pred)
        scores[col] = r2_col
        logger.info(f"{col} R2: {r2_col}")
    logger.info(f"Mean R2: {np.mean(list(scores.values()))}")

    submission = pd.DataFrame({"id": test["id"]})
    submission[TARGET_COLUMNS] = 0
    submission.columns = submission.columns.str.replace("_mean", "")

    for i, col in enumerate(TARGET_COLUMNS):
        test_pool = Pool(test_final_feat)
        col_pred = models[col].predict(test_pool)
        submission[col.replace("_mean", "")] = col_pred

    Path(args.output_csv).parent.mkdir(exist_ok=True, parents=True)
    submission.to_csv(args.output_csv, index=False)
    submission.head()
