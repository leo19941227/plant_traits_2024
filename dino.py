import random
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
from sklearn.model_selection import train_test_split


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
    parser.add_argument("--n_valid", type=int, default=4096)
    parser.add_argument("--filter_low", type=float, default=0.001)
    parser.add_argument("--filter_high", type=float, default=0.981)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    seed_all(args.seed)

    train_image_dir = Path(args.train_image_dir)
    test_image_dir = Path(args.test_image_dir)

    # load pickled dataframes from a public dataset; split to train-val
    train0 = pd.read_csv(args.train_csv)
    train0["file_path"] = train0["id"].apply(
        lambda s: str(train_image_dir / f"{s}.jpeg")
    )

    test = pd.read_csv(args.test_csv)
    test["file_path"] = test["id"].apply(lambda s: str(test_image_dir / f"{s}.jpeg"))

    train, val = train_test_split(
        train0, test_size=args.n_valid, shuffle=True, random_state=args.seed
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    labels_describe_df = (
        train[TARGET_COLUMNS]
        .describe(percentiles=[args.filter_low, args.filter_high])
        .T
    )
    # Masks
    mask_train = get_mask(train, labels_describe_df)
    mask_val = get_mask(val, labels_describe_df)
    # Masked DataFrames
    train_mask = train[mask_train].reset_index(drop=True)
    val_mask = val[mask_val].reset_index(drop=True)

    for m, subset, full in zip([train_mask, val_mask], ["train", "val"], [train, val]):
        print(f"===== {subset} shape: {m.shape} =====")
        n_masked = len(full) - len(m)
        perc_masked = (n_masked / len(full)) * 100
        print(f"{subset} \t| # Masked Samples: {n_masked}")
        print(f"{subset} \t| % Masked Samples: {perc_masked:.3f}%")

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
        valid_image_embeddings = get_image_embeddings_dino(
            model, preprocess, args.batch_size, val, args.device
        )
        test_image_embeddings = get_image_embeddings_dino(
            model, preprocess, args.batch_size, test, args.device
        )

        np.save(train_emb_path, np.array(train_image_embeddings))
        np.save(valid_emb_path, np.array(valid_image_embeddings))
        np.save(test_emb_path, np.array(test_image_embeddings))

    train_image_embeddings = np.load(train_emb_path)[mask_train]
    val_image_embeddings = np.load(valid_emb_path)[mask_val]
    test_image_embeddings = np.load(test_emb_path)

    train_features_mask_df = pd.DataFrame(train_image_embeddings)
    val_features_mask_df = pd.DataFrame(val_image_embeddings)
    test_features_mask_df = pd.DataFrame(test_image_embeddings)

    models = {}
    scores = {}
    y_train_mask = train_mask[TARGET_COLUMNS].values
    y_val_mask = val_mask[TARGET_COLUMNS].values

    for i, col in tqdm(enumerate(TARGET_COLUMNS), total=len(TARGET_COLUMNS)):
        y_curr = y_train_mask[:, i]
        y_curr_val = y_val_mask[:, i]
        train_pool = Pool(train_features_mask_df, y_curr)
        val_pool = Pool(val_features_mask_df, y_curr_val)

        # tried to tune these parameters but without real success
        model = CatBoostRegressor(
            iterations=50,
            learning_rate=0.06,
            loss_function="RMSE",
            verbose=0,
            random_state=args.seed,
        )
        model.fit(train_pool)
        models[col] = model

        y_curr_val_pred = model.predict(val_pool)

        r2_col = r2_score(y_curr_val, y_curr_val_pred)
        scores[col] = r2_col
        print(f"Target: {col}, R2: {r2_col:.3f}")
    # this val score somewhat correlates with submission score bit I didn't really bother
    print(f"Mean R2: {np.mean(list(scores.values())):.3f}")

    submission = pd.DataFrame({"id": test["id"]})
    submission[TARGET_COLUMNS] = 0
    submission.columns = submission.columns.str.replace("_mean", "")

    for i, col in enumerate(TARGET_COLUMNS):
        test_pool = Pool(test_features_mask_df)
        col_pred = models[col].predict(test_pool)
        submission[col.replace("_mean", "")] = col_pred

    Path(args.output_csv).parent.mkdir(exist_ok=True, parents=True)
    submission.to_csv(args.output_csv, index=False)
    submission.head()
