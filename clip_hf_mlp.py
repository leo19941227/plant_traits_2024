import sys
import random
import logging
import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModel

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


def get_image_embeddings(model, processor, batch_size, df, device: str = "cuda:0"):
    image_embeddings = []
    all_pooled_hs = []
    for i in tqdm(range(0, len(df), batch_size)):
        paths = df["file_path"][i : i + batch_size]
        inputs = processor(
            [Image.open(path) for path in paths], return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs, output_hidden_states=True)
        image_embeddings.append(image_emb.detach().cpu())
    image_embeddings = torch.cat(image_embeddings, dim=0)
    return image_embeddings, image_embeddings


def log_message(message: str, log_file: str):
    logger.info(message)
    with open(log_file, "a") as f:
        print(message, file=f)


class Mlp(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size: int = 768,
        hidden_layer: int = 3,
        lr: float = 1.0e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert hidden_layer >= 1
        self.lr = lr

        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layer - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        pred = self.layers(x).view(-1)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("loss", loss, prog_bar=True, on_step=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PlantTraitsDataset(Dataset):
    def __init__(self, features, labels=None) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]


def train_mlp(args, embed_size, train_dataset, valid_dataset, test_dataset):
    model = Mlp(embed_size, args.hidden_size, args.hidden_layer, args.lr)

    def train_collate_fn(samples):
        features, labels = zip(*samples)
        features = torch.stack([torch.FloatTensor(f) for f in features], dim=0)
        labels = torch.FloatTensor(labels)
        return features, labels

    def infer_collate_fn(samples):
        features = torch.stack([torch.FloatTensor(f) for f in samples], dim=0)
        return features

    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=infer_collate_fn,
        num_workers=args.num_workers,
    )
    test_laoder = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=infer_collate_fn,
        num_workers=args.num_workers,
    )

    trainer = L.Trainer(max_steps=args.n_iter, accelerator="gpu", devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
    valid_preds = trainer.predict(model=model, dataloaders=valid_loader)
    test_preds = trainer.predict(model=model, dataloaders=test_laoder)
    valid_preds = torch.cat(valid_preds)
    test_preds = torch.cat(test_preds)
    return valid_preds, test_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--train_csv", default="./data/train.csv")
    parser.add_argument("--test_csv", default="./data/test.csv")
    parser.add_argument("--train_image_dir", default="./data/train_images/")
    parser.add_argument("--test_image_dir", default="./data/test_images/")
    parser.add_argument("--recompute_embedding", action="store_true")
    parser.add_argument("--embedding_dir", default="./embeddings/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--filter_outlier", action="store_false")
    parser.add_argument("--filter_low", type=float, default=0.012)
    parser.add_argument("--filter_high", type=float, default=0.991)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_auxiliary", action="store_true")
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--hidden_layer", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--use_hidden_state", type=int)
    parser.add_argument("--use_ws", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / "predict.csv"
    output_log = output_dir / "log.txt"
    with open(output_log, "w") as f:
        print(" ".join(sys.argv), file=f)
    log = partial(log_message, log_file=output_log)

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
    log(f"[train/val split] train: {len(train)}; valid: {len(val)}")

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
    log(f"[outlier filtering] train: {len(train_masked)}; valid: {len(val_masked)}")

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

    model_name = args.model_name.replace("/", "__")
    emb_dir = Path(args.embedding_dir) / model_name
    emb_dir.mkdir(exist_ok=True, parents=True)
    train_emb_path = emb_dir / "train.npy"
    valid_emb_path = emb_dir / "valid.npy"
    test_emb_path = emb_dir / "test.npy"
    train_pooled_path = emb_dir / "train_pooled.npy"
    valid_pooled_path = emb_dir / "valid_pooled.npy"
    test_pooled_path = emb_dir / "test_pooled.npy"

    if args.recompute_embedding:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name).to(args.device)
        model.eval()

        train_image_embeddings, train_pooled = get_image_embeddings(
            model, processor, 64, train, args.device
        )
        val_image_embeddings, val_pooled = get_image_embeddings(
            model, processor, 64, val, args.device
        )
        test_image_embeddings, test_pooled = get_image_embeddings(
            model, processor, 64, test, args.device
        )

        np.save(train_emb_path, np.array(train_image_embeddings))
        np.save(valid_emb_path, np.array(val_image_embeddings))
        np.save(test_emb_path, np.array(test_image_embeddings))

        np.save(train_pooled_path, np.array(train_pooled))
        np.save(valid_pooled_path, np.array(val_pooled))
        np.save(test_pooled_path, np.array(test_pooled))

    if isinstance(args.use_hidden_state, int):
        train_final_feat = np.load(train_pooled_path)[train_masking][
            :, args.use_hidden_state, :
        ]
        val_final_feat = np.load(valid_pooled_path)[val_masking][
            :, args.use_hidden_state, :
        ]
        test_final_feat = np.load(test_pooled_path)[:, args.use_hidden_state, :]
    elif args.use_ws:
        train_final_feat = np.load(train_pooled_path)[train_masking]
        val_final_feat = np.load(valid_pooled_path)[val_masking]
        test_final_feat = np.load(test_pooled_path)
    else:
        train_final_feat = np.load(train_emb_path)[train_masking]
        val_final_feat = np.load(valid_emb_path)[val_masking]
        test_final_feat = np.load(test_emb_path)

    if args.use_auxiliary:
        train_final_feat = np.concatenate(
            (train_features_mask, train_final_feat), axis=1
        )
        val_final_feat = np.concatenate((val_features_mask, val_final_feat), axis=1)
        test_final_feat = np.concatenate((test_features, test_final_feat), axis=1)

    scores = {}
    all_test_preds = {}
    y_train_masked = train_masked[TARGET_COLUMNS].values
    y_val_masked = val_masked[TARGET_COLUMNS].values

    for i, col in tqdm(enumerate(TARGET_COLUMNS), total=len(TARGET_COLUMNS)):
        y_curr = y_train_masked[:, i]
        y_curr_val = y_val_masked[:, i]

        train_dataset = PlantTraitsDataset(train_final_feat, y_curr)
        valid_dataset = PlantTraitsDataset(val_final_feat)
        test_dataset = PlantTraitsDataset(test_final_feat)

        valid_preds, test_preds = train_mlp(
            args, train_final_feat.shape[1], train_dataset, valid_dataset, test_dataset
        )

        r2_col = r2_score(y_curr_val, valid_preds.tolist())
        scores[col] = r2_col
        log(f"{col} R2: {r2_col}")

        all_test_preds[col] = test_preds.tolist()

    log(f"Mean R2: {np.mean(list(scores.values()))}")

    # prepare final submission file
    submission = pd.DataFrame({"id": test["id"]})
    submission[TARGET_COLUMNS] = 0
    submission.columns = submission.columns.str.replace("_mean", "")
    for i, col in enumerate(TARGET_COLUMNS):
        submission[col.replace("_mean", "")] = all_test_preds[col]
    submission.to_csv(output_csv, index=False)
    submission.head()
