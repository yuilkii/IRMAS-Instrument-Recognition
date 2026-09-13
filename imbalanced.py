import os
import ast
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, recall_score
from tqdm import tqdm

# config
TARGET_CLASSES = ['Piano', 'Acoustic Guitar', 'Violin', 'Flute', 'Trumpet', 'Saxophone']
NUM_CLASSES = len(TARGET_CLASSES)
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"{DEVICE}")


def load_and_prepare_data(csv_path, is_train=True):
    df = pd.read_csv(csv_path)

    X_paths = df['wav_path'].tolist()

    if is_train:
        y = []
        for cls in df['class']:
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            if cls in TARGET_CLASSES:
                label[TARGET_CLASSES.index(cls)] = 1.0
            y.append(label)
    else:
        y = []
        for classes_str in df['classes']:
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            try:
                classes_list = ast.literal_eval(classes_str)
                for cls in classes_list:
                    if cls in TARGET_CLASSES:
                        label[TARGET_CLASSES.index(cls)] = 1.0
            except (ValueError, SyntaxError):
                pass
            y.append(label)

    return X_paths, np.array(y)


def extract_mfcc_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    except Exception as e:
        print(f"Ошибка при обработке {wav_path}: {e}")
        return np.zeros(80, dtype=np.float32)


class AudioDataset(Dataset):
    def __init__(self, wav_paths, labels, cache_features=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.cache_features = cache_features
        self.features_cache = {}

        if self.cache_features:
            print("Кэширование признаков MFCC")
            for i, path in enumerate(tqdm(wav_paths, desc="Извлечение фич")):
                self.features_cache[i] = extract_mfcc_features(path)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        if self.cache_features:
            features = self.features_cache[idx]
        else:
            features = extract_mfcc_features(self.wav_paths[idx])

        return torch.tensor(features), torch.tensor(self.labels[idx])


class SimpleAudioMLP(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128):
        super(SimpleAudioMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    try:
        auc = roc_auc_score(all_targets, all_preds, average='macro')
    except ValueError:
        auc = 0.0
    predictions_binary = (all_preds > 0.5).astype(int)

    print("Метрики на Test наборе")
    print(f"Macro ROC-AUC: {auc:.4f}")
    print("Sensitivity (Recall) по классам:")

    sensitivities = {}
    for i, cls in enumerate(TARGET_CLASSES):
        true_positives = np.sum((all_targets[:, i] == 1) & (predictions_binary[:, i] == 1))
        actual_positives = np.sum(all_targets[:, i] == 1)

        if actual_positives > 0:
            sens = true_positives / actual_positives
        else:
            sens = 0.0

        sensitivities[cls] = sens
        print(f"   {cls:15}: {sens:.4f} (положительных примеров в test: {actual_positives})")

    return auc, sensitivities


def main():
    print("1. Загрузка данных")
    # balanced/imbalanced
    train_csv = "train_imbalanced.csv"
    test_csv = "test_full.csv"

    print(f"Используем: {train_csv}")
    X_train, y_train = load_and_prepare_data(train_csv, is_train=True)
    X_test, y_test = load_and_prepare_data(test_csv, is_train=False)

    print(f"Train: {len(X_train)} примеров, Test: {len(X_test)} примеров")

    print("2. Создание Dataset и DataLoader")
    train_dataset = AudioDataset(X_train, y_train, cache_features=True)
    test_dataset = AudioDataset(X_test, y_test, cache_features=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("3. Инициализация модели")
    model = SimpleAudioMLP(input_dim=80, hidden_dim=128).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Идеально для multi-label и single-label
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("4. Начало обучения")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}")

    print("5. Финальная оценка модели")
    auc, sensitivities = evaluate_model(model, test_loader)

    with open("imbalanced_metrics.txt", "w") as f:
        f.write(f"Mode: {train_csv}\n")
        f.write(f"Macro AUC: {auc:.4f}\n")
        for cls, sens in sensitivities.items():
            f.write(f"{cls}: {sens:.4f}\n")
    print("Метрики сохранены в imbalanced_metrics.txt")


if __name__ == "__main__":
    main()