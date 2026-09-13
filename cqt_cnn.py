import os
import ast
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# config
TARGET_CLASSES = ['Piano', 'Acoustic Guitar', 'Violin', 'Flute', 'Trumpet', 'Saxophone']
NUM_CLASSES = len(TARGET_CLASSES)
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Используем устройство: {DEVICE}")


# ЗАГРУЗКА ДАННЫХ
def load_data(train_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    X_train, y_train = [], []
    for _, row in df_train.iterrows():
        X_train.append(row['wav_path'])
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[TARGET_CLASSES.index(row['class'])] = 1.0
        y_train.append(label)

    df_test = pd.read_csv(test_csv)
    X_test, y_test = [], []
    for _, row in df_test.iterrows():
        X_test.append(row['wav_path'])
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        try:
            classes_list = ast.literal_eval(row['classes'])
            for cls in classes_list:
                if cls in TARGET_CLASSES:
                    label[TARGET_CLASSES.index(cls)] = 1.0
        except:
            pass
        y_test.append(label)

    return X_train, np.array(y_train), X_test, np.array(y_test)


# ИЗВЛЕЧЕНИЕ CQT
def extract_cqt_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=3.0)
        # 84 бина = 7 октав * 12 полутонов
        cqt = librosa.cqt(y, sr=sr, bins_per_octave=12, n_bins=84)

        # переводим в децибелы
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        # нормализация в диапазон [0, 1] для стабильности
        cqt_db = np.clip(cqt_db, -80, 0)
        cqt_norm = (cqt_db + 80) / 80.0

        return cqt_norm.astype(np.float32)
    except Exception as e:
        # заглушка 84 бина по частоте, 130 фреймов по времени для 3 сек при sr=22050
        return np.zeros((84, 130), dtype=np.float32)


class CQTDataset(Dataset):
    def __init__(self, wav_paths, labels, cache_features=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.cache_features = cache_features
        self.features_cache = {}

        if self.cache_features:
            print("Кэширование CQT спектрограмм")
            for i, path in enumerate(tqdm(wav_paths, desc="Извлечение CQT")):
                self.features_cache[i] = extract_cqt_features(path)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        cqt_img = self.features_cache[idx]
        # добавляем канал (как у grayscale картинки): (1, 84, 130)
        cqt_tensor = torch.tensor(cqt_img).unsqueeze(0)
        return cqt_tensor, torch.tensor(self.labels[idx])


# МОДЕЛЬ 2D CNN С BATCH NORMALIZATION
class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # КРИТИЧЕСКИ ВАЖНО для стабильности
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 x 42 x 65

            # 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 21 x 32

            # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 x 10 x 16
        )

        # адаптивный пулинг подгонит любой размер к фиксированному
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 8))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 64 * 4 * 8 = 2048
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x


# ОЦЕНКА
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_targets = [], []

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

    print("\nМетрики на Test наборе (CQT + CNN Fixed)")
    print(f"Macro ROC-AUC: {auc:.4f}")
    print(f"{'Класс':<15} | {'Sensitivity (Recall)':<20} | {'Specificity':<15}")
    print("-" * 55)

    for i, cls in enumerate(TARGET_CLASSES):
        # Sensitivity = TP / (TP + FN)
        tp = np.sum((all_targets[:, i] == 1) & (predictions_binary[:, i] == 1))
        fn = np.sum((all_targets[:, i] == 1) & (predictions_binary[:, i] == 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity = TN / (TN + FP)
        tn = np.sum((all_targets[:, i] == 0) & (predictions_binary[:, i] == 0))
        fp = np.sum((all_targets[:, i] == 0) & (predictions_binary[:, i] == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"{cls:<15} | {sensitivity:<20.4f} | {specificity:<15.4f}")

    return auc


def main():
    print("1. Загрузка данных")
    train_csv = "train_balanced.csv"
    test_csv = "test_full.csv"

    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    print("2. Создание Dataset")
    train_dataset = CQTDataset(X_train, y_train, cache_features=True)
    test_dataset = CQTDataset(X_test, y_test, cache_features=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("3. Инициализация CNN")
    model = SimpleAudioCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("4. Обучение")
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
            print(f"   Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}")

    print("5. Финальная оценка")
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()