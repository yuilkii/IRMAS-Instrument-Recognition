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
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# config
TARGET_CLASSES = ['Piano', 'Acoustic Guitar', 'Violin', 'Flute', 'Trumpet', 'Saxophone']
NUM_CLASSES = len(TARGET_CLASSES)
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Используем устройство: {DEVICE}")
if DEVICE.type == 'cpu':
    print("Внимание: Первое извлечение признаков Wav2Vec2 на CPU займет 5-10 минут.")
    print("Но мы их закэшируем, и повторные запуски будут мгновенными!")

# глобальные переменные для модели, чтобы не перезагружать её для каждого файла
processor = None
wav2vec_model = None


def init_wav2vec2():
    global processor, wav2vec_model
    if processor is None:
        print("   Загрузка Wav2Vec2 (это может занять минуту)...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
        wav2vec_model.eval()  # модель только для извлечения фич


def extract_wav2vec2_features(wav_path):
    global processor, wav2vec_model
    init_wav2vec2()

    try:
        # wav2vec2 16kHz
        y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=3.0)

        # процессор нормализует и токенизирует аудио
        inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            # усредняем по времени (mean pooling): (1, seq_len, 768) -> (1, 768)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy().squeeze(0)

        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"Ошибка при обработке {wav_path}: {e}")
        return np.zeros(768, dtype=np.float32)


class AudioDataset(Dataset):
    def __init__(self, wav_paths, labels, cache_features=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.cache_features = cache_features
        self.features_cache = {}

        if self.cache_features:
            print("Кэширование эмбеддингов Wav2Vec2")
            for i, path in enumerate(tqdm(wav_paths, desc="Извлечение Wav2Vec2")):
                self.features_cache[i] = extract_wav2vec2_features(path)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        features = self.features_cache[idx] if self.cache_features else extract_wav2vec2_features(self.wav_paths[idx])
        return torch.tensor(features), torch.tensor(self.labels[idx])


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(Wav2Vec2Classifier, self).__init__()
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

    print("Метрики на Test наборе")
    print(f"Macro ROC-AUC: {auc:.4f}")
    print(" Sensitivity (Recall) по классам:")

    for i, cls in enumerate(TARGET_CLASSES):
        tp = np.sum((all_targets[:, i] == 1) & (predictions_binary[:, i] == 1))
        actual = np.sum(all_targets[:, i] == 1)
        sens = (tp / actual) if actual > 0 else 0.0
        print(f"{cls:15}: {sens:.4f} (положительных примеров в test: {actual})")

    return auc


def calculate_class_weights(y_train):
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts

    weights = np.minimum(neg_counts / (pos_counts + 1e-6), 10.0)

    print(" Рассчитанные веса классов (pos_weight для BCE):")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"{cls:15}: {weights[i]:.2f} (примеров в train: {int(pos_counts[i])})")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def main():
    print("1. Загрузка данных")
    train_csv = "train_imbalanced.csv"
    test_csv = "test_full.csv"

    print(f"Используем: {train_csv}")
    df_train = pd.read_csv(train_csv)
    X_train, y_train = [], []
    for _, row in df_train.iterrows():
        X_train.append(row['wav_path'])
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[TARGET_CLASSES.index(row['class'])] = 1.0
        y_train.append(label)
    y_train = np.array(y_train)

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
    y_test = np.array(y_test)

    print(f"Train: {len(X_train)} примеров, Test: {len(X_test)} примеров")

    print("2. Создание Dataset и DataLoader")
    train_dataset = AudioDataset(X_train, y_train, cache_features=True)
    test_dataset = AudioDataset(X_test, y_test, cache_features=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("3. Инициализация модели и весов")
    model = Wav2Vec2Classifier(input_dim=768, hidden_dim=256).to(DEVICE)

    # WEIGHTS
    pos_weights = calculate_class_weights(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("4. Начало обучения (с учетом весов классов)")
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
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()