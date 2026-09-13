import os
import glob
import pandas as pd
import numpy as np
import re
import random

TRAIN_DIR = "IRMAS-TrainingData"
TEST_DIR = "IRMAS-TestingData"

CLASS_MAP = {
    'pia': 'Piano',
    'gac': 'Acoustic Guitar',
    'vio': 'Violin',
    'flu': 'Flute',
    'tru': 'Trumpet',
    'sax': 'Saxophone'
}

TARGET_CLASSES = list(CLASS_MAP.values())


def build_train_dataset():
    data = []

    for folder_name, class_name in CLASS_MAP.items():
        folder_path = os.path.join(TRAIN_DIR, folder_name)

        if not os.path.exists(folder_path):
            print(f"️Папка {folder_path} не найдена")
            continue
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"{folder_name} ({class_name}): найдено {len(wav_files)} файлов")

        for wav_path in wav_files:
            filename = os.path.basename(wav_path).replace(".wav", "")
            match = re.search(r'\[(\d+)\]', filename)
            track_id = f"track_{match.group(1)}" if match else filename

            data.append({
                'track_id': track_id,
                'fragment_id': filename,
                'wav_path': wav_path,
                'class': class_name,  # Single label
                'label': TARGET_CLASSES.index(class_name)  # Индекс класса
            })

    return pd.DataFrame(data)


def parse_test_txt(filepath):
    classes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                abbr = line.strip().lower()
                abbr_map = {
                    'pia': 'Piano', 'gac': 'Acoustic Guitar', 'vio': 'Violin',
                    'flu': 'Flute', 'tru': 'Trumpet', 'sax': 'Saxophone'
                }
                if abbr in abbr_map:
                    classes.append(abbr_map[abbr])
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")

    return classes


def build_test_dataset():
    data = []
    test_part1 = os.path.join(TEST_DIR, "Part1")

    if not os.path.exists(test_part1):
        print(f"Папка {test_part1} не найдена")
        return pd.DataFrame()

    txt_files = glob.glob(os.path.join(test_part1, "*.txt"))
    print(f"Найдено {len(txt_files)} .txt файлов в Part1")

    for txt_path in txt_files:
        wav_path = txt_path.replace(".txt", ".wav")
        if not os.path.exists(wav_path):
            continue

        filename = os.path.basename(txt_path).replace(".txt", "")
        match = re.match(r'\((\d+)\)\s*(.+?)-\d+', filename)
        if match:
            track_id = f"track_{match.group(1)}_{match.group(2).replace(' ', '_')}"
        else:
            track_id = filename

        classes = parse_test_txt(txt_path)
        classes = [c for c in classes if c in TARGET_CLASSES]

        if not classes:
            continue
        labels = np.zeros(len(TARGET_CLASSES), dtype=np.float32)
        for cls in classes:
            labels[TARGET_CLASSES.index(cls)] = 1.0

        data.append({
            'track_id': track_id,
            'fragment_id': filename,
            'wav_path': wav_path,
            'classes': classes,
            'labels': labels
        })

    return pd.DataFrame(data)

print("=" * 60)
print("1. ОБРАБОТКА TRAINING DATA (Single-label)")
print("=" * 60)
df_train = build_train_dataset()

if df_train.empty:
    print("\n Train пуст!")
    exit()

print(f"Всего в Train: {len(df_train)} файлов")
print("Распределение по классам:")
print(df_train['class'].value_counts())

print("\n" + "=" * 60)
print("2. ОБРАБОТКА TESTING DATA (Multi-label)")
print("=" * 60)
df_test = build_test_dataset()

if df_test.empty:
    print("Test пуст!")
    exit()

print(f"Всего в Test: {len(df_test)} файлов")

# СОЗДАНИЕ ПОДВЫБОРОК
random.seed(42)

print("\n" + "=" * 60)
print("3. СОЗДАНИЕ ВЫБОРОК ДЛЯ ЭКСПЕРИМЕНТА")
print("=" * 60)

# Сбалансированная
print("Сбалансированная выборка (по 300 каждого класса)")
balanced_indices = []
for class_name in TARGET_CLASSES:
    mask = df_train['class'] == class_name
    valid_idx = df_train[mask].index.tolist()
    sampled = random.sample(valid_idx, min(300, len(valid_idx)))
    balanced_indices.extend(sampled)

df_train_balanced = df_train.loc[balanced_indices].reset_index(drop=True)
print(f"Размер: {len(df_train_balanced)} файлов")

# Несбалансированная
print("Несбалансированная выборка (Long-tail)")
target_counts = {
    'Piano': 400, 'Acoustic Guitar': 300, 'Violin': 150,
    'Flute': 100, 'Trumpet': 50, 'Saxophone': 20
}

imbalanced_indices = []
for class_name, count in target_counts.items():
    mask = df_train['class'] == class_name
    valid_idx = df_train[mask].index.tolist()
    sampled = random.sample(valid_idx, min(count, len(valid_idx)))
    imbalanced_indices.extend(sampled)

df_train_imbalanced = df_train.loc[imbalanced_indices].reset_index(drop=True)
print(f"Размер: {len(df_train_imbalanced)} файлов")

# Сохраняем
df_train_balanced.to_csv("train_balanced.csv", index=False)
df_train_imbalanced.to_csv("train_imbalanced.csv", index=False)
df_test.to_csv("test_full.csv", index=False)

print("\n" + "=" * 60)
print("CSV файлы сохранены.")
print("=" * 60)

print("Распределение в train_balanced:")
print(df_train_balanced['class'].value_counts().sort_index())

print("Распределение в train_imbalanced:")
print(df_train_imbalanced['class'].value_counts().sort_index())