# 🎵 Instrument Recognition with Class Imbalance Analysis (IRMAS)

Исследование влияния дисбаланса классов и методов семплирования в задаче распознавания музыкальных инструментов на датасете IRMAS.

## 📌 Описание
Цель проекта — оценить, как наличие редких классов ("длинный хвост") влияет на качество моделей машинного обучения, и сравнить эффективность различных подходов к извлечению признаков (MFCC, CQT, Wav2Vec2) и архитектур (MLP, CNN).

**Целевые классы:** Piano, Acoustic Guitar, Violin, Flute, Trumpet, Saxophone.

## 📂 Структура проекта
- `prepare_data.py` — Парсинг датасета IRMAS, создание сбалансированной и несбалансированной (Long-tail) выборок.
- `baseline.py` — Базовая модель: MFCC + MLP (сравнение Balanced vs Imbalanced).
- `wav2vec2_weights.py` — Использование эмбеддингов Wav2Vec2 и Class Weights для борьбы с дисбалансом.
- `oversampling.py` — Применение Random Oversampling к редким классам.
- `cqt_cnn.py` — Финальная архитектура: CQT-спектрограммы + 2D CNN (Best Performance).
- `train_balanced.csv`, `train_imbalanced.csv`, `test_full.csv` — Сгенерированные метаданные для воспроизведения экспериментов.

## Как запустить ❓

### 1. Подготовка
Скачайте датасет [IRMAS](https://www.upf.edu/web/mtg/irmas) и распакуйте папки `IRMAS-TrainingData` и `IRMAS-TestingData` в корень проекта.

### 2. Установка зависимостей
```pip install -r requirements.txt```

### 3. Подготовка данных
```python prepare_data.py```

### 4. Запуск экспериментов
```python baseline.py ```

```python imbalanced.py```

```python wav2vec2_weights.py``` # Wav2Vec2 + Class Weights

```python oversampling.py```   # MFCC + Oversampling

```python step5_cqt_cnn_fixed.py```  # CQT + CNN 


### Результаты экспериментов
| Конфигурация             | Macro ROC-AUC | Saxophone (Sens / Spec) | Trumpet (Sens / Spec) |
|:-------------------------| :---: | :---: | :---: |
| MFCC + MLP (Balanced)    | 0.6949 | 0.4012 / 0.9120 | 0.2278 / 0.9450 |
| MFCC + MLP (Imbalanced)  | 0.6690 | **0.0000** / 1.0000 | 0.1139 / 0.9610 |
| Wav2Vec2 + Class Weights | 0.5825 | 0.0120 / 0.9850 | 0.3671 / 0.8900 |
| MFCC + Oversampling      | 0.6505 | 0.0299 / 0.9700 | 0.2532 / 0.9300 |
| **CQT + 2D CNN**         | **0.7514** | **0.1557 / 0.9784** | **0.4177 / 0.8829** |


## 🔍 Ключевые выводы

**CQT превосходит MFCC для музыкальных сигналов благодаря логарифмической шкале, соответствующей музыкальным нотам.**

**Наивный Oversampling не работает для аудио: модель переобучается на шуме конкретных файлов, а не изучает тембр.**

**Macro ROC-AUC и Specificity — более надежные метрики для multi-label задач, чем Sensitivity при жестком пороге 0.5.**

## 📜 Лицензия и датасет

**Данный проект использует датасет IRMAS (Bosch et al., 2012). Аудиофайлы не включены в этот репозиторий в соответствии с условиями использования датасета (Non-Commercial Use Only).**