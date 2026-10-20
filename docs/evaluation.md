# Evaluation report

- **Dataset:** `images` (split: `validation`)
- **Images evaluated:** 3589
- **Model:** `emotiondetector.keras`
- **Face detection:** none (whole image)
- **Accuracy:** 63.64%
- **Balanced accuracy:** 64.46%
- **Macro F1:** 0.617
- **Weighted F1:** 0.607
- **Top-2 accuracy:** 80.61%
- **Throughput:** 460 images/s (CPU)

## Per-class metrics

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.606 | 0.707 | 0.652 | 491 |
| Disgust | 0.694 | 0.782 | 0.735 | 55 |
| Fear | 0.683 | 0.379 | 0.487 | 528 |
| Happy | 0.755 | 0.939 | 0.837 | 879 |
| Neutral | 0.347 | 0.179 | 0.236 | 626 |
| Sad | 0.547 | 0.684 | 0.608 | 594 |
| Surprise | 0.698 | 0.844 | 0.764 | 416 |

## Confusion matrix

| true \ predicted | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise |
|---|---|---|---|---|---|---|---|
| **Angry** | 347 | 2 | 16 | 16 | 46 | 56 | 8 |
| **Disgust** | 7 | 43 | 0 | 1 | 1 | 1 | 2 |
| **Fear** | 54 | 3 | 200 | 20 | 53 | 132 | 66 |
| **Happy** | 10 | 0 | 6 | 825 | 21 | 11 | 6 |
| **Neutral** | 97 | 11 | 45 | 164 | 112 | 133 | 64 |
| **Sad** | 51 | 2 | 11 | 34 | 84 | 406 | 6 |
| **Surprise** | 7 | 1 | 15 | 33 | 6 | 3 | 351 |

## Most confused pairs

| True | Predicted | Count |
|---|---|---|
| Neutral | Happy | 164 |
| Neutral | Sad | 133 |
| Fear | Sad | 132 |
| Neutral | Angry | 97 |
| Sad | Neutral | 84 |
