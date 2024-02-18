# Intuitive learning
Реализация феномена двойного спуска в глубоком обучении по материалам статьи Grokking: Generalization beyond overfitting (см. ниже точное указание).
Прилагается обзорная статья (очерк), где исследуется взаимосвязь переобучения с нахождением закономерностей в данных. Исследована взаимосвязь SGD и DD.
| Функция потерь            | Точность          |
|------------------------|------------------------|
| ![](results/loss.png) | ![](results/acc.png) |

## Запуск
```python
train.py # С настройкой параметров модели
```
## Результаты

(1) Операция $\frac{x}{y} \ mod \ p \ $, $(x + y) \ mod \ p \$, $x \cdot y \ mod \ p \$

(2) Операции с перестановками в группе $S_{97}$: $xyx^{-1}, \ x \cdot y$

| Номер опыта | Batch size | Budget | Вид операции
|:-----------:|:-----------:|:-----------:|:-----------:|
| 1   |         512  |        3e5 | (1) | 
| 2   |         256  |         3e4    | (1) |
| 3   |         256  |         3e4 | (2) |

### Опыт 1
| Точность | Функция потерь |
|----------|----------|
| <img src="results/acc_512.gif" alt="" width="535" height="400"> | <img src="results/loss_512.gif" alt="" width="535" height="400"> |

### Опыт 2

| Точность | Функция потерь |
|----------|----------|
| <img src="results/acc_256.gif" alt="" width="535" height="400"> | <img src="results/loss_256.gif" alt="" width="535" height="400"> |

### Опыт 3

<img src="results/Perm_loss_100.png" alt="" width="640" height="480">

## Цитирование
```
@inproceedings{power2021grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  booktitle={ICLR MATH-AI Workshop},
  year={2021}
}
```
