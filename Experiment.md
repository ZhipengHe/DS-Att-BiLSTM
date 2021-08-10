### Experiment Results

#### Sepsis - Predictor 1

##### Table

```
# Training

Epoch 1/5
37/37 [==============================] - 13s 210ms/step - loss: 0.4929 - accuracy: 0.7939 - auc: 0.6998 - val_loss: 0.4032 - val_accuracy: 0.8404 - val_auc: 0.7726
Epoch 2/5
37/37 [==============================] - 4s 100ms/step - loss: 0.3564 - accuracy: 0.8657 - auc: 0.8178 - val_loss: 0.3212 - val_accuracy: 0.8762 - val_auc: 0.8766
Epoch 3/5
37/37 [==============================] - 4s 105ms/step - loss: 0.2695 - accuracy: 0.9093 - auc: 0.8880 - val_loss: 0.2460 - val_accuracy: 0.9224 - val_auc: 0.9034
Epoch 4/5
37/37 [==============================] - 4s 104ms/step - loss: 0.2372 - accuracy: 0.9231 - auc: 0.9071 - val_loss: 0.2293 - val_accuracy: 0.9287 - val_auc: 0.9110
Epoch 5/5
37/37 [==============================] - 4s 105ms/step - loss: 0.2240 - accuracy: 0.9297 - auc: 0.9121 - val_loss: 0.2178 - val_accuracy: 0.9311 - val_auc: 0.9157

# Evaluation
Evaluate on test data
1846/1846 [==============================] - 42s 23ms/step - loss: 0.2178 - accuracy: 0.9311 - auc: 0.9157
test loss, test acc: [0.21783477067947388, 0.9310867786407471, 0.9157195091247559]
```

##### Plot

![](/img/exp/sepsis_p1_exp1_ac.svg)
![](/img/exp/sepsis_p1_exp1_loss.svg)

When training dataset `Sepsis`, the accuracy of model started with a very high level, and then reach to optimal results.

##### Some hypotheses for leading to this results:

- **Hypothesis 1**: The dataset is very small and imbalanced for DL models and easy to overfit with the dataset.

```
# outcome distribution for predictor 1
Raw dataset:
False    940
True     110

Filter all imcompleted cases:
False    675
True     107
```

- **Hypothesis 2**: The predictor is ralated to theprocess activities or the trajectories.

    - **Predictor 1:** if the patient has admission to intensive care or not (True or False)

![sepsis-map](/img/dataset/Sepsis.png)


**Create criteria for the dataset and predictors**

1. The outcome predictor should not be a part of the business process.
2. The dataset should include enough cases and have balanced distribution of the predictor types.

