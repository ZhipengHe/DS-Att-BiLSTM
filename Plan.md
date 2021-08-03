### Research Plan

Plan by @ZhipengHe

<!-- Supervised by @ChunOuyang and @CatarinaMoreira -->

#### Research Task

> Target: IEEE ACCESS Submission

The baseline will only include dynamic features with basic LSTM NNs

Then,
- Static features can be added by selecting from correlation analysis 
- Attention layers are used to increase the accuracy of predictions and extract the attention weights

Experiments:

- Dynamic features 
- Dynamic features + Static features
- Dynamic features + Attention
- Dynamic features + Static features + Attention

Three Dataset:

- MIMIC-IV
- Sepsis
- Bpic11

#### Stage 1

Based on the dataset ***Sepsis***, constructing the general experiment architecture.

##### Clean Method for Sepsis

- [x] Step 1: Update missing attributes values for static attributes
    - [x] Export processed dataframe to CSV file
- [x] Step 2: Choose suitable outcome predictors
- [x] Step 3: Based on the selected predictors, conduct feature correlation analysis to select static features

##### Build four neural network architectures

- [x] Dynamic features 
- [x] Dynamic features + Static features
- [x] Dynamic features + Attention
- [x] Dynamic features + Static features + Attention

##### Evaluation

- [ ] Experiment 1: Dynamic (Ongoing)
- [ ] Experiment 2: 
- [ ] Experiment 3: 
- [ ] Experiment 4: 

#### Stage 2

- Based on the results of stage 1, update neural network architectures.
- Extend the evaluation to mimic-iv and bpic'11

#### Stage 3

- Summarize the evaluation outcomes
- Write the manuscript

