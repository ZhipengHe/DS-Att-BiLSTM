### Research Plan

Plan by @ZhipengHe

Supervised by @ChunOuyang and @CatarinaMoreira

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
- [ ] Step 2: Choose suitable outcome predictors
- [ ] Step 3: Based on the selected predictors, conduct feature correlation analysis to select 

##### Build four neural network architectures

- [ ] Dynamic features 
- [ ] Dynamic features + Static features
- [ ] Dynamic features + Attention
- [ ] Dynamic features + Static features + Attention

