### Dataset Library

#### Sepsis

> Information provided by the dataset publisher

DOI: [doi:10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)

##### Description

This real-life event log contains events of sepsis cases from a hospital. Sepsis is a life threatening condition typically caused by an infection. One case represents the pathway through the hospital.  The events were recorded by the ERP system of the hospital. There are about 1000 cases with in total 15,000 events that were recorded for 16 different activities. Moreover, 39 data attributes are recorded, e.g., the group responsible for the activity, the results of tests and information from checklists.  Events and attribute values have been anonymized. The time stamps of events have been randomized, but the time between events within a trace has not been altered.

##### Attributes

|Name			| Description|
|---------------|-------------------------|
|Age			| Age in 5-year groups|
|Diagnostic*	| Various checkboxes on the triage document|
|DisfuncOrg		| Checkbox: Disfunctional organ|
|Hypotensie		| Checkbox: Hypotension |
|Hypoxie 		| Checkbox: Hypoxia |
|InfectionSuspected 	| Checkbox: Suspected infection |
|Infusion 		| Checkbox: Intravenous infusion required |
|Oligurie		| Checkbox: Oliguria |
|SIRSCritHeartRate	| Checkbox: One of the SIRS criteria |
|SIRSCritLeucos		| Checkbox: One of the SIRS criteria |
|SIRSCritTachypnea	| Checkbox: One of the SIRS criteria |
|SIRSCritTemperature	| Checkbox: One of the SIRS criteria |
|SIRSCriteria2OrMore	| Checkbox: Two or more of the SIRS criteria |
|Leucocytes		| Leucocytes measurement |
|CRP			| CRP measurement |
|LacticAcid		| Lactic-acid measurement |

---

> Information discovered by Disco

##### Process Map (100% Activity 0% Path)

![sepsis-map](/img/dataset/Sepsis.png)

##### Dynamic Attributes

|Name			| Description|
|---------------|-------------------------|
|Activity| |
|Complete Timestamp| |
|org:group||
|Leucocytes		| Leucocytes measurement |
|CRP			| CRP measurement |
|LacticAcid		| Lactic-acid measurement |


##### Static Attributes

|Name			| Description|
|---------------|-------------------------|
|Age			| Age in 5-year groups |
|Diagnostic*	| Various checkboxes on the triage document|
|DisfuncOrg		| Checkbox: Disfunctional organ|
|Hypotensie		| Checkbox: Hypotension |
|Hypoxie 		| Checkbox: Hypoxia |
|InfectionSuspected 	| Checkbox: Suspected infection |
|Infusion 		| Checkbox: Intravenous infusion required |
|Oligurie		| Checkbox: Oliguria |
|SIRSCritHeartRate	| Checkbox: One of the SIRS criteria |
|SIRSCritLeucos		| Checkbox: One of the SIRS criteria |
|SIRSCritTachypnea	| Checkbox: One of the SIRS criteria |
|SIRSCritTemperature	| Checkbox: One of the SIRS criteria |
|SIRSCriteria2OrMore	| Checkbox: Two or more of the SIRS criteria |


Diagnostic*

|Name			| Description|
|---------------|-------------------------|
|Diagnose| Label |
|DiagnosticArtAstrup| CheckBox|
|DiagnosticBlood| CheckBox|
|DiagnosticECG |CheckBox|
|DiagnosticIC	|CheckBox|
|DiagnosticLacticAcid	|CheckBox|
|DiagnosticLiquor |CheckBox|
|DiagnosticOther    |CheckBox|
|DiagnosticSputum |CheckBox|
|DiagnosticUrinaryCulture	|CheckBox|
|DiagnosticUrinarySediment	|CheckBox|
|DiagnosticXthorax |CheckBox|

---

##### Clean Method for Sepsis

- [x] **Step 1:** Update missing attributes values for static attributes
    - [x] Preprocess in `sepsis_clean.py`
    - [x] Export processed dataframe to `sepsis_processed.csv`
- [x] **Step 2:** Choose suitable outcome predictors
    - ❌ Choose **diagnosis** related features as predictors
    - ✔️ Based on process model, choose reasonable predictors
        - [x] Read this two paper carefully
            - [x] Mannhardt, F., & Blinde, D. (2017). [Analyzing the Trajectories of Patients with Sepsis using Process Mining](http://ceur-ws.org/Vol-1859/bpmds-08-paper.pdf)
            - [x] [Outcome-Oriented Predictive Process Monitoring: Review and Benchmark](https://arxiv.org/pdf/1707.06766.pdf)
        - [x] Analysing the information from this two papers.
        - [x] Identify potential predictors

- [x] Step 3: Based on the selected predictors, conduct feature correlation analysis 
    - [x] Label dataset for selected predictors
    - [x] Feature coorelation analysis for static features
    - [x] Select highly relative static features

##### How to find suitable outcome predictors

In the first paper, they identify these questions for this dataset:

> - ❌ are particular medical guidelines for the treatment of sepsis patients followed:
>   - patients should be administered antibiotics within one hour,
>   - lactic acid measurements should be done within three hours;
> > This problem is related to time prediction, not for outcome prediction.
> - ✔️ visualize and investigate the following specific trajectories:
>   - discharge without admission,
>   - admission to the normal care,
>   - admission to the intensive care,
>   - admission to the normal care and transfer to intensive care;
> - ❌ investigate the trajectory of patients that return within 28 days.
> > From this paper, their do not find reasonable rules to identify the different cases from decision mining techniques.

In the second paper, they create three different labelings for this log:

> - ❌ sepsis_1: the patient returns to the emergency room within 28 days from the discharge,
> > The AUC of sepsis_1 in the experiment are in a relatively low level.
> - ✔️ sepsis_2: the patient is (eventually) admitted to intensive care,
> - ✔️ sepsis_3: the patient is discharged from the hospital on the basis of something other than Release A (i.e., the most common release type).

Based on both paper, some potential outcome predictors are proposed here:

-  **Predictor 1:** if the patient has admission to intensive care or not (True or False)

-  **Predictor 2:** the release type from hospital (Release A or Others) 

##### Predictor 1: IC admission

1. Label the dataset by `pos_label = True` and `neg_label = False` by function `check_if_activity_exists` in `data/data_cleaning.py`.

2. Conduct feature coorelation analysis for static features
    - `label` means the Predictor 1

![Sepsis IC Admission](/img/dataset/Sepsis_IC_Admission.png)

3. Select highly relative static features
    - `DisfuncOrg`
    - `Hypotensie`
    - `Oligurie`


##### Predictor 2: Release type
