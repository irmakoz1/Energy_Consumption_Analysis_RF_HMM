# HVAC Energy Forecasting — Modular ML Pipeline with HMM + LSTM

This repository implements a fully modular machine-learning pipeline for forecasting building energy consumption using multiple models, including:

- Baseline models (Linear Regression, Random Forest, XGBoost)

- Hidden Markov Model (HMM)–augmented models

- Deep learning (LSTM sequence model)

- Feature importance and hidden-state analysis

- Forecasting for 1-step and long-term horizons (up to 2 months)

*The goal is to evaluate the capability of empirical modelling techniques and identify key environmental drivers of energy usage.*

### 📁 Dataset
*Data Set A*

Hourly institutional building data (1989-09 → 1989-12):

**WBE** — Whole Building Electricity (kWh/h)

**WBCW** — Chilled Water Load

**WBHW** — Hot Water Load

**Environmental variables:** Temperature, Humidity, Solar Radiation, Wind

**Time features:** Year, Month, Day, Hour

**Task:**
Predict the building’s energy usage for 2 months beyond the 4-month training period.




### Research Questions

1- Which environmental variables most influence building energy consumption?

2- How accurately can we predict hourly consumption one step ahead?

3- How well can we forecast long-term consumption (up to 2 months)?

Most analysis focuses on WBE (electricity) from Dataset A due to its strong connection to environmental and HVAC behaviour.

### Exploratory Data Analysis

 <img width="963" height="466" alt="image" src="https://github.com/user-attachments/assets/93272912-8b32-4f42-a911-acecb1138cf9" />

 <img width="966" height="483" alt="image" src="https://github.com/user-attachments/assets/39c752ca-cb5c-494d-9388-f3d989656fff" />

<img width="970" height="483" alt="image" src="https://github.com/user-attachments/assets/67821f97-27b3-4cdc-b0cd-ce72e28d52a2" />


The data exhibits seasonal patterns (daily and weekly cycles), strong temperature- and solar-driven fluctuations, significant noise and non-linearity, a pronounced drift near the end (Christmas → New Year), which complicates forecasting.

**Correlations** 

<img width="773" height="541" alt="image" src="https://github.com/user-attachments/assets/c2191e62-3415-4d2c-8499-736459154ecb" />


<img width="1131" height="854" alt="image" src="https://github.com/user-attachments/assets/24253df8-274c-4cc8-87ee-2110dee53dcc" />




- Temperature and Solar Radiation show the strongest positive correlation with electricity and chilled water usage

- Humidity and Wind have negligible influence

- Time-of-day has a clear cyclic impact

### Hidden Markov Model (HMM) Analysis

<img width="1831" height="361" alt="image" src="https://github.com/user-attachments/assets/ce873c66-c668-462a-ab64-7fa2c447891a" />




*We applied HMMs to extract latent operational modes of the building.*
Best performance found with 5 hidden states, interpreted as:

| State | Interpretation                                                             |
| ----- | -------------------------------------------------------------------------- |
| **0** | Sunny, warm weekday afternoons → High cooling + high electricity load      |
| **1** | Cool late-fall mornings → Strong heating load (weekdays/weekends)          |
| **2** | Warm weekend afternoons → High solar + moderate cooling                    |
| **3** | Cool, low-sunlight weekend mornings → Mixed HVAC usage                     |
| **4** | Similar to State 3 but on weekdays → Moderate typical weekday energy usage |


<img width="635" height="513" alt="image" src="https://github.com/user-attachments/assets/1d9d29fc-c096-4cd9-95f6-805abd08f25c" />


Transitions form two major clusters representing:

- Warm/bright high-energy periods

- Cool/low-sunlight low-energy periods

- Humidity and wind do not influence state transitions significantly.

Hidden states were then used as additional features in augmented ML models.

### Modular ML Pipeline

The repository includes a clean, modularized pipeline:

src/

 ├── models/            # HMM, LSTM, XGBoost, RF, LR, Augmented models
 
 ├── utils/             # Preprocessing, forecasting, evaluation
 
 ├── plots/             # Visualization utilities
 
 └── main notebook     # Analysis pipeline
 


| Category           | Models                                      |
| ------------------ | ------------------------------------------- |
| **Baseline**       | Linear Regression, Random Forest, XGBoost   |
| **Sequence Model** | LSTM with sliding window (seq_len)          |
| **HMM Models**     | HMM only, HMM + AR, HMM + RF, HMM + XGBoost |
| **Hybrid**         | HMM features integrated with ML models      |


All models output predictions, residuals, and evaluation metrics stored in the pipeline artifacts.

### Forecasting Performance

**Forecasting from the best performing model: Random Forest without hidden states**
<img width="1307" height="932" alt="image" src="https://github.com/user-attachments/assets/a60c56e7-5658-4932-b1ce-c07323fbaf3d" />

**Feature Importances for the Random Forest Model**

<img width="914" height="459" alt="image" src="https://github.com/user-attachments/assets/a6263332-320a-428b-95cd-51133e4f9edf" />

<img width="951" height="463" alt="image" src="https://github.com/user-attachments/assets/1b40836c-0586-4a28-bc6f-5d08e2f093b5" />

<img width="876" height="432" alt="image" src="https://github.com/user-attachments/assets/5b9d393a-f83c-4456-bd96-550acea28c66" />



Short-Term (1-Step Ahead)

- Random Forest & XGBoost perform best on noisy non-linear structure

- HMM-augmented models help smooth sequence transitions however, does not perform better than baseline models.

- LSTM performs solidly but constrained by limited training time range

Long-Term (Up to 2 Months Ahead)

The forecast task is extremely challenging because, only 4 months of training data are available. The last days contain holiday season anomalies. Model must extrapolate beyond seasonal patterns not observed in training.

*Strong non-linearity + abrupt demand drift*


-> Random Forest shows the best generalization for long-term forecasting in this dataset.

### Summary

This project demonstrates a complete end-to-end ML forecasting pipeline, modularized models allowing easy experimentation, hidden-state discovery of building operating modes, comparison of classical ML, HMM, and deep learning models and the practical challenges of long-term time series forecasting with limited data. Despite the difficulty of predicting beyond the training horizon, the approach provides valuable insights in some trends and patterns.

Hot water (WBHW) and overall electricity increase in winter months due to heating load and there are seasonal and periodic behavior related to weekend/weekdays, day/night and is holiday or not but dataset is too short to reliably estimate these trends. We identified that solar radiation, temperature, night/day time and weekend/week day has influence on building energy consumption levels.​
