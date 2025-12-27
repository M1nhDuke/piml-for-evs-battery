# piml-for-evs-battery

## 1. Introduction

This is our final project for the course Introduction to AI in USTH. 

Topic: Physics-Informed Machine Learning (PIML) model for battery temperature prediction and smart charging control in electric vehicles (EVs)

Team name: **USTH.HoLaDream** (#CR7)

Team members: 
- Kiều Minh Đức (Duke): Leader, 
- Nguyễn Đình Giang (Zang): member, model trainer
- Hoàng Viết Đức (Florian): member, model trainer
- Nguyễn Mỹ (Miidai): member, data collector, 
- Nguyễn Minh Đức (Gr1mEd): member, data collector.


## 2. Problem statement

The world is witnessing a powerful wave of green transition. In the face of this trend, electric vehicles have become one of the most scrutinized, thanks to their ability to reduce fossil fuel consumption and greenhouse gas emissions. Major automakers such as Tesla (USA), Honda (Japan), Hyundai (Korea), and Vinfast (Vietnam) have been investing in the production of hybrid and electric vehicles to capitalize on this trend. However, they are all facing obstacles in electric vehicle battery technology: suboptimal battery materials or technical specifications.

In this project, we will apply a PIML model on data about a specific lithium-ion battery samples, provided by Hong Kong Polytechnic University (China), to optimize battery temperature and charging power. In other words, with this battery sample, we will find an optimized way to charge the battery quickly while ensuring it doesn't overheat, preventing fire hazards and extending battery life.


## 3. Proposed model

We are planning to not use any pre-trained model. We train our own PINN with pytorch and some physical equations. The trained model has to meet the following requirements:
- Expected Input: state of charge (SoC), temperature, voltage, current, label
- Expected Output: Optimized time to charge, maximum battery temerature during the charge


## 4. Related papers

We have found numerous high-quality papers on this topic published over the past five years from reputable sources such as MDPI, IEEE, ResearchGate, etc. Below are the papers that we unanimously agree are most relevant to our project:
 - Heat Transfer Modeling and Optimal Thermal Management of Electric Vehicle Battery Systems [(MDPI, 2025)](https://www.mdpi.com/1996-1073/17/18/4575)
 - Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature [(IEEE, 2022)](https://ieeexplore.ieee.org/document/9858911)
 - Critical Review of Temperature Prediction for Lithium-Ion Batteries in Electric Vehicles [(MDPI, 2024)](https://www.mdpi.com/2313-0105/10/12/421)
 - Thermal Management Optimization in EV Battery Pack Assembly: A Data-Driven Approach Using AI- Based Feedback Loops [(ResearchGate, 2024)](https://www.researchgate.net/publication/394306531_Thermal_Management_Optimization_in_EV_Battery_Pack_Assembly_A_Data-Driven_Approach_Using_AI-_Based_Feedback_Loops)

Here is the link to our [summary for each papers](https://docs.google.com/document/d/1fd2wR5t4fOKbkepD0492d2HveSrJscDjacWE2ZeskKg/edit?usp=sharing).

## 5. Dataset

In this project, we will need the following data about lithium-ion batteries: state of charge (SoC), temperature, charging time, voltage, current, and label (faulty or normal). The dataset used will be divided into two parts: one part (80%) will be used to train the model, and the remaining part (20%) will be used to evaluate the accuracy of it.

We conducted research and found many free high-quality online datasets, such as those from [NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), [Stanford university](https://data.mendeley.com/datasets/7vdkzpnjgj/2) and [Kaggle](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns/data). Unfortunately, most of them did not meet out requirements regarding parameters related to battery temperature and their charging cycles. After testing and evaluating, we decided to use the dataset from [Shenzhen Auto Electric Power Plant Co., Ltd (Autosun) and Hong Kong Polytechnic University](https://data.mendeley.com/datasets/c7gg94tmvz/3) on Mendeley Data as it contained (more than) everything we need in this project. 


## 6. Tech stack

We used **Python** as the primary programming language for this project because it is easy to read, easy to debug, and has many libraries that support machine learning and deep learning-related tasks.

Supporting libraries:
- **numpy**: executing heavy numerical works
- **pandas**: working with the dataframe
- **pytorch**: training the PIML model
(there may be more, we will update this soon)


## 7. Our slides and scripts for the final presentation (we are coding and have not begun this part yet)


## 8. How to run the project
