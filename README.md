# piml-for-evs-battery

## 1. Introduction

This is our final project for the course Introduction to AI in USTH. 

Topic: Physics-Informed Machine Learning (PIML) model for battery temperature prediction and smart charging control in electric vehicles (EVs)

Team name: **USTH.HoLaDream**

Team members: 
- Kiều Minh Đức (Duke): Leader, 
- Nguyễn Đình Giang (Zang): member, model trainer
- Hoàng Viết Đức (Florian): member, model trainer
- Nguyễn Mỹ (Miidai): member, data collector, 
- Nguyễn Minh Đức (Gr1mEd): member, jerker, gooner :))


## 2. Summary on the topic

The world is witnessing a powerful wave of green transition. In the face of this trend, electric vehicles have become one of the most scrutinized, thanks to their ability to reduce fossil fuel consumption and greenhouse gas emissions. Major automakers such as Tesla (USA), Honda (Japan), Hyundai (Korea), and Vinfast (Vietnam) have been investing in the production of hybrid and electric vehicles to capitalize on this trend. However, they are all facing obstacles in electric vehicle battery technology: suboptimal battery materials or technical specifications.

In this project, we will apply a PIML model on data about a specific lithium-ion battery samples, which is used in Vinfast's VF9 and Tesla Model 3, to optimize battery temperature and charging power. In other words, with this battery sample, we will find an optimized way to charge the battery quickly while ensuring it doesn't overheat, preventing fire hazards and extending battery life.


## 3. Proposed model

We do not use any pre-trained model. We train our own PIML model with pytorch and some physical equations. For more information about our model, please read our [report]().


## 4. Related papers

We have found numerous high-quality papers on this topic published over the past five years from reputable sources such as MDPI, IEEE, ResearchGate, etc. Below are the papers that we unanimously agree are most relevant to our project:
 - Heat Transfer Modeling and Optimal Thermal Management of Electric Vehicle Battery Systems [(MDPI, 2025)](https://www.mdpi.com/1996-1073/17/18/4575)
 - Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature [(IEEE, 2022)](https://ieeexplore.ieee.org/document/9858911)
 - Critical Review of Temperature Prediction for Lithium-Ion Batteries in Electric Vehicles [(MDPI, 2024)](https://www.mdpi.com/2313-0105/10/12/421)
 - Physics-informed machine learning for lithium-ion batteries: A review [(ScienceDirect, 2023)](https://www.sciencedirect.com/science/article/pii/S037877532300062X)
 - Physics-Informed Neural Networks for Battery Thermal Modeling [(IEEE, 2023)](https://ieeexplore.ieee.org/document/10100412)
 - Thermal Management Optimization in EV Battery Pack Assembly: A Data-Driven Approach Using AI- Based Feedback Loops [(ResearchGate, 2024)](https://www.researchgate.net/publication/394306531_Thermal_Management_Optimization_in_EV_Battery_Pack_Assembly_A_Data-Driven_Approach_Using_AI-_Based_Feedback_Loops)
 - Fast-Charging Optimization Method for Lithium-Ion Battery Packs Based on Deep Deterministic Policy Gradient Algorithm [(MDPI, 2025)](https://www.mdpi.com/2313-0105/11/5/199)
 - Machine learning-based fast charging of lithium-ion battery by perceiving and regulating internal microscopic states [(ScienceDirect, 2023)](https://www.sciencedirect.com/science/article/abs/pii/S2405829722006912)
 - A Review on Advanced Battery Thermal Management Systems for Fast Charging in Electric Vehicles [(MDPI, 2024)](https://www.mdpi.com/2313-0105/10/10/372)


## 5. Dataset

We conducted research and found many high-quality online datasets, such as those from [NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [Kaggle](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns/data). Unfortunately, most of them did not meet out requirements regarding parameters related to battery temperature and their charging cycles. After testing and evaluating, we decided to use the dataset from [Stanford university](https://data.mendeley.com/datasets/7vdkzpnjgj/2) on Mendeley Data as it contained everything we need in this project. Here is some information about it:


## 6. Tech stack

We used **Python** as the primary programming language for this project because it is easy to read, easy to debug, and has many libraries that support machine learning and deep learning-related tasks.

Supporting libraries:
- **numpy**: executing heavy numerical works
- **pandas**: working with the dataframe
- **pytorch**: training the PIML model
(there may be more, we will update this soon)


## 7. Our slides and scripts for the final presentation (we are coding and have not begun this part yet


## 8. Running this project
