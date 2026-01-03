# piml-for-evs-battery

## 1. Introduction

This is our final project for the course Introduction to AI in USTH. 

Topic: Physics-Informed Machine Learning (PIML) model for battery temperature prediction and smart charging control in electric vehicles (EVs)

Team name: **USTH.HoLaDream** (#CR7)

Team members: 
- Kiều Minh Đức (**Duke**): Leader, your favourite ni**er
- Nguyễn Đình Giang (**Zang**): member, model trainer, ben10
- Hoàng Viết Đức (**Florian**): member, model trainer, Chelsea fan
- Nguyễn Mỹ (**Miidai**): member, data collector, dancer
- Nguyễn Minh Đức (**Gr1mEd**): member, data collector, gymer


## 2. Problem statement

The world is witnessing a powerful wave of green transition. In the face of this trend, electric vehicles have become one of the most scrutinized, thanks to their ability to reduce fossil fuel consumption and greenhouse gas emissions. Major automakers such as Tesla (USA), Honda (Japan), Hyundai (Korea), and Vinfast (Vietnam) have been investing in the production of hybrid and electric vehicles to capitalize on this trend. However, they are all facing obstacles in electric vehicle battery technology: suboptimal battery materials or technical specifications.

In this project, we will apply a PIML model on data about the two most popular Lithium-based battery types for electric vehicles nowadays, Lithium-ion and Lithium Iron Phosphate (LiFePO4). In other words, with this battery sample, we will find an optimized way to charge the battery quickly while ensuring it doesn't overheat, preventing fire hazards and extending battery life.


## 3. Proposed model

We are planning to not use any pre-trained model. We train our own PINN with pytorch or deepXDE with some physical constraints, which are represented by a first-order linear non-homogeneous differential equation called the **Lumped Capacitance Model (LCM)**. The trained model has to meet the following requirements in I/O:
- Expected Input: battery type, state of charge (SoC), temperature, voltage, current, label
- Expected Output: Optimized time to charge, maximum battery temerature during the charge


## 4. Related papers

We have found numerous high-quality papers on this topic published over the past five years from reputable sources such as MDPI, IEEE, ResearchGate, etc. Below are the papers that we unanimously agree are most relevant to our project:
 - Heat Transfer Modeling and Optimal Thermal Management of Electric Vehicle Battery Systems [(MDPI, 2025)](https://www.mdpi.com/1996-1073/17/18/4575)
 - Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature [(IEEE, 2022)](https://ieeexplore.ieee.org/document/9858911)
 - Critical Review of Temperature Prediction for Lithium-Ion Batteries in Electric Vehicles [(MDPI, 2024)](https://www.mdpi.com/2313-0105/10/12/421)
 - Thermal Management Optimization in EV Battery Pack Assembly: A Data-Driven Approach Using AI- Based Feedback Loops [(ResearchGate, 2024)](https://www.researchgate.net/publication/394306531_Thermal_Management_Optimization_in_EV_Battery_Pack_Assembly_A_Data-Driven_Approach_Using_AI-_Based_Feedback_Loops)

Here is the link to our [summary for each papers](https://docs.google.com/document/d/1fd2wR5t4fOKbkepD0492d2HveSrJscDjacWE2ZeskKg/edit?usp=sharing).


## 5. Dataset

In this project, we will need the following data about lithium-ion batteries: battery type, state of charge (SoC), temperature, charging time, voltage, current. The dataset used will be divided into two parts: one part (80%) will be used to train the model, and the remaining part (20%) will be used to evaluate the accuracy of it. In addition, there will be some customed tests specifically designed by **Duke** to ensure that we have met the initial expectations for this project.

We conducted research and found many free high-quality online datasets. There are extremely promising candidates with very rich, diverse, and reputable datasets:
- [NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Stanford University](https://data.mendeley.com/datasets/7vdkzpnjgj/2)
- [Technische Universität Berlin (TU Berlin)](https://depositonce.tu-berlin.de/items/7f68932b-4d43-4f49-a5d8-914b00039f87)
- [Kaggle(1)](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns/data)
- [Shenzhen Auto Electric Power Plant Co., Ltd (Autosun) and Hong Kong Polytechnic University](https://data.mendeley.com/datasets/c7gg94tmvz/3)
- [Oxford University](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)

Unfortunately, most of them did not meet out requirements regarding parameters related to battery temperature, initial state of charge and charging tme. After testing and evaluating, we decided to use the dataset from [Kaggle(2)](https://www.kaggle.com/datasets/ziya07/ev-battery-charging-data) as it contained everything we need in this project. 


## 6. Tech stack

**Python** is used as the primary programming language for this project because it is easy to read, easy to debug, and has many libraries that support machine learning and deep learning-related tasks.

Supporting libraries:
- **numpy**: executing heavy numerical works
- **pandas**: working with the dataframe
- **pytorch** and **deepXDE**: training the PIML model
(there may be more, we will update this soon)


## 7. Our slides and scripts for the final presentation (we are coding and have not begun this part yet)


## 8. How to run the project
