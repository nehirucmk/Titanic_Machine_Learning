# Titanic - machine learning from disaster

this project predicts the survival of passengers aboard the Titanic using machine learning. it's a classic data science project that involves data cleaning, exploratory data analysis, and predictive modeling

my goal is to see if you could survive the incident with the remaining data we have about the Titanic 

# built with
- **Python**
- **Pandas** for data manipulation
- **NumPy** for numeric operations
- **Seaborn & Matplotlib** for data visualization
- **Scikit-learn** for machine learning modeling

## dataset
the dataset used in this project is the classic **Titanic - Machine Learning from Disaster**.
- **source:** [Titanic.csv on GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

# methodology
1. *EDA* for understanding the impact of features like gender and class on survival rates
2. *data cleaning* for handling missing values for 'Age' and 'Embarked' columns
3. *feature engineering* for dropping unnecessary columns and encoding categorical data
4. *modeling* for using the random forest classifier to train the data

# results
- **model accuracy**: 82.68%
- **key insight**: gender was the most important aspect for surviving percentage. despite taking up 22% of the crew, 50% of the survivors were women since the ship had 20 lifeboats, 18 of them being used with the reasoning (according to norm) "women and children first". that's how mostly women survived the wreck. if you're a woman, you're more likely to survive using the boats (or jack's true love).

# how to run
1. clone the repository.
2. install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
3. run the following: python analysis.py

# license
this project is licensed under the MIT License.



