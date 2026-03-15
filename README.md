# Recipes and Ratings Analysis
DSC80 Final Project

By: Nikita Jain 

## Overview 
This Data Science project, conducted at UCSD, is to explore the relationship between features of food recipes and how those features can help predict the ratings of recipes. 

## Introduction 

Platforms to share recipes have become more apparent, as food and cooking remains a large part of everyday life. However, based on recent trends, the complexity and time commitement of popular recipes has decreased, as many people are deterred from taking time out of their busy lives in order to cook a meal. According to the National Library of Medicine, recent surveys have revealed that the time spent on cooking and food preparation has declined substantially since the 1960s, as Americans currently spending an estimated 33 minutes per day on food preparation and cleanup.** Thus, this study raises the question of whether the average ratings of recipes may decrease with longer cooking times and more complex recipes. ** This may be due to Americans becoming busier, or preferring simpler recipes when cooking. In order to answer this question, I will analyze two datasets from [food.com](https://www.food.com/). 

The first of these datasets, 'recipe', has a total of 83,782 rows and 14 columns containing the following information about each recipe: 

| Column             | Description                                                                                                                                                                                       |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `'name'`           | Recipe name                                                                                                                                                                                       |
| `'id'`             | Recipe ID                                                                                                                                                                                         |
| `'minutes'`        | Minutes to prepare recipe                                                                                                                                                                         |
| `'contributor_id'` | User ID who submitted this recipe                                                                                                                                                                 |
| `'submitted'`      | Date recipe was submitted                                                                                                                                                                         |
| `'tags'`           | Food.com tags for recipe                                                                                                                                                                          |
| `'nutrition'`      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `'n_steps'`        | Number of steps in recipe                                                                                                                                                                         |
| `'steps'`          | Text for recipe steps, in order                                                                                                                                                                   |
| `'description'`    | User-provided description                                                                                                                                                                         |
| `'ingredients'`    | Text for recipe ingredients                                                                                                                                                                       |
| `'n_ingredients'`  | Number of ingredients in recipe   

The second dataset, 'interactions', has 731,927 rows and 5 columns containing the following information about each recipe: 

| Column        | Description         |
| :------------ | :------------------ |
| `'user_id'`   | User ID             |
| `'recipe_id'` | Recipe ID           |
| `'date'`      | Date of interaction |
| `'rating'`    | Rating given        |
| `'review'`    | Review text         |


With this information, I hope to answer the question if simpler recipes are more likely to gain higher ratings. 

## Data Cleaning and Exploratory Data Analysis 

### Data Cleaning 

In order to best be able to implement and analyze this data, I must process and clean the dataframes.

1. First, left merge the two data files in order to create one cohesive dataframe, 'recipes', which contains one row for each recipe and all of the information in a singular structure.

3. Then, fill all 0 ratings with np.nan. Ratings are normally scaled from 1-5, so a 0 rating indicates that the recipe obtained no ratings. Thus, in order to remove bias and be more accurate with the data, all 0 ratings should be filled with np.nan

These steps produce the following columns and column types: 

| Column             | Type |
     | :----------------- | :---------- |
     | `'name'`           | object      |
     | `'id'`             | int64       |
     | `'minutes'`        | int64       |
     | `'contributor_id'` | int64       |
     | `'submitted'`      | object      |
     | `'tags'`           | object      |
     | `'nutrition'`      | object      |
     | `'n_steps'`        | int64       |
     | `'steps'`          | object      |
     | `'description'`    | object      |
     | `'ingredients'`    | object      |
     | `'n_ingredients'`  | int64       |
     | `'user_id'`        | float64     |
     | `'recipe_id'`      | float64     |
     | `'date'`           | object      |
     | `'rating'`         | float64     |
     | `'review'`         | object      |

4. Next, after checking the types of all columns, convert the following columns into lists: `"nutrition"`, `"tags"`, `"steps"`, and `"ingredients"` to better be able to parse over the data. While the objects look like lists initially, they are actually string objects.

5. Add an `"avg_rating"` column per recipe, instead of having individual ratings, to obtain a more comprehensive image of each recipe. 

6. In order to best use the nutrition column, split the values to be more accurate to each element's meaning. Split the column into the 7 different following columns: `"calories"`, `"fat"`, `"sugar"`, `"sodium"`, `"protein"`, `"saturated_fat"`, and `"carbs"`.

7. Convert the `"submitted"` column into a DateTime object in order to conduct more extensive analysis over time.

8. Add a "greater_than_three" column - a boolean value indicating whether the total recipe cooking time takes longer than three hours. This column will be extremely beneficial for further analysis between the relationship of longer cooking times and average ratings of recipes. 


### Resulting DataFrame
With this data cleaning, the resulting dataframe has 83782 rows and the following columns are produced:

| Column                  | Description    |
| :---------------------- | :------------- |
| `'name'`                | object         |
| `'id'`                  | int64          |
| `'minutes'`             | int64          |
| `'contributor_id'`      | int64          |
| `'submitted'`           | datetime64[ns] |
| `'tags'`                | object         |
| `'nutrition'`           | object         |
| `'n_steps'`             | int64          |
| `'steps'`               | object         |
| `'description'`         | object         |
| `'ingredients'`         | object         |
| `'n_ingredients'`       | int64          |
| `'avg_rating'`          | object         |
| `'calories'`            | float64        |
| `'fat'`                 | float64        |
| `sugar'`                | float64        |
| `'sodium '`             | float64        |
| `'protein '`            | float64        |
| `'saturated fat '`      | float64        |
| `'carbohydrates '`      | float64        |
|  `'greater_than_three '`| boolean        |

The head of the relevant columns for this analysis of the resulting dataframe is shown below: 

| name                                 |   calories |   minutes | submitted           |   n_steps |   n_ingredients |   carbs |   avg_rating | greater_than_three   |
|:-------------------------------------|-----------:|----------:|:--------------------|----------:|----------------:|--------:|-------------:|:---------------------|
| 1 brownies in the world    best ever |      138.4 |        40 | 2008-10-27 00:00:00 |        10 |               9 |       6 |            4 | False                |
| 1 in canada chocolate chip cookies   |      595.1 |        45 | 2011-04-11 00:00:00 |        12 |              11 |      26 |            5 | False                |
| 412 broccoli casserole               |      194.8 |        40 | 2008-05-30 00:00:00 |         6 |               9 |       3 |            5 | False                |
| millionaire pound cake               |      878.3 |       120 | 2008-02-12 00:00:00 |         7 |               7 |      39 |            5 | False                |
| 2000 meatloaf                        |      267   |        90 | 2012-03-06 00:00:00 |        17 |              13 |       2 |            5 | False                |

## Exploratory Analysis 



### Univariate Analysis

First, I created a visualization of the distribution of average ratings amongst all recipes. 

<iframe
  src="assets/distribution-rating.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>
The histogram is heavily skewed right, demonstrating recipes tend to have higher reviews rather than lower reviews, with relatively very few recipes having ratings between 1-3. 

Next, I developed a visualization of the distribution of the number of ingredients amongst all recipes, in order to guage the complexity of most recipes. 

<iframe
  src="assets/distribution-ingredients.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>
This histogram is primarily centered around 8-9 ingredients, with a slight right skew. This suggests that the data is relatively well behaved, possibly being a good indicator to predict average ratings. 

### Bivariate Analysis

In order to measure the relationship between cooking time and average rating, I created a scatterplot corresponding to the average rating for each recipe by the cooking time in minutes.

<iframe
  src="assets/time-rating.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>
This visualization shows that the entire distribution of average ratings have low cooking times data points, but higher average ratings seem to be more likely to have higher cooking times. This visualization does not adhere to my initial hypothesis, but in later sections, I will see how cooking time can help predict average rating for recipes to further explore this relationship. 

### Interesting Aggregates 

Next, in order to explore the relationship between the number of steps and average rating for recipes, I created a dataframe measuring the average rating of all recipes that have each corresponding number of steps. A few rows of the resulting dataframe in ascending order is shown below. 

Head of Table: 

|   n_steps |   avg_rating |
|----------:|-------------:|
|         1 |      4.64813 |
|         2 |      4.66612 |
|         3 |      4.65546 |
|         4 |      4.64004 |
|         5 |      4.61038 |

Tail of Table:

|   n_steps |   avg_rating |
|----------:|-------------:|
|        87 |      5       |
|        88 |      3.66667 |
|        93 |      5       |
|        98 |      5       |
|       100 |      5       |

Based on the general trend (ignoring the outlier shown for average rating for 88 steps), it seems that more steps corresponds to a higher average rating for recipes.  

## Assessment of Missingness

The columns '"description"', corresponding to the description of the recipe, '"name"', stating the name of the recipe, and `"avg_rating"` all have missing values. In order to determine the cause of the missingness, I performed a missingness analysis on the columns.

### MNAR Analysis 

I believe that the "description" column is NMAR. If there is no description of the recipe, that means the person who posted to recipe is likely to believe that the name is already self explanatory as it is a simple recipe, and thus, a description is unneccesary. Adding a description would require more effort, and if the developer of the recipe believed a description was neccesary to better explain the final product, they would take the time to add a description. Thus, a missing description suggests that the recipe is relatively simple, and the description would just have repetitive information, adding nothing more than the data already collected. Some additional data I may want to collect is how the people who posted the recipe rank its difficulty/simplicity on a numeric scale. This may show that simpler recipes tend to have more missing descriptions than more difficult recipes. 

### Missingness Dependency 

> Year and Rating

Next, in order to explore the missingness of the "avg_rating" column, I examined the column "submitted", which contains information about when the recipe was first posted to the website. Recipes posted on later dates may be more likely to have missing values, as people haven't had time to discover the recipe yet. In order to conduct this test, I first added a column to the recipes DataFrame, `"year"`, which I created by extracting the year from each DateTime object in the `"submitted"` column. 

In order to confirm this theory, I created an overlaid histogram mapping the density of missing and non-missing data over time. 
<iframe
  src="assets/year_vis.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>

From the visualization, we can conclude that non-missing average ratings are heavily concentrated in the earlier years while the missing average rating values are relatively more in the later years. Thus, I decided to conduct a permutation test examining if missingness of "avg_rating" is dependent on the year the recipe was posted.

**Null Hypothesis** : The missingness of rating does not depend on the year the recipe was posted. 

**Alternative Hypothesis** : The missingness of rating does depend on the year the recipe was posted. 

**Test Statistic**: The absolute difference in means between the average year for recipes with missing rating values and the average year for recipes without missing rating values.

**Significance Level** : 0.05 

I ran this permutation test with 1000 simulations by shuffling the missingness of `"avg_rating"` and checking if the simulated test statistics were as extreme as the observed test statistic.

<iframe
  src="assets/missing.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>
The observed test statistic of 0.7297 is represented by the red line on the graph. Since the **p-value of 0.0** < significance level of 0.05, we can **reject the null hypothesis.** Thus, there is significant evidence that the missingness of "average rating" does depend on "year."

> Sodium and Rating

Next, I conducted a permutation test to determine if the missingness of "avg_rating" is dependent on the sodium content of the recipes.

**Null Hypothesis** : The missingness of rating does not depend on the sodium content of the recipe. 

**Alternative Hypothesis** : The missingness of rating does depend on the sodium content of the recipe. 

**Test Statistic**: The absolute difference in means between the sodium content for recipes with missing rating values and the sodium content for recipes without missing rating values.

**Significance Level** : 0.05 

I ran this permutation test with 1000 simulations by shuffling the missingness of "avg_rating" and checking if the simulated test statistics were as extreme as the observed test statistic.

<iframe
  src="assets/sodium.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>
The observed test statistic of 0.3404 is represented by the red line of the graph. This results in a** p-value of 0.883 **which is greater than the significance level of 0.05. Thus, we **fail to reject the null hypothesis**, and cannot conclude that the missingness of "avg_rating" is dependent on "sodium." 

## Hypothesis Testing 

As mentioned in the introduction, this project is mainly focused on how a recipe's increased complexity could possibly impact or lower the average rating of that recipe. In order to further explore this question, I conducted a **permutation test** using the `"greater_than_three"` column and `"avg_rating"` column. 

**Null Hypothesis**: The average ratings for recipes that have a cooking time of greater than or equal to three hours is equal to the average ratings for cooking times of less than three hours 

**Alternate Hypothesis**: The average rating for recipes that have a cooking time of greater than or equal to three hours is less than the average ratings for cooking times of less than three hours

**Test Statistic**: Difference in means between "avg_rating" for recipes with greater than or equal to three hours cooking time and "avg_rating" for recipes with less than three hours cooking time. 

**Significance Level**: 0.05

I decided to conduct a permutation test with these choices because we have no information about actual population, rather are trying to see if longer cooking times have a different average rating distributions from shorter cooking times. According to research conducted prior to the test, people are more likely to favor less complex recipes, and I wanted to test this hypothesis with the data. Thus, I conducted a one-sided permutation test as I believe that shorter cooking times would have higher average ratings. This is why my test statistic was a difference in means rather than an absolute difference in means, as I had a directional hypothesis. 

To run this test, I first split up the data into two groups - greater than or equal to three hours cooking time and less than three hours cooking time. **This resulted in a test statistic of 0.3504**. I ran 1000 simulations in which I shuffled the labels of the two groups and calculated the difference in means for average rating. This resulted in a ** p-value of 0.0**. The empirical distribution of the difference in means is shown below. 
<iframe
  src="assets/permutation.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>


### Conclusion
Since the p-value of 0.0 < 0.05 (the significance level), we can reject the null hypothesis. This suggests that the average rating for recipes that have a cooking time of greater than or equal to three hours is less than the average ratings for cooking times of less than three hours. This may be because people tend to prefer faster recipes, as they are less of a time commitement. Because a recipe take less time and effort, people may be more likely to rate it higher. 


## Framing a Prediction Problem 

I plan to predict the **average rating** of a recipe. This would be a classification problem as you can treat `average_rating` as a categorical variable by rounding average ratings into 5 separate bins [1-5]. This will be a multi-classifier problem, as we have to classifiy recipes into one of the 5 bins. I chose average rating as it is a very good representation of how people feel about a certain recipe, and can help show trends of what kinds of recipes people tend to like. Additionally, there seems to be a relationship between the cooking time and rating of a recipe, and this prediction model will help further explore that relationship. 

I will be using the **F1** score in order to measure the performance of my model. This is primarily because there are many more high ratings than low ratings in the dataset, causing a left skew. Thus, using accuracy may incorrectly reflect how well the model is actually performing. Using the f1 score will allow me to maximize both recall and precision in my model. 

In order to predict the average rating, I will only be using the information available in the columns in the recipes dataset. All of this information would be available before average rating is determined and the recipe is posted, so it is valid to use these columns as predictors. Every column is something you can determine before knowing average rating of the recipe, even if no one has left ratings of the recipe yet. 

## Baseline Model 

For the baseline model, I trained a RandomForestClassifier in order to clasify recipes into the 5 separate rating bins. I incorporated the following features: `"greater_than_three"` and `"calories."` I implemented these features because I found that recipes with lower calories tend to have higher ratings. Thus, I believed the quantitative column of `'calories'` may be a good predictor for average rating. As mentioned previously, `"greater_than_three"` and `"avg_rating"` also seem to have a strong relationship, so I implemented this cateogorical variable into my baseline model. 

I one hot encoded the `"greater_than_three"` column in order to transform it from a boolean value to a column with 0s and 1s, and dropped one of the columns to avoid repetitive information. Thus, I can train the model properly. I left the calories column as is.

After training my model, I tested the model on unseen data. My model obtained an f1 score of [0.01 0.02 0.05 0.27 0.55], the items in the list corresponding to the f1 score of 1s, 2s, 3s, 4s, and 5s. It resulted in an average F1 score of ** 0.453.** I believe that this model has much room for improvement and is not a particularly good model, as it has a relatively low average f1 score, correctly predicting ratings less than half of the time. Additionally, it seems to predict higher ratings much better than lower ratings, which could possibly be because there are fewer lower ratings in the dataset as a whole. 

## Final Model

In order to expand and improve on the baseline model, I incorporated and transformed new features to make my final model. The final model was trained using the following features: `'greater_than_three'`, `'calories'`, `'submitted'`, `'n_steps'`, `'sugar'`. 

`'greater_than_three'`

This column contains boolean values as to whether the recipe takes longer than three hours to prepare. As per the hypothesis test I performed earlier, recipes that take longer have a lower average rating, while recipes that take less time have a higher average rating. Thus, it seems to be a useful feature in predicting the average rating for a recipe. Like the baseline model, I one hot encoded this column in order to translate it to training a model. 

`'calories'`

This column contains the total number of calories for each recipe. The pivot table attached below summarizes the mean calories for each given recipe rating from 1-5. There seems to be a trend that higher rated recipes have a lower amount of calories. Thus, calories may be a good predictor for `'avg_rating'`. 

|   avg_rating |   calories |
|-------------:|-----------:|
|            1 |    447.964 |
|            2 |    445.258 |
|            3 |    442.251 |
|            4 |    422.67  |
|            5 |    427.668 |

I implemented a RobustScalarTransformer on this column as there are many outliers that could possibly introduce bias on the results of the model. 

`'n_steps'`

This column contains the number of steps per recipe. In order to see how this column could impact the average rating, I created a pivot table describing the average rating for recipes that have more than 40 steps (index = True) and less than 40 steps (index = False). I used 40 as the threshold as this is the median number of steps amongst all recipes in the datasest. The average rating for recipes with 40 steps or more seem to be higher than those with less than forty steps. This may be because more steps means the recipes are more descriptive and exact, allowing users with little experience to better make the recipe. 

| >=40 steps   |   avg_rating |
|:-------------|-------------:|
| False        |      4.62499 |
| True         |      4.72653 |

I implemented a Binarizer transformer on this column to classify each recipe into one of two groups: less than 40 steps and greater than or equal to 40 steps. 

`'submitted'`

This column contains information about when the recipe was submitted to the network. I created a stacked bar chart representing the proportions of 1-5s per year, and the proportions seem to vary per year. Specifically, there seems to be a trend of more 1s and 2s in later years than in earlier years. This may be because people have had more time to get used to recipes posted earlier, and newer recipes may not add much more to the already posted recipes. Because the average weighted f1 scores for the lower ratings in my baseline model were extremely low, I believe that this column can help better classify the lower rated recipes. 
<iframe
  src="assets/final_mod_year.html"
  width="100%"
  height="450"
  frameborder="0"
></iframe>
I implemented a FunctionTransformer on this column to extract the year from the already transformed DateTime objects of the submitted column. 

`'sugar'`

Finally, I implemented the sugar column, which contains information about the total sugar content in each given recipe. Based on the pivot table attached below, it seems that higher rated recipes tend to have a lower sugar content. This may be because people lean towards healthier recipes when cooking for themselves, and thus are more likely to rate healthy recipes higher. Thus, it could be a good predictor for average rating. 

|   avg_rating |   sugar |
|-------------:|--------:|
|            1 | 76.7458 |
|            2 | 73.9923 |
|            3 | 84.4529 |
|            4 | 61.3649 |
|            5 | 69.2212 |

For this column, I did not implement a transformer. 

I used a RandomForestClassifier model so that my model could average out the decisions of multiple trees. In order to create the best model, I had to choose the best hyperparameters - specifically, the max tree depth of each tree in the random forest. This is so that my model does not overfit nor underfit the training data, and can generalize to unseen data. Additionally, it will help control the variance of the forests. I tuned this hyperparameter by implementing a GridSearchCV, and learned that the optimal max_depth for my model is 17. With a max_depth of 17, and a hundred estimators, I ended up getting a **f1 test score of 0.549**, an overall **increase of 0.096** from the baseline model. However, the individual f1 scores for each rating 1-5 was [0, 0.01,0.04 ,0.31,0.68]. While the model did get better overall, it still incorrectly estimates the lower ratings, which may still be because there is minimal data for lower rated recipes.

## Fairness Analysis 

For the fairness analysis, I split up my data into two groups: recipes posted before 2009 and recipes posted after 2009. I split the year at 2009 because that is the median of all years in the dataset. I chose an evaluation metric of precision, as precision is the most relevant metric for predicting average ratings. For example, if the model consistently incorrectly predicts a very high rating for recipes posted in earlier years, people would be deterred from trying newer recipes. Incorrectly labeled ratings would falsesly inform users, and false positives would be bad. Thus, my parity metric is precision.

**Null Hypothesis**: Our model is fair. Its precision for recipes posted before 2009 and after 2009 are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: Our model is unfair. Its precision for recipes posted after 2009 is higher than those posted before 2009.

**Test Statistic**: Difference in precision scores (after 2009 -  before 2009

**Significance Level**: 0.05

<iframe
  src="assets/fairness.html"
  width="100%"
  height="400"
  frameborder="0"
></iframe>

First, I calculated the observed test statistic. I did this by transforming the "year" column into a boolean column, `"after_2009"`, which represents whether or not the recipe was posted after 2009. Then I grouped the data by "after_2009", and calculated the difference in precision, obtaining a test statistic of 0.0807. After performing a permutation test by simulating the null hypothesis 1000 times, I obtained a ** p-value of 0.0**. Because 0.0 < 0.05, the signficance level, we reject the null hypothesis that the model is fair. There is sufficient evidence that the precision of the prediction model for recipes posted after 2009 is higher than the precision of the model posted before 2009. 

