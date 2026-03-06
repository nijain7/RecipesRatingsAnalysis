# Recipes and Ratings Analysis
DSC80 Final Project

By: Nikita Jain 

## Overview 
This Data Science project, conducted at UCSD, is to explore the relationship between features of food recipes, and how those features can help predict the ratings of recipes. 

## Introduction 

Platforms to share recipes have become more apparent, as food and cooking remains a large part of everyday life. However, while watching recent trends, the complexity and time commitement of popular recipes has decreased, as many people are deterred from taking time out of their busy lives in order to cook a meal. According to the National Library of Medicine, recent surveys from the U.S. have revealed that time spent on cooking and food preparation has declined substantially since the 1960s, as Americans currently spending an estimated 33 minutes per day on food preparation and cleanup.** Thus, this study raises the question of whether the average ratings of recipes may decrease with longer cooking times. ** This may be due to Americans becoming busier, or preferring simpler recipes when cooking. In order to answer this question, I will analyze two datasets from [food.com](https://www.food.com/). 

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

4. Next, after checking the types of all columns, convert the following columns into lists: "nutrition", "tags", "steps", and "ingredients" to better be able to parse over the data. While the objects look like lists initially, they are actually string objects.

5. Add an "avg_rating" column per recipe, instead of having individual ratings, to obtain a more comprehensive image of each recipe. 

6. In order to best use the nutrition column, split the values to be more accurate to each elements' meaning. Split the column into the 7 different following columns: "calories", "fat", "sugar", "sodium", "protein", "saturated_fat", and "carbs."

7. Convert the "submitted" column into a DateTime object in order to conduct more extensive analysis over time.

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

<iframe
  src="distribution-ingredients.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


