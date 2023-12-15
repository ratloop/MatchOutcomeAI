# MatchOutcomeAI
A data-driven approach to predicting football match outcomes using advanced machine learning techniques. This project integrates various algorithms to forecast game results, providing insights for sports betting, team performance analysis, and sports enthusiasts.

# Getting Started
Prerequisites: `Python 3.x`

## Usage

Your own API key from [API Football](https://www.api-football.com/) must be imported into api_data_scraper.py.

```git clone https://github.com/ratloop/MatchOutcomeAI```

```pip install -r requirements.txt```

```py main.py```

# Introduction

## Problem Statement
Predicting football match outcomes has become increasingly popular due to the rising interest in sports betting and the desire to improve team performance. Accurate predictions can benefit various stakeholders, including fans, coaches, analysts, and bookmakers. However, predicting match results remains a complex task due to numerous factors that influence a game's outcome, such as team form, player performance, and historical data. This project aims to develop a machine learning model capable of providing accurate and logical football match outcome predictions, comparable to those of popular bookmakers.

## Objectives
Starting with the primary goal - developing a predictive model capable of generating probabilities that align with the offerings of established bookmakers - the model's performance was both robust and reliable. By leveraging the Gradient Boosting algorithm, it offered predictions that were not only on par with commercial bookmakers but also grounded in statistical accuracy. This accomplishment underscores the effectiveness of the chosen approach and validates the model's utility.

Secondly, the model achieved a commendable test accuracy of over 50%, a clear indicator of its success. This figure was more than a mere benchmark; it represented a significant achievement in the challenging field of sports predictions, where the complexity and unpredictability of football matches often make precise forecasting a daunting task. The accomplishment in surpassing this threshold indicates the model's credibility and its practical applicability for users looking to understand football match outcomes better.
In addition to the model's predictive power, the application's interface was also a success. The command-line interface provided users with an intuitive and straightforward way to make predictions and review past predictions, ensuring the application was not only accurate but also user-friendly. The adoption of clear and organized menu options further enhanced the user experience, allowing seamless navigation through the application's various features.

Lastly, the model was designed with an eye towards continuous improvement. By incorporating new data and refining existing features, the system ensures that its predictive power does not stagnate but improves over time. This approach to iterative improvement is essential for staying relevant and accurate in the fast-paced and ever-changing world of football.

# System Design

## Data Scraper
For this predictive model, data was gathered using a purpose-built data scraper. The scraper was designed to fetch data from a specified API (Application Programming Interface), which provides a comprehensive collection of football match data for the Premier League.
The data scraper works by making GET requests to the API's endpoints for each football season. These requests retrieve data in JSON format which includes various details about each match, such as the teams involved, the final score, the date of the match, and other match-specific statistics. The decision to use this particular API was driven by its extensive dataset that offers a wide range of features beneficial for the predictive model. The API is reliable, consistently updated, and provides granular data which is critical for the analysis.

Available statistics:
* Shots on Goal
* Shots off Goal
* Shots inside box
* Shots outside box
* Total Shots
* Blocked Shots
* Fouls
* Corner Kicks
* Offsides
* Ball Possession
* Yellow Cards
* Red Cards
* Goalkeeper Saves
* Total passes
* Passes accurate
* Passes %

The data scraping process was facilitated using Python, a language renowned for its robust data manipulation capabilities and extensive libraries. Libraries such as 'requests' and 'json' were employed for HTTP requests and JSON data manipulation respectively. The decision to utilize Python was driven by its ability to streamline the data collection process, allowing us to amass a substantial dataset in a time-efficient manner.

Upon retrieval, the data is stored in its raw JSON format for each football season, earmarking it for further processing. This method of storage retains an unaltered copy of the original data from the API, enabling us to revisit or troubleshoot the data if necessary. Moreover, it ensures that the data scraping stage needs to be executed only once, thereby conserving resources for subsequent phases of the project.

The API used was [API Football](https://www.api-football.com/)

## Data Engineering

Data Combination: Since I have data for each Premier League season in separate CSV files, I combine all these datasets into one single CSV file. This allows us to build a model that considers historical data from multiple seasons, thereby improving its predictive capability.

Feature Creation: The API provides us with a wide range of features, such as goals scored, shots on target, and possession percentage. I utilize these features and create some of my own, such as form from the past 5 games. The addition of these engineered features aims to capture more complex patterns and relationships that can potentially improve the accuracy of the predictive model.

Data Cleaning: The final step in the data engineering process is cleaning the data. This involves dealing with missing or inconsistent data and ensuring that the dataset is reliable and accurate. It's an important step as the quality of the data used to train the model significantly influences the model's performance.

The data engineering process was primarily implemented using Python due to its powerful data manipulation libraries like pandas and NumPy. This choice of language ensured a smooth and efficient data processing phase, providing us with a clean, organized dataset ready for visualisation, analysis, and model training

## Data Visualisation

Having a clear, visual understanding of the data is a fundamental part of this project. This step aids in interpreting the dataset, revealing potential patterns, trends, and correlations among different variables. It provides valuable insights before I dive into building the machine learning models.

Several data visualisation strategies were exploited to delve into the relationships within the features:

Correlation Matrix: I created a correlation matrix to identify the interdependence between different features in my dataset. This step is crucial as it helps determine which variables exhibit strong positive or negative correlations. This information is valuable during the phase of feature selection for model training.

![Correlation Matrix of the Relevant Variables for the Home Team](https://i.imgur.com/ddPhm9b.png)

This heatmap presents a visual depiction of the correlations among a host of home team statistics such as shots on target, total shots, fouls, corners, offsides, ball possession, yellow cards, red cards, saves made by the goalkeeper, attempted passes, and successful passes. The magnitude and orientation of the correlation between these elements are signified by the depth of the colour and the proportions of the squares. This illustration provides valuable insights into the interplay and potential influence of these varied factors within a game scenario.

Scatter Plots: I used scatter plots to illustrate the relationships between different pairs of variables. These plots can emphasize correlations, reveal trends, and help spot any outliers or anomalies in the data.

![Scatter Plot Matrix of the Relevant Variables for the Home Team](https://i.imgur.com/vuRYwtB.png)

This scatterplot matrix visualizes the pairwise relationships of shots on target, total shots, fouls, corners, offsides, possession, yellow cards, red cards, goalkeeper saves, attempted passes, and successful passes for the home team. Each plot in the matrix shows the relationship between a pair of these attributes, which allows for a detailed exploration of potential correlations or patterns. This comprehensive view provides critical insights for understanding the multifaceted dynamics of the game.

I leveraged libraries such as matplotlib and seaborn for data visualisation, which offer an extensive set of tools for creating informative and aesthetically pleasing statistical graphics. With these visual insights, it was better equipped to interpret the data and make more informed decisions during the model training phase.

## Model Training

Logistic Regression: I chose to start with Logistic Regression due to its simplicity and power. Known as a linear classifier, Logistic Regression calculates the probability of a binary outcome based on input features. It then classifies the data accordingly. Despite its simplicity, it can model complex relationships when the input features are nonlinearly transformed or when interaction terms are included. The model was trained using maximum likelihood estimation to find the best fitting model to the data.

Support Vector Machine (SVM): Next, I employed the Support Vector Machine model. SVM is a versatile machine learning model typically used for classification and regression analysis. Its strength lies in its ability to handle high dimensional data and create complex decision boundaries, even in cases where the number of dimensions exceeds the number of samples. It works by constructing a hyperplane in a high-dimensional space that distinctly classifies the data points.

K-Nearest Neighbours (KNN): The K-Nearest Neighbours algorithm was another model utilized in this project. Known as a lazy learning algorithm, KNN stores all instances corresponding to training data in a multidimensional space. When a prediction is required for an unseen instance, it searches through the entire training dataset for the K-most similar instances (the neighbours) and returns the most common outcome (in the case of classification) or average (in the case of regression).

Decision Tree: I also made use of Decision Trees, a type of model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A decision tree model provides a practical approach to decision-making, capturing the possible outcomes, resources, and utilities in a structured format.

Bagging Classifier, Gradient Boosting, XGBoost: These ensemble models were also incorporated into this project. Ensemble methods, which combine the predictions from several base machine learning algorithms, can provide more accurate predictions than any individual model. Bagging and Boosting are two of the most commonly used ensemble techniques. Bagging (Bootstrap Aggregating) reduces variance by running multiple models in parallel and averaging their predictions, while boosting reduces bias by running multiple models sequentially, each trying to correct the mistakes made by the previous one.

Naive Bayes: Lastly, I used the Naive Bayes model, a classification technique based on applying Bayes’ Theorem with a strong assumption of independence among predictors. Despite its simplicity and the naive design, Naive Bayes can perform well in many complex real-world situations.

Each model was trained using the processed dataset and implemented using Python's scikit-learn library. Scikit-learn is a popular tool for machine learning in Python, offering simplicity, efficiency, and flexibility in conducting model training. The performance of each model was evaluated using a hold-out validation set, which was not part of the training data. This strategy ensures a more accurate measure of real-world performance, as it evaluates how well the model can generalize to new, unseen data.

The diverse selection of models was intentionally chosen to understand which model, and which features from the dataset would yield the most accurate predictions for football match outcomes. By examining different types of models, I had a higher chance of finding one that could capture the unique characteristics and patterns in the data.

## Model Comparison

In the Model Comparison phase, I assessed and compared the performance of each trained model. This comparison allowed us to determine which model performed the best in terms of predicting the outcomes of football matches. The evaluation metrics used for this comparison were Accuracy, Precision, Recall, and F1-Score.

Accuracy: This is the proportion of true results (both true positives and true negatives) among the total number of cases examined. It is suitable when the target classes are well balanced.
Precision: Precision calculates the percentage of correct positive predictions (True Positives divided by the sum of True Positives and False Positives). It is a good measure to determine when the costs of a False Positive are high.
Recall (Sensitivity): Recall calculates the percentage of correct positive predictions from the actual positives (True Positives divided by the sum of True Positives and False Negatives). It is a good measure to determine when there is a high cost associated with a False Negative.
F1-Score: The F1 Score is the harmonic mean of Precision and Recall and tries to balance both. It is a good measure when you want to seek a balance between Precision and Recall.

A visual representation of the model comparison was made using a bar chart and heatmap, clearly showing the performance of each model based on the above metrics.

![Model Performance Comparison Bar Chart](https://i.imgur.com/UfkDt4U.png)

![Model Performance Comparison Heatmap](https://i.imgur.com/yODk4Ik.png)

In the evaluation process, I also drew comparisons with the predictions provided by Google's football match predictor. Yet, it is crucial to bear in mind that, much like any other predictive model, Google's forecasts are not infallible. The predictions they offer stem from their unique machine learning models and the specific data they utilize, which may differ significantly from mine. Hence, the aim in making this comparison was not to establish an ultimate benchmark for precision but rather to provide another point of reference in this analysis.

## Model Performance

As a fundamental tool in the field of machine learning, Logistic Regression offered a decent baseline for this project. With an accuracy of 61.15%, it demonstrated a reasonable ability to classify match outcomes correctly. Its precision of 48.77% indicates that when it predicted a particular outcome, it was correct nearly half of the time. The recall of 50.94% reflects that it identified about half of the actual outcomes correctly, and the F1-Score of 46.37% provides a balanced measure of the model's precision and recall. While these results were satisfactory, the performance of Logistic Regression was overshadowed by some of the more sophisticated models I employed.

The Support Vector Machine (SVM), a powerful classification algorithm, achieved an accuracy of 54.42%, revealing a modest capacity to predict match outcomes correctly. Its precision of 39.65% and recall of 40.28% were slightly below my expectations, pointing towards a higher rate of misclassifications compared to Logistic Regression. The F1-Score of 35.05%, a harmonic mean of precision and recall, was likewise lower. These figures led us to explore other machine learning models with greater predictive power.

The K-Nearest Neighbours (KNN) algorithm, despite its simplicity and intuitive nature, achieved an accuracy of 40.96%, the lowest among the models we tested. This means it classified the outcomes correctly less than half of the time. The precision, recall, and F1-Score were similarly low, standing at 38.09%, 38.12%, and 37.18% respectively. These figures highlight the limitations of simpler models in the face of complex, real-world data sets.

The Decision Tree model performed better than the KNN model, boasting an accuracy of 60.96%, but it fell short of outperforming Logistic Regression. Its precision and recall scores, at 56.28% and 55.77% respectively, suggest that it classified over half of the instances correctly, but also misclassified a significant portion. Its F1-Score of 55.92% reflects the balance between precision and recall, placing it in the mid-range of performance among the models.

As an ensemble model, the Random Forest algorithm showed a noticeable improvement, achieving an accuracy of 63.85%. Its precision of 52.58%, recall of 54.60%, and F1-Score of 52.03% were also superior to those of the individual Decision Tree model. This improvement showcases the strength of ensemble methods, which combine the predictions of multiple models to produce more robust results.

The Bagging Classifier model further demonstrated the power of ensemble methods. It delivered an accuracy of 69.23%, the third highest among the models, and a precision of 63.80%. The recall score of 64.50% and the F1-Score of 63.96% were also impressive, reflecting the model's balanced performance.

The Gradient Boosting model outperformed all other models in this study. It achieved the highest accuracy, at 77.12%, signifying that it predicted match outcomes correctly over three-quarters of the time. Its precision of 70.62% suggests that when it predicted an outcome, it was right about 70% of the time. Its recall of 69.40% means that it was able to identify nearly 70% of the actual outcomes correctly. The F1-Score of 69.22% is a testament to the model's balanced performance in terms of precision and recall. These outstanding figures underscore the potency of the Gradient Boosting model in tackling the complexity of football match prediction.

The XGBoost model, a more advanced implementation of gradient boosting, was close behind. It achieved an accuracy of 76.54%, making correct predictions just slightly less often than the Gradient Boosting model. Its precision of 70.82%, recall of 70.49%, and F1-Score of 70.48% were all just marginally below those of the Gradient Boosting model, reflecting its solid performance across the board.

The Naive Bayes model, a classic machine learning algorithm, offered an accuracy of 56.15%, placing it in the mid-range among the models I tested. Its precision of 50.12%, recall of 50.47%, and F1-Score of 49.35% were all close to 50%, suggesting a balanced, though not outstanding, performance.

After a thorough analysis of these metrics, I decided to adopt the Gradient Boosting model for the predictor, despite XGBoost having slightly better metric values. The reason for this choice is that the Gradient Boosting model yielded more realistic probabilities, which closely matched the odds given by major bookmakers. This calibration of predicted probabilities is an essential aspect of this project, as it allows the predictions to be interpretable and comparable with external standards.

## Predictor

The project's "Predictor" component is a pivotal module that is embodied in a Python file and designed to work symbiotically with the main.py script. This module plays a central role, harmonizing the diverse facets of this system to provide users with easily digestible predictions for football matches.

Once the laborious but crucial process of model training and selection concludes, we weave the star performer - the Gradient Boosting model, in this case - into the fabric of the predictor system. When called into action, this module requires the names of two teams set to compete in an upcoming match as inputs. These inputs are then channelled through the meticulously trained Gradient Boosting model. This model, in turn, exploits the rich historical and statistical data we've assembled on the teams to forecast the most probable outcome of the upcoming match.

The results of the match prediction are presented in a simple yet comprehensive manner. Each game's outcome is partitioned into three discrete possibilities: a win for the home team (Team A), a win for the away team (Team B), and a draw. For instance, when predicting a game between Bournemouth (home) and West Ham (away), the output might appear as follows:
-	Bournemouth (Home) Win: 36.12%
-	West Ham (Away) Win: 48.24%
-	Draw: 15.64%

These percentages represent the likelihood of each outcome as per the model's calculations. The advantage of this presentation format is twofold. Firstly, it simplifies the complex underlying computations, making the results easily understandable for the users. Secondly, it parallels the odds structure that bookmakers typically use, making the system's output relatable and intuitive even for those who may not be familiar with machine learning or data science concepts.

In essence, the goal is to make this powerful predictive tool user-friendly and accessible to a broad spectrum of users, whether they are hardcore football enthusiasts, casual viewers, or even professional bettors. This simplicity of understanding, combined with the system's predictive accuracy, can potentially provide valuable insights for users looking to make informed decisions related to football matches.

The Predictor module is engineered to be resilient and capable of handling missing or atypical inputs gracefully. It incorporates sophisticated error-management mechanisms to insulate the system from failure in the face of unexpected or unforeseen inputs. Instead of succumbing to these anomalies, it responds by providing informative feedback to the user and suggests corrective measures where applicable.

In designing the Predictor, I prioritized not just robustness but also speed. The Predictor is optimized to churn out predictions in real-time, thereby enabling users to make well-informed decisions promptly. This feature is of particular significance to users who intend to leverage the system for making real-time bets or decisions during a live football match.

Finally, the Predictor module was conceived with an eye on the future. It is adaptable and scalable, designed to assimilate new data effortlessly as it becomes available. This capability ensures that the model's predictions remain current, accurate, and reflective of the latest trends. Such flexibility renders the system a potent and reliable tool in the ever-evolving and unpredictable realm of football.

## User Interface

The interface of the system has been meticulously designed, giving due consideration to usability and accessibility. Being aware that this tool could be used by individuals with varying levels of technical expertise, I opted for a Command-Line Interface (CLI). This interface, despite its simplicity, provides a powerful conduit for user-system interaction, bridging the gap between the user's inputs and the complex computational operations taking place in the backend.

Upon initiating the program, the user encounters a main menu, which comprises four clearly labelled options:
1.	Scrape All Datasets
2.	Scrape Current Season Dataset
3.	Make a Prediction
4.	View Previous Prediction

![User Interface - Main Screen](https://i.imgur.com/wVyoOjv.png)

The first two options are centred around data acquisition and processing, which form the backbone of this system. Opting for the first choice, 'Scrape All Datasets', triggers the data scraping process for all historical seasons available. The tool then fetches raw data from the API in the form of JSON files. After the data scraping phase comes to an end, the amassed JSON files undergo a transformation process to convert them into a structured CSV format. This not only allows for a more streamlined storage but also facilitates the data processing tasks in the later stages.

Option two, named 'Scrape Current Season Dataset', is uniquely dedicated to procuring data for the latest football season. This choice allows users to swiftly update the dataset for the current season, eliminating the need to re-scrape or reprocess data from previous seasons. This task is completed with the production of a well-structured CSV file, poised for further exploration and analysis.

![User Interface - Data Scraper](https://i.imgur.com/TXPV6Dt.png)

'Make a Prediction', is where users can truly tap into the potential of this predictive system. The user has the liberty to input the names of two football teams, leaving the rest to the system. The prediction model, which has been groomed using pre-processed data and fine-tuned by the Gradient Boosting technique, takes this user input and churns out probabilities for three possible outcomes - Team A clinching a win, Team B securing a victory, or the match concluding in a draw. The output does not stop at mere probabilities. It also includes key performance metrics of the prediction model, including but not limited to, accuracy and the F1-score. This information presents the user with a holistic understanding of the prediction and its underlying credibility.

![User Interface - Predictor](https://i.imgur.com/aV5OyYU.png)

The fourth and final option, 'View Previous Prediction', provides the user with a retrospective lens. All previous predictions made by the user are stored in a JSON file, which can be accessed and reviewed anytime using this option. The tool retrieves the desired prediction from the JSON file and displays it for the user, enabling them to reflect on the past predictions or to compare them with the actual outcomes.

![User Interface - View Previous Predictions](https://i.imgur.com/G2HDFXb.png)

# License

This project is licensed under the MIT License.
