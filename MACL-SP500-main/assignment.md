For this assignment you will need to form groups of 3 people, officially I see 72 students enrolled so it is nicely divisible to 24 teams. I assume that some people will not do the assignment so I expect some 2-people teams but that should be the minority. You should e-mail me with your team composition by Friday 8/11. The e-mail topic should be [MACLE2425] Team declaration. Please send one e-mail per team. If your name doesn't appear in a team by Friday 08/11 23:59 CET, you automatically get 0.

Now to the actual assignment part, you will work with a dataset of an S&P500 ETF. The dataset has 4 features, namely the date, the opening price, the high price of the day, and the low price of the day. Your goal is to make a model that will be able to predict the closing price of a specific amount dates as provided in a testing file based on the historical data given in the training file. For example, to develop your model, you should split the CSV into two files with the testing file containing the data of October 2024. I WILL test your model with different data so it should be flexible in its inputs and outputs. You should assume that the testing will always be around 20-30 points (i.e. I essentially want to predict the next month). Following is your dataset

    historicalData_IE00B5BMR087_clean.csv

To evaluate the performance (and train your model) you should use the mean absolute percentage error.

You are allowed to use any modelling method you want. In general this form of problems have to to with time-series analysis, regression etc. You can use such keywords to find relevant literature.

The deliverable of the project is your python file that I will run to evaluate your model, and a report that will be 6 pages long, including citations, using the IEEE double column format. You should zip them and upload them here. Each member of the team should upload the same zip. To start writing your project I have attached a barebones project in a zip file.

    macle_barebones.zip

The program should accept two CSV files, one with the training data and one with the testing data. It should print the MAPE of the testing data and any plots you might find useful. From lilbraries you are allowed to use numpy, matplotlib, scikit learn, pandas, pytorch, jax, and pytest in case you want to write unit tests.

The report should describe your choices, why you made them, etc. It should have a structure of Introduction, Method, Results, Citations. In general, it should reflect on your process and show me that you understand the topic.

The grading will be done in two parts, the first part will evaluate the report with a 0.66 weight, and the second part will be based on the performance of your model compared to the other teams. The most performant team will get full marks (of the remaining weight) while the last team will get 0. The other teams will get something in-between scaled linearly between the first and the last team. If you want a grading formula it should look like this:

    (0.66 _ percentage of report + 0.34 _ percentage of performance) \* 6

where 6 is the max grade you can get out of the total 20 for the course. If the code gives me any error and I cannot run it you get a 0 in the performance. So make it robust and test well.

Good luck and have fun!
