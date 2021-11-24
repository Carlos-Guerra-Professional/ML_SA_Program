# ML_SA_Program
Machine Learning Sentiment Analysis Program


This school-assigned project tasked us to provide a Machine Learning software solution for any scenario we choose. 

I decided to create a software solution in a scenario where a fictitious clothing eCommerce company could use that solution to accurately determine the sentiment analysis of individual reviews for products as well as the overall sentiment of all reviews for each individual product simply by having the software solution automatically analyze the vocabulary customers use in their reviews. Thus, providing a company with very useful and important information in order to make better business decisions quickly and efficiently.

The software solution can provide sentiment analysis on thousands of reviews within a matter of seconds. Something that would take humans manually reviewing each review, to determine the review’s overall sentiment, much longer... from hours to possibly days. It also provides a way to determine quickly what the total sentiment of a product is by averaging out the sentiment score, on a float value scale, for each product. This would allow a company to remove products that are underperforming or provide more products that are doing well. The value of this type of information, and all of that information’s applications, is essential in today's eCommerce industry.

I used the Machine Learning method of Sentiment Analysis through the Rule-Based Machine Learning method. I also used a Lexicon (Nielsen, AFINN-en-165.txt 2015) someone else had already created. A Lexicon is a dictionary of words and their associated sentiment values score as determined by human input. This would allow the computer software to use a dictionary of words and their associated sentiment value score to automatically determine the sentiment of a review by calculating the average value of all words’ sentiment scores found in both the review and in the Lexicon. For this to work, the raw data must be cleaned and unnecessary information, such as columns that contain information not needed for the analysis, are filtered out and removed manually. The program itself then filters, cleans, and processes the data to complete the sentiment analysis.

The program also allows for additional information to be determined and provides reports and visualizations in the form of various tables as well as various charts and graphs that each provides unique insight into the analyzed data.

The dataset used was found on the website Kaggle.com. (Brooks, Women's E-Commerce Clothing Reviews 2018)

The program itself is user-friendly but here are some simple instructions to get it started.

Once the programming is running...

1.) Login using username: "test" and password: "test"
2.) Select the raw_dataset.csv found in the programs ML_SA_Project file folder, next to his README.txt file. You may need to save this file first.
3.) Run through each section of the program, noticing the changes to the table, by cleaning the dataset using the "Clean Dataset" button, running the sentiment analysis using the "Run Sentiment Analysis" button. You will notice that the columns for "Numerical_Rating" and "Sentiment_Rating" are now populated with their corresponding values based on the sentiment analysis.
4.) Run a query using the following syntax... "Clothing_ID == (ENTER A CLOTHING_ID VALUE HERE)". Choose a "Clothing_ID" value from the column in the table.
5.) Review the Queried Data Table and view the information provided in the pie chart, bar graph, and box plot using their corresponding buttons located underneath the "Queried Data Table".
6.) Go back to the "Sentiment Rated Data Table" and enter a "Clothing_ID" value into the second input box and click its corresponding "Run Average" button to see the average numerical rating and total sentiment rating of that particular product.
7.) You can repeat this entire process as many times as needed for different products.


Sources:

Nielsen, F. Ã…. (2015, September 29). AFINN-en-165.txt. GitHub.
https://github.com/fnielsen/afinn/blob/master/afinn/data/AFINN-en-165.txt.

Brooks, N. (2018, February 3). Women's E-Commerce Clothing Reviews. Kaggle.
https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews. 




