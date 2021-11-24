# Load libraries
import PySimpleGUI as sg
import matplotlib.pyplot
import numpy as np
import pandas as pd
import re
import nltk.classify.util
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

filename = ""


def average_query_window(item_id, numerical_rating_avg, total_item_sentiment):
    layout = [[sg.Text("Item " + str(item_id) + " has an average numerical rating of: "), sg.Text(str(numerical_rating_avg))],
              [sg.Text("Item " + str(item_id) + " has a total sentiment rating of: "), sg.Text(str(total_item_sentiment))],
              [sg.Text("")],
              [sg.Text("")],
              [sg.Button("Go Back")]
              ]
    window = sg.Window("Item Average Numerical & Sentiment Rating", layout, modal=True, margins=(150, 100))

    while True:
        event, values = window.read()
        if event == "Go Back":
            window.close()
            show_rated_data_table()
        if event == sg.WIN_CLOSED:
            window.close()
            break

    window.close()


def average_query(item_id):
    rated_dataset_file_address = 'venv\sentiment_rated_dataset'
    rated_dataset = pd.read_csv(rated_dataset_file_address, header=0, index_col=0)

    total_item_count = 0
    total_item_value = 0
    numerical_rating_avg = 0
    for rated_dataset_item in rated_dataset.itertuples():
        if rated_dataset_item[1] == item_id:
            total_item_value = total_item_value + rated_dataset_item[3]
            total_item_count = total_item_count + 1

            numerical_rating_avg = total_item_value / total_item_count

    if numerical_rating_avg >= 1:
        total_item_sentiment = "Positive"
    elif numerical_rating_avg <= 0:
        total_item_sentiment = "Negative"
    else:
        total_item_sentiment = "Neutral"

    print(numerical_rating_avg)
    print(total_item_sentiment)

    average_query_window(item_id, numerical_rating_avg, total_item_sentiment)


def query_data(query_syntax):
    rated_dataset_file_address = 'venv\sentiment_rated_dataset'
    rated_dataset = pd.read_csv(rated_dataset_file_address, header=0, index_col=0)

    rated_dataset.query(query_syntax, inplace=True)

    rated_dataset.to_csv('venv\query_dataset')


def pie_chart(dataset_file_address):
    dataset = pd.read_csv(dataset_file_address, header=0, index_col=0)

    total_positive = 0
    total_negative = 0
    total_neutral = 0

    for item in dataset.itertuples():
        sentiment = str(item[4])
        if sentiment == 'Positive':
            total_positive = total_positive + 1
        if sentiment == 'Negative':
            total_negative = total_negative + 1
        if sentiment == 'Neutral':
            total_neutral = total_neutral + 1

        total_rated = total_positive + total_negative + total_neutral

        percentage_of_positive = total_positive / total_rated
        percentage_of_negative = total_negative / total_rated
        percentage_of_neutral = total_neutral / total_rated

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [percentage_of_positive, percentage_of_negative, percentage_of_neutral]
    explode = (0, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    matplotlib.pyplot.title("Pie Chart of Sentiment Percentages")
    plt.legend(title='Sentiment Labels:')
    plt.show()

    return


def bar_graph(dataset_file_address):
    dataset = pd.read_csv(dataset_file_address, header=0, index_col=0)

    total_positive = 0
    total_negative = 0
    total_neutral = 0

    for item in dataset.itertuples():
        sentiment = str(item[4])
        if sentiment == 'Positive':
            total_positive = total_positive + 1
        if sentiment == 'Negative':
            total_negative = total_negative + 1
        if sentiment == 'Neutral':
            total_neutral = total_neutral + 1

    labels = ['Positive', 'Negative', 'Neutral']
    rating_quantity = [total_positive, total_negative, total_neutral]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, rating_quantity, width)

    ax.set_ylabel('Total Amount of Each Sentiment Rating')
    ax.set_title('Bar Graph of of Each Sentiment Rating Amount')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.tight_layout()

    plt.show()

    return


def box_plot(dataset_file_address):
    dataset = pd.read_csv(dataset_file_address, header=0, index_col=0)

    box_plot_values = dataset['Numerical_Rating']

    fig, ax = plt.subplots()
    ax.set_title("Box Plot of the Numerical_Rating for Entire Dataset")
    ax.boxplot(box_plot_values, showfliers=False)
    plt.xticks([1], ["Numerical_Rating"])
    plt.show()

    return


def log_in(username, password):
    with open('venv\log_in_credentials_dataset') as csv:
        credentials = [[item.strip() for item in row.split(",")]
                       for row in csv]
        return any(username == details[0] and password == details[1]
                   for details in credentials)


def choose_a_csv_file_for_table():
    sg.set_options(auto_size_buttons=True)
    global filename
    filename = sg.popup_get_file(
        'filename to open', no_window=True, file_types=(("CSV Files", "*.csv"),))
    if filename == '':
        return

    df = pd.read_csv(filename, sep=',', engine='python', header=None)
    header_list = df.iloc[0].tolist()
    data = df[1:].values.tolist()
    show_raw_data_table(data, header_list)


def main_screen_window():
    layout = [[sg.Text("Choose a dataset file for Sentiment Analysis.")], [sg.Button("Choose a File")],
              ]
    window = sg.Window("Main Screen", layout, modal=True, margins=(200, 100))

    while True:
        event, values = window.read()
        if event == "Choose a File":
            window.close()
            choose_a_csv_file_for_table()
        if event == sg.WIN_CLOSED:
            break

    window.close()


def log_in_error_window():
    layout = [[sg.Text("Incorrect username and/or password.")],
              [sg.Button('Okay.')]]

    window = sg.Window("Log In Error Window", layout, margins=(150, 100))
    while True:
        event, values = window.read()
        if event == 'Okay.':
            window.close()
            log_in_window()
        elif event == sg.WIN_CLOSED:
            break

    window.close()


def log_in_window():
    window = sg.Window(title="Sentiment Analysis Application",
                       layout=[[sg.Text("LOG IN")], [sg.Text("Username"), sg.Input(key='-username-')],
                               [sg.Text("Password"), sg.Input(key='-password-')], [sg.Button("Log In")]],
                       margins=(200, 100)
                       )
    while True:
        event, values = window.read()
        username = values['-username-']
        password = values['-password-']
        # print(username, password)
        if event == "Log In":
            if not log_in(username, password):
                window.close()
                log_in_error_window()
                break
            else:
                window.close()
                main_screen_window()
                break

        elif event == sg.WIN_CLOSED:
            break

    window.close()


def clean_dataset():
    stop_words = stopwords.words('english')
    clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt', 'jeans', 'pant',
               'skirt', 'order', 'white', 'black', 'fabric', 'blouse', 'sleeve', 'even', 'jacket']
    lemm = WordNetLemmatizer()

    def clean_up_text(text):
        # The function to return all words in the text and then lower case them.
        words = re.sub("[^a-zA-Z]", " ", text)
        text = words.lower().split()
        return " ".join(text)

    def remove_stopwords_text(text):
        # The function to remove all the stopwords in the text.
        edited_text = [word.lower() for word in text.split() if
                       word.lower() not in stop_words and word.lower() not in clothes]
        return " ".join(edited_text)

    def remove_numbers_text(text):
        # The function to remove all numbers in the text.
        new_text = []
        for word in text.split():
            if not re.search('\d', word):
                new_text.append(word)
        return ' '.join(new_text)

    def apply_lemmatize(text):
        # The function to apply lemmatizing to all words in the text.
        lemm_text = [lemm.lemmatize(word) for word in text.split()]
        return " ".join(lemm_text)

    # Load dataset
    raw_data_file_address = filename
    dataset = pd.read_csv(raw_data_file_address, header=0, index_col=0)

    dataset['Review'] = dataset['Review'].astype(str)
    dataset['Review'] = dataset['Review'].apply(clean_up_text)
    dataset['Review'] = dataset['Review'].apply(remove_stopwords_text)
    dataset['Review'] = dataset['Review'].apply(remove_numbers_text)
    dataset['Review'] = dataset['Review'].apply(apply_lemmatize)

    dataset.to_csv('venv\clean_dataset')


# Build a Sentiment Dictionary using Lexicon and run Sentiment Analysis
def sentiment_analysis():
    clean_data_file_address = 'venv\clean_dataset'
    clean_dataset = pd.read_csv(clean_data_file_address, header=0, index_col=0)
    print(clean_dataset['Review'])

    #Build Sentiment Dictionary
    sentiment_dictionary = {}
    for line in open('venv\AFINN-en-165'):
        word, score = line.split('\t')
        sentiment_dictionary[word] = int(score)
    print(sentiment_dictionary)

    # Sentiment Analysis(Descriptive) using the Rules-Based Method(Prescriptive)
    review_id = 0
    for clean_dataset_item in clean_dataset.itertuples():
        review_text = str(clean_dataset_item[2])

        word_per_review = review_text.split()

        review_value_total = 0
        word_count = 0
        for word in word_per_review:
            if word in sentiment_dictionary:
                word_value = sentiment_dictionary.get(word)
                review_value_total = review_value_total + word_value
                word_count = word_count + 1

                review_average_sentiment = review_value_total / word_count

        if review_average_sentiment >= 1:
            review_sentiment_polarity = "Positive"
        elif review_average_sentiment <= 0:
            review_sentiment_polarity = "Negative"
        else:
            review_sentiment_polarity = "Neutral"

        clean_dataset.loc[review_id, "Numerical_Rating"] = review_average_sentiment
        clean_dataset.loc[review_id, "Sentiment_Rating"] = review_sentiment_polarity

        review_id = review_id + 1

        print(str(review_id - 1) + ":     " + str(review_sentiment_polarity))

    clean_dataset.to_csv('venv\sentiment_rated_dataset')


def progress_bar_func():
    layout = [[sg.Text('Processing...')],
              [sg.Text("This may take a few moments. Please be patient.")],
              [sg.Text("New data table will appear once processing is complete.")],
              [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progressbar')],
              [sg.Cancel()]]

    window = sg.Window('Processing Data', layout)
    progress_bar = window['progressbar']

    for i in range(400):
        event, values = window.read(timeout=10)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break
        progress_bar.UpdateBar(i + 1)

    window.close()


def show_query_data_table():
    query_data_file_address ='venv\query_dataset'
    df = pd.read_csv(query_data_file_address, sep=',', engine='python', header=None)
    header_list = df.iloc[0].tolist()
    data = df[1:].values.tolist()

    layout = [[sg.Table(values=data,
              headings=header_list,
              display_row_numbers=False,
              max_col_width=150,
              justification="left",
              num_rows=min(25, len(data)))],
              [sg.Text("IMPORTANT - PLEASE NOTE THE FOLLOWING:")],
              [sg.Text("        If the query was \"Clothing_ID == (SOME NUMBER)\", then the pie chart, bar graph, and box plot visualizations will produce relevant and useful information.")],
              [sg.Text("        No other form of query can use pie chart, bar graph, or box plot to produce relevant and useful information.")],
              [sg.Text("")],
              [sg.Text("View pie chart of the total percentages of each Sentiment Rating in the table based on query parameters:"), sg.Button("View Pie Chart")],
              [sg.Text("View bar graph of the total quantity of each Sentiment Rating in the table based on query parameters:"), sg.Button("View Bar Graph")],
              [sg.Text("View box plot of the entire Numerical_Rating value range in the table based on query parameters:"), sg.Button("View Box Plot")],
              [sg.Text("___________________________________________________________________________________")],
              [sg.Text("Go back to Sentiment Rated Data Table:"), sg.Button("Go Back")]]

    window = sg.Window('Queried Data Table', layout, grab_anywhere=False)

    while True:
        event, values = window.read()
        if event == "View Pie Chart":
            pie_chart(query_data_file_address)
        if event == "View Bar Graph":
            bar_graph(query_data_file_address)
        if event == "View Box Plot":
            box_plot(query_data_file_address)
        if event == "Go Back":
            window.close()
            show_rated_data_table()
        if event == sg.WIN_CLOSED:
            break

    window.close()


def show_rated_data_table():
    rated_data_file_address ='venv\sentiment_rated_dataset'
    df = pd.read_csv(rated_data_file_address, sep=',', engine='python', header=None)
    header_list = df.iloc[0].tolist()
    data = df[1:].values.tolist()

    layout = [[sg.Table(values=data,
              headings=header_list,
              display_row_numbers=False,
              max_col_width=150,
              justification="left",
              num_rows=min(25, len(data)))],
              [sg.Text("Execute a query of data in dataset:"), sg.Input(key='-query_syntax-'), sg.Button("Run Query")],
              [sg.Text("__________________________________________________________")],
              [sg.Text("Enter a Clothing_ID number to get the average numerical and total sentiment rating for a particular item:")],
              [sg.Input(key='-item_id-'), sg.Button("Run Average")],
              [sg.Text("__________________________________________________________")],
              [sg.Text("Go back to Rated Data Table:"), sg.Button("Go Back")]]

    window = sg.Window('Sentiment Rated Data Table', layout, grab_anywhere=False)

    while True:
        event, values = window.read()

        query_syntax = values['-query_syntax-']
        item_id = values['-item_id-']

        if event == "Run Average":
            window.close()
            average_query(int(item_id))
        if event == "Run Query":
            try:
                query_data(str(query_syntax))
                window.close()
                show_query_data_table()
            except ValueError:
                sg.Popup('Error', 'Incorrect Syntax. Try again.')
        if event == "Go Back":
            window.close()
            show_clean_data_table()
        if event == sg.WIN_CLOSED:
            window.close()
            break

    window.close()


def show_clean_data_table():
    clean_data_file_address = 'venv\clean_dataset'
    df = pd.read_csv(clean_data_file_address, sep=',', engine='python', header=None)
    header_list = df.iloc[0].tolist()
    data = df[1:].values.tolist()

    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  display_row_numbers=False,
                  max_col_width=150,
                  justification="left",
                  num_rows=min(25, len(data)))],
        [sg.Text("Run Sentiment Analysis on chosen dataset file:"), sg.Button("Run Sentiment Analysis")],
        [sg.Text("------------------------------------------------")],
        [sg.Text("Go back to Main Screen:"), sg.Button("Go Back")],
    ]

    window = sg.Window('Clean Data Table', layout, grab_anywhere=False)

    while True:
        event, values = window.read()
        if event == "Run Sentiment Analysis":
            progress_bar_func()
            sentiment_analysis()
            window.close()
            show_rated_data_table()
        if event == "Go Back":
            window.close()
            main_screen_window()
        if event == sg.WIN_CLOSED:
            break

    window.close()


def show_raw_data_table(data, header_list):
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  display_row_numbers=False,
                  max_col_width=150,
                  justification="left",
                  num_rows=min(25, len(data)))],
        [sg.Text("Clean dataset for sentiment analysis process:"), sg.Button("Clean Dataset")],
        [sg.Text("------------------------------------------------")],
        [sg.Text("Go back to Main Screen:"), sg.Button("Go Back")],
    ]

    window = sg.Window('Raw Data Table', layout, grab_anywhere=False)

    while True:
        event, values = window.read()
        if event == "Clean Dataset":
            progress_bar_func()
            clean_dataset()
            window.close()
            show_clean_data_table()
        if event == "Go Back":
            window.close()
            main_screen_window()
        if event == sg.WIN_CLOSED:
            break


log_in_window()

