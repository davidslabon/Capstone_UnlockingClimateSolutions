import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# define global styling params
rcParams = {
    "axes.titlesize" : 20,
    "axes.titleweight" :"bold",
    "axes.labelsize" : 12,
    "lines.linewidth" : 3,
    "lines.markersize" : 10,
    "xtick.labelsize" : 16,
    "ytick.labelsize" : 16,
            }


def get_response_pivot(data, year, questionnumber, columnnumber='all', pivot=True):
    '''A query function that creates a pivot with multilevel index on questions and columns'''
    
    # create list of unique column numbers if no numbers were given
    if columnnumber == 'all':
        columnnumber = data.query('question_number == @questionnumber & year == @year').column_number.unique().tolist()
        columnnumber = sorted(columnnumber)
        
    # get data from basic dataframe
    df = data.query(
     ('question_number == @questionnumber & column_number == @columnnumber & year == @year')
         ).loc[:,[
                'account_number',
                'row_number',
                'row_name',
                'question_number',
                'question_name',
                'column_number',
                'column_name',
                'response_answer'
                ]]
       
    # print question
    print_question(df, questionnumber, columnnumber)

    # sort values by QuestionNumber, ColumnNumber and Account for optimized indexing.
    df = df.sort_values(by=['question_number', 'column_number', 'account_number'])
    
    # create a Key to identify multiple combinations of the same datapoint
    df['Key'] = df.groupby([
                            'account_number', 
                            'row_number', 
                            'row_name', 
                            'question_number', 
                            'column_number'
                            ]).cumcount()
   
    if pivot:     # return pivot table if pivot == True
        # build a pivot table
        pivot_df = df.pivot(
                    index= [                      # define multilevel row index
                            'account_number', 
                            'row_number', 
                            'row_name', 
                            'Key'
                            ], 
                    columns= [                    # group by Question and Column
                            'question_number', 
                            'column_number'
                            ], 
                    values='response_answer'       # set answers as values
                    )

        return pivot_df

    else:       # return filtered dataframe if pivot == False
        return df
        


def print_question(data, questionnumber, columnnumber):
    """Print unique column / question combination"""
    
    print(f'Question {questionnumber}:')
    print(str(data.loc[data.question_number==questionnumber]
                                .question_name.unique())
                                .replace("[","")
                                .replace("]",""))
    print('------------------------------------------------------------------------------------------')
        
    for c in columnnumber:
        print(f'''{c}: {(str(data.loc[data.column_number== c]
                                    .column_name.unique()))
                                    .replace("[","")
                                    .replace("]","")}''')


# First, we start by importing the different csv files and continue by concatenating the files into one dataframe.
def get_data(path, filename_start):
    '''a function to store the content of a directory into a pd dataframe'''
    
    # checking the contents of the directory using the os-module. 
    files = [
        file for file in os.listdir(path) 
        if file.startswith(filename_start)
        ]
    
    print(files)  
    
    # iterate through files and add to the data frame
    all_data = pd.DataFrame()
    for file in files:
        current_data = pd.read_csv(path+"/"+file, dtype={'comments': str})
        all_data = pd.concat([all_data, current_data], ignore_index=True)

    # replace whitespaces from column names 
    all_data.columns = [i.lower().replace(' ', '_') for i in all_data.columns]
        
    print(f'''\nA dataframe with {all_data.shape[0]} rows and {all_data.shape[1]} columns has been created!\nColumn names are now lower case and spaces are replaced by "_".''')
    
    return all_data




# This function comes from Silas and could be an idea for further development.
def meta(df, transpose=True):
    """
    This function returns a dataframe that lists:
    - column names
    - nulls abs
    - nulls rel
    - dtype
    - duplicates
    - number of diffrent values (nunique)
    """
    metadata = []
    dublicates = sum([])
    for elem in df.columns:

        # Counting null values and percantage
        null = df[elem].isnull().sum()
        rel_null = round(null/df.shape[0]*100, 2)

        # Defining the data type
        dtype = df[elem].dtype

        # Check dublicates
        duplicates = df[elem].duplicated().any()

        # Check number of nunique vales
        nuniques = df[elem].nunique()


        # Creating a Dict that contains all the metadata for the variable
        elem_dict = {
            'varname': elem,
            'nulls': null,
            'percent': rel_null,
            'dtype': dtype,
            'dup': duplicates,
            'nuniques': nuniques
        }
        metadata.append(elem_dict)

    meta = pd.DataFrame(metadata, columns=['varname', 'nulls', 'percent', 'dtype', 'dup', 'nuniques'])
    meta.set_index('varname', inplace=True)
    meta = meta.sort_values(by=['nulls'], ascending=False)
    if transpose:
        return meta.transpose()
    print(f"Shape: {df.shape}")

    return metadata

    return all_data



def get_responses(data, question_number, column_number=[1], row_number=[1], theme='combined',year=[2018,2019,2020]):
    '''’A query function that creates a new dataframe with responses from the given data.'''
    # Reduktion auf ausgewählte Menge:
    responses = data[(data.theme == theme) &
                     (data.year.isin(year)) &
                     #(data.q_nr == q_nr) &
                     (data.question_number == question_number) &
                     (data.column_number.isin(column_number)) &
                     (data.row_number.isin(row_number)) 
                    ].copy()

    # Ausgabe der Haupt-Frage:
    print(f'AnswerCount = {responses.shape[0]}')
    #quest_num = data[(data.q_nr == q_nr)].question_number.iat[0]
    quest_num = data[(data.question_number == question_number)].question_number.iat[0]
    #question = data[(data.q_nr == q_nr)].question_name.iat[0]
    question = data[(data.question_number == question_number)].question_name.iat[0]
    print(f'QuestionNumber = {quest_num}:\n{question}')

    # Sortierung:
    result = responses.sort_values(by=['type',
                                       'theme',
                                       #'year',
                                       'account_number',
                                       'response_pnt'])[[#'type',
                                                         #'theme',
                                                         #'year',
                                                         'account_number',
                                                         'response_pnt',
                                                         'column_name',
                                                         'row_name',
                                                         'response_answer']]
    return result


def question_number_cleaning(question_number_string):
    dict_l3 = {'a':'1', 'b':'2', 'c':'3', 'd':'4', 'e':'5', 
               'f':'6', 'g':'7', 'h':'8', 'i':'9', 'j':'10', 
               'k':'11', 'l':'12', 'm':'13', 'n':'14', 'o':'15', 
               'p':'16', 'q':'17', 'r':'18', 's':'19', 't':'20', 
               'u':'21', 'v':'22', 'w':'23', 'x':'24', 'y':'25', 'z':'26'}
    last_char = question_number_string[-1]
    
    if question_number_string == 'Response Language':
        q_nr_l1, q_nr_l2, q_nr_l3 = '00','00','01'
    elif question_number_string == 'Amendments_question':
        q_nr_l1, q_nr_l2, q_nr_l3 = '00','00','02'
    elif last_char in  dict_l3:
        question_number_string = question_number_string[0:-1]
        q_nr_l1 = question_number_string.split('.')[0].zfill(2)
        q_nr_l2 = question_number_string.split('.')[1].zfill(2)
        q_nr_l3 = dict_l3[last_char].zfill(2)
    else:
        q_nr_l1 = question_number_string.split('.')[0].zfill(2)
        q_nr_l2 = question_number_string.split('.')[1].zfill(2)
        q_nr_l3 = '00'
    return q_nr_l1, q_nr_l2, q_nr_l3


def plot_freq_of_cv(data, title, xlabel, ylabel, orient="v"):
    """Creates a frequency plot based on a count values function.
    
    Attributes:
        - data: a count values object which has the x-separation as index
        and the y-values as values.
        - title, xlabel, ylabel: Textstrings for visualization
        - orient: Choose between vertical and horizontal barplot
    """
    if orient == "v":
        x = data.index
        y = data.values
    elif orient == "h":
        x = data.values
        y = data.index
        
    fig = sns.barplot(x=x, y=y, palette="hls", orient=orient)
    plt.title(
        label=title, 
        fontdict={
            'fontsize': rcParams['axes.titlesize'],
            'fontweight' : rcParams['axes.titleweight'],
                }
            )
    plt.xlabel(
        xlabel=xlabel,
        fontdict={
        'fontsize': rcParams['axes.labelsize'],
            }
        )
    plt.ylabel(
        ylabel=ylabel,
        fontdict={
        'fontsize': rcParams['axes.labelsize'],
            }
        )
    
    return fig



def get_pct_freq(data):
    """Returns the absolute and relativ frequncy as count and % for the values of a series.
    
    Attributes:
        - data: has to be a series
        
    Output:
        - Series of absolut values, Series of % values
    """
    
    val_c = data.value_counts
    perc = round((data.value_counts(normalize=True)*100),1)
    
    return val_c, perc 
    


