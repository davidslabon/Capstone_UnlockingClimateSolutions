import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display_html, display ,Image


# define global styling params
rcParams = {
    "axes.titlesize" : 20,
    "axes.titleweight" :"bold",
    "axes.labelsize" : 12,
    "axes.palette": "mako",
    "lines.linewidth" : 3,
    "lines.markersize" : 10,
    "xtick.labelsize" : 16,
    "ytick.labelsize" : 16,
    "patches.labelsize": 12,
    "axes.small_titlesize" : 10,
    "axes.small_titleweight" :"bold",
    "axes.small_labelsize" : 8,
    "lines.small_linewidth" : 3,
    "lines.small_markersize" : 8,
    "xtick.small_labelsize" : 8,
    "ytick.small_labelsize" : 8,
    "patches.small_labelsize": 8,
            }

# QUERY FUNCTIONS

def get_response_pivot(data, questionnumber, columnnumber='all', pivot=True, add_info=False, year=["2018", "2019", "2020"]):
    '''A query function that creates a pivot with multilevel index on questions and columns'''
    
    # create list of unique column numbers if no numbers were given
    if columnnumber == 'all':
        columnnumber = data.query('question_number == @questionnumber & year == @year').column_number.unique().tolist()
        columnnumber = sorted(columnnumber)
        
    # get data from basic dataframe
    if add_info:
        
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
                    'response_answer',
                    'year',
                    'entity',
                    'city',
                    'population',
                    'region',
                    'country',
                      ]]
   
    elif not add_info:
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
                    'response_answer',
                    'year',
                    'entity',
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
    
    q = f'''Q{questionnumber}:{str(data.loc[data.question_number==questionnumber]
                                .question_name.unique())
                                .replace("[","")
                                .replace("]","")}'''
    
    print(q)
    
    print('------------------------------------------------------------------------------------------')
        
    for c in columnnumber:
        col = f'''{c}: {(str(data.loc[data.column_number== c]
                                    .column_name.unique()))
                                    .replace("[","")
                                    .replace("]","")}'''
        print(col)
    
    q_string = f'{q} /C{col}'
    
    return q_string


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


def get_var_indexed_responses(df, question_number, select_col, select_answer_param, show_col):
    """A function that allows to query tables-like responses with variable number of rows.
    
    Arguments:
     - df: dataframe to filter, should be either like cir or cor
     - question_number: top level question to observe, e.g. "C2.3a"
     - select_col: column that holds the filter criteria, e.g. "1"
     - select_answer_param: criteria to filter the select_col with, e.g. "Risk 1", "Risk 2", etc.
     - show_col: The colum that should be used to show the answer
     
    Output:
    a filtered data-frame
    
    """
    
    # create base_df and add selection_key
    base_df = df.copy().query('question_number == @question_number & (column_number == @select_col | column_number == @show_col)')
    base_df["select_key"] = base_df.year.astype(str)+"_"+base_df.account_number.astype(str)+"_"+base_df.row_number.astype(str)
    
    # get selection_keys from select_col
    selection = list(base_df.copy().query('question_number == @question_number & column_number == @select_col & response_answer == @select_answer_param').select_key)
    
    # query dataframe based on show_col and selection
    response_df = base_df.copy().query('question_number == @question_number & column_number == @show_col and select_key == @selection')
    
    return response_df



def get_responses(data, question_number, column_number=[1], row_number=[1], theme='combined',year=["2018","2019","2020"]):
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



def get_pct_freq(data):
    """Returns the absolute and relativ frequncy as count and % for the values of a series.
    
    Attributes:
        - data: has to be a series
        
    Output:
        - Series of absolut values, Series of % values
    """
    
    val_c = data.value_counts()
    perc = round((data.value_counts(normalize=True)*100),1)
    
    return val_c, perc 




## PLOTTING FUNCTIONS

def create_3x3grid(size=(15,10), orient="vertical"):
    """Creates an 3x3 plotting grid with one big and three small plots.
    Attributes:
    - size: tuple(height, width)
    - type: 'vertical', 'horizontal' 
      #vertical: big plot spans on columns 1 + 2
      #horizontal: big plot spans on rows 1+2
    
    Output:
    - returns 5 variables containging the grid specification and the 4 plotting grids information.
    """
    
       
    # Create basic figure 
    fig = plt.figure(1, figsize=(size))

    # set up grid with subplots of different sizes
    gs=GridSpec(3,3) # 3 rows, 3 columns
    
    if orient == "vertical":
        ax_b1=fig.add_subplot(gs[:,:2]) # span all rows and columns 1 + 2
        ax_s1=fig.add_subplot(gs[0,2]) # first row, third column
        ax_s2=fig.add_subplot(gs[1,2]) # second row, third column
        ax_s3=fig.add_subplot(gs[2,2]) # third row, third columns
    elif orient == "horizontal":
        ax_b1=fig.add_subplot(gs[:2,:]) # span all columns and rows 1 + 2
        ax_s1=fig.add_subplot(gs[2,0]) # third row, first column
        ax_s2=fig.add_subplot(gs[2,1]) # third row, second column
        ax_s3=fig.add_subplot(gs[2,2]) # third row, third columns
        
    return fig, ax_b1, ax_s1, ax_s2, ax_s3


def plot_freq_of_cv(data, title, xlabel, ylabel, orient="v", ax=None):
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
        
    fig = sns.barplot(x=x, y=y, palette=rcParams["axes.palette"], orient=orient, ax=ax)
    fig.set_title(
        label=title, 
        fontdict={
            'fontsize': rcParams['axes.titlesize'],
            'fontweight' : rcParams['axes.titleweight'],
                }
            )
    fig.set_xlabel(
        xlabel=xlabel,
        fontdict={
        'fontsize': rcParams['axes.labelsize'],
            }
        )
    fig.set_ylabel(
        ylabel=ylabel,
        fontdict={
        'fontsize': rcParams['axes.labelsize'],
            }
        )
    
    return fig


def plot_small_no_responses(df, ax=None):
    '''Create a small subplot with the number total number of responses
     Attributes:
     - ax: Define ax if you want to use in subplot / facetgrid
     '''
    
    # calculate number of responses
    no_responses = [len(df)]
    
    # plot results
    x = ["Evaluable"]
    y =  no_responses
    xlabel="" 
    ylabel="Count" 
    title="No of Responses"
    orient="v"
    fig = sns.barplot(x=x, y=y, palette=rcParams["axes.palette"], orient="v", ax=ax)
    
    fig.set_title(
        label=title, 
        fontdict={
            'fontsize': rcParams['axes.small_titlesize'],
            'fontweight' : rcParams['axes.small_titleweight'],
                }
            )
    fig.set_xlabel(
        xlabel=xlabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
    fig.set_ylabel(
        ylabel=ylabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
    for p in fig.patches:
         fig.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=rcParams["patches.small_labelsize"], fontweight="bold", color='black', xytext=(0, 5),
             textcoords='offset points')
   
    return fig   


def plot_small_responses_yoy(df, ax=None, plt_type="total"):
    '''Create a small subplot with the number of responses per year
    Attributes:
     - ax: Define ax if you want to use in subplot / facetgrid
     - plt_type: choose either "total" or "perc"
    '''
    
    # calculate responses per year
    # preprocess / calculate data for visualization
    years = df.year
    counts, perc = get_pct_freq(years)
    values = counts if plt_type == "total" else perc
                          
    # plot results
    x = values.index
    y =  values.values
    xlabel="Year" 
    ylabel="Count" if plt_type == "total" else "% of Total Count"
    title="Responses per Year"
    orient="v"
    fig = sns.barplot(x=x, y=y,palette=rcParams["axes.palette"], ax=ax)
    
    fig.set_title(
        label=title, 
        fontdict={
            'fontsize': rcParams['axes.small_titlesize'],
            'fontweight' : rcParams['axes.small_titleweight'],
                }
            )
    fig.set_xlabel(
        xlabel=xlabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
    fig.set_ylabel(
        ylabel=ylabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
    
    for p in fig.patches:
             fig.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=rcParams["patches.small_labelsize"], fontweight="bold", color='black', xytext=(0, 5),
                 textcoords='offset points')

    return fig   


def plot_small_responses_per_ptcp(df, ax=None):
    '''Create a small subplot with the number of responses per year
    Attributes:
     - ax: Define ax if you want to use in subplot / facetgrid
     - plt_type: choose either "total" or "perc"
    '''
    
    # calculate responses per year
    # preprocess / calculate data for visualization
    data = df.groupby(["account_number", "year"], as_index=False)["response_answer"].count()
    data.year = data.year.astype(str)
                          
    # plot results
    xlabel="No Responses to this question" 
    ylabel= "Count"
    title="Responses per Participant"
    orient="v"
    
    # Exclude questions with max 1 response per participant from KDE-plot
    if data.response_answer.nunique() > 1:
        fig = sns.histplot(data, x="response_answer", hue="year", palette=rcParams["axes.palette"], bins=20, kde=True, ax=ax, multiple="stack")
    else:
        fig = sns.histplot(data, x="response_answer", hue="year", palette=rcParams["axes.palette"], bins=20, kde=False, ax=ax, multiple="stack")
    
    fig.set_title(
        label=title, 
        fontdict={
            'fontsize': rcParams['axes.small_titlesize'],
            'fontweight' : rcParams['axes.small_titleweight'],
                }
            )
    fig.set_xlabel(
        xlabel=xlabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
    fig.set_ylabel(
        ylabel=ylabel,
        fontdict={
        'fontsize': rcParams['axes.small_labelsize'],
            }
        )
      
    return fig   


def plot_pareto(data, xlabel, ylabel, title, orient="v"):
    """Plots a pareto bar chart
    Attributes:
    - data: list or pd.series
    - xlabel: labeltext as string
    - ylabel: labeltext as string
    - title: labeltext as string
    - orient: orientation "h" or "v"
    
    Function is not working in combination with GridSpec-Tool!"""

    # calculate pareto values 
    weights = data / data.sum()
    cumsum = weights.cumsum()

    # create subplot fig and first ax
    fig, ax1 = plt.subplots()
    
    # Configure main plot
    ax1 = plot_freq_of_cv(data=data, xlabel=xlabel, ylabel=ylabel,
                            title=title, orient=orient)
    
    if orient == "v":
        # add 2nd graph to plot
        ax2 = ax1.twinx()
        ax2 = sns.lineplot(x = data.index, y = cumsum, palette=rcParams["axes.palette"])
        ax2.set_yticks([])
        ax2.set_ylabel(None)
    elif orient == "h":
        # add 2nd graph to plot
        ax2 = ax1.twinx()
        ax2 = sns.lineplot(x = cumsum, y = data.index, palette=rcParams["axes.palette"])
        ax2.set_xticks([])
        ax2.set_xlabel(None)
  
    formatted_weights = ["{0:.0%}".format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (data.index[i], cumsum[i]))  
        
    return ax1



## HELPER FUNCTIONS

def sorter(column):
    """A small helper function to sort in a specific order, e.g. for categorical data"""
    mapper = {name: order for order, name in enumerate(order)}
    return column.map(mapper)


def identify_theme(strng):
    if strng[0] == 'C':
        result = 'climate'
    elif strng[0] == 'W':
        result = 'water'
    else:
        result = 'other'
    return result


def cut_labels(fig, axis, max_length=10):
    '''Shortens the labels of an axis to a given length.'''
    
    if axis == "x":
        new_labels = [i.get_text()[0:max_length] if len(i.get_text()) > max_length else i.get_text() 
              for i in fig.xaxis.get_ticklabels()]

        return fig.xaxis.set_ticklabels(new_labels)  
    
    elif axis == "y":
        new_labels = [i.get_text()[0:max_length] if len(i.get_text()) > max_length else i.get_text() 
              for i in fig.yaxis.get_ticklabels()]

        return fig.yaxis.set_ticklabels(new_labels)  
    
    
def add_patches(fig, orient='v'):
    """adding value patches to plot
    Attributes:
    - fig = plot figure, preferably barplot, histplot countplot or equivalent
    - orient = orientation'v' or 'h'
    """
    if orient =='v':
        for p in fig.patches:
             fig.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=rcParams["patches.small_labelsize"], color='black', xytext=(0, 5),
                 textcoords='offset points')
        return fig 
    elif orient == 'h':
        for p in fig.patches:
             fig.annotate("%.0f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height()),
                 ha='center', va='center', fontsize=rcParams["patches.small_labelsize"], color='black', xytext=(5, 7),
                 textcoords='offset points')
        return fig 
    
    
def rotate_labels(fig, axis, rotation):
    """A function to rotate axis labels
    Attributes:
    - fig: figure, plot to work on
    - axis: "y" or "x" - axis to rotate
    - rotation: integer - degrees rotation"""
        
    if axis == "x":
        for item in fig.get_xticklabels():
              item.set_rotation(rotation)
        return fig
    
    elif axis == "y":
        for item in fig.get_yticklabels():
              item.set_rotation(rotation)
        return fig
    
    
def get_distribition_df(data):
    """Creates a dateframe showing value counts and relative distribution of the series' values
    Attributes: 
    - data: pd.Series or array / list of values"""

    print("Unique answers: " +str(data.nunique()))
    print("Total count: " +str(len(data)))
    df = pd.DataFrame()
    df["counts"] = data.value_counts()
    df["perc"] = df.counts / df.counts.sum()
    df.perc = df.perc.apply(lambda x:"{0:.1%}".format(x))
    return df

def display_side_by_side(*args):
    """show multiple dataframes in one output
    Attributes:
    - *args: dataframes
    """
    
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    

def compare_columns(data, questionnumber, select_col, compare_col):
    """Takes a question with multiple columns / rows and presents the input columns next to each other.
    Attributes:
    - data: A question dataframe, e.g. cir / cor
    - questionnumber: string - question_number to query
    - select_col: first column to merge on
    - compare_col: column to merge side by side"""
    
    df = data.copy()
    df.response_answer = df.response_answer.apply(lambda x: x.split(">")[0]) 
    answers = df.response_answer     # get responses from data frame

    # build a dateframe with two answer columns next to each other
    # query information, create common key per row
    base_df = df.copy().query('column_number == @select_col | column_number == @compare_col')
    base_df["select_key"] = base_df.year.astype(str)+"_"+base_df.account_number.astype(str)+"_"+base_df.row_number.astype(str)

    # create "left" dataframe 
    select_df = base_df.query('column_number ==@select_col').loc[:,["year", "response_answer", "select_key"]]
    select_df.rename(columns={'response_answer': "column_"+str(select_col)}, inplace=True)
    select_df.set_index('select_key', inplace=True)

    # create corrosponding "right" dataframe
    compare_df = base_df.query('column_number ==@compare_col').loc[:,["response_answer", "select_key"]]
    compare_df.rename(columns={'response_answer': "column_"+str(compare_col)}, inplace=True)
    compare_df.set_index('select_key', inplace=True)

    #concat dfs
    result = pd.merge(select_df, compare_df, left_index=True, right_index=True).reset_index()
    return result
    

def rename_and_merge(original_df, feature_df, feature, left_on=None, right_on=None, how='left'):

    '''this function helps to quickly rename the new feature column and to merge it to the original disclosure dataframe, drop duplicates as well as the key_0 column
    original_df: disclosure dataframe that the feature is mapped onto
    feature: information that is supposed to be added to the original df. Needs to be passed on as a list.
    feature_df: dataframe with the information that is supposed to be added to the original df
    left_on: information used for mapping the data. Default set to 'select_key'
    right_on: information from feature df used for mapping. Default set to 'select_key'
    how: default set to left'''

    #create unique select key from year and account number in feature dataframe
    feature_df["select_key"] = feature_df.year.astype(str)+"_"+feature_df.account_number.astype(str)
    original_df["select_key"] = original_df.year.astype(str)+"_"+original_df.account_number.astype(str)

    # rename response answer to the new feature in feature df
    feature_df[feature] = feature_df["response_answer"]

    # merge feature column to disclosure dataframe
    if left_on is None:
        left_on = original_df["select_key"]
    
    if right_on is None:
        right_on = feature_df["select_key"]
    

    original_df = pd.merge(left=original_df,
                          right=feature_df[feature],
                          left_on=left_on,
                          right_on=right_on,
                          how=how)
    original_df.drop_duplicates(inplace=True)
    original_df.drop("key_0", axis=1, inplace=True)
    original_df.drop("select_key", axis=1, inplace=True)
    return original_df

def split_response(df, column, sep=';', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df
