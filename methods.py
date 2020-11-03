def get_response_pivot(data, year, questionnumber, columnnumber='all'):
    '''A query function that creates a pivot with multilevel index on questions and columns'''
    
    # create list of unique column numbers if no numbers were given
    if columnnumber == 'all':
        columnnumber = data.query('QuestionNumber == @questionnumber').ColumnNumber.unique().tolist()
        columnnumber = sorted(columnnumber)
        
    # get data from basic dataframe
    df = data.query(
     ('QuestionNumber == @questionnumber & ColumnNumber == @columnnumber')
         ).loc[:,[
                'AccountNumber',
                'RowNumber',
                'RowName',
                'QuestionNumber',
                'QuestionName',
                'ColumnNumber',
                'ColumnName',
                'ResponseAnswer'
                ]]
     
    
    # sort values by QuestionNumber, ColumnNumber and Account for optimized indexing.
    df = df.sort_values(by=['QuestionNumber', 'ColumnNumber', 'AccountNumber'])
    
    # create a Key to identify multiple combinations of the same datapoint
    df['Key'] = df.groupby([
                            'AccountNumber', 
                            'RowNumber', 
                            'RowName', 
                            'QuestionNumber', 
                            'ColumnNumber'
                            ]).cumcount()
    
    # build a pivot table
    pivot_df = df.pivot(
                index= [                      # define multilevel row index
                        'AccountNumber', 
                        'RowNumber', 
                        'RowName', 
                        'Key'
                        ], 
                columns= [                    # group by Question and Column
                        'QuestionNumber', 
                        'ColumnNumber'
                        ], 
                values='ResponseAnswer'       # set answers as values
                )
    
    return pivot_df
        