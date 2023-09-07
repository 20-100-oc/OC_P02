# This file has all the functions used in the project 2 notebook

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import missingno as msno
    


def heatmap(df_x, df_y, x_start=48.7, y_start=2.2, d=0.28, n=500):
    x_end, y_end = x_start+d, y_start+d    

    # theses 3 lines are to plot a bigger image (more pixels)
    img_proportion = np.array([6, 4])
    img_scale = 10
    img_size = img_proportion * img_scale

    heatmap = np.zeros((n, n), dtype='int')

    for x_tree, y_tree in zip(df_x, df_y):
        x_percent = (x_tree - x_start) / d
        y_percent = (y_tree - y_start) / d
        x_heatmap = int(x_percent * n)
        y_heatmap = int(y_percent * n)


        try:
            # stange formula because axis from graph don't correspond to a satelite view
            # (the 0 of vertical axis is at the top on the grah)
            heatmap[n - y_heatmap, x_heatmap] += 1
        except:
            #index out of bound if n too small
            if x_heatmap == n:
                heatmap[n - y_heatmap, n-1] += 1
            else:
                heatmap[1, x_heatmap] += 1

    fig, ax = plt.subplots(figsize=img_size) 
    heatmap_graph = sns.heatmap(heatmap)




def plot_nominal_column(column_name, 
                        df, 
                        include_na=True, 
                        percentage_limit=0, 
                        plot_type='bar', 
                        too_much_indices=True):
    '''
    Optional inputs
    
     - include_na:
        if True, keep NaN values as an index
        if False, drop NaN values
     - percentage_limit: 
         On the graph, if a represented index is considered
         to small (under percentage_limit), it is grouped in 
         the column 'other'.
     - plot_type:
         Can be 'bar' or 'pie'
    '''
    
    na_index = 'NaN'
    if include_na:
        column_data = df[column_name].fillna(na_index)
    else:
        column_data = df[column_name].dropna()
    
    ids = {}
    for index in column_data:
        try:
            ids[index] += 1
        except:
            ids[index] = 1

    size = len(column_data)
    id_to_del = []
    for index in ids:
        if ids[index] / size * 100 < percentage_limit:
            id_to_del.append(index)
    
    if len(id_to_del) > 0:
        ids['other'] = 0
        for index in id_to_del:
            ids['other'] += ids[index]
            del ids[index]
    
    # transform the dict into a list 
    # (to sort by nb of occurences so it's pretty to plot)
    item_list = list(ids.items())
    sorted_items = sorted(item_list, key=lambda x: x[1])
    x, y = [], []
    for item in sorted_items:
        x.append(item[0])
        y.append(item[1])
    
    # print info & graph
    print('-'*20)
    print(column_name)
    print('-'*20, '\n')

    nb_duplicates = column_data.duplicated().sum()
    nb_unique_values = len(column_data.unique())
    print('unique values :', nb_unique_values)
    print('duplicates :   ', nb_duplicates)
    
    plt.title('Distribution ' + column_name)
    if plot_type == 'pie':
        plt.pie(y, labels=x, autopct='%0.0f%%')
    elif plot_type == 'bar':
        plt.xlabel(column_name)
        plt.ylabel('count')
        if too_much_indices:
            plt.xticks(rotation=270)
        sns.barplot(x=x, y=y)
    
    smallest_bar = x[0]
    smallest_nb_index = y[0]
    print(f'\nnb of index \'{smallest_bar}\':', (ids[smallest_bar]))




def plot_ordinal_column(column_name, 
                        ordinal_order, 
                        df, 
                        include_na=True, 
                        percentage_limit=0, 
                        plot_type='bar'):
    '''
    Optional inputs
    
     - include_na:
        if True, keep NaN values as an index
        if False, drop NaN values
     - percentage_limit: 
         On the graph, if a represented index is considered
         to small (under percentage_limit), it is grouped in 
         the column 'other'.
     - plot_type:
         Can be 'bar' or 'pie'

    '''
    
    na_index = 'NaN'
    if include_na:
        column_data = df[column_name].fillna(na_index)
        ordinal_order.append(na_index)
    else:
        column_data = df[column_name].dropna()
    
    ordinal_list = [[index, 0] for index in ordinal_order]
    
    ids = {}
    for index in column_data:
        try:
            ids[index] += 1
        except:
            ids[index] = 1
    
    #print(ids)

    size = len(column_data)
    id_to_del = []
    for index in ids:
        if ids[index] / size * 100 < percentage_limit:
            id_to_del.append(index)
    
    if len(id_to_del) > 0:
        ids['other'] = 0
        for index in id_to_del:
            ids['other'] += ids[index]
            del ids[index]
    
    # transform the dict into a list 
    # (to sort by ordinal order)
    sorted_items = ordinal_list
    for i, sub_list in enumerate(ordinal_list):
        index = sub_list[0]
        sorted_items[i][1] = ids[index]
    #sorted_items = sorted(item_list, key=lambda x: x[1])
    x, y = [], []
    for item in sorted_items:
        x.append(item[0])
        y.append(item[1])
    
    # print info & graph
    print('-'*20)
    print(column_name)
    print('-'*20, '\n')
    
    nb_duplicates = column_data.duplicated().sum()
    nb_unique_values = len(column_data.unique())
    print('unique values :', nb_unique_values)
    print('duplicates :   ', nb_duplicates)
    
    plt.title('Distribution ' + column_name)
    if plot_type == 'pie':
        plt.pie(y, labels=x, autopct='%0.0f%%')
    else:
        plt.xlabel(column_name)
        plt.ylabel('count')
        sns.barplot(x=x, y=y)
    
    smallest_bar = x[0]
    smallest_nb_index = y[0]
    print(f'\nnb of index \'{smallest_bar}\'=', (ids[smallest_bar]))
    
    return ids




def get_quartiles(column_name, df):
    l = len(df[column_name])
    
    q1 = df[column_name][:l//2 + 1].median()
    q2 = df[column_name].median()
    q3 = df[column_name][l//2 + 1:].median()
    
    return q1, q2, q3




def quartile_exclusion(column_name, df):
    factor = 1.5
    
    q1, q2, q3 = get_quartiles(df[column_name])
    iq = q3 - q1
    max_value_accepted = q3 + factor*iq
    return max_value_accepted




def print_statistics(column_name, df):
    mean = df[column_name].mean()
    q1, q2, q3 = get_quartiles(column_name, df)
    
    print('mean =', round(mean, 2))
    print(f'Q1 = {q1}    median = {q2}    Q3 = {q3}')




def remove_impossible_values(column_name, impossible_limit, df, show_process):
    if impossible_limit is not None:
        df.loc[df[column_name] > impossible_limit, column_name] = np.nan
        
        if show_process:
            plt.figure()
            plt.title('Remove impossible values')
            sns.distplot(df[column_name], kde=True)
            plt.show()
            plt.figure()
            plt.title('Box plot')
            sns.boxplot(x=df[column_name])
            plt.show()
            
            print('max value in dataset=', max(df[column_name]))
            print_statistics(column_name, df)
            print('\n'*5)
            
    return df




def remove_conflictual_values(df, show_process):
    if show_process:
        plt.figure()
        plt.title('Original distribution')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()
        
        plt.figure()
        plt.title('Original distribution colored by "stade_developpement"')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m', 
                        hue='stade_developpement')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()

    replacement = np.nan
    df.loc[(df['hauteur_m'] == 0) & (df['circonference_cm'] != 0), 
           ['hauteur_m']] = replacement
    df.loc[(df['hauteur_m'] != 0) & (df['circonference_cm'] == 0), 
           ['circonference_cm']] = replacement
    
    if show_process:
        plt.figure()
        plt.title('Remove conflictual values')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m', 
                        hue='stade_developpement')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()
        print('\n'*5)
        
    return df




def get_multivariate_outliers(df, percent, nb_neighbors):
    # unsupervised KNN algorithm
    ratio = percent / 100
    
    lof = LocalOutlierFactor(n_neighbors=nb_neighbors, n_jobs=-1)
    lof.fit_predict(df.select_dtypes(['float64', 'int64']).dropna())
    indices = df.select_dtypes(['float64', 'int64']).dropna().index
    df_lof = pd.DataFrame(index=indices,
                          data=lof.negative_outlier_factor_, 
                          columns=['lof'])
    indices = df_lof[df_lof['lof'] < np.quantile(
                     lof.negative_outlier_factor_, ratio)].index
    return list(indices)




def remove_multivariate_outliers(columns_for_analysis, 
                                 column_indices, 
                                 df, 
                                 show_process):
    if show_process:       
        # plot correlation matrix before multivariate analysis
        corr = df[columns_for_analysis].corr()
        plt.figure()
        plt.title('Correlation matrix')
        corr.style.background_gradient(cmap='coolwarm')
        sns.heatmap(corr, annot=True)
        plt.show()

        #plot joint distribution before modification
        plt.figure()
        plt.title('before multivariate analysis')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m', 
                        hue='stade_developpement')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()
        for column_name in columns_for_analysis:
            print('NaN', column_name, '=', df[column_name].isna().sum())
    
    percent = 1
    nb_neighbors = 2
    impossible_indices = get_multivariate_outliers(df[columns_for_analysis], 
                                                   percent, 
                                                   nb_neighbors)
    df.iloc[impossible_indices, column_indices] = [np.nan]*len(column_indices)
    
    if show_process:
        plt.figure()
        plt.title('After multivariate analysis')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m', 
                        hue='stade_developpement')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()
        for column_name in columns_for_analysis:
            print('NaN', column_name, '=', df[column_name].isna().sum())
        print('\n'*5)
        
    return df




def impute_nan(column_name, method, df, show_process):
    median = df[column_name].median()
    
    if method == 'median':
        df.loc[df[column_name].isna(), column_name] = median
        
    elif method == 'group_medians':
        # group target column by values of another column and get median for each group
        column_for_groups = 'stade_developpement'
        
        if show_process:
            print('\n'*3, '-'*20, '\n', column_name, '\n', '-'*20)
        
        groups = {}
        for group in df[column_for_groups].dropna().unique():
            group_median = df.loc[df[column_for_groups] == group, column_name].median()
            groups[group] = group_median
            df.loc[df[column_name].isna() & 
                   (df[column_for_groups] == group), column_name] = group_median
        # for the ones where 'group' is NaN, impute with global median
        df.loc[df[column_name].isna(), column_name] = median
    
    if show_process:
        plt.figure()
        plt.title('Impute all NaN values')
        sns.distplot(df[column_name], kde=True)
        plt.show()
        for group in groups:
            print('median', group, '=', groups[group])
        print('\n')
    
    return df




def clean_quantitative_column(columns, 
                              data, 
                              show_result=True, 
                              show_process=True):
    
    column_indices = [12, 13]
    columns_for_analysis = list(columns.keys())
    
    # deep copy to be able to run the code multiple times 
    # without modifying the original dataframe
    df = data.copy(deep=True)
    
    
    for column_name in columns:
        if show_process:
            print('\n'*3, '-'*20, '\n', column_name, '\n', '-'*20)
            plt.figure()
            plt.title('Original distribution')
            sns.distplot(df[column_name], kde=True)
            
            plt.show()
            
        # remove impossible values
        impossible_limit = columns[column_name]
        df = remove_impossible_values(column_name, impossible_limit, df, show_process) 

    # remove trees with height of 0 but circumference not 0 (and vice-versa)
    df = remove_conflictual_values(df, show_process)

    # remove a percentage of outliers with multivariate analysis
    df = remove_multivariate_outliers(columns_for_analysis, 
                                      column_indices, 
                                      df, 
                                      show_process)

    for column_name in columns:
        # impute all NaN values
        df = impute_nan(column_name, 'group_medians', df, show_process)
            
        if show_result:
            plt.figure()
            plt.title('After cleaning')
            sns.distplot(df[column_name], kde=True)
            plt.show()
            print_statistics(column_name, df)
            print('\n')
    
    if show_result:
        plt.figure()
        plt.title('After cleaning')
        sns.scatterplot(data=df, 
                        x='circonference_cm', 
                        y='hauteur_m', 
                        hue='stade_developpement')
        plt.xlabel('circonference_cm')
        plt.ylabel('hauteur_m')
        plt.show()

    return df





