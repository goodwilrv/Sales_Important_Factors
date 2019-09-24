# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:09:13 2019

@author: GAUTKUM
"""

import os
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from collections import Counter
import matplotlib.pyplot as plt

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats
import statsmodels.tools.eval_measures as eval_mes
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn import tree

np.set_printoptions(linewidth=200)

pd.get_option("display.max_rows")
pd.set_option("display.max_rows",200)
pd.set_option("display.max_colwidth",200)
pd.get_option("display.max_colwidth")


def write_example_sourrounding(main_df):
    with open('Mydata2.txt', 'w') as outfile:  
        json.dump(main_df.surroundings[12], outfile)
        
    
## this method extracts information from surrounding.json and converts to a dataframe
## which can be later used for modelling.
def extract_from_surrounding(main_df):
    fist_level_keys = main_df.surroundings[1].keys()
    all_columns = ['Store_Code']
    all_columns.extend(fist_level_keys)
    
    modelling_df = pd.DataFrame(columns = all_columns)
    print("shape of main_df =========>>>>>>> ",str(main_df.shape))
    for index, row in main_df.iterrows():
        this_store_code = [main_df.store_code[index]]
        sourrounding_list = main_df.surroundings[index]

    
        len_list = list()
        extra_column_names = list()
        extra_column_values = list()
        for str_1 in fist_level_keys:
            value_obj = sourrounding_list.get(str_1)
            no_value_obj = len(sourrounding_list.get(str_1))
            len_list.append(no_value_obj)
    #        print("Type of no_value_obj ==========>>>>>> ")
            
            print("Value of index",index)
            
            
    #        print(type(no_value_obj))
    #        print("value_obj=========>>>>>>>" , value_obj)
            longi_list = list()
            lati_list =list()
            top_address_comp_type = ""
            top_level_type = ""
            for element in value_obj:
    #            print("Value of element ===>>> ",element)
                
                
                longi_list.append(element.get('longitude'))
                lati_list.append(element.get('latitude'))
                
                if((element.get('address_components') is not None) & (len(element.get('address_components'))>0)):
                    first_address_comp = element.get('address_components')[0]
                    a_type = str(first_address_comp.get('types')[0])
                    top_address_comp_type = a_type
                
                if((element.get('types') is not None) & (len(element.get('types'))>0)):
                    top_level_type = str(element.get('types')[0])
                    

            
            longi_ave = np.mean(longi_list) if len(longi_list) > 0 else 0.0
            lati_ave = np.mean(lati_list) if len(lati_list) > 0 else 0.0
            

            new_columns_names = [str_1+'_longi_ave' ,str_1+'_lati_ave',str_1 + '_top_level_type', str_1 + '_top_address_comp_type']
            extra_column_names.extend(new_columns_names)
            new_column_values = [longi_ave,lati_ave,top_level_type,top_address_comp_type]
            extra_column_values.extend(new_column_values)

        this_store_code.extend(len_list)

        
        this_store_code.extend(extra_column_values)
        new_all_columns = all_columns + extra_column_names
        
        x_df = pd.DataFrame([this_store_code])
        
        
        x_df.columns = new_all_columns
        
        modelling_df = modelling_df.append(x_df)
    return modelling_df,all_columns
    
## this method just fills na with zeros and takes a transpose of the sales data frame
def extract_from_sales_data(main_df):
    print("shape of main_df ====>>>>>>>>> ",main_df.shape)
    main_df.head()
    
    main_df = main_df.fillna(0.0)
    
    main_df_t = main_df.transpose().reset_index().rename(index=str, columns={"index": "Sales_Date"})
    
    
    return main_df_t
    
## this method divides the input dataframe into train and test data.
def create_train_test_df(x_df,text_percentage):
    x_df = x_df.loc[:, ~x_df.columns.str.contains('^Unnamed')]
    train_X = x_df.drop(['Store_Code','Last_Three_Quarter_Ave'],axis=1)
    train_Y = x_df[['Last_Three_Quarter_Ave']]   
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=text_percentage, random_state=42)

    return X_train,X_test,y_train,y_test

## this method divides the input dataframe into train and test data, it included log normal transformation of response variable
## But it did not yield good results so remained unsed.
def create_train_test_df_2(x_df,text_percentage):
    x_df = x_df.drop(['Last_Three_Quarter_Ave','norm_sales'],axis=1)
    x_df = x_df.loc[:, ~x_df.columns.str.contains('^Unnamed')]
    train_X = x_df.drop(['Store_Code','lognorm_sales'],axis=1)
    train_Y = x_df[['lognorm_sales']]   
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=text_percentage, random_state=42)

    return X_train,X_test,y_train,y_test
    
## this method calculates variable importance based on random forest and plots the variable importance
def calculatePlot_Variable_Importance(tr_x,tr_y):
    regr = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=100)
    regr.fit(tr_x,tr_y.lognorm_sales.tolist())

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(tr_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(tr_x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    plt.xticks(range(tr_x.shape[1]), indices)
    plt.xlim([-1, tr_x.shape[1]])
    plt.show()
    
## NOT WORKING as of now, due to an issue in the library
def forward_step_feature_selection(x_train_1,y_train_1):
    # Build RF classifier to use in feature selection
    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    # Build step forward feature selection
    sfs1 = sfs(clf,k_features=10,forward=True,floating=False,verbose=2,scoring='accuracy',cv=5)
    # Perform SFFS
    sfs1 = sfs1.fit(x_train_1, y_train_1)


## this method does backward elimination feature selection and clcultes the
## RMSE starting from 70 variables till 35 variables.
def eliminationRMSEValue(x_train, y_train, x_test, y_test, columns):
    my_columns = columns.tolist()
    
    final_col_rmse_df = pd.DataFrame(columns = {'No_columns','RMSE','Adj-RSquare'})
    for item in my_columns:
        regressor_OLS = sm.OLS(y_train,x_train[my_columns] ).fit()
        y_pred = regressor_OLS.predict(x_test[my_columns])
        
        x1 = np.asanyarray(y_pred)
        x2 = np.asanyarray(y_test)
        this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
        
        this_adjrsq = regressor_OLS.rsquared_adj

        this_rmse_df = pd.DataFrame(data={'No_columns':[len(my_columns)],'RMSE':[this_rmse],'Adj-RSquare':[this_adjrsq]})
   
        final_col_rmse_df = final_col_rmse_df.append(this_rmse_df)

        my_columns.remove(item)

                    
    
    return final_col_rmse_df

## this method does step forward feature selection and calcultes the
## RMSE and adjusted R-square starting from 1 variables till 70 variables.
def stepForwardRMSEValue(x_train, y_train, x_test, y_test, columns):
    current_columns = [(columns.tolist()[0])]
    
    final_col_rmse_df = pd.DataFrame(columns = {'No_columns','RMSE','Adj-RSquare'})
    for item in columns.tolist():
        regressor_OLS = sm.OLS(y_train,x_train[current_columns] ).fit()

        y_pred = regressor_OLS.predict(x_test[current_columns])
        
        x1 = np.asanyarray(y_pred)
        x2 = np.asanyarray(y_test)
        this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
        
        this_adjrsq = regressor_OLS.rsquared_adj

        this_rmse_df = pd.DataFrame(data={'No_columns':[len(current_columns)],'RMSE':[this_rmse],'Adj-RSquare':[this_adjrsq]})
   
        final_col_rmse_df = final_col_rmse_df.append(this_rmse_df)
        current_columns.extend([item])
        print("Type of columns ======>>>>>> ",len(current_columns))
                    
    
    return final_col_rmse_df

## this methods fits decision tree model with max. dept from 2,5,6,7,9
## and calculates corresponding RMSEs.
def apply_decision_tree_regression(x_train_1,y_train_1,x_test_1,y_test_1):
    
    # Fit Decision Tree model
    depth_list = [2,5,6,7,9]
    final_col_rmse_df = pd.DataFrame(columns = {'Tree_Max_Depth','RMSE'})
    
    for x in depth_list:
        regr_1 = DecisionTreeRegressor(max_depth=x)
        regr_1.fit(x_train_1, y_train_1)
        y_1_pred = regr_1.predict(x_test_1)
        
        x1 = np.asanyarray(y_1_pred)
        x2 = np.asanyarray(y_test)
        this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
        
        this_rmse_df = pd.DataFrame(data={'Tree_Max_Depth':[x],'RMSE':[this_rmse]})
   
        final_col_rmse_df = final_col_rmse_df.append(this_rmse_df)
    
    return final_col_rmse_df
        
        
    
    
    
    

 
def apply_Liner_Regression(x_train_1,y_train_1,x_test_1,y_test_1):

    X2 = sm.add_constant(x_train_1)
    est = sm.OLS(y_train_1, X2)
    est2 = est.fit()
    print(est2.summary())
    
    print("Type of p values",type(est2.pvalues))
    print("p values ",est2.pvalues)
    
    
    

if __name__ == '__main__':
    
    third_lastQ_weight = 0.2
    second_lastQ_weight = 0.3
    lastQ_weight = 0.5
      
    
    ## Change to work Directory
    work_project_path = "C:\\Users\\Dell 3450\\Desktop\\LnT_Documents\\Phillip_Morris_Assignment\\UseCase_3_Datasets"
    os.chdir(work_project_path)
    
    surrounding_df = pd.read_json("Surroundings.json", orient='records')
    surrounding_df = surrounding_df.loc[:, ~surrounding_df.columns.str.contains('^Unnamed')]
    
    print(surrounding_df.head())
    
    surrounding_changed_df,concise_columns = extract_from_surrounding(surrounding_df)
    surrounding_changed_df.to_csv("surrounding_changed_long_df.csv")
    
    surrounding_concise_df = surrounding_changed_df[concise_columns]
    surrounding_concise_df.to_csv("surrounding_concise_df.csv")
    
    surrounding_concise_df = pd.read_csv("surrounding_concise_df.csv")
    
    granular_mod_df = pd.read_csv("granular_mod_df.csv")
    
    ## Manually deleting the fisrt row and index column
    
    granular_mod_df = pd.read_csv("granular_mod_df.csv")
    granular_mod_df['Sales_Date_Time'] =  pd.to_datetime(granular_mod_df['Sales_Date_Time'], format='%m/%d/%Y %H:%M')
    granular_mod_df = granular_mod_df.assign(Sales_Date_Quarter = pd.PeriodIndex(granular_mod_df.Sales_Date_Time, freq='Q'))
    
    granular_mod_df = granular_mod_df.drop(['Sales_Date_Time'],axis=1)
    granular_mod_df.Sales_Date_Quarter = granular_mod_df.Sales_Date_Quarter.apply(str)
    
    granular_agg_mod_df = granular_mod_df.groupby(['Sales_Date_Quarter']).sum().reset_index()
    granular_agg_mod_T_df = granular_agg_mod_df.set_index('Sales_Date_Quarter').transpose().reset_index().rename(columns={'index':'Store_Code'}) 
    
    granular_agg_mod_T_df = granular_agg_mod_T_df.assign(Last_Three_Quarter_Ave = (third_lastQ_weight * granular_agg_mod_T_df['2016Q4'] + second_lastQ_weight * granular_agg_mod_T_df['2017Q1'] + lastQ_weight * granular_agg_mod_T_df['2017Q2'])/2)
    granular_agg_mod_T_df.to_csv("granular_agg_mod_T_df.csv")
    ##.rename(index=str, columns={"index": "Sales_Date"})
    granular_agg_mod_T_df.Store_Code = granular_agg_mod_T_df.Store_Code.apply(str)
    surrounding_concise_df.Store_Code = surrounding_concise_df.Store_Code.apply(str)
    
    missing_store_list = np.setdiff1d(surrounding_concise_df.Store_Code.tolist(),granular_agg_mod_T_df.Store_Code.tolist())
    missing_store_list_1 = np.setdiff1d(granular_agg_mod_T_df.Store_Code.tolist(),surrounding_concise_df.Store_Code.tolist())

    
    merged_train_df = surrounding_concise_df.merge(granular_agg_mod_T_df[['Store_Code','Last_Three_Quarter_Ave']],on='Store_Code',how='inner') 
    merged_train_df.to_csv("ThreeQAv_merged_train_df.csv")
    
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'. Tried two lines, 
    ## to remove the outliers and check the Rsquare and RMSE.
#    merged_train_df_1 = merged_train_df[np.abs(merged_train_df.Last_Three_Quarter_Ave-merged_train_df.Last_Three_Quarter_Ave.mean()) <= (3*merged_train_df.Last_Three_Quarter_Ave.std())]
#    merged_train_df_2 = merged_train_df[~(np.abs(merged_train_df.Last_Three_Quarter_Ave-merged_train_df.Last_Three_Quarter_Ave.mean()) > (3*merged_train_df.Last_Three_Quarter_Ave.std()))]
##    
    
    
    ############## Below 3 lines were used to try lon normal transformation of the response variable##
#    merged_train_df_2['norm_sales'] = (1+merged_train_df_2.Last_Three_Quarter_Ave)/2 # (-1,1] -> (0,1]
#    merged_train_df_2['lognorm_sales'] = np.log(merged_train_df_2['norm_sales'])
#    x_train, x_test, y_train, y_test =  create_train_test_df_2(merged_train_df_2,0.20)
    
    x_train, x_test, y_train, y_test =  create_train_test_df(merged_train_df,0.20)
#    
    ## Create a Heat Map of the correlation matrix.
    plt.figure(figsize = (100,50))
    x_corr=x_train.corr()
    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(x_corr)
    
    ## select only those columns which dont not have correlation higher than 0.8 
    columns = np.full((x_corr.shape[0],), True, dtype=bool)
    for i in range(x_corr.shape[0]):
        for j in range(i+1, x_corr.shape[0]):
            if x_corr.iloc[i,j] >= 0.8:
                if columns[j]:
                    columns[j] = False
    selected_columns = x_train.columns[columns]
    
    removed_columns = np.setdiff1d(x_train.columns,selected_columns)
    
    ###############
    ################# Finding Significant columns based on p-value in linear regression
    X2 = sm.add_constant(x_train[selected_columns])
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())
    
    print("Type of p values",type(est2.pvalues))
    print("p values ",est2.pvalues)
    Variable_p_value_df = est2.pvalues.to_frame().reset_index().rename(columns={'index':'Variable',0:'P-Value'})
    Variable_p_value_df.to_csv("ThreeQAve_Variable_p_value_df.csv")
    
    important_Variables_LR = Variable_p_value_df.loc[Variable_p_value_df['P-Value'] < 0.05,:].Variable.tolist()
    
    
    ########## Start Back step feature selection #######################
    rmse_df = eliminationRMSEValue(x_train[selected_columns],y_train,x_test[selected_columns],y_test,selected_columns)
    rmse_df.to_csv("ThreeQAve_rmse_df.csv")
    
    ax_1 =rmse_df.plot(x='No_columns', y='RMSE')
    ax_1.set_xlabel("ThreeQAve Number of columns")
    ax_1.set_ylabel("ThreeQAve RMSE")
    
    ax_2 =rmse_df.plot(x='No_columns', y='Adj-RSquare')
    ax_2.set_xlabel("ThreeQAve Number of columns")
    ax_2.set_ylabel("ThreeQAve AdjustedRSquare")
    
    #################End Back step feature selection ###################
    
    ## start Step forward RMSE
    stp_rmse_df = stepForwardRMSEValue(x_train[selected_columns],y_train,x_test[selected_columns],y_test,selected_columns)
    stp_rmse_df.to_csv("ThreeQAve_stp_rmse_df.csv")
    
    ax_3 =stp_rmse_df.plot(x='No_columns', y='RMSE')
    ax_3.set_xlabel("ThreeQAve Number of columns")
    ax_3.set_ylabel("ThreeQAve RMSE")
    
    ax_4 =stp_rmse_df.plot(x='No_columns', y='Adj-RSquare')
    ax_4.set_xlabel("ThreeQAve Number of columns")
    ax_4.set_ylabel("ThreeQAve AdjustedRSquare")
    
    ## END Step forward RMSE
    
    ## Step forward RMSE after root Square transformation of explanatory variables
    x_train_rt = np.sqrt(x_train)
    x_test_rt = np.sqrt(x_test)
    rmse_rt_df = stepForwardRMSEValue(x_train_rt[selected_columns],y_train,x_test_rt[selected_columns],y_test,selected_columns)
    rmse_rt_df.to_csv("ThreeQAvestp_rmse_rt_df.csv")
    
    ax_5 =rmse_rt_df.plot(x='No_columns', y='RMSE')
    ax_5.set_xlabel("ThreeQAve Number of columns_RS")
    ax_5.set_ylabel("ThreeQAve RMSE_RS")
    
    ax_6 =rmse_rt_df.plot(x='No_columns', y='Adj-RSquare')
    ax_6.set_xlabel("ThreeQAve Number of columns_RS")
    ax_6.set_ylabel("ThreeQAve AdjustedRSquare_RS")
    
    ## Step forward RMSE after normalization transformation of explanatory variables
    x_train_nz = ((x_train-x_train.min())/(x_train.max()-x_train.min() + 1))
    x_test_nz = ((x_test-x_train.min())/(x_test.max()-x_test.min() + 1))
    
    rmse_nz_df = stepForwardRMSEValue(x_train_nz[selected_columns],y_train,x_test_nz[selected_columns],y_test,selected_columns)
    rmse_nz_df.to_csv("ThreeQAve_stp_rmse_NZ_df.csv")
    
    ax_8 =rmse_nz_df.plot(x='No_columns', y='RMSE')
    ax_8.set_xlabel("ThreeQAve Number of columns_NZ")
    ax_8.set_ylabel("ThreeQAve RMSE_NZ")
    
    ax_9 =rmse_nz_df.plot(x='No_columns', y='Adj-RSquare')
    ax_9.set_xlabel("ThreeQAve Number of columns_NZ")
    ax_9.set_ylabel("ThreeQAve AdjustedRSquare_NZ")
    
    ## Trying transformations of the response variable
    
    
    
#    rmse_nz_df = stepForwardRMSEValue(x_train[selected_columns],y_train['lognorm_sales'],x_test[selected_columns],y_test['lognorm_sales'],selected_columns)
#    rmse_nz_df.to_csv("ThreeQAve_stp_rmse_NZ_df.csv")
#    
#    ax_8 =rmse_nz_df.plot(x='No_columns', y='RMSE')
#    ax_8.set_xlabel("ThreeQAve Number of columns_Y_logNorm")
#    ax_8.set_ylabel("ThreeQAve RMSE_Y_logNorm")
#    
#    ax_9 =rmse_nz_df.plot(x='No_columns', y='Adj-RSquare')
#    ax_9.set_xlabel("ThreeQAve Number of columns_Y_logNorm")
#    ax_9.set_ylabel("ThreeQAve AdjustedRSquare_Y_logNorm")

    
    
     ## Calculate variable importance based on Variable importance statistics of random forest.
    calculatePlot_Variable_Importance(x_train[selected_columns],y_train)
    
    
#   important_feature_indices_RF = [0,7,42,3,8,46,16,52,33,54,65,58,48,60,39,40,34,1,12,60,39,45,53,38,35]
#    important_feature_indices_RF = [0,7,42,3,8,33,52,48,16,54,58,65,46,34,60,1,39,40,69,38,12,53,45,35,37]
    important_feature_indices_RF = [47,71,40,7,34,26,3,54,69,38,16,28,60,12,2,1,62,68,41,22,21,35,15,31,57]
    
    
    important_columns_RF = x_train.columns[important_feature_indices_RF]
   
   
   ##########################################################################################
   ################################## START Final Modelling #################################
   ##########################################################################################
    common_columns = list(set(important_columns_RF).intersection(set(important_Variables_LR)))
    
    extra_columns = ['liquor_store', 'accounting', 'museum', 'pharmacy', 'restaurant',
       'post_office', 'beauty_salon', 'meal_delivery']
    final_columns = common_columns + extra_columns
   
    tree_rmse_df = apply_decision_tree_regression(x_train[final_columns],y_train,x_test[final_columns],y_test)
    ax_10 =tree_rmse_df.plot(x='Tree_Max_Depth', y='RMSE')
    ax_10.set_xlabel("ThreeQAve Tree_Max_Depth")
    ax_10.set_ylabel("ThreeQAve RMSE")
   
  
   
   ########### Linear Regression #####################################
   
    regressor_OLS = sm.OLS(y_train,x_train[final_columns] ).fit()

    y_pred = regressor_OLS.predict(x_test[final_columns])
    regressor_OLS.summary()
   
        
    x1 = np.asanyarray(y_pred)
    x2 = np.asanyarray(y_test)
    this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
        
    this_adjrsq = regressor_OLS.rsquared_adj
    this_rsq = regressor_OLS.rsquared
   
   ######## Deicision tree modelling ################################
    tree_mod = DecisionTreeRegressor(max_depth=5)
    tree_mod.fit(x_train[final_columns], y_train)
   
    y_1_pred = tree_mod.predict(x_test[final_columns])
        
    x1 = np.asanyarray(y_1_pred)
    x2 = np.asanyarray(y_test)
    this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
    
    from sklearn import tree
    tree.export_graphviz(tree_mod, out_file='tree.dot') #produces dot file
    
    import pydot
    dotfile = StringIO('tree.dot')
    tree.export_graphviz(tree_mod, out_file=dotfile)
    pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")
   
   ############## Random Forest modelling ########################### 
   
    fr_regr = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=100)
    fr_regr.fit(x_train[final_columns],y_train)
   
    y_1_pred = fr_regr.predict(x_test[final_columns])
        
    x1 = np.asanyarray(y_1_pred)
    x2 = np.asanyarray(y_test)
    this_rmse_rf = np.sqrt(np.mean(np.square( x1 - x2)))
   
    
    ##########################################################################################
   ################################## END Final Modelling #################################
   ##########################################################################################
    
    
   
    
    
    
    
    
    
    


    

