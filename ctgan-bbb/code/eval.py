import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm

def cal_mean_cio(name,orig,syn):
    if name == 'UK':
        target_cols = ['TENURE','MSTATUS']
        key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
        orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
        syn['MSTATUS'] = (syn['MSTATUS'] == 'Married' ) | (syn['MSTATUS'] == 'Remarried' )
        syn['TENURE'] = (syn['TENURE'] == 'Own occ-buying' ) | (syn['TENURE'] == 'Own occ-outright' )
    elif name=='Canada':
        target_cols = ['TENURE','MARST']
        key_cols = ['ABIDENT','AGE','CLASSWK','DEGREE','EMPSTAT','SEX','URBAN','TENURE','MARST']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a4' )| (orig['MARST'] == '2' ) | (orig['MARST'] == '4' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' ) | (orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a4' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '4' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Fiji':
        target_cols = ['TENURE','MARST']
        key_cols = ['AGE','CLASSWKR','ETHNIC','RELIGION','EDATTAIN','SEX','PROV','TENURE','MARST']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' )|(orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' )|(orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Rwanda':
        target_cols = ['OWNERSH','MARST']
        key_cols = ['AGE','DISAB1','EDCERT','CLASSWK','LIT','RELIG','SEX','OWNERSH','MARST']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['OWNERSH'] =((orig['OWNERSH'] == 'a1' ) | (orig['OWNERSH'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['OWNERSH'] = ((syn['OWNERSH'] == 'a1' ) | (syn['OWNERSH'] == '1' )).astype('int')
    elif name=='Indonesia':
        target_cols = ['OWNERSHIP','MARST']
        key_cols = ['LANDOWN', 'AGE', 'RELATE', 'SEX', 'HOMEFEM', 'HOMEMALE', 'RELIGION', 
                    'LIT', 'SCHOOL', 'EDATTAIND', 'DISABLED','OWNERSHIP','MARST']
        orig = orig[key_cols]
        syn = syn[key_cols]
        
        orig['MARST'] = ((orig['MARST'] == '3' ) | (orig['MARST'] == '4' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == 'a4' )).astype('int')
        orig['OWNERSHIP'] =((orig['OWNERSHIP'] == 'a1' ) | (orig['OWNERSHIP'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == '3' ) | (syn['MARST'] == '4' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == 'a4' )).astype('int')
        syn['OWNERSHIP'] = ((syn['OWNERSHIP'] == 'a1' ) | (syn['OWNERSHIP'] == '1' )).astype('int')
    elif name=='Adult':
        target_cols = ['income','marital-status']
        key_cols = ['workclass', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
        ## 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['income'] = ((orig['income'] == '>50K' ) | (orig['income'] == '>50K.' )).astype('int')
        orig['marital-status'] =((orig['marital-status'] == 'Married-civ-spouse' ) | (orig['marital-status'] == 'Married-spouse-absent' ) | (orig['marital-status'] == 'Married-AF-spouse')).astype('int')
        syn['income'] = ((syn['income'] == '>50K' ) | (syn['income'] == '>50K.' )).astype('int')
        syn['marital-status'] =((syn['marital-status'] == 'Married-civ-spouse' ) | (syn['marital-status'] == 'Married-spouse-absent' ) | (syn['marital-status'] == 'Married-AF-spouse')).astype('int')
    elif name=='Churn':
        target_cols = ['Exited']
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
        ## 'CreditScore', 'Age', 'Balance', 'EstimatedSalary'
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['Exited'] = ((orig['Exited'] == '1' )).astype('int')
        syn['Exited'] = ((syn['Exited'] == '1' )).astype('int')
    elif name=='Insurance':
        target_cols = ['charges']
        key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
        ## 'age','bmi'
        median_cutoff = orig['charges'].median()
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['charges'] = ((orig['charges'] > median_cutoff )).astype('int')
        syn['charges'] = ((syn['charges'] > median_cutoff )).astype('int')
    elif name=='Credit':
        target_cols = ['checking_balance', 'default']
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
        ## 'months_loan_duration', 'amount', 'age'
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['checking_balance'] = ((orig['checking_balance'] == '1 - 200 DM') | (orig['checking_balance'] == '> 200 DM')).astype('int')
        orig['credit_history'] =((orig['credit_history'] == 'repaid') | (orig['credit_history'] == 'fully repaid') | (orig['credit_history'] == 'fully repaid this bank')).astype('int')
        orig['savings_balance'] = ((orig['savings_balance'] == '501 - 1000 DM' ) | (orig['savings_balance'] == '> 1000 DM')).astype('int')
        syn['checking_balance'] = ((syn['checking_balance'] == '1 - 200 DM') | (syn['checking_balance'] == '> 200 DM')).astype('int')
        syn['credit_history'] =((syn['credit_history'] == 'repaid' ) | (syn['credit_history'] == 'fully repaid') | (syn['credit_history'] == 'fully repaid this bank')).astype('int')
        syn['savings_balance'] =((syn['savings_balance'] == '1 - 200 DM') | (syn['savings_balance'] == '> 1000 DM') ).astype('int')



    orig.fillna('a0',inplace=True)
    syn.fillna('a0',inplace=True)
    encoder = OrdinalEncoder()
    encoder.fit(pd.concat([orig.astype(str),syn.astype(str)],axis=0))
    orig = pd.DataFrame(encoder.transform(orig.astype(str)),columns=key_cols)
    syn = pd.DataFrame(encoder.transform(syn.astype(str)),columns=key_cols)
    scores = []
    for target in target_cols:
        orig_glm = sm.GLM(orig[target].astype(float),orig.drop(columns=target).astype(float),family=sm.families.Binomial() )
        syn_glm = sm.GLM(syn[target].astype(float),syn.drop(columns=target).astype(float),family=sm.families.Binomial() )
        results = CIO_function(orig_glm, syn_glm)
        scores.append(results['mean_ci_overlap_noNeg'])
    return np.mean(scores)

def CIO_function(orig_glm,syn_glm):
    # # put them into a form so it is easier to extract the coefficients etc.
    try:
        syn_glm = syn_glm.fit()
        orig_glm = orig_glm.fit()
    except:
        return {'mean_std_coef_diff':0, 
                'median_std_coef_diff' : 0,
                'mean_ci_overlap':0, 
                'median_ci_overlap' : 0,
                'mean_ci_overlap_noNeg' :0, 
                'median_ci_overlap_noNeg':0}  # when there is a perfect separation in syn dataset

    syn_glm = pd.DataFrame(syn_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    orig_glm = pd.DataFrame(orig_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    syn_glm = syn_glm.iloc[:,:3] # take the first three columns
    orig_glm = orig_glm.iloc[:,:3]
    
    # join the original and synth
    combined = orig_glm.merge(syn_glm,how='left',on='names',suffixes=('_orig', '_syn'))
    for i in combined.columns[1:]:
        combined[i] = combined[i].astype('float')
    combined['std.coef_diff'] = abs(combined['Estimate_orig']-combined['Estimate_syn']) / (combined['stderr_orig'])
    combined['orig_lower'] = combined['Estimate_orig'] - 1.96 * combined['stderr_orig']
    combined['orig_upper'] = combined['Estimate_orig'] + 1.96 * combined['stderr_orig']
    combined['syn_lower'] = combined['Estimate_syn'] - 1.96 * combined['stderr_syn']
    combined['syn_upper'] = combined['Estimate_syn'] + 1.96 * combined['stderr_syn']
    combined['ci_overlap'] = 0.5 * (
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['orig_upper']-combined['orig_lower']) + 
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['syn_upper']-combined['syn_lower'])
                                    )
    for index,row in combined.iterrows():
        if row['orig_lower'] == row['orig_upper'] and row['orig_upper'] == row['syn_lower'] and row['syn_upper'] == row['syn_lower']:
            combined.loc[index,'ci_overlap'] = 1.0
    combined = combined[['names','std.coef_diff','ci_overlap']]
    
    combined.fillna(0,inplace=True) # set negative overlaps to zero
    combined['ci_overlap_noNeg'] = [0 if i<0 else i for i in combined['ci_overlap']]

    results = {'mean_std_coef_diff':combined['std.coef_diff'].mean(), 
                'median_std_coef_diff' : combined['std.coef_diff'].median(),
                'mean_ci_overlap': combined.ci_overlap.mean(), 
                'median_ci_overlap' : combined.ci_overlap.median(),
                # add in the overlaps where negatives were changed to zeros
                'mean_ci_overlap_noNeg' :combined.ci_overlap_noNeg.mean(), 
                'median_ci_overlap_noNeg':combined.ci_overlap_noNeg.median()}
    # now compute std. diff and ci overlap
    return results


def cal_mean_roc(name,orig,syn,bi=True):
    if name == 'UK':
        key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS',
                    'TENURE','MSTATUS']
    elif name=='Canada':
        key_cols = ['AGE','ABIDENT','SEX','TENURE','URBAN','BPLMOM','BPLPOP',
                    'CITIZEN','LANG','MARST','RELATE','MINORITY','RELIG','BPL']
    elif name=='Fiji':
        key_cols = ['PROV','TENURE','RELATE','SEX','AGE','ETHNIC','MARST',
                    'RELIGION','BPLPROV','RESPROV',
                    'RESSTAT','SCHOOL','TRAVEL']
    elif name=='Rwanda':
        key_cols = ['AGE','STATUS','SEX','URBAN','OWNERSH','DISAB2','DISAB1',
                    'RELATE','RELIG','HINS','NATION','BPL']
    elif name=='Indonesia':
        key_cols = ['OWNERSHIP', 'LANDOWN', 'AGE', 'RELATE', 'SEX', 'MARST', 
                    'HOMEMALE', 'RELIGION', 'SCHOOL', 'LIT', 'EDATTAIND', 'DISABLED']
    elif name=='Adult':
        key_cols = ['workclass', 'education',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
    elif name=='Churn':
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
    elif name=='Credit':
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']

    orig = orig[key_cols]
    syn = syn[key_cols]
    uni_scores = []
    bi_scores = []
    for i in range(len(key_cols)):
        uni_scores.append(roc_univariate(orig, syn, i) )
      
        if bi and i+1<len(key_cols):# max i == len(key_cols)-1
            for j in range(i+1,len(key_cols)):
                bi_scores.append(roc_bivariate(orig, syn, i, j))
    if bi:
        return np.mean(uni_scores),np.mean(bi_scores)
    else:
        return np.mean(uni_scores),0

def roc_univariate(original,synthetic,var_num):
    # create frequency tables for the original and synthetic data, on the variable
    orig_table = original.iloc[:,var_num].value_counts().reset_index()
    syn_table = synthetic.iloc[:,var_num].value_counts().reset_index()
    orig_table.columns = ['value','Freq']
    syn_table.columns = ['value','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variable
    combined = orig_table.merge(syn_table,on=['value'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


def roc_bivariate(original, synthetic, var1, var2):
    # create frequency tables for the original and synthetic data, on the two variable cross-tabulation
    orig_table = pd.crosstab(index=original.iloc[:,var1],columns=original.iloc[:,var2]).stack().reset_index()
    syn_table = pd.crosstab(index=synthetic.iloc[:,var1],columns=synthetic.iloc[:,var2]).stack().reset_index()
    orig_table.columns = ['Var1','Var2','Freq']
    syn_table.columns = ['Var1','Var2','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variables
    combined = orig_table.merge(syn_table,on=['Var1', 'Var2'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


'''
function:     replace_missing()   
description:  replaces missing values dependant on data type. Categorical or object NAs are replaced with 'blank', numerical NAs with -999. Can be modified as required
input:        pandas dataframe
output:       pandas dataframe with missing values replaced
'''
def replace_missing(dataset):
    # get a dictionary of the different data types
    types = dataset.dtypes.to_dict()
    # replace object or categorical NAs with 'blank', and numerical with -999
    for col_nam, typ in types.items():
        if (typ == 'O' or typ == 'c'):
            dataset[col_nam] = dataset[col_nam].fillna('blank')
        if (typ == 'float64' or typ == 'int64'):
            dataset[col_nam] = dataset[col_nam].fillna(-999)
    return(dataset)

def cal_mean_tcap(name,orig,syn):
    if name=='UK':
        target_cols = ['LTILL','FAMTYPE','TENURE']
        key_cols = ['AREAP','AGE','SEX','MSTATUS','ETHGROUP','ECONPRIM']
    elif name=='Canada':
        target_cols = ['RELIG','CITIZEN','TENURE']
        key_cols = ['AGE','SEX','MARST','MINORITY','EMPSTAT','BPL']
    elif name=='Fiji':
        target_cols = ['RELIGION','WORKTYPE','TENURE']
        key_cols = ['PROV','AGE','SEX','MARST','ETHNIC','CLASSWKR']
    elif name=='Rwanda':
        target_cols = ['RELIG','WKSECTOR','OWNERSH']
        key_cols = ['AGE','SEX','MARST','CLASSWK','URBAN','BPL']
    elif name=='Indonesia':
        target_cols = ['RELIGION','OWNERSHIP','EDATTAIND']
        key_cols = ['AGE', 'SEX', 'MARST', 'HOMEFEM','SCHOOL', 'LANDOWN']
    elif name=='Adult':
        target_cols = ['native-country', 'race', 'occupation']
        key_cols = ['education-num', 'marital-status', 'workclass', 
                    'relationship', 'sex', 'income']
    elif name=='Churn':
        target_cols = ['Geography']
        key_cols = ['Gender', 'Tenure', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        target_cols = ['children']
        key_cols = ['sex','region', 'smoker']
    elif name=='Credit':
        target_cols = ['checking_balance', 'credit_history', 'savings_balance']
        key_cols = ['purpose', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']

    scores = []
    for target in target_cols:
        for i in range(3,len(key_cols)+1):
            score,baseline = tcap(orig,syn,target,key_cols[:i],verbose=False)
            # print(score,baseline)
            score_scaled = (score - baseline)/(1-baseline)
            scores.append(score_scaled)
    return np.mean(scores)


'''
function:     tcap()   
description:  takes the original and synthetic dataset filenames and a set of keys/target variables and calculates the TCAP score
input:        original = location/filename of original dataset
              synth = location/filename of synthetic dataset
              num_keys = number of key variables
              target = target variable
              key = key variable as the baseline 
              verbose = if set to True it will print out more detailed results
output:       TCAP score and the baseline value for that target variable
'''
def tcap(orig, syn, target, key, verbose=False):
       
    # read in the data
    #orig = pd.read_csv(original)
    #syn = pd.read_csv(synth)
    
    # define the keys and target. using the num_keys parameter means that a dataset with any number of columns can
    # be used, and only the relevant keys analysed
    keys_target = key + [target]
    num_keys = len(key)
    # print(keys_target)
    
    # select just the required columns (keys and target)    
    orig = orig[keys_target]
    syn = syn[keys_target]
    # replace any missing values
    orig = replace_missing(orig)
    syn = replace_missing(syn)
    # count the categories for the target (for calculating baseline)
    uvd = orig[target].value_counts()
    
    # use groupby to get the equivalance classes for synthetic data
    eqkt_syn = pd.DataFrame({'count' : syn.groupby( keys_target ).size()}).reset_index()           # with target
    eqk_syn = pd.DataFrame({'count' : syn.groupby( keys_target[:num_keys] ).size()}).reset_index() # without target
    # equivalance classes for original data without target
    eqk_orig = pd.DataFrame({'count' : orig.groupby( keys_target[:num_keys] ).size()}).reset_index()

    # merge with original to calculate baseline    
    orig_merge_eqk = pd.merge(orig, eqk_orig, on= keys_target[:num_keys]) 
    orig_merge_eqk.rename({'count': 'count_eqk_orig'}, axis=1, inplace=True)
    # calculate the baseline
    uvt = sum(uvd[orig_merge_eqk[target]]/sum(uvd))
    baseline = uvt/len(orig)
    
    # calculate synthetic cap score. merge syn eq classes (with keys) with syn eq classes (with keys/target)
    syn_merge = eqk_syn.merge(eqkt_syn, on=keys_target[:num_keys])
    syn_merge['prop'] = syn_merge['count_y']/syn_merge['count_x']
    # filter out those less than tau=1
    syn_merge = syn_merge[syn_merge['prop'] >= 1]
    # merge with original, if in syn eq classes (just keys) then this is a matching record (Taub)
    syn_merge = syn_merge.merge(orig_merge_eqk, on=keys_target[:num_keys], how='inner')
    matching_records = len(syn_merge)

    # drop records where the targets are not equal
    syn_merge = syn_merge[syn_merge[target + '_x']==syn_merge[target + '_y']]
    dcaptotal = len(syn_merge)

    if matching_records == 0:
        tcap_undef = 0
    else:
        tcap_undef = dcaptotal/matching_records
   
    # output is [the TCAP as used by Taub, and the baseline]. Modify as required
    output = ([tcap_undef,baseline])
    
    if verbose==True:
        print('TCAP calculation')
        print('===============')
        print('Source dataset is: ',orig)
        print('Target dataset is: ',syn)
        print('The total number of records in the source dataset is: ', len(orig))
        print('The total number of records in the target dataset is: ', len(syn))
        print('The target variable is: ', target)
        print('The key size is: ', num_keys)
        print('The keys are: ', key)
        print('Number of matching records: ', matching_records)
        print('DCAP total is: ', dcaptotal)
        print('TCAP with non-matches undefined is: ', tcap_undef)
        print('The baseline is: ', baseline)

    return(output)