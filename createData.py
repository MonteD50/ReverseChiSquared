import pandas as pd  
import numpy as np
import random, time
from scipy.stats import chi2, chisquare

class InverseChi:
    def __init__(self, df, num_new_additions, num_samples, alpha=0.05):
        """Create new data. (HELPER CLASS)

        Uses the Chi Squared Goodness of Fit Test to estimate 
        a non- significant distribution and create new observed values.

        H_0: the distribution of observed values follows the "real" distribution
        H_a: the distribution of observed values doesn't follow the "real" distribution

        Args: 
            df: pandas data frame containing the categorical data
            num_new_additions: the number of new cases
            num_samples: the number of cases in the data
            alpha: significance level

        """
        self.df = df 
        self.num_new_additions = num_new_additions
        self.num_samples = num_samples
        self.alpha = alpha

    def create(self):
        """ Creates new dataframe with new data

        Returns: 
            pandas dataframe with new data

        """
        new_df = {}
        for i in range(len(self.df.columns)):
            new = self.create_(self.df[self.df.columns[i]].tolist())
            new_df[self.df.columns[i]] = new

        return pd.DataFrame(new_df)

    def create_(self, cat_column):
        """ Creates new data given only one column. (HELPER FUNCTION)

        Args: 
            cat_column: categorical distribution

        Returns: 
            new data

        """

        """Notes:

        Restrictions in the equation:
            O_1 + O_2 + O_3 + ... + O_n = num samples

            pvalue > alpha

        """
        new_data = []
        while len(new_data) != self.num_new_additions:
            rand = np.random.multinomial(self.num_samples, np.array(cat_column) / self.num_samples, size=1)[0]
            pvalue = self.chiTest(cat_column, rand).pvalue
            if pvalue > self.alpha:
                new_data.append(rand.tolist())

        return new_data

    def chiTest(self, expected, observed):
        """ Performs the chi test.

        Note: only likes the count, not proportions 

        Args: 
            expected: the already established proportions
            observed: the created data meant for testing

        Returns: 
            chi test result

        """
        return chisquare(f_obs=observed, f_exp=expected)
        
    def pvalue(self, x_squared, dof):
        """ Computes the pvalue from the x_squared value.

        Args: 
            x_squared: the result of the chi test
            dof: degrees of freedom

        Returns:
            pvalue of the chi test: float

        """
        return chi2.sf(x_squared, dof)

class ToCategorical:
    def __init__(self, df, columns, num_column_vars):
        """ Converts dataframe to categorical (HELPER CLASS)

        Warning: 
            Currently only percents categories are supported. Other categories are untested
            TODO: test or addition for other categories not percents

        Args:
            df: the pandas dataframe containing the data   
            columns: a list of column names to be converted
            num_column_vars: number of variables to make 

        Returns: 
            new pandas dataframe with specified categorical columns 
        """
        self.df = df
        self.columns = columns
        self.num_column_vars = num_column_vars

    def transform(self):
        """ Main function to transform data to categorical

        Returns: 
            1. new pandas dataframe with specified categorical columns
            2. the interval array for categorical variable
            3. the interval to distribute the categorical variables by

        """
        new_df = {}
        for i in range(len(self.columns)):
            new_col = np.zeros(self.num_column_vars)
            spec = self.df[self.columns[i]].tolist()

            if "Percentage" in self.columns[i]:
                dividing_threshold = 100.0 / self.num_column_vars
            else:
                dividing_threshold = max(spec) / self.num_column_vars

            dividing_array = np.zeros((self.num_column_vars + 1))
            for j in range(1, self.num_column_vars + 1):
                dividing_array[j] = dividing_array[j - 1] + dividing_threshold
            
            for k in spec:
                for l in range(1, self.num_column_vars + 1):
                    if k <= dividing_array[l]:
                        new_col[l - 1] += 1
                        break
            
            for g in range(len(new_col)):
                if new_col[g] == 0:
                    new_col[g] = 0.01

            new_df[self.columns[i]] = new_col
        return pd.DataFrame(new_df), dividing_array, dividing_threshold

class ToDiscrete:
    def __init__(self, df, num_column_vars, dividing_array, dividing_threshold, og_cases):
        """ Converts dataframe to discrete (HELPER CLASS)

        Args: 
            df: pandas dataframe containing the data  
            num_column_vars: number of variables to make (should be the same as in ToCategorical function)
            dividing_array: the intervals array for categorical variable
            dividing_threshold: the interval to distribute the categorical variables by
            og_cases: the number of original cases

        Returns:
            pandas dataframe with created values converted to discrete (ready for deep learning)

        """
        self.df = df
        self.num_column_vars = num_column_vars
        self.dividing_array = dividing_array
        self.dividing_threshold = dividing_threshold
        self.og_cases = og_cases

    def convert(self):
        """ Main function to convert to discrete

        Returns: 
            pandas dataframe with created values converted to discrete (ready for deep learning)
        """
        all_new = {}
        for i in range(len(self.df.columns)):
            rows = self.df[self.df.columns[i]]
            all_new[self.df.columns[i]] = []
            for j in range(len(rows)):
                new = np.random.choice(self.dividing_array[1:], size=self.og_cases, p=np.array(rows[j]) / self.og_cases)
                for l in range(len(new)):
                    randInterval = np.random.randint(low=new[l] - self.dividing_threshold, high=new[l] + 1, size=1)[0]
                    new[l] = randInterval
                all_new[self.df.columns[i]].append(new.tolist())
            all_new[self.df.columns[i]] = np.array(all_new[self.df.columns[i]]).flatten().tolist()
        return pd.DataFrame(all_new)

class Main:
    def __init__(self, df, columns, num_new_samples, num_categories=5):
        """ Main class to creates all the data in one continuous motion using the Helper classes.

        Args:
            df: the original pandas dataframe
            columns: list of desired column names
            num_new_samples: the number of new cases too create
            num_categories: the number of categorical variables to have

        Returns:
            Combined pandas dataframe with new and old data of the desired columns

        """
        self.df = df
        self.columns = columns
        self.num_new_samples = num_new_samples
        self.num_categories = num_categories

    def run(self):
        """ The main run function

        """

        categorical_class = ToCategorical(self.df, self.columns, self.num_categories)
        one_cat_df, dividing_array, dividing_threshold = categorical_class.transform()

        inverse = InverseChi(one_cat_df, self.num_new_samples, len(self.df))
        new_cats = inverse.create()

        final = ToDiscrete(new_cats, self.num_categories, dividing_array, dividing_threshold, len(self.df))
        created_data = final.convert()

        wanted_df = df[self.columns]

        result = wanted_df.append(created_data)
        
        return result

filename = # csv file
df = pd.read_csv(filename)

"""
Use only a subset of df to create data, the rest will be used for prediction
"""
subset_df = df.sample(int(len(df)/2))

ins = []
for i in range(len(subset_df)):
    ins.append(subset_df.iloc[i]['Unnamed: 0'])

testing_df = df.loc[df['Unnamed: 0'].isin(ins) == False]

num_categories = 100
num_new_samples = 10 # Note: this will be multiplied by len(df) for the actual number of new cases.

columns = [] # columns you want


main = Main(subset_df, columns, num_new_samples, num_categories)
result = main.run()

print(result)
print(result.describe().transpose())
result.to_csv("data2.csv", index=False) # returns the new data + data used for creation
testing_df.to_csv("subset_testing.csv") # returns the data to be used for testing (not included in data creation)
