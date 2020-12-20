# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:09:36 2020

@author: santi

Cohort simulation

Scripts used:
Random number list generator
Create a list of numbers in a given range who's sum is a specific value
http://sunny.today/generate-random-integers-with-fixed-sum/

09/02/2020: Version 1.1
Added functionality to modify time before first repayment

"""
import numpy as np
import random as rd
import pandas as pd
#import os
#import scipy.stats as sta


def generate_random_integers(_sum, n):
    "return n sized array sum of which = _sum"  
    mean = _sum / n
    variance = int(0.5 * mean)

    min_v = mean - variance
    max_v = mean + variance
    array = [min_v] * n

    diff = _sum - min_v * n
    while diff > 0:
        a = np.random.randint(0, n - 1)
        if array[a] >= max_v:
            continue
        array[a] += 1
        diff -= 1
    return np.array(array)

def generate_bernoulli(n, p):
    return rd.choices([0, 1], weights = [1-p, p], k=n)

def generate_random_list_w_bounds(n, l):
    array = np.empty(n)
    for i in range(n):
        array[i] = rd.uniform(l[0], l[1])
    return array

def bool_prob(p):
    #returns True with p probability
    return rd.random()<p

#--------------------------------------------------------------------------------------------------------------#
#Parameters

n = 5000                  #size of cohort
_sum = 75000              #amount disbursed (k)

months_before_1st_payment = 18      #wait time until first payments

#vectorise for different values for all students

CIP_FIRST4 = np.full(n, 5202)
#option of different CIP, credlev uni id for all

CREDLEV = np.full(n, 5)

Uni_ID = np.full(n, 199120)


GR_filepath = r"../../1 - Model/Income modelling/Growth rates/growth_rates_12y.csv"
std_filepath = r"..\..\1 - Model\Income modelling\Quartiles\distribution_var_weighted_avg.csv"
inc_med_filepath = r"..\..\2 - Data\Education Outcome Data\CollegeScorecard\scorecard_FOS1517_clean.csv"

#LF participation and unemployment parameters

p_labor_participation = 0.94                       #prob of participation in labor force (LF)
p_first_year_empl = 0.9                             #prob of empl;oyment straight after first year

p_unempl = 0.04
p_laid_off = 0.03                                   #Prob of getting laid off
p_job_back = (1/p_unempl -1)*(p_laid_off)           #Prob of getting job back on a given month after getting laid off

                                                    #Conditions: p_laid_off < p_unempl / (1-p_unempl)

p_labor_part_first_year_empl = p_first_year_empl/p_labor_participation

#Initiate contract terms

principal = generate_random_integers(_sum, n)*1000

cut_per_dollar = 0.25/(100*1000)                       #Set for mba at unc


# automate API call based on characteristics


ISA_cut = np.full(n, cut_per_dollar*principal)

ContractLength = 60                         #contract length months
#same for all
ISA_floor = 25000                           #floor for ISA under which no revenue is taken.
#same for all
max_repayment_multiple = 3                  #amount to be paid max
#same for all


max_repayment = principal * max_repayment_multiple

inc_GR_df = pd.read_csv(GR_filepath)
inc_GR = np.empty(n)


Params = [n, _sum, p_first_year_empl, p_labor_participation, p_unempl, p_laid_off, p_job_back, ContractLength ]

#-------------------------------------------------------

init_income = np.empty(n)           #initiate starting incomes
inc_std_df = pd.read_csv(std_filepath)
inc_med_df = pd.read_csv(inc_med_filepath)
for i in range(n):
    inc_GR[i] = inc_GR_df.loc[(inc_GR_df['CIP_FIRST4']==CIP_FIRST4[i]) & (inc_GR_df['CREDLEV']==CREDLEV[i]), 'AGR']     #define anual sal gr
    
    inc_std = inc_std_df.loc[(inc_std_df['CIP_FIRST4']==CIP_FIRST4[i]) & (inc_std_df['CREDLEV']==CREDLEV[i]), 'WEIGHTED_AVG_STDDEV']
    
    inc_med = inc_med_df.loc[(inc_med_df['Major/Program Code']==CIP_FIRST4[i]) & \
                             (inc_med_df['Degree Level']==CREDLEV[i]) &\
                            (inc_med_df['University ID Number']==Uni_ID[i]), 'Median Earnings']
    init_income[i] = np.random.normal(loc = inc_med, scale = inc_std) / 12              #take monthly income

#-------------------------------------------------------------------------------


#---------------------------------------------------
#Identify labor_particip and first year labor particip

labor_particip_ind = np.array(generate_bernoulli(n, p_labor_participation))

first_year_particip_ind = np.array(generate_bernoulli(n, p_labor_part_first_year_empl))

#-----------------------------------------
#initiate student dataframe

stu_df = pd.DataFrame({'principal':principal, 'income share':ISA_cut, \
                       'LF participation':labor_particip_ind , '1st y LF participation': first_year_particip_ind, 'Max to be paid': max_repayment,\
                          'income GR': inc_GR, 'Months work count': np.zeros(n), 'total paid':np.zeros(n),  'Init_income':init_income,\
                        'current_income':init_income, 'temp unempl':np.zeros(n)})
    
#initiate monthly payments array
max_months = 12*10
revenue = np.empty((max_months, n))
revenue[0] = -1*principal

#initiate income array
income = np.empty((max_months, n))
income[0]  = np.zeros(n)


#-----------------------------------------
#Run first year. Assuming no payments at all during first 6 months (still at school)


for i in range(months_before_1st_payment, months_before_1st_payment+12):
    month_rev = stu_df['1st y LF participation']*stu_df['current_income']*stu_df['income share'].to_numpy()
    
    month_inc = stu_df['1st y LF participation']*stu_df['current_income'].to_numpy()
    
    for j in range(n):
        #Temp unempl
        
        if stu_df.loc[j, 'temp unempl'] == 0 and bool_prob(p_laid_off) and stu_df.loc[j, '1st y LF participation']==1:
            stu_df.loc[j, 'temp unempl'] =1    
        elif stu_df.loc[j, 'temp unempl'] ==1 and bool_prob(p_job_back) and stu_df.loc[j, '1st y LF participation']==1:
            stu_df.loc[j, 'temp unempl']=0 
        if stu_df.loc[j, 'temp unempl'] == 1:
            month_rev[j] = 0 ; month_inc[j] = 0
            
        if stu_df.loc[j, 'current_income'] <= ISA_floor/12:
            month_rev[j] = 0
        
        
        #Check if alrd paid back cap
        if stu_df.loc[j, 'total paid']+ month_rev[j] >= stu_df.loc[j, 'Max to be paid']:
            month_rev[j] = stu_df.loc[j, 'Max to be paid']-stu_df.loc[j, 'total paid']
        #Increase tracking stats if paying
        if month_rev[j] > 0:
            stu_df.loc[j, 'Months work count'] = stu_df.loc[j, 'Months work count']+1
            stu_df.loc[j, 'total paid'] += month_rev[j]
    
    revenue[i+1] = month_rev
    income[i+1] = month_inc    
#-------------------------------------------
#Now run loop until end
    
for i in range(12+months_before_1st_payment, max_months-1):
    for j in range(n):
        if stu_df.loc[j, 'Months work count']>0 and stu_df.loc[j, 'Months work count']%12 ==0:
            stu_df.loc[j, 'current_income'] = stu_df.loc[j, 'current_income']*(1+stu_df.loc[j, 'income GR'])
    
    month_rev = stu_df['LF participation']*stu_df['current_income']*stu_df['income share'].to_numpy()
    month_inc = stu_df['LF participation']*stu_df['current_income'].to_numpy()
    
    for j in range(n):
        #Temp unempl
        if stu_df.loc[j, 'temp unempl'] == 0 and bool_prob(p_laid_off):
            stu_df.loc[j, 'temp unempl'] =1    
        elif stu_df.loc[j, 'temp unempl'] ==1 and bool_prob(p_job_back):
            stu_df.loc[j, 'temp unempl']=0
            
        
        if stu_df.loc[j, 'temp unempl'] == 1:
            month_rev[j] = 0 ; month_inc[j] = 0
            
        if stu_df.loc[j, 'current_income'] <= ISA_floor/12:
            month_rev[j] = 0
        
        #Check if alrd completed terms
        if stu_df.loc[j, 'Months work count'] >= ContractLength:
            month_rev[j] = 0
        elif stu_df.loc[j, 'total paid']+ month_rev[j] > stu_df.loc[j, 'Max to be paid']:
            month_rev[j] = stu_df.loc[j, 'Max to be paid']-stu_df.loc[j, 'total paid'] 
            
        if month_rev[j] > 0:
            stu_df.loc[j, 'Months work count'] = stu_df.loc[j, 'Months work count']+1
            stu_df.loc[j, 'total paid'] += month_rev[j]
    
    revenue[i+1] = month_rev
    income[i+1] = month_inc 
#-----------------------------------------
#save the array as a csv
    
pd.DataFrame(revenue).T.to_excel("SimulatedRevenue_{}_size_{}k_funding.xlsx".format(n, _sum), index = None, header = np.arange(0, max_months))
print("SimulatedRevenue_{}_size_{}k_funding.xlxs".format(n, _sum),  'written')

pd.DataFrame(income).T.to_excel("SimulatedIncome_{}_size_{}k_funding.xlsx".format(n, _sum), index = None, header = np.arange(0, max_months))
print("SimulatedIncome_{}_size_{}k_funding.xlsx".format(n, _sum),  'written')

stu_df[['principal', 'income share', 'LF participation', '1st y LF participation', 'income GR', 'total paid', 'Init_income']].to_csv(\
            "SimulatedCohortData_{}_size_{}k_funding.csv".format(n, _sum), index = None)

print("SimulatedCohortData_{}_size_{}k_funding.csv".format(n, _sum), 'written')
    
#-----------------------------------------
#Write txt with parameters
f= open("SimulatedCohortParameters_{}_size_{}k_funding.txt".format(n, _sum),"w+")

f.write('cohort size = {} \n'.format(n))
f.write('funds disbursed = {} \n'.format(_sum))
f.write('months until first repayments = {} \n'.format(months_before_1st_payment))
f.write("labor lf participation = {} \n".format(p_labor_participation))
f.write("1st y out of college lf participation = {} \n".format(p_first_year_empl))
f.write("unemplyment rate = {} \n".format( p_unempl))
f.write('laid off rate = {} \n'.format(p_laid_off))
f.write('job back rate = {} \n'.format(p_job_back))
f.write('contract length = {} \n'.format(ContractLength))

f.close()

print('Parameters txt written')
