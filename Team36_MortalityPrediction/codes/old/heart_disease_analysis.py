import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mt
import seaborn as sns
import numpy as np
import datetime as dt
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Comprises a description of what each ICD9 code stands for, with both a short title and a long title to search
D_ICD_Diagnoses = pd.read_csv('data/D_ICD_DIAGNOSES.csv')

# Comprises all ICD9 codes for each subject as identified by subject ID and HADM ID
Diagnoses_ICD = pd.read_csv('data/DIAGNOSES_ICD.csv')

# List of all patients, their sex, DOB, DOD and associated information, by subject ID only
Patients = pd.read_csv('data/PATIENTS.csv')

# Additional information for each patient including additional demographic info, diagnosis etc.
Admissions = pd.read_csv('data/ADMISSIONS.csv')

# Find Number of Unique Patients in over diagnosis database
unique_patients = Diagnoses_ICD.SUBJECT_ID.unique()
print(len(unique_patients))
# 46,520 unique patients in the data set
unique_visits = Diagnoses_ICD.HADM_ID.unique()
print(len(unique_visits))
# 58,976 unique visits in the data set

# Find the most common diagnoses in ICD9 data
unique_diagnoses = Diagnoses_ICD.ICD9_CODE.value_counts()
print(unique_diagnoses.head())


# 414.01, which is Coronary atherosclerosis of native coronary artery, has 12,429 diagnoses
# 401 = Hypertension with 20,703 diagnoses
# 428 = Heart Failure
# 427 = Cardiac Dysrythmia
# 414.01 = Coronary atherosclerosis of native coronary artery
# First, we will compile a bunch of metrics to get patient ages, stay lengths and several other metrics

# We need to calculate parameters for each patient including their age at death, age at discharge etc.

# Add Patient DOB in usable format and Merge with patient information
Patients_DOB = Patients['DOB']
Patients_DOB2 = []
for n in range(len(Patients_DOB)):
    prelim = Patients_DOB[n]
    Patients_DOB2.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())
Birth_Date_Series = pd.Series(Patients_DOB2)
Patients['DOB-2'] = Birth_Date_Series

# Add Data for Patient's Date of Death
Patients_DODeath = Patients['DOD']
Patients_DOD = []
for n in range(len(Patients_DODeath)):
    if pd.isnull(Patients_DODeath[n]) == True:
        Patients_DOD.append(np.nan)
    else:
        prelim = Patients_DODeath[n]
        Patients_DOD.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())

# Add Data for Patient's Date of Admission
Patients_Admit = Admissions['ADMITTIME']
Patients_Admissions = []
for n in range(len(Patients_Admit)):
    if pd.isnull(Patients_Admit[n]) == True:
        Patients_Admissions.append(np.nan)
    else:
        prelim = Patients_Admit[n]
        Patients_Admissions.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())

# Add Data for Patient's Date of Release
Patients_leave = Admissions['DISCHTIME']
Patients_Release = []
for n in range(len(Patients_leave)):
    if pd.isnull(Patients_leave[n]) == True:
        Patients_Release.append(np.nan)
    else:
        prelim = Patients_leave[n]
        Patients_Release.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())

    # Add Data to Series
Birth_Date_Series = pd.Series(Patients_DOB2)
Death_Series = pd.Series(Patients_DOD)

Admit_Series = pd.Series(Patients_Admissions)
Release_Series = pd.Series(Patients_Release)

# Find Age at Death
Age_Death = (Death_Series - Birth_Date_Series)
death_age = []

for entry in range(len(Age_Death)):
    if type(Age_Death[entry]) == float:
        death_age.append(float('NAN'))
    else:
        death_age.append(Age_Death[entry].days)

Age_Death = pd.Series(death_age)

# Find Total Admission Times
Admissions_Time = (Release_Series - Admit_Series)

Admit_Time = []

for entry in range(len(Admissions_Time)):
    if type(Admissions_Time[entry]) == float:
        Admit_Time.append(float('NAN'))
    else:
        Admit_Time.append(Admissions_Time[entry].days)

Admissions_Time = pd.Series(Admit_Time)

# Add to Pandas Dataframe
Patients['DOB-2'] = Birth_Date_Series
Patients['DOD-2'] = Death_Series
Patients['AOD'] = Age_Death

Admissions['Admit'] = Admit_Series
Admissions['Release'] = Release_Series
Admissions['Total Admission Time'] = Admissions_Time

Admissions_culled = Admissions.drop_duplicates('SUBJECT_ID', keep='last')
Admissions_culled_Admits = Admissions_culled['ADMITTIME']

Admit_Ages = (Admit_Series - Birth_Date_Series)

# Combine admissions information and patient's information
Admissions_culled = Admissions.copy(deep=False)
Admissions_culled = Admissions_culled.drop_duplicates('SUBJECT_ID', keep='last')

Admissions_long = Admissions_culled.merge(Patients, on='SUBJECT_ID')
Admissions_long.head()
Admit_Series = pd.Series(Patients_DOD)
Admissions_culled_Admits = Admissions_culled['ADMITTIME']

Admit_times2 = pd.Series(Admissions_long['ADMITTIME'])
Atimes = []
for n in range(len(Admit_times2)):
    if pd.isnull(Admit_times2[n]) == True:
        Atimes.append(np.nan)
    else:
        prelim = Admit_times2[n]
        Atimes.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())

DOB_Admit_times = pd.Series(Admissions_long['DOB'])
DOBAtimes = []
for n in range(len(DOB_Admit_times)):
    if pd.isnull(DOB_Admit_times[n]) == True:
        DOBAtimes.append(np.nan)
    else:
        prelim = DOB_Admit_times[n]
        DOBAtimes.append(dt.datetime.strptime(prelim[0:10], '%Y-%m-%d').date())

Atimes = pd.Series(Atimes)
DOBAtimes = pd.Series(DOBAtimes)

admit_births = (Atimes - DOBAtimes)

Admit_Timet = []
for entry in range(len(admit_births)):
    if type(admit_births[entry]) == float:
        Admit_Timet.append(float('NAN'))
    else:
        Admit_Timet.append(admit_births[entry].days)

Admit_Timet = pd.Series(Admit_Timet)

# Find total number of visits for each subject ID
visit_count = pd.DataFrame(Admissions.SUBJECT_ID.value_counts())
visit_count = pd.DataFrame(Admissions.SUBJECT_ID.value_counts())
visit_count.reset_index(level=0, inplace=True)
visit_count.columns = ['SUBJECT_ID', 'ADMISSIONS']

Admissions_long = Admissions_long.merge(visit_count, on='SUBJECT_ID')

# Find causes of death for patients based on final admission reason
cdeath = []
subjectid = []
ldeath = []
ideath = []
edeath = []
tdeath = []
hamdiddeath = []
admitt = []

for entry in range(len(Admissions)):
    if Admissions['HOSPITAL_EXPIRE_FLAG'][entry] == 0:
        continue
    elif Admissions['HOSPITAL_EXPIRE_FLAG'][entry] == 1:
        subjectid.append(Admissions['SUBJECT_ID'][entry])
        cdeath.append(Admissions['DIAGNOSIS'][entry])
        ldeath.append(Admissions['ADMISSION_LOCATION'][entry])
        tdeath.append(Admissions['ADMISSION_TYPE'][entry])
        ideath.append(Admissions['INSURANCE'][entry])
        edeath.append(Admissions['ETHNICITY'][entry])
        hamdiddeath.append(Admissions['HADM_ID'][entry])
        admitt.append(Admissions['Total Admission Time'][entry])

cdeath = pd.Series(cdeath)
subjectid = pd.Series(subjectid)
ldeath = pd.Series(ldeath)
ideath = pd.Series(ideath)
edeath = pd.Series(edeath)
tdeath = pd.Series(tdeath)
hamdiddeath = pd.Series(hamdiddeath)
admitt = pd.Series(admitt)

Death = pd.DataFrame()
Death['SUBJECT_ID'] = subjectid
# Death['HADM_ID'] = hamdiddeath
Death['CAUSE'] = cdeath
# Death['INSURANCE'] = ideath
# Death['ETHNICITY'] = edeath
# Death['ADMISSION_TYPE'] = tdeath
# Death['ADMISSION_LOCATION'] = ldeath
# Death['TOTAL_ADMISSION_TIME'] = admitt
print(Death.head())

heart_death_desc = Death[Death['CAUSE'].str.contains("HEART|MYOCARDIAL|CARDIAC", na=False)==True]
heart_attack = Death[Death['CAUSE'].str.contains("MYOCARDIAL", na=False)==True]
del heart_attack['CAUSE']
heart_attack['HEART_ATTACK_FLAG'] = 1


# Combine admissions information and patients information for an improved demographic dataframe
demographics = pd.DataFrame()
demographics['SUBJECT_ID'] = Admissions_long['SUBJECT_ID']
demographics['GENDER'] = Admissions_long['GENDER']
demographics['DOB'] = Admissions_long['DOB-2']
demographics['DOD'] = Admissions_long['DOD-2']
demographics['DOA'] = Admissions_long['ADMITTIME']
demographics['ADMIT_AGE'] = Admit_Timet
demographics['ETHNICITY'] = Admissions_long['ETHNICITY']
demographics['MARITAL_STATUS'] = Admissions_long['MARITAL_STATUS']
demographics['LANGUAGE'] = Admissions_long['LANGUAGE']
demographics['RELIGION'] = Admissions_long['RELIGION']
demographics['INSURANCE'] = Admissions_long['INSURANCE']
demographics['ADMISSION_LOCATION'] = Admissions_long['ADMISSION_LOCATION']
demographics['#ADMISSIONS'] = Admissions_long['ADMISSIONS']
print(demographics.head())

# Merge Death and demographics on subject ID to add relevant data about patient deaths
demographics = pd.merge(Death, demographics, on= 'SUBJECT_ID', how = 'outer')

# Put "outside hospital" and create flag for patients who died outside of hospital and we don't have information on
death_cause = []
outside_death_cause = []
for entry in range(len(demographics)):
    if pd.isnull(demographics['DOD'][entry]) == False and pd.isnull(demographics['CAUSE'][entry]) == True:
        death_cause.append('Death Outside of Hospital')
        outside_death_cause.append(1)
    else:
        death_cause.append(demographics['CAUSE'][entry])
        outside_death_cause.append(0)

death_cause = pd.Series(death_cause)
outside_death_flag = pd.Series(outside_death_cause)

demographics['CAUSE'] = death_cause
demographics['OUTSIDE_DEATH_FLAG'] = outside_death_flag
print(demographics.head())

# Add general death flag
death_flag = []
for entry in range(len(demographics)):
    if pd.isnull(demographics['DOD'][entry]) == False:
        death_flag.append(1)
    else:
        death_flag.append(0)

death_flag = pd.Series(death_flag)
demographics['DEATH_FLAG'] = death_flag

# Add too old flag
old_flag = []
for entry in range(len(demographics)):
    if demographics['ADMIT_AGE'][entry] > 32850:
        old_flag.append(1)
    else:
        old_flag.append(0)

old_flag = pd.Series(old_flag)
demographics['OLD_FLAG'] = old_flag

# Develop no letter list
list_values = Diagnoses_ICD['ICD9_CODE'].values.tolist()
no_letter_list = []
for i in range(len(list_values)):
    Value = str(list_values[i])
    Letter_stripped_value = Value.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    no_letter_list.append(Letter_stripped_value)

    # Develop no number list
list_values = Diagnoses_ICD['ICD9_CODE'].values.tolist()
no_number_list = []
for i in range(len(list_values)):
    Value = str(list_values[i])
    number_stripped_value = Value.lstrip('1234567890')
    if number_stripped_value:
        final_number = number_stripped_value[0]
        no_number_list.append(final_number)
    else:
        no_number_list.append('')

    # Correct three digit codes for V codes where decimal is only in first two spots
three_number_list = []
for i in range(len(list_values)):
    if no_number_list[i] == 'V':
        Value = no_letter_list[i]
        three_digits = Value[0:2]
        three_number_list.append(float(three_digits))
    else:
        Value = no_letter_list[i]
        three_digits = Value[0:3]
        three_number_list.append(float(three_digits))

# Add three number list to diagnoses codes
three_number_series = pd.Series(three_number_list)
Diagnoses_ICD['Three Numbers'] = three_number_series  # Pull only first three numbers (need to fix for letter codes)

# Fix ages for over 90 population
age = []
for entry in range(len(demographics)):
     if demographics['OLD_FLAG'][entry] == 0:
            age.append(demographics['ADMIT_AGE'][entry])
     else:
        age.append(np.nan)
age = pd.Series(age)
demographics['ADMIT_AGE'] = age
print(demographics.head())

# Add myocardial infarction flag to data
demographics = pd.merge(demographics, heart_attack, on= 'SUBJECT_ID', how = 'outer')
print(demographics.head())



attack_flag = []

for entry in range(len(demographics)):
    if pd.isnull(demographics['HEART_ATTACK_FLAG'][entry]) == True:
        attack_flag.append(0)
    else:
        attack_flag.append(1)

attack_flag = pd.Series(attack_flag)
demographics['HEART_ATTACK_FLAG'] = attack_flag


# Pull Data for all Coronary artery disease patients
Cor_Diagnoses_ICD = Diagnoses_ICD.loc[Diagnoses_ICD['ICD9_CODE'] == '41401'] # All Coronary Artery disease patients
Athero_Diagnoses = pd.DataFrame(Cor_Diagnoses_ICD.SUBJECT_ID)
Athero_Diagnoses = Athero_Diagnoses.drop_duplicates()
Athero_Diagnoses['ATHERO_DIAGNOSIS_FLAG'] = 1


# Merge with demographics file:
demographics = pd.merge(demographics, Athero_Diagnoses, on= 'SUBJECT_ID', how = 'outer')


# Add Atherosclerosis diagnosis flags to demographic data
athero_flag = []
for entry in range(len(demographics)):
    if demographics['ATHERO_DIAGNOSIS_FLAG'][entry] == 1:
        athero_flag.append(1)
    else:
        athero_flag.append(0)
athero_flag = pd.Series(athero_flag)
demographics['ATHERO_DIAGNOSIS_FLAG'] = athero_flag

# Add heart cause of death flag to demographic data
demographics = pd.merge(demographics, heart_death_desc, on= 'SUBJECT_ID', how = 'outer')
print(demographics.head())




heart_death_flag = []

for entry in range(len(demographics)):
    if pd.isnull(demographics['CAUSE_y'][entry]) == True:
        heart_death_flag.append(0)
    else:
        heart_death_flag.append(1)



heart_death_flag = pd.Series(heart_death_flag)
demographics['HEART_DEATH_FLAG'] = athero_flag

del demographics['CAUSE_y']

demographics['CAUSE'] = demographics['CAUSE_x']
del demographics['CAUSE_x']
demographics['HEART_DEATH_FLAG'] = heart_death_flag


# Add Age in years to simplify interpretation
demographics['ADMIT_AGE'] = demographics['ADMIT_AGE']/365

# Test: Find all demographic information on patients who died from heart condition
print(demographics[demographics['HEART_DEATH_FLAG'] == 1].head())


# Split List into those whose age we know, those still alive, and those too old to have age listed
# Combine those still alive with those who died at a known age
Patients_old = demographics[demographics['OLD_FLAG'] == 1]
Patients_young = demographics[demographics['OLD_FLAG'] == 0]
Patients_alive = demographics[demographics['DEATH_FLAG'] == 0]
Patients_dead = demographics[demographics['DEATH_FLAG'] == 1]
print('# of total patients is', len(demographics))
print('# of old patients is ', len(Patients_old))
print('# of young patients is ', len(Patients_young))
print('# of living patients is ',len(Patients_alive))
print('# of dead patients is ', len(Patients_dead))


# Fill na values with "uknown" outside of age
demographics = demographics.fillna({'ETHNICITY':'UKNOWN','MARITAL_STATUS':'UKNOWN', 'RELIGION': 'UKNOWN', 'LANGUAGE':'UKNOWN', 'INSURANCE':'UKNOWN', 'ADMISSION_LOCATION':'UKNOWN'})

print(demographics.groupby('GENDER').mean())

print(demographics.groupby('INSURANCE').mean())

print(demographics.groupby('#ADMISSIONS').mean())

print(demographics[demographics['ATHERO_DIAGNOSIS_FLAG'] == 1].groupby('#ADMISSIONS').size())

print(demographics.groupby('ATHERO_DIAGNOSIS_FLAG').mean())

print(demographics.groupby('MARITAL_STATUS').mean())

print(demographics.groupby('ETHNICITY').mean())

print(demographics.groupby('ADMISSION_LOCATION').mean())

lang = demographics.groupby('LANGUAGE').mean()
language = pd.DataFrame()
language['English'] = lang.loc['ENGL']
lang = lang.drop('ENGL')
language['Others'] = lang.mean()
print(language)

# Export data set
demographics.to_csv('data/Demographics.csv')




