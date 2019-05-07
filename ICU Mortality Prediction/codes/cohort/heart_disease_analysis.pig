-- ***************************************************************************
-- TASK
-- Aggregate d_icd9_diagnoses into features of patient and generate demographics csv for baseline prediction.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- set the debug mode on
--SET debug 'on'

-- ***************************************************************************
-- To RUN:
-- cd /mnt/host/ICU-Mortality-Prediction-Team36/pig
-- sudo pig -x local heart_disease_analysis.pig
-- ***************************************************************************


-- load D_ICD_DIAGNOSES.csv file
-- "ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
d_icd9_diagnoses = LOAD '/mnt/data/D_ICD_DIAGNOSES.csv' USING PigStorage(',') AS (row_id:int, icd9_code:chararray,
                                                                                  short_title:chararray,
                                                                                  long_title:chararray);

d_icd9_diagnoses = FOREACH d_icd9_diagnoses GENERATE row_id, icd9_code, short_title, long_title;
--DUMP d_icd9_diagnoses;

-- load DIAGNOSES_ICD.csv file
-- "ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"
diagnoses_icd = LOAD '/mnt/data/DIAGNOSES_ICD.csv' USING PigStorage(',') AS (row_id:int, subject_id:int, hadm_id:int,
                                                                             seq_num:int, icd9_code:chararray);

diagnoses_icd = FOREACH diagnoses_icd GENERATE row_id, subject_id, hadm_id, seq_num, icd9_code;
--DUMP diagnoses_icd;

-- load PATIENTS.csv file
-- "ROW_ID","SUBJECT_ID","GENDER","DOB","DOD","DOD_HOSP","DOD_SSN","EXPIRE_FLAG"
patients = LOAD '/mnt/data/PATIENTS.csv' USING PigStorage(',') AS (row_id:int, subject_id:int, gender:chararray,
                                                                   dob:chararray, dod:chararray, dod_hosp:chararray,
                                                                   dod_ssn:chararray, expire_flag:int);

patients = FOREACH patients GENERATE row_id, subject_id, gender,
                                     ((dob is null or IsEmpty(dob)) ? dob:ToDate(dob, 'YYYY-MM-dd HH:mm:ss')) AS dob_2,
                                     ((dod is null or IsEmpty(dod)) ? dod:ToDate(dod, 'YYYY-MM-dd HH:mm:ss')) AS dod_2,
                                     expire_flag;

--DUMP patients;
age_of_death = FOREACH patients GENERATE subject_id, YearsBetween(dod_2, dob_2) AS aod; -- Age of Death

patients = JOIN patients BY subject_id, age_of_death BY subject_id;



--patients_aod = FOREACH patients GENERATE patients::subject_id AS subject_id,
--                                         YearsBetween(patients::dod_timestamp, patients::dob_timestamp) AS aod;
--
--patients = JOIN patients BY subject_id, patients_aod BY subject_id;

-- load ADMISSIONS.csv file
-- "ROW_ID","SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME","ADMISSION_TYPE","ADMISSION_LOCATION",
-- "DISCHARGE_LOCATION", "INSURANCE","LANGUAGE","RELIGION","MARITAL_STATUS","ETHNICITY","EDREGTIME","EDOUTTIME",
-- "DIAGNOSIS","HOSPITAL_EXPIRE_FLAG","HAS_CHARTEVENTS_DATA"
admissions = LOAD '/mnt/data/ADMISSIONS.csv' USING PigStorage(',') AS (row_id:int, subject_id:int, hadm_id:int,
                                                                       admittime:chararray, dischtime:chararray,
                                                                       deathtime:chararray, admission_type:chararray,
                                                                       admission_location:chararray,
                                                                       discharge_location:chararray,
                                                                       insurance:chararray, language:chararray,
                                                                       religion:chararray, marital_status:chararray,
                                                                       ethnicity:chararray, edregtime:chararray,
                                                                       edouttime:chararray, diagnosis:chararray,
                                                                       hospital_exp_flag:int;

admissions = FOREACH admissions GENERATE row_id, subject_id, hadm_id, ToDate(admittime, 'yyyy-MM-dd') AS admit,
                                         ToDate(dischtime, 'yyyy-MM-dd') AS release,
                                         ToDate(deathtime, 'yyyy-MM-dd') AS deathtime,
                                         admission_type, admission_location, discharge_location, insurance, language,
                                         religion, marital_status, ethnicity,
                                         ToDate(edregtime, 'yyyy-MM-dd') AS edregtime,
                                         ToDate(edouttime, 'yyyy-MM-dd') AS edouttime, diagnosis, hospital_expire_flag,
                                         DaysBetween(dischtime, admittime) as total_admission_time;

--DUMP admissions;
--admissions_admittime = FOREACH admissions GENERATE admissions::hadm_id AS hadm_id,
--                                         DaysBetween(admissions::dischtime, admissions::admittime) as admittime;
--
--admissions = JOIN admissions BY hadm_id, admissions_admittime BY hadm_id;


--last_admissions_event = FOREACH (GROUP admissions BY subject_id DESC)
--                            GENERATE group AS subject_id, MAX(admissions.admit) as admit;

admissions_by_subject_id = GROUP admissions BY subject_id;
admissions_by_subject_id_count = FOREACH admissions_by_subject_id GENERATE GROUP AS subject_id, COUNT($1) AS admissions;

--
--admissions_culled = FOREACH admissions GENERATE row_id, subject_id, hadm_id, admit, release,
--                            deathtime, admission_type, admission_location, discharge_location, insurance, language,
--                            religion, marital_status, ethnicity, edregtime, edouttime, diagnosis,
--                            hospital_expire_flag, total_admission_time;



admissions_culled = FOREACH admissions {
    admissions_sorted_by_subject_id = ORDER admissions BY subject_id DESC;
    subject_id = LIMIT admissions_sorted_by_subject_id 1;
    GENERATE row_id, subject_id, hadm_id, admit, release, deathtime, admission_type, admission_location,
             discharge_location, insurance, language, religion, marital_status, ethnicity, edregtime, edouttime,
             diagnosis, hospital_expire_flag, total_admission_time;
}

-- Combine patients and admissions tables
patients_admissions = JOIN patients BY subject_id, admissions_culled BY subject_id;

-- Combine patients + admissions and add count of admissions
patients_admissions = JOIN patients_admissions BY subject_id, admissions_by_subject_id_count BY subject_id;

admissions_culled = FOREACH patients_admissions GENERATE subject_id, hadm_id, admit, release, deathtime,
                                         admission_type, admission_location, discharge_location, insurance, language,
                                         religion, marital_status, ethnicity, edregtime, edouttime, diagnosis,
                                         hospital_expire_flag, total_admission_time,
                                         YearsBetween(admit, dob_2) AS admit_age; -- Age of Admission
--DUMP admissions_culled;

-- ***************************************************************************
-- Find Number of Unique Patients, Visits and Diagnoses
-- ***************************************************************************

-- unique_patients = Diagnoses_ICD.SUBJECT_ID.unique()
unique_patients = DISTINCT (FOREACH diagnoses_icd GENERATE subject_id);

-- unique_visits = Diagnoses_ICD.HADM_ID.unique()
unique_visits = DISTINCT (FOREACH diagnoses_icd GENERATE hadm_id);

-- unique_diagnoses = Diagnoses_ICD.ICD9_CODE.value_counts()
unique_diagnoses = DISTINCT (FOREACH diagnoses_icd GENERATE icd9_code);


-- Find causes of death for patients based on final admission reason
-- cause of death (diagnosis)
death = FOREACH admissions_long GENERATE subject_id, (hospital_expire_flag == 1 ? diagnosis:null) AS cdeath,
                                           (hospital_expire_flag == 1 ? admission_location:null) AS ldeath,
                                           (hospital_expire_flag == 1 ? insurance:null) AS ideath,
                                           (hospital_expire_flag == 1 ? ethnicity:null) AS edeath,
                                           (hospital_expire_flag == 1 ? admission_type:null) AS tdeath,
                                           (hospital_expire_flag == 1 ? hadm_id:null) AS hamdiddeath,
                                           (hospital_expire_flag == 1 ? total_admission_time:null) AS admitt;

demographics = JOIN admissions_culled BY subject_id OUTER, death BY hadm_id;

--DUMP demographics;

STORE demographics INTO '/mnt/data/Demographics.csv' USING PigStorage(',');
