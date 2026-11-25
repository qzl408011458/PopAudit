import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import argparse
import copy
import itertools
import warnings

import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from collections import Counter

warnings.filterwarnings('ignore')
# real_test_data = df = pd.read_csv('data/traindata_exact_ctx.csv')



def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='modelSave/tvae_2025-12-03 10:47:08/final_generated_samples.csv', type=str,
                        help='')

    args = parser.parse_args(argv)
    return args




def transform_cat_columns(real_test_data, synthetic_data, cat_cols, num_cols):
    # cat_cols = ['Municipality_Categories', 'Gender', 'Marital_status', 'Employment_status', 'Studenthood_status', 'Income_class', 'Household_Type', ]
    # num_cols = ['Age', 'Number_of_cars_of_person', 'Household_Size', 'Number_of_children', 'Number_of_cars_of_household', 'Trips_of_Car', 'Trips_of_CarPassenger', 'Trips_of_PublicTransport',
    #             'Trips_of_Walking', 'Trips_of_Bike', 'Activities_of_Home', 'Activities_of_Work', 'Activities_of_Other', 'Activities_of_School',
    #             'First_Activity_Start_Time', 'Last_Activity_End_Time']



    for col in cat_cols:
        le = LabelEncoder()
        # fit on the UNION of real + synthetic categories so a category that appears
        # only in the generated data (e.g. a rare 'Ferryboat' commute mode absent from a
        # state's test split) does not raise "previously unseen labels" during transform.
        rc = real_test_data[col].astype(str)
        sc = synthetic_data[col].astype(str)
        le.fit(pd.concat([rc, sc], ignore_index=True))
        real_test_data[col] = le.transform(rc)
        synthetic_data[col] = le.transform(sc)
    all_cols = cat_cols + num_cols
    real_test_data = real_test_data[all_cols]
    synthetic_data = synthetic_data[all_cols]

    return real_test_data, synthetic_data

def discretize(df, dic_col_disunit):
    for col in dic_col_disunit:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(x // dic_col_disunit[col]))
    return df

def generate_combinations(k, elements):
    return list(itertools.combinations(elements, k))

def evaluate_marg(real_df, model_df, cat_cols, num_cols):  # Evaluate the real data and the model data using certain evaluation metrics and plot
    # if 'Age' in agg_vars:
    real_df, model_df = transform_cat_columns(real_df, model_df, cat_cols, num_cols)

    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]

    real_df = discretize(real_df, dic_col_disunit)
    model_df = discretize(model_df, dic_col_disunit)

    metrics = {'MAE': 0., 'MSE': 0., 'RMSE': 0., 'SRMSE': 0.}

    vars_list = real_df.columns.tolist()

    comb_list = generate_combinations(len(vars_list) - 1, vars_list)
    len_combs = len(comb_list)


    ori_model_df = model_df.copy(deep=True)
    ori_real_df = real_df.copy(deep=True)

    ##### Adding a counter variable and aggregating to retrieve probabilities
    for agg_vars in comb_list:
        agg_vars = list(agg_vars)
        model_df = ori_model_df.copy(deep=True)
        model_df['count'] = 1
        model_df = model_df.groupby(agg_vars, observed=True).count()
        model_df /= model_df['count'].sum()

        real_df = ori_real_df.copy(deep=True)
        real_df['count'] = 1
        real_df = real_df.groupby(agg_vars, observed=True).count()
        real_df /= real_df['count'].sum()

        ##### Merge and difference
        real_and_sampled = pd.merge(real_df, model_df, suffixes=['_real', '_sampled'], on=agg_vars, how='outer',
                                    indicator=True)
        real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
        real_and_sampled['diff'] = real_and_sampled.count_real - real_and_sampled.count_sampled
        diff = np.array(real_and_sampled['diff'])

        N = 1

        for var in agg_vars:
            N *= len(ori_real_df[var].unique())

        # metrics = {}
        metrics['MAE'] += np.sum(abs(diff)) / N
        metrics['MSE'] += np.sum(diff ** 2) / N
        metrics['RMSE'] += np.sqrt(metrics['MSE'])
        metrics['SRMSE'] += np.sqrt(metrics['MSE'] * N)


    metrics['MAE'] /= len_combs
    metrics['MSE'] /= len_combs
    metrics['RMSE'] /= len_combs
    metrics['SRMSE'] /= len_combs
    # print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{:4f}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))

    return metrics['SRMSE']

def evaluate_bivar(real_df, model_df, cat_cols, num_cols):  # Evaluate the real data and the model data using certain evaluation metrics and plot
    # if 'Age' in agg_vars:
    real_df, model_df = transform_cat_columns(real_df, model_df, cat_cols, num_cols)

    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]


    metrics = {'MAE': 0., 'MSE': 0., 'RMSE': 0., 'SRMSE': 0.}

    vars_list = real_df.columns.tolist()

    comb_list = generate_combinations(len(vars_list) - 2, vars_list)
    len_combs = len(comb_list)


    ori_model_df = model_df.copy(deep=True)
    ori_real_df = real_df.copy(deep=True)

    ##### Adding a counter variable and aggregating to retrieve probabilities
    for agg_vars in comb_list:
        agg_vars = list(agg_vars)
        model_df = ori_model_df.copy(deep=True)
        model_df['count'] = 1
        model_df = model_df.groupby(agg_vars, observed=True).count()
        model_df /= model_df['count'].sum()

        real_df = ori_real_df.copy(deep=True)
        real_df['count'] = 1
        real_df = real_df.groupby(agg_vars, observed=True).count()
        real_df /= real_df['count'].sum()

        ##### Merge and difference
        real_and_sampled = pd.merge(real_df, model_df, suffixes=['_real', '_sampled'], on=agg_vars, how='outer',
                                    indicator=True)
        real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
        real_and_sampled['diff'] = real_and_sampled.count_real - real_and_sampled.count_sampled
        diff = np.array(real_and_sampled['diff'])

        N = 1

        for var in agg_vars:
            N *= len(ori_real_df[var].unique())

        # metrics = {}
        metrics['MAE'] += np.sum(abs(diff)) / N
        metrics['MSE'] += np.sum(diff ** 2) / N
        metrics['RMSE'] += np.sqrt(metrics['MSE'])
        metrics['SRMSE'] += np.sqrt(metrics['MSE'] * N)


    metrics['MAE'] /= len_combs
    metrics['MSE'] /= len_combs
    metrics['RMSE'] /= len_combs
    metrics['SRMSE'] /= len_combs
    # print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{:4f}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))

    return metrics['SRMSE']



def evaluate(real_df, model_df, type='marg'):  # Evaluate the real data and the model data using certain evaluation metrics and plot
    # if 'Age' in agg_vars:
    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]
    real_df = discretize(real_df, dic_col_disunit)
    model_df = discretize(model_df, dic_col_disunit)

    metrics = {'MAE': 0., 'MSE': 0., 'RMSE': 0., 'SRMSE': 0.}

    vars_list = real_df.columns.tolist()
    if type == 'marg':
        comb_list = generate_combinations(len(vars_list) - 1, vars_list)
        len_combs = len(comb_list)
    if type == 'bivar':
        comb_list = generate_combinations(len(vars_list) - 2, vars_list)
        len_combs = len(comb_list)

    ori_model_df = model_df.copy(deep=True)
    ori_real_df = real_df.copy(deep=True)

    ##### Adding a counter variable and aggregating to retrieve probabilities
    for agg_vars in comb_list:
        agg_vars = list(agg_vars)
        model_df = ori_model_df.copy(deep=True)
        model_df['count'] = 1
        model_df = model_df.groupby(agg_vars, observed=True).count()
        model_df /= model_df['count'].sum()

        real_df = ori_real_df.copy(deep=True)
        real_df['count'] = 1
        real_df = real_df.groupby(agg_vars, observed=True).count()
        real_df /= real_df['count'].sum()

        ##### Merge and difference
        real_and_sampled = pd.merge(real_df, model_df, suffixes=['_real', '_sampled'], on=agg_vars, how='outer',
                                    indicator=True)
        real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
        real_and_sampled['diff'] = real_and_sampled.count_real - real_and_sampled.count_sampled
        diff = np.array(real_and_sampled['diff'])

        N = 1

        for var in agg_vars:
            N *= len(ori_real_df[var].unique())

        # metrics = {}
        metrics['MAE'] += np.sum(abs(diff)) / N
        metrics['MSE'] += np.sum(diff ** 2) / N
        metrics['RMSE'] += np.sqrt(metrics['MSE'])
        metrics['SRMSE'] += np.sqrt(metrics['MSE'] * N)


    metrics['MAE'] /= len_combs
    metrics['MSE'] /= len_combs
    metrics['RMSE'] /= len_combs
    metrics['SRMSE'] /= len_combs
    print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{:4f}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))

    # return real_and_sampled

def eval_travelattrs(real_df, model_df, cat_cols, num_cols):
    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]
    real_df = discretize(real_df, dic_col_disunit)
    model_df = discretize(model_df, dic_col_disunit)

    metrics = {'MAE': 0., 'MSE': 0., 'RMSE': 0., 'SRMSE': 0.}
    # agg_vars = ['Municipality_Categories', 'Age', 'Gender', 'Marital_status', 'Employment_status', 'Studenthood_status', 'Income_class', 'Household_Type',
    #             'Number_of_cars_of_person', 'Household_Size', 'Number_of_children', 'Number_of_cars_of_household']
    agg_vars = cat_cols + num_cols

    ori_model_df = model_df.copy(deep=True)
    ori_real_df = real_df.copy(deep=True)

    model_df['count'] = 1
    model_df = model_df.groupby(agg_vars, observed=True).count()
    model_df /= model_df['count'].sum()

    real_df['count'] = 1
    real_df = real_df.groupby(agg_vars, observed=True).count()
    real_df /= real_df['count'].sum()

    ##### Merge and difference
    real_and_sampled = pd.merge(real_df, model_df, suffixes=['_real', '_sampled'], on=agg_vars, how='outer',
                                indicator=True)
    real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
    real_and_sampled['diff'] = real_and_sampled.count_real - real_and_sampled.count_sampled
    diff = np.array(real_and_sampled['diff'])

    N = 1

    for var in agg_vars:
        N *= len(ori_real_df[var].unique())

    # metrics = {}
    metrics['MAE'] = np.sum(abs(diff)) / N
    metrics['MSE'] = np.sum(diff ** 2) / N
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['SRMSE'] = np.sqrt(metrics['MSE'] * N)

    print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{:4f}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'],
                                                        metrics['SRMSE']))
    return metrics['SRMSE']

def eval_zerocells(ori_real_df, ori_model_df, cat_cols, num_cols):
    # compute recall, precision, F1 score
    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]

    cat_cols_sel = cat_cols
    num_cols_sel = num_cols

    # cat_cols_sel = ['Municipality_Categories', 'Gender', 'Marital_status', 'Employment_status', 'Studenthood_status', 'Income_class', 'Household_Type', ]
    # num_cols_sel = ['Age', 'Household_Size', 'First_Activity_Start_Time', 'Last_Activity_End_Time',
    #             'Number_of_cars_of_person', 'Number_of_children', 'Number_of_cars_of_household', 'Trips_of_Car', 'Trips_of_CarPassenger', 'Trips_of_PublicTransport',
    #             'Trips_of_Walking', 'Trips_of_Bike', 'Activities_of_Home', 'Activities_of_Work', 'Activities_of_Other', 'Activities_of_School',
    #             ]


    real_arr = discretize(ori_real_df[cat_cols_sel + num_cols_sel], dic_col_disunit).to_numpy()
    model_arr = discretize(ori_model_df[cat_cols_sel + num_cols_sel], dic_col_disunit).to_numpy()
    # real_arr = ori_real_df.to_numpy()
    # model_arr = ori_model_df.to_numpy()
    real_arr_text = [str(list(_)) for _ in real_arr]
    model_arr_text = [str(list(_)) for _ in model_arr]
    counter_real = Counter(real_arr_text)
    counter_model = Counter(model_arr_text)
    PRECISION = 0.
    RECALL = 0.

    for comb in counter_model:
        if comb in counter_real:
            PRECISION += counter_model[comb] / len(model_arr_text)
    for comb in counter_real:
        if comb in counter_model:
            RECALL += counter_real[comb] / len(real_arr_text)

    # for comb in model_arr_text:
    #     if comb in counter_real:
    #         PRECISION += 1 / len(model_arr_text)
    # for comb in real_arr_text:
    #     if comb in counter_model:
    #         RECALL += 1 / len(real_arr_text)
    try:
        F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        print(f'PRECISION: {PRECISION * 100:2f}, RECALL: {RECALL * 100:2f}, F1: {F1 * 100:2f}')
    except:
        F1 = 0.
        print(f'PRECISION: {PRECISION * 100:2f}, RECALL: {RECALL * 100:2f}, F1: {F1 * 100:2f}')

    return PRECISION * 100, RECALL * 100, F1 * 100

def eval_zerocells_travel(ori_real_df, ori_model_df, cat_cols, num_cols):
    # compute recall, precision, F1 score
    dic_col_disunit = {
        'Age': 5,
        'Number_of_cars_of_household': 2,
        'Number_of_activities': 2,
        'First_Activity_Start_Time': 60,
        'Last_Activity_End_Time': 60
    }

    cols_cand = copy.deepcopy(list(dic_col_disunit.keys()))
    for col in cols_cand:
        if col not in num_cols:
            del dic_col_disunit[col]


    cat_cols_sel = cat_cols
    num_cols_sel = num_cols


    real_arr = discretize(ori_real_df[cat_cols_sel + num_cols_sel], dic_col_disunit).to_numpy()
    model_arr = discretize(ori_model_df[cat_cols_sel + num_cols_sel], dic_col_disunit).to_numpy()
    # real_arr = ori_real_df.to_numpy()
    # model_arr = ori_model_df.to_numpy()
    real_arr_text = [str(list(_)) for _ in real_arr]
    model_arr_text = [str(list(_)) for _ in model_arr]
    counter_real = Counter(real_arr_text)
    counter_model = Counter(model_arr_text)
    PRECISION = 0.
    RECALL = 0.

    for comb in counter_model:
        if comb in counter_real:
            PRECISION += counter_model[comb] / len(model_arr_text)
    for comb in counter_real:
        if comb in counter_model:
            RECALL += counter_real[comb] / len(real_arr_text)

    # for comb in model_arr_text:
    #     if comb in counter_real:
    #         PRECISION += 1 / len(model_arr_text)
    # for comb in real_arr_text:
    #     if comb in counter_model:
    #         RECALL += 1 / len(real_arr_text)
    try:
        F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        print(f'PRECISION: {PRECISION * 100:2f}, RECALL: {RECALL * 100:2f}, F1: {F1 * 100:2f}')
    except:
        F1 = 0.
        print(f'PRECISION: {PRECISION * 100:2f}, RECALL: {RECALL * 100:2f}, F1: {F1 * 100:2f}')

    return PRECISION * 100, RECALL * 100, F1 * 100

if __name__ == '__main__':
    args = parse_args()
    # args.result_path = 'modelSave/opt_study_tvae-llm_FiLM_2025-12-16 15:28:46/final_generated_samples_3.csv'
    # args.result_path = 'modelSave/wgan_llm_2025-12-24_16-48-51/final_generated_samples.csv'
    # args.result_path = 'modelSave/wgan_2025-12-23_16-03-00/final_generated_samples.csv'

    # rel_root = args.result_path
    # print(rel_root)

    cat_cols = ['Age', 'Household_Size', 'Number_of_children', 'Number_of_cars_of_household']
    num_cols = ['Region_Category', 'Gender', 'Marital_status',
       'Employment_status', 'Studenthood_status', 'Household_Type',
       'Commute_Mode', 'Personal_Income_Class', 'Household_Income_Class']

    real_test_data = pd.read_csv(_PA.R("/data/pums/testdata_pers.csv"))
    df_final = pd.read_csv(_PA.R("/modelSave/wgan_llm_lam(0.4)_pums_2026-04-22_23-05-07/final_generated_samples.csv"))
    save_path = _PA.R("/modelSave/wgan_llm_lam(0.4)_pums_2026-04-22_23-05-07")

    eval_SRMSE_marg = evaluate_marg(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_SRMSE_bivar = evaluate_bivar(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_SRMSE_joint_travel = eval_travelattrs(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_zc = eval_zerocells_travel(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_zc_sa = eval_zerocells(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)

    config = {}

    config['eval_SRMSE_marg'] = str(float(eval_SRMSE_marg))
    config['eval_SRMSE_bivar'] = str(float(eval_SRMSE_bivar))
    config['eval_SRMSE_joint(travel)'] = str(float(eval_SRMSE_joint_travel))
    config['eval_precision(travel)'] = str(float(eval_zc[0]))
    config['eval_recall(travel)'] = str(float(eval_zc[1]))
    config['eval_f1(travel)'] = str(float(eval_zc[2]))

    config['eval_precision(spatial_age)'] = str(float(eval_zc_sa[0]))
    config['eval_recall(spatial_age)'] = str(float(eval_zc_sa[1]))
    config['eval_f1(spatial_age)'] = str(float(eval_zc_sa[2]))

    with open(f'{save_path}/model_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)




