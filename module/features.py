import pickle

from module.pipeline import PipeLine
from module.kayano import *


###################### PipeLineによる比較用データセット準備 #########################
def train_or_test(pipe, train_flg, split_kwrg):
    if train_flg:
        pack = pipe.fold_out_split(**split_kwrg)
        return pack
    else:
        if split_kwrg['to_array']:
            return pipe.df_num.values
        else:
            return pipe.df_num


def pipe_1(df, train_flg, split_kwrg, retrain=False):
    print('Base(pipe_1)'.center(75))
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_2(df, train_flg, split_kwrg, retrain=False):
    df = stSlope_categolize(df)
    print('StSlpoeCat(pipe_2)'.center(75))

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_3(df, train_flg, split_kwrg, retrain=False):
    print('CholestMean(pipe_3)'.center(75))
    df = cholesterol_mean(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_4(df, train_flg, split_kwrg, retrain=False):
    print('AgeCat(pipe_4)'.center(75))
    df = age_categolize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_5(df, train_flg, split_kwrg, retrain=False):
    print('StSlpoeCat CholestMean AgeCAt(pipe_5)'.center(75))
    df = stSlope_categolize(df)
    df = cholesterol_mean(df)
    df = age_categolize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack



def pipe_6(df, train_flg, split_kwrg, retrain=False):
    print('RestingBpCat(pipe_6)'.center(75))
    df = restingBP_categorize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_7(df, train_flg, split_kwrg, retrain=False):
    print('OldPeakCat(pipe_7)'.center(75))
    df = oldpeak_categolize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_8(df, train_flg, split_kwrg, retrain=False):
    print('RestingBpCat OldPeakCat(pipe_8)'.center(75))
    df = oldpeak_categolize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_9(df, train_flg, split_kwrg, retrain=False):
    print('Onehot(pipe_9)'.center(75))
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_10(df, train_flg, split_kwrg, retrain=False):
    print('CholestMean AgeCat Onehot(pipe_10)'.center(75))
    df = cholesterol_mean(df)
    df = age_categolize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_11(df, train_flg, split_kwrg, retrain=False):
    print('CholCut(pipe_11)'.center(75))
    if train_flg:
        df = cholestrol_zero(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_12(df, train_flg, split_kwrg, retrain=False):
    print('CholCut Onehot(pipe_12)'.center(75))
    if train_flg:
        df = cholestrol_zero(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_13(df, train_flg, split_kwrg, retrain=False):
    print('DropByShap(pipe_13)'.center(75))

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    columns = ['Age', 'Sex', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',\
               'ChestPainType_ASY', 'ST_Slope_Flat', 'ST_Slope_Up']
    pipe.df_num = pipe.df_num[columns]
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_14(df, train_flg, split_kwrg, retrain=False):
    print('CholCut DropByShap(pipe_14)'.center(75))
    if train_flg:
        df = cholestrol_zero(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    columns = ['Age', 'Sex', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',\
               'ChestPainType_ASY', 'ST_Slope_Flat', 'ST_Slope_Up']
    pipe.df_num = pipe.df_num[columns]
    display(pipe.df_num.head())
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_15(df, train_flg, split_kwrg, retrain=False):
    print('Chol Regression(pipe_15)'.center(75))
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    pipe.df_num

    with open('./data/reg_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)

    pred_trg = pipe.df_num[pipe.df_num.Cholesterol==0]
    pred_trg = pred_trg.drop('Cholesterol', axis=1)
    pred = reg_model.predict(pred_trg)
    pipe.df_num.loc[pipe.df_num.Cholesterol==0, 'Cholesterol'] = pred.astype('int')
    display(pipe.df_num.head())

    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack