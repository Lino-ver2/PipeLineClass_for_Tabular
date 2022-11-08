from pathlib import Path
import sys

rank = 0 
#sys.path[0] = f'{Path().resolve().parents[rank]}' # mymoduleが上の階層にある場合rankを調整してコメント解除
from module.mymodule import PipeLine, train_or_test
from module.kayano import *


def pipe_1(df, train_flg, split_kwrg, retrain=False):
    print('Base(pipe_1)'.center(75))
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_2(df, train_flg, split_kwrg, retrain=False):
    df = stSlope_categolize(df)
    print('StSlpoeCat Standard(pipe_2)'.center(75))

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_3(df, train_flg, split_kwrg, retrain=False):
    print('CholestMean Standard(pipe_3)'.center(75))
    df = cholesterol_mean(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_4(df, train_flg, split_kwrg, retrain=False):
    print('AgeCAt Standard(pipe_4)'.center(75))
    df = age_categolize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()

    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_5(df, train_flg, split_kwrg, retrain=False):
    print('StSlpoeCat CholestMean AgeCAt Standard(pipe_5)'.center(75))
    df = stSlope_categolize(df)
    df = cholesterol_mean(df)
    df = age_categolize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack



def pipe_6(df, train_flg, split_kwrg, retrain=False):
    print('RestingBpCat Standard(pipe_6)'.center(75))
    df = restingBP_categorize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_7(df, train_flg, split_kwrg, retrain=False):
    print('OldPeakCat Standard(pipe_7)'.center(75))
    df = oldpeak_categolize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_8(df, train_flg, split_kwrg, retrain=False):
    print('RestingBpCat OldPeakCat Standard(pipe_8)'.center(75))
    df = oldpeak_categolize(df)
    
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_9(df, train_flg, split_kwrg, retrain=False):
    print('Onehot Standard(pipe_9)'.center(75))
    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack


def pipe_10(df, train_flg, split_kwrg, retrain=False):
    print('CholestMean AgeCat Onehot Standard(pipe_10)'.center(75))
    df = cholesterol_mean(df)
    df = age_categolize(df)

    pipe = PipeLine()
    pipe.train_flg = train_flg
    pipe(df)
    pipe.one_hot(pipe.df_cat.columns)
    display(pipe.df_num.head(3))
    pipe.standard_scaler()
    
    if retrain:
        return pipe.df_num, pipe.df_target
    pack = train_or_test(pipe, train_flg, split_kwrg)
    return pack