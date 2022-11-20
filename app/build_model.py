import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings


class Model:
    def __init__(self,
                 sp500_file="../data/SP500.csv",
                 moex_file="../data/MOEX.csv",
                 historical_file="../data/USDRUB_TOM.csv",
                 model_file="../model/model.pkl"):
        self.sp500_data = pd.read_csv(sp500_file)
        self.moex_data = pd.read_csv(moex_file)
        self.historical_data = pd.read_csv(historical_file)
        self.prepare_moex_data()
        self.prepare_s500_data()
        self.prepare_historical_data()

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

    def prepare_historical_data(self):
        self.historical_data.rename(columns={'<TICKER>': 'TICKER', '<PER>': 'PER', '<DATE>': 'DATE', '<TIME>': 'TIME', '<OPEN>': 'OPEN', '<HIGH>': 'HIGH', '<LOW>': 'LOW', '<CLOSE>': 'CLOSE', '<VOL>': 'VOL'}, inplace=True)
        self.historical_data['DATE'] = pd.to_datetime(self.historical_data['DATE'], format='%Y%m%d')
        self.historical_data = self.historical_data.drop(['TICKER', 'PER', 'TIME'], axis=1)
        self.historical_data.columns = ['DATE', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'CLOSE_cur', 'VOL_cur']

    def prepare_moex_data(self):
        self.moex_data.Price = self.moex_data.Price.str.replace(',', '', regex=True)
        self.moex_data.Open = self.moex_data.Open.str.replace(',', '', regex=True)
        self.moex_data.High = self.moex_data.High.str.replace(',', '', regex=True)
        self.moex_data.Low = self.moex_data.Low.str.replace(',', '', regex=True)
        self.moex_data[['Price', 'Open', 'High', 'Low']] = self.moex_data[['Price', 'Open', 'High', 'Low']].apply(pd.to_numeric, downcast='float', errors='coerce')
        self.moex_data['Moex_movement'] = self.moex_data.apply(moex_movement, axis=1)
        self.moex_data['Moex_spread'] = self.moex_data.apply(moex_spread, axis=1)
        self.moex_data = self.moex_data.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
        self.moex_data['Date'] = pd.to_datetime(self.moex_data['Date'])
        self.moex_data.columns = ['DATE', 'Moex_close', 'Moex_movement', 'Moex_spread']

    def prepare_s500_data(self):
        self.sp500_data.Price = self.sp500_data.Price.str.replace(',', '', regex=True)
        self.sp500_data.Open = self.sp500_data.Open.str.replace(',', '', regex=True)
        self.sp500_data.High = self.sp500_data.High.str.replace(',', '', regex=True)
        self.sp500_data.Low = self.sp500_data.Low.str.replace(',', '', regex=True)
        self.sp500_data[['Price', 'Open', 'High', 'Low']] = self.sp500_data[['Price', 'Open', 'High', 'Low']].apply(pd.to_numeric, downcast='float', errors='coerce')
        self.sp500_data['SP500_spread'] = self.sp500_data.apply(SP500_spread, axis=1)
        self.sp500_data['SP500_movement'] = self.sp500_data.apply(SP500_movement, axis=1)
        self.sp500_data = self.sp500_data.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
        self.sp500_data['Date'] = pd.to_datetime(self.sp500_data['Date'])
        self.sp500_data.columns = ['DATE', 'SP500_close', 'SP500_spread', 'SP500_movement']

    def data_prepare(self, currency_current):
        df = pd.DataFrame([currency_current])
        date_str = currency_current["DATE"]
        find_entry_index = self.historical_data[self.historical_data["DATE"] == date_str].index.values[0]
        previous_period_df = self.historical_data.loc[find_entry_index - 12:find_entry_index - 1]
        df = pd.concat([previous_period_df, df])
        df['MOVEMENT_cur'] = df.apply(currency_movement, axis=1)
        df['SPREAD_cur'] = df.apply(currency_spread, axis=1)

        close_df = df.filter(items = ['DATE', 'CLOSE_cur'])
        for i in range(1, 11):
            close_df["lag_{}".format(i)] = close_df.CLOSE_cur.shift(i)
        close_df = close_df.iloc[-3:]
        close_drop_close_df = close_df.drop(['CLOSE_cur', 'DATE'], axis=1)
        data_long = pd.DataFrame({0: close_drop_close_df.values.flatten(),
                          1: np.arange(close_drop_close_df.shape[0]).repeat(close_drop_close_df.shape[1])})
        
        settings_minimal = settings.MinimalFCParameters()
        settings_time = settings.TimeBasedFCParameters()
        settings_time.update(settings_minimal)

        X = extract_features(
            data_long, column_id=1, 
            impute_function=impute, 
            default_fc_parameters=settings_time,
            n_jobs=2
        )
        X = X.drop(['0__length'], axis=1)

        close_df.reset_index(drop=True, inplace=True)
        close_df_and_x = pd.concat([close_df, X], axis=1)

        all_data = df.merge(close_df_and_x.drop(['CLOSE_cur'], axis=1), on='DATE', how='inner')

        all_data['DATE'] = pd.to_datetime(all_data['DATE'])
        all_data = all_data.merge(self.moex_data, on='DATE', how='inner')
        all_data = all_data.merge(self.sp500_data, on='DATE', how='inner')

        half_data = all_data[['DATE', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'MOVEMENT_cur', 'SPREAD_cur',
                     'SP500_close', 'SP500_spread', 'SP500_movement', 'Moex_close', 'Moex_movement', 'Moex_spread']]
        half_data['DATE_shift'] = half_data.DATE.shift(-1)
        half_data = half_data.drop(['DATE'], axis=1)
        half_data[['DATE']] = half_data[['DATE_shift']]
        half_data = half_data.drop(['DATE_shift'], axis=1)
        all_data_drop = all_data.drop(['OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'MOVEMENT_cur', 'SPREAD_cur',
                               'SP500_close', 'SP500_spread', 'SP500_movement', 
                               'Moex_close', 'Moex_movement', 'Moex_spread'], axis=1)

        data_for_classification = all_data_drop.merge(half_data, on='DATE', how='inner')
        data_for_classification = data_for_classification.set_index('DATE')
        X = data_for_classification.dropna().drop(['CLOSE_cur'], axis=1)

        scaler = StandardScaler()
        df_scale = scaler.fit_transform(X[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', '0__sum_values', '0__median', '0__mean', '0__standard_deviation', '0__variance',
                                           '0__root_mean_square', '0__maximum', '0__minimum', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'MOVEMENT_cur', 'SPREAD_cur', 'SP500_close', 'SP500_spread', 'SP500_movement', 'Moex_close', 'Moex_movement', 'Moex_spread']])
        X[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', '0__sum_values', '0__median', '0__mean', '0__standard_deviation', '0__variance', '0__root_mean_square', '0__maximum',
        '0__minimum', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'MOVEMENT_cur', 'SPREAD_cur', 'SP500_close', 'SP500_spread', 'SP500_movement', 'Moex_close', 'Moex_movement', 'Moex_spread']] = df_scale

        data_for_classification['Cur_movement_up_or_down'] = data_for_classification.apply(up_or_down, axis=1)
        data_for_classification['Cur_movement_up_or_down_shift'] = data_for_classification.Cur_movement_up_or_down.shift(-1)
        data_for_classification = data_for_classification.dropna()
        data_for_classification = data_for_classification.drop(['Cur_movement_up_or_down', 'CLOSE_cur', 'MOVEMENT_cur'], axis=1)
        X = data_for_classification.drop(['Cur_movement_up_or_down_shift'], axis=1)
        df_res = scaler.fit_transform(X[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', '0__sum_values', '0__median', '0__mean', '0__standard_deviation', '0__variance',
                                           '0__root_mean_square', '0__maximum', '0__minimum', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'SPREAD_cur', 'SP500_close', 'SP500_spread', 'SP500_movement', 'Moex_close', 'Moex_movement', 'Moex_spread']])
        X[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', '0__sum_values', '0__median', '0__mean', '0__standard_deviation', '0__variance', '0__root_mean_square', '0__maximum',
         '0__minimum', 'OPEN_cur', 'HIGH_cur', 'LOW_cur', 'VOL_cur', 'SPREAD_cur', 'SP500_close', 'SP500_spread', 'SP500_movement', 'Moex_close', 'Moex_movement', 'Moex_spread']] = df_res

        return X

    def predict(self, currency_current):
        df = self.data_prepare(currency_current)
        pred = self.model.predict(df)
        return int(pred[0])

def currency_movement(row):
    return row['CLOSE_cur'] - row['OPEN_cur']

def currency_spread(row):
    return row['HIGH_cur'] - row['LOW_cur']

def moex_movement(row):
    return row['Price'] - row['Open']

def moex_spread(row):
    return row['High'] - row['Low']

def SP500_spread(row):
    return row['High'] - row['Low']

def SP500_movement(row):
    return row['Price'] - row['Open']

def up_or_down(row):
    if row['MOVEMENT_cur'] > 0:
        return 1
    else:
        return 0