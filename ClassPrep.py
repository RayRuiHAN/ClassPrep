# BS function
import math
import pandas as pd
from ClassBSM import BSM


class IOPrep(object):

    def __init__(self, file):
        """
        Fields needed: TradingDate, CallOrPut, StrikePrice, ClosePrice, UnderlyingScrtClose, RemainingTerm, RisklessRate
        file: file path. The file needs to contain only one time period
        data1,2: StrikePrice, IV_Merge, ClosePrice_Merge
        data_iv: IV1, IV2, StdIV(Monthly Standardize Volatility)
        info: terms, t1 t2, S, r
        """

        self.fields = ['TradingDate', 'CallOrPut', 'StrikePrice', 'ClosePrice', 'UnderlyingScrtClose', 'RemainingTerm',
                       'RisklessRate']
        df = pd.read_csv(file).loc[:, self.fields]
        TradingDate = df.loc[0, 'TradingDate']
        self.S = df.loc[0, 'UnderlyingScrtClose']
        self.r = math.log(1 + df.loc[0, 'RisklessRate'] / 100)
        self.df = df
        self.terms = self.df.RemainingTerm.unique()
        if self.terms[1] < 5 / 365:
            self.t1 = self.terms[1]
            self.t2 = self.terms[2]
        else:
            self.t1 = self.terms[0]
            self.t2 = self.terms[1]
        df['IV'] = df.apply(
            lambda x: BSM(asset_price=self.S, call_put=x.CallOrPut, exercise_price=x.StrikePrice,
                          remaining=x.RemainingTerm, rf_rate=self.r).vol(x.ClosePrice),
            axis=1)
        self.data1 = df.loc[df['RemainingTerm'] == self.t1, ['CallOrPut', 'StrikePrice', 'ClosePrice', 'IV']]
        self.data2 = df.loc[df['RemainingTerm'] == self.t2, ['CallOrPut', 'StrikePrice', 'ClosePrice', 'IV']]

        df_1iv = df.loc[
            (df['RemainingTerm'] == self.t1) & (df['IV'] != '-1'), ['CallOrPut', 'StrikePrice', 'ClosePrice', 'IV']]
        df_2iv = df.loc[
            (df['RemainingTerm'] == self.t2) & (df['IV'] != '-1'), ['CallOrPut', 'StrikePrice', 'ClosePrice', 'IV']]

        df_1iv_c = df_1iv.loc[df['CallOrPut'] == 'C', ['StrikePrice', 'ClosePrice', 'IV']]
        df_1iv_p = df_1iv.loc[df['CallOrPut'] == 'P', ['StrikePrice', 'ClosePrice', 'IV']]
        df_2iv_c = df_2iv.loc[df['CallOrPut'] == 'C', ['StrikePrice', 'ClosePrice', 'IV']]
        df_2iv_p = df_2iv.loc[df['CallOrPut'] == 'P', ['StrikePrice', 'ClosePrice', 'IV']]
        df_1iv_c.columns = ['StrikePrice', 'ClosePrice_C', 'IV_C']
        df_1iv_p.columns = ['StrikePrice', 'ClosePrice_P', 'IV_P']
        df_2iv_c.columns = ['StrikePrice', 'ClosePrice_C', 'IV_C']
        df_2iv_p.columns = ['StrikePrice', 'ClosePrice_P', 'IV_P']

        # Order by StrikePrice
        df_1k = df_1iv_c.merge(df_1iv_p, how='inner', on='StrikePrice')
        df_2k = df_2iv_c.merge(df_2iv_p, how='inner', on='StrikePrice')
        df_1k['ClosePriceDiff'] = df_1k.apply(lambda x: abs(x.ClosePrice_C - x.ClosePrice_P), axis=1)
        df_2k['ClosePriceDiff'] = df_2k.apply(lambda x: abs(x.ClosePrice_C - x.ClosePrice_P), axis=1)

        f_idx1 = df_1k['ClosePriceDiff'].idxmin()
        f_idx2 = df_2k['ClosePriceDiff'].idxmin()
        f1 = df_1k.loc[f_idx1, 'StrikePrice'] + math.exp(self.r * self.t1) * df_1k.loc[f_idx1, 'ClosePriceDiff']
        f2 = df_2k.loc[f_idx2, 'StrikePrice'] + math.exp(self.r * self.t2) * df_2k.loc[f_idx2, 'ClosePriceDiff']

        k0_1 = df_1k.loc[df_1k.StrikePrice < f1, 'StrikePrice'].max()
        k0_2 = df_2k.loc[df_2k.StrikePrice < f2, 'StrikePrice'].max()

        def iv_merge(x, k0):
            if x.StrikePrice == k0:
                return (x.IV_C + x.IV_P) / 2
            if x.StrikePrice < k0:
                return x.IV_P
            if x.StrikePrice > k0:
                return x.IV_C

        df_1k['IV_Merge'] = df_1k.apply(lambda x: iv_merge(x, k0_1), axis=1)
        df_2k['IV_Merge'] = df_2k.apply(lambda x: iv_merge(x, k0_2), axis=1)

        def price_merge(x, k0):
            if x.StrikePrice == k0:
                return (x.ClosePrice_C + x.ClosePrice_P) / 2
            if x.StrikePrice < k0:
                return x.ClosePrice_P
            if x.StrikePrice > k0:
                return x.ClosePrice_C

        df_1k['ClosePrice_Merge'] = df_1k.apply(lambda x: price_merge(x, k0_1), axis=1)
        df_2k['ClosePrice_Merge'] = df_2k.apply(lambda x: price_merge(x, k0_2), axis=1)

        # Monthly Standardize Volatility
        df1 = df_1k.loc[:, ['StrikePrice', 'IV_Merge']]
        df2 = df_2k.loc[:, ['StrikePrice', 'IV_Merge']]
        df1.columns = ['StrikePrice', 'IV1']
        df2.columns = ['StrikePrice', 'IV2']
        df_iv = df1.merge(df2, how='inner', on='StrikePrice')

        def Std_iv(x):
            w1 = (self.t2 - 1 / 12) / (self.t2 - self.t1)
            w2 = -(self.t1 - 1 / 12) / (self.t2 - self.t1)
            a = math.sqrt((self.t1 * w1 * x.IV1 * x.IV1 + self.t2 * w2 * x.IV2 * x.IV2)*12)
            return a

        df_iv['StdIV'] = df_iv.apply(lambda x: Std_iv(x), axis=1)

        self.data1 = df_1k.loc[:, ['StrikePrice', 'IV_Merge', 'ClosePrice_Merge']]
        self.data2 = df_2k.loc[:, ['StrikePrice', 'IV_Merge', 'ClosePrice_Merge']]
        self.data_iv = df_iv
        self.info = 'Trading Date:\t' + str(TradingDate) + '\nTerm list:\t\t' + str(self.terms) \
                    + '\nTerm Chosen:\t' + str(self.t1) + ' ' + str(self.t2) + '\nUAsset Price:\t' + str(self.S) \
                    + '\nRisk-free Rate:\t' + str(self.r) + '\n'


if __name__ == '__main__':
    Data = IOPrep("20220228.csv")
    print(Data.info)
    print(Data.data1)
    print(Data.data2)
    print(Data.data_iv)
