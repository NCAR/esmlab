import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

import esmlab

data1 = np.arange(30, dtype='float32').reshape(6, 5)
data1 = np.where(data1 < 4, np.nan, data1)
data2 = np.linspace(start=0, stop=100, num=30, dtype='float32').reshape(6, 5)
wgts1 = np.arange(30, 0, -1).reshape(6, 5)
wgts2 = np.arange(6, 0, -1)
wgts3 = np.linspace(start=12, stop=14, num=5, dtype='int32')
time = pd.date_range(start='2000', freq='1D', periods=6)
state = ['CO', 'CA', 'NH', 'MA', 'WA']

da1 = xr.DataArray(
    data1,
    dims=['time', 'state'],
    coords={'time': time, 'state': state},
    name='da1',
    attrs={'units': 'ampere'},
)
da2 = xr.DataArray(data2, dims=['time', 'state'], coords={'time': time, 'state': state}, name='da2')
dset = xr.Dataset(
    data_vars={'da1': da1, 'da2': da2},
    attrs={'year_recorded': datetime.datetime.now().year, 'creator': 'foo', 'reviewer': 'bar'},
)


wgts = xr.Dataset(
    data_vars={
        't_s_wgts': xr.DataArray(wgts1, dims=['time', 'state']),
        't_wgts': xr.DataArray(wgts2, dims=['time']),
        's_wgts': xr.DataArray(wgts3, dims=['state']),
    },
    coords={'time': time, 'state': state},
)


def wavg(x, weights, col_names):
    def _np_ma_avg(data, weights):
        ma = np.ma.MaskedArray(data, mask=np.isnan(data))
        np_w_mean = np.ma.average(ma, axis=0, weights=weights)
        return np_w_mean

    ds = []
    for col in col_names:
        ds.append(_np_ma_avg(x[col], weights))

    return pd.Series(ds, index=x.columns)


@pytest.mark.parametrize(
    'dim, level, wgts_name', [(['time'], ['state'], 't_wgts'), (['state'], ['time'], 's_wgts')]
)
def test_weighted_sum(dim, level, wgts_name):
    df = dset.to_dataframe()
    df_w = wgts.to_dataframe()
    res = esmlab.weighted_sum(dset, dim=dim, weights=wgts[wgts_name]).to_dataframe()
    expected = df.multiply(df_w[wgts_name], axis='index').sum(level=level)
    assert_frame_equal(res.sort_index(), expected.sort_index())


@pytest.mark.parametrize(
    'dim, level, wgts_name', [(['time'], ['state'], 't_wgts'), (['state'], ['time'], 's_wgts')]
)
def test_weighted_mean(dim, level, wgts_name):
    res = esmlab.weighted_mean(dset, dim=dim, weights=wgts[wgts_name]).to_dataframe()
    df = dset.to_dataframe()
    expected = df.groupby(level=level).apply(
        wavg, weights=wgts[wgts_name].data, col_names=['da1', 'da2']
    )
    assert_frame_equal(res.sort_index(), expected.sort_index())


@pytest.mark.parametrize(
    'dim, level, wgts_name', [(['time'], ['state'], 't_wgts'), (['state'], ['time'], 's_wgts')]
)
def test_weighted_std(dim, level, wgts_name):
    res = esmlab.weighted_std(dset, dim=dim, weights=wgts[wgts_name]).to_dataframe()
    df = dset.to_dataframe()
    df_w = wgts.to_dataframe()
    df_w_mean = df.groupby(level=level).apply(
        wavg, weights=wgts[wgts_name].data, col_names=['da1', 'da2']
    )
    temp_df = (df - df_w_mean) ** 2
    temp_df = temp_df.multiply(df_w[wgts_name], axis='index').sum(level=level)
    total_weights_da1 = df_w[df['da1'].notnull()][wgts_name].sum(level=level)
    total_weights_da2 = df_w[df['da2'].notnull()][wgts_name].sum(level=level)
    expected = pd.DataFrame(columns=res.columns)
    expected['da1'] = np.sqrt(temp_df['da1'] / total_weights_da1)
    expected['da2'] = np.sqrt(temp_df['da2'] / total_weights_da2)
    assert_frame_equal(res.sort_index(), expected.sort_index())


@pytest.mark.parametrize(
    'dim, level, wgts_name', [(['time'], ['state'], 't_wgts'), (['state'], ['time'], 's_wgts')]
)
def test_weighted_rmsd(dim, level, wgts_name):
    res = esmlab.weighted_rmsd(
        x=dset['da1'], y=dset['da2'], dim=dim, weights=wgts[wgts_name]
    ).to_dataframe('rmsd')
    df = dset.to_dataframe()
    dev = (df['da1'] - df['da2']) ** 2
    dev = dev.to_frame(name='rmsd')
    dev_mean = dev.groupby(level=level).apply(
        wavg, weights=wgts[wgts_name].data, col_names=['rmsd']
    )
    expected = np.sqrt(dev_mean)
    assert_frame_equal(res.sort_index(), expected.sort_index())


@pytest.mark.parametrize(
    'dim, level, wgts_name', [(['time'], ['state'], 't_wgts'), (['state'], ['time'], 's_wgts')]
)
def test_weighted_cov(dim, level, wgts_name):
    res = esmlab.weighted_cov(
        x=dset['da1'], y=dset['da2'], dim=dim, weights=wgts[wgts_name]
    ).to_dataframe('cov')
    df = dset.to_dataframe()
    means = df.groupby(level=level).apply(
        wavg, weights=wgts[wgts_name].data, col_names=['da1', 'da2']
    )
    dev_da1 = df['da1'] - means['da1']
    dev_da2 = df['da2'] - means['da2']
    dev_da1_da2 = (dev_da1 * dev_da2).to_frame('cov')
    expected = dev_da1_da2.groupby(level=level).apply(
        wavg, weights=wgts[wgts_name].data, col_names=['cov']
    )
    assert_frame_equal(res.sort_index(), expected.sort_index())