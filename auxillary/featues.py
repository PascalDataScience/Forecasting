
import holidays
import numpy as np
import pandas as pd


def get_temporal_features_(index, features, n_cos_sin_hour=6, n_cos_sin_weekday=42, n_cos_sin_year=2):
    """
    get a df of temporal features specified for an index and the specified features
    """
    temporal_features = pd.DataFrame(index=index)
    if 'hour' in features:
        hour = index.hour
        for i in range(1, int(n_cos_sin_hour) + 1):
            hour_sin = np.sin(i * (2 / 24) * np.pi * hour).astype(float)
            hour_cos = np.cos(i * (2 / 24) * np.pi * hour).astype(float)
            temporal_features['hour_sin_{}'.format(i)] = hour_sin
            temporal_features['hour_cos_{}'.format(i)] = hour_cos

    if 'weekhour' in features:
        hour = index.hour + index.weekday * 24
        for i in range(1, int(n_cos_sin_weekday) + 1):
            hour_sin = np.sin(i * (2 / 24) * np.pi * hour).astype(float)
            hour_cos = np.cos(i * (2 / 24) * np.pi * hour).astype(float)
            temporal_features['weekhour_sin_{}'.format(i)] = hour_sin
            temporal_features['weekhour_cos_{}'.format(i)] = hour_cos

    if 'weekday' in features:
        weekday = index.weekday
        weekdays = pd.get_dummies(weekday, prefix='weekday').astype(float)
        weekdays.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, weekdays], axis=1)

    if 'month' in features:
        month = index.month
        months = pd.get_dummies(month, prefix='month').astype(float)
        months.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, months], axis=1)

    if 'dayofyear' in features:
        dayofyear = index.dayofyear
        for i in range(1, int(n_cos_sin_year) + 1):
            dayofyear_sin = np.sin(i * (2 / 365) * np.pi * dayofyear).astype(float)
            dayofyear_cos = np.cos(i * (2 / 365) * np.pi * dayofyear).astype(float)
            temporal_features['dayofyear_sin_{}'.format(i)] = dayofyear_sin
            temporal_features['dayofyear_cos_{}'.format(i)] = dayofyear_cos

    if 'holiday' in features:
        ch_holidays = holidays.CountryHoliday("CH")
        # ch_holidays = CountryHoliday('CH')
        # holiday = pd.Series(index.date).apply(lambda x: x in ch_holidays)
        # temporal_features['holiday'] = holiday.astype(int)
        temporal_features['holiday'] = pd.Series(index, index=index, name='holiday').apply(
            lambda x: x in ch_holidays).astype(int)
    if 'tz_shift' in features:
        """
        if there is a time shift (CET <-> CEST) then all the timestamp at this day = 1, no shift at this day = 0 
        """
        df_tz_info = pd.DataFrame()
        df_iteration = pd.DataFrame(data=[np.nan] * len(index), index=index).groupby(pd.Grouper(freq='D'))
        for key, daily_values in df_iteration:
            lst_tzinfo = [timestamp.tzinfo for timestamp in daily_values.index]
            if (all(tz_info == lst_tzinfo[0] for tz_info in lst_tzinfo)):
                df_tz_append = pd.DataFrame(data=[0] * len(daily_values), index=daily_values.index)
            else:
                df_tz_append = pd.DataFrame(data=[1] * len(daily_values), index=daily_values.index)
            df_tz_info = pd.concat([df_tz_info, df_tz_append], axis=0)
        df_tz_info.rename(columns={0: 'tz_shift'}, inplace=True)
        temporal_features['tz_shift'] = df_tz_info
    return temporal_features