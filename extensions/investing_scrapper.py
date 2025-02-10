from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

"""
bold act blackFont
bold act greenFont
bold act redFont
"""


def news_verification():
    r = Request('https://pl.investing.com/economic-calendar/', headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(r).read()
    soup = BeautifulSoup(response, "html.parser")
    table = soup.find_all(class_="js-event-item")

    result = []
    base = {}
    free_day = []

    for bl in table:
        time = bl.find(class_ ="first left time js-time").text
        evento = bl.find(class_ ="left event").text
        evento = " ".join(evento.split())
        currency = bl.find(class_ ="left flagCur noWrap").text.split(' ')
        intensity = bl.find_all(class_="left textNum sentiment noWrap")
        id_hour = currency[1] + '_' + time

        if not id_hour in base:
            base.update({id_hour : {'currency' : currency[1], 'time' : time, 'evento':evento ,
                                    'intensity' : {"low": 0, "mid": 0, "high": 0} } })

        intencity = base[id_hour]['intensity']

        for intense in intensity:
            _true = intense.find_all(class_="grayFullBullishIcon")
            _false = intense.find_all(class_="grayEmptyBullishIcon")

            if len(_true) == 1:
                intencity['low'] += 1
            elif 'CTFC' in evento:
                intencity['low'] += 1
            elif len(_true) == 2:
                intencity['mid'] += 1
            elif len(_true) == 3:
                intencity['high'] += 1

        base[id_hour].update({'intensity': intencity})

    for b in base:
        result.append(base[b])

    return result


def df_calendar(news):
    df = pd.read_json(json.dumps(news))
    # df.set_index('currency', inplace=True)

    mapper = {
        'low': lambda x: x['low'],
        'mid': lambda x: x['mid'],
        'high': lambda x: x['high']
    }

    for i in mapper.keys():
        df[i] = df['intensity'].apply(mapper[i])
    df = df.drop(columns=['intensity'], axis=0)
    return df


def blocker(df):
    return df[df['high'] > 1]['currency'].to_list()


class Scraper:
    def __init__(self):
        self.df = df_calendar(news_verification())

    def give_me_economics(self):
        df = self.df.copy()
        df['sum'] = df['mid'] + df['high'] *3
        try:
            sum_ = df[df['time'] == '20:00']['sum'].iloc[-1]
            if sum_ > 10:# or 'Oświadczenie FOMC' in df['evento'].to_list():
                return True
        except IndexError:
            pass
        return False

    def give_me_hardcore_hours(self):
        df = self.df.copy()
        df = df[(df['mid']>1)|(df['high']!=0)]
        try:
            df['sum'] = df['mid'] + df['high'] *3
            print(df)
            sum_ = df[df['sum'] > 10]
            hours = sum_['time'].to_list()
            print(f"Hardcore hours: {hours}")
            hours = [pd.to_datetime(i) for i in hours]
        except Exception:
            return []
        return hours

if __name__=='__main__':
    import icecream as ic
    from datetime import datetime as dt
    from datetime import timedelta as td
    #get_me_economics()
    scrapper = Scraper()
    hours = scrapper.give_me_hardcore_hours()
    def too_much_risk():
        test = [i - td(minutes=15) < dt.now() < i + td(minutes=60) for i in hours]
        if any(test):
            return True
        return False
    
    print(too_much_risk())
        