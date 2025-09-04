
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pandas as pd
from FiinQuantX import FiinSession
import datetime



username = 'DSTC_36@fiinquant.vn'
password = 'Fiinquant0606'

client = FiinSession(
    username=username,
    password=password,
).login()


tickers = ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']
    
data = client.Fetch_Trading_Data(
    realtime = False,
    tickers = tickers,    
    fields = ['open', 'high', 'low', 'close', 'volume'],
    adjusted=True,
    by = '1d', 
    from_date='2022-9-01 09:00',
    to_date='2025-9-01 09:00'
).get_data()

data.to_csv('DSTC_3y.csv', index=False)

