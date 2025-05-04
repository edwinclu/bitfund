import datetime
from ib_insync import *

# util.startLoop()  # uncomment this line when in a notebook
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1, readonly=True)

contract = Stock('TSLA', 'SMART', 'USD')
BEGIN_DATE_TIME = datetime.datetime(2019, 9, 1)
END_DATE_TIME = datetime.datetime(2020, 9, 1)
bars = ib.reqHistoricalData(
    contract,
    endDateTime=END_DATE_TIME,
    durationStr='180 D',
    barSizeSetting='1 day',
    whatToShow='MIDPOINT',
    useRTH=True,
    formatDate=1)
df = util.df(bars)
print(df[['date', 'open', 'high', 'low', 'close']])

ib.disconnect()
