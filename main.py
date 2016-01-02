import logging
import tushare as ts
import numpy as np
import pandas as pd
import datetime
import pymongo

from pymongo import MongoClient

logging.basicConfig(filename='maincompute.log', level=logging.DEBUG)


import time
class Timer(object):
	def __init__(self, verbose=False):
		self.verbose = verbose
 
	def __enter__(self):
		self.start = time.time()
		return self
 
	def __exit__(self, *args):
		self.end = time.time()
		self.secs = self.end - self.start
		self.msecs = self.secs * 1000  # millisecs
		if self.verbose:
			print 'elapsed time: %f ms' % self.msecs


class dbManager:

	def __init__(self, host='localhost', port=27017):
		self._dbclient = MongoClient(host, port)

	def connectDB(self, dbName):
		self._db = self._dbclient[dbName]

	@property
	def DB(self):
	    return self._db
	
	def insert(self, colName, doc):
		_col = self._db[colName]
		result = None
		if isinstance(doc, dict):
			result = _col.insert_one(doc)
		else:
			result = _col.insert_many(doc)

		return result

def get_stock_his_day_Data(code, startDay, endDay):###generator for the stock data share by year
	df = ts.get_stock_basics()
	tmDate = df.ix[code]['timeToMarket']

	if '-' in startDay:
		_d = startDay.split('-')
		startDay = _d[0]+_d[1]+_d[2]

	if '-' in endDay:
		_d = endDay.split('-')
		endDay = _d[0]+_d[1]+_d[2]

	if not isinstance(startDay, np.int64):
		startDay = np.int64(startDay)
	if not isinstance(endDay, np.int64):
		endDay = np.int64(endDay)

	if startDay < tmDate:
		startDay = tmDate

	today = np.int64( str(datetime.date.today()).replace('-','') )

	if endDay > today:
		endDay = today
 
 	#search by year, for the reliability
 	nyears = endDay/10000 - startDay/10000 + 1
 	sstartDay, sendDay = str(startDay), str(endDay)
	for nyear in xrange(startDay/10000,endDay/10000+1):
		tmpStart = sstartDay[0:4]+'-'+sstartDay[4:6]+'-'+sstartDay[6:8] if nyear==startDay/10000 else str(nyear)+'-01-01'
		tmpEnd = sendDay[0:4]+'-'+sendDay[4:6]+'-'+sendDay[6:8] if nyear==(endDay/10000) else str(nyear)+'-12-31'
		logging.debug("get code:%s history data from %s to %s" %(code, tmpStart, tmpEnd))
		tmpdata = ts.get_h_data(code, start=tmpStart, end=tmpEnd)
		yield(tmpdata)


class stockAccount:
    def __init__(self, cash=1000000):
    	self._initCash = cash
        self._cash = cash #initial money
        self._stock_account = dict() #stock share {code:account}
        self._shareValues = dict()
        self._stockValue = 0
    
    @property
    def cash(self):
        return self._cash

    @property
    def stock_account(self):
        return self._stock_account
    
    def buy(self, code, cashcount, price):
    	amount = int(int(cashcount/price)/100) *100
    	self._buy(code, amount, price)

    def _buy(self, code, amount, price):
        if code in self._stock_account:
            self._stock_account[code] += amount
        else:
            self._stock_account[code] = amount
        self._cash -= amount * price
        self._shareValues[code] = price

    def sell(self, code, amount, price):
        self._stock_account[code] -= amount
        if self._stock_account[code]==0:
            del self._stock_account[code]

        self._cash += amount * price
        self._shareValues[code] = price


    def updateStockValue(self, **shareValues): 
        self._shareValues[shareValues['code']] = shareValues['price']

        self._stockValue = 0
        for _c, _a in self._stock_account.items():
            self._stockValue += self._shareValues[_c] * _a

    def marketValue(self):
        self._stockValue = 0
        for _c, _a in self._stock_account.items():
            self._stockValue += self._shareValues[_c] * _a
        return self._stockValue + self._cash

    def accountChangeRate(self):
        return round(self.marketValue()/float(self._initCash),3)

@profile
def alphaOneStockCompute(code, startDay, endDay, db):
	#warining:  this version only handle the day trade data!!!
	
	###algrithm for computing the alpha value for one stock share

	#the stock_data column for the day and tick is different
	#the tick data columns are 'time like [13:15:10] price change volume amount type', the index is [0,1,2...]
	#the day data columns are 'open high close low volume amount', the index is date like '2015-01-01'
	stock_data = fill_DB_day_data(code, startDay, endDay, db)# make sure all the data has kept in the db
	
	if stock_data is None or stock_data.count()==0:
		return

	isDaysTrade = False
	if 'open' in stock_data[0]:
		isDaysTrade = True

	#seperate the cash into 5 parts, sell and buy the stock when condition meets 
	TOTAL_PARTS = 5
	cashPart, stockPart = TOTAL_PARTS,0 #use cash as 5 parts

	initTrade, initprice = stock_data[stock_data.count()-1], 0
	if isDaysTrade:
		initprice = initTrade['open']
	else:
		initprice = initTrade['price']
	if isDaysTrade:
		curtimestamp = initTrade['date']
		cur_time = str(datetime.date(curtimestamp.year, curtimestamp.month, curtimestamp.day)) 
	else:
		cur_time = initTrade['time'] 

	valueMatrix = pd.DataFrame(index=range(1,11), columns=range(1,11))
	
	for _up in range(1,11):
		for _down in range(1,11):
			cashPart, stockPart = 3,2 #use 2/5 cash to initiate the customer account
			customerAccount = stockAccount()	
			policyValue = pd.DataFrame(columns=['time', 'operation','targetprice','tradeprice','totalrate'])

			#add the inital operation policy value
			for _ in range(stockPart):#record the inital trade
				customerAccount.buy(code,  customerAccount.cash/TOTAL_PARTS, initprice)
				policyValue = policyValue.append({'time':cur_time,'operation':1,'targetprice':initprice, 'tradeprice':initprice, 'totalrate':1},ignore_index=True)
						
			#policy(_up,_down) matrix, including columns: time, operation, totalValue
			up_price = initprice * (100+_up) / 100
			down_price = initprice * (100-_down)/100
			# the default order of the stock_data matrix is descending;
			for _i in range(stock_data.count()-2, -1, -1):
				trade_item = stock_data[_i]
				cur_price = trade_item['close'] if isDaysTrade else trade_item['price']
				if isDaysTrade:
					curtimestamp = trade_item['date']
					cur_time = str(datetime.date(curtimestamp.year, curtimestamp.month, curtimestamp.day)) 
				else:
					cur_time = trade_item['time'] 

				if cur_price >= up_price:
					if stockPart > 0:# have stock to sell
						customerAccount.sell(code, customerAccount.stock_account[code]/stockPart,cur_price)
						cur_totalValue = customerAccount.accountChangeRate()
						logging.debug("[trade record] sell 1 stock part, stockPart=%i, cashPart=%i at price:%f" %(stockPart, cashPart,cur_price))
						policyValue = policyValue.append({'time':cur_time,'operation':-1,'targetprice':up_price, 'tradeprice':cur_price, 'totalrate':cur_totalValue},ignore_index=True)
						stockPart -= 1
						cashPart += 1
						up_price = cur_price * (100+_up) / 100
						down_price = cur_price * (100-_down)/100
					else:
						pass
				elif cur_price <= down_price:
					if cashPart > 0:#have cash to buy
						customerAccount.buy(code, customerAccount.cash/cashPart, cur_price)
						cur_totalValue = customerAccount.accountChangeRate()
						logging.debug("[trade record] buy 1 stock part, stockPart=%i, cashPart=%i at price:%f" %(stockPart, cashPart,cur_price))
						policyValue = policyValue.append({'time':cur_time,'operation':1, 'targetprice':down_price, 'tradeprice':cur_price,'totalrate':cur_totalValue},ignore_index=True)
						stockPart += 1
						cashPart -= 1
						up_price = cur_price * (100+_up) / 100
						down_price = cur_price * (100-_down)/100
					else:
						pass

			#update the market value on the last day of the account
			trade_item = stock_data[0]#last trading day
			if isDaysTrade:
				curtimestamp = trade_item['date']
				cur_time = str(datetime.date(curtimestamp.year, curtimestamp.month, curtimestamp.day)) 
			else:
				cur_time = trade_item['time'] 
			cur_price = trade_item['close'] if isDaysTrade else trade_item['price']
			customerAccount.updateStockValue(code=code, price=cur_price)
			cur_totalValue = customerAccount.accountChangeRate()
			policyValue = policyValue.append({'time':cur_time,'operation':0, 'targetprice':cur_price, 'tradeprice':cur_price,'totalrate':cur_totalValue},ignore_index=True)
			#add policy value into the policy matrix
			valueMatrix.iat[_up-1, _down-1] = policyValue

	return valueMatrix


def computeBestpolicy(yieldValue):
	maxRate = 1
	policy_up, policy_down = 0, 0
	for _i in range(10):
		for _j in range(10):
			policyMatrix = yieldValue.iat[_i, _j]
			finalRate = policyMatrix.tail(1).loc[policyMatrix.index[len(policyMatrix.index)-1],'totalrate'] if len(policyMatrix.index)>0 else 1
			if finalRate>maxRate:
				maxRate = finalRate
				policy_up, policy_down = _i+1, _j+1
			else:
				pass
	#print maxRate, policy_up, policy_down
	#print yieldValue.iat[policy_up-1, policy_down-1]


def fill_DB_day_data(code, startDay, endDay, db):
	'''
	fill the trade day data into the database, download the lacked part from the network(by tushare)
	
	return the dataframe of the day trade data (startDay->endDay) 
	'''
	collectionName = '%s_his_data'%(code)
	if '-' in startDay:
		_day = startDay.split('-')
		startDay = _day[0]+_day[1]+_day[2]

	if '-' in endDay:
		_day = endDay.split('-')
		endDay = _day[0]+_day[1]+_day[2]


	startDay = datetime.datetime(int(startDay[0:4]),int(startDay[4:6]),int(startDay[6:8]))
	endDay = datetime.datetime(int(endDay[0:4]),int(endDay[4:6]),int(endDay[6:8]))

	result = db.DB[collectionName].find({'$and':[{'date': {'$gte':startDay}},{'date': {'$lte':endDay}}]}).sort('date', pymongo.DESCENDING)

	if result.count()>0:
		resultEndDay =  result[0]['date']
		resultStartDay =  result[result.count()-1]['date']
	else:
		#download from network, break
		download_his_day_data(code, str(startDay)[0:10], str(endDay)[0:10], db)
		fullresult = db.DB[collectionName].find({'$and':[{'date': {'$gte':startDay}},{'date': {'$lte':endDay}}]}).sort('date', pymongo.DESCENDING)
		return fullresult

	resultEndDay =  result[0]['date']
	resultStartDay =  result[result.count()-1]['date']
	#fill the gap between database and the query range period by downloading from network(tushare api)
	if startDay < resultStartDay:
		deltaOneDay = datetime.timedelta(days=1)
		newEnd = str(resultStartDay - deltaOneDay)[0:10]	
		print 'download %s - %s' %(str(startDay)[0:10], newEnd)   
		download_his_day_data(code, str(startDay)[0:10], newEnd, db)

	if endDay > resultEndDay:
		deltaOneDay = datetime.timedelta(days=1)
		newStart = str(resultEndDay + deltaOneDay)[0:10]	
		print 'download %s - %s' %(newStart, str(endDay)[0:10])     
		download_his_day_data(code, newStart, str(endDay)[0:10], db)	

	#return the dataframe of the queried trade day data 
	fullresult = db.DB[collectionName].find({'$and':[{'date': {'$gte':startDay}},{'date': {'$lte':endDay}}]}).sort('date', pymongo.DESCENDING)
	return fullresult


def download_his_day_data(code, startDay, endDay, db):
	#download from the network and insert into database
	
	collectionName = '%s_his_data'%(code)
	for partData in get_stock_his_day_Data(code, startDay, endDay):#search by year
		if partData is None:
			break;
		#partData.to_csv('%s.csv'%(collectionName), mode='a')##back up in the csv files
		
		#for the data download from network is descending
		#db.DB[collectionName].create_index([('date', pymongo.DESCENDING)], unique=True)
		for _i, date in enumerate(partData.index.date):
			##convert type datetime.date to datetime.datetime for mongodb only konws how to encode datetime.datetime
			date = datetime.datetime(date.year,date.month,date.day) 
			dayData = dict([('date', date)] + 
				[(partData.columns[_j],partData.iloc[_i][_j]) for _j in range(len(partData.columns))])
			try:
				db.insert(collectionName, dayData)##insert the year data into the db
			except Exception as e:
				logging.warning("insert into db %s failed, %s" %(collectionName, str(e)))	

if __name__ == '__main__':

	#connect to database
	db = dbManager()
	db.connectDB('stock_data')

	#analyze the code and related period
	code, startDay, endDay = '600000', '20150129','20150808'
	#for the data download from network is descending
	collectionName = '%s_his_data'%(code)
	db.DB[collectionName].create_index([('date', pymongo.DESCENDING)], unique=True)
	with Timer(True) as t:
		yieldValue = alphaOneStockCompute(code, startDay, endDay, db)
	computeBestpolicy(yieldValue)

			