import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import multiprocessing
import random
import datetime as dt
import math
import pickle
import random
from dateutil.relativedelta import relativedelta

def Load(storemap, SaveText):
    with open("E:\\hello\\" + storemap + "\\" + SaveText + '.Pkl', 'rb') as Doc:
        return pickle.load(Doc)

class LoadDataV2():
    #This class is used to make stored data easy accessible.
    #
    #
    #The class load data, makes correct size vectors and gives a posibility for data augmentation.

    def __init__(self, StoreLocation, DataSetConfig, ScalingFactor, VerificationScalingFactor, training):
        self.StoreLocation = StoreLocation
        self.DataSetConfig = DataSetConfig
        self.training = training
        self.InputKeys = dict()
        self.InputKeys['eod'] = dict()

        for Colume in self.DataSetConfig['eod'].keys():
            self.InputKeys['eod'][Colume] = len(self.DataSetConfig['eod'][Colume]['Inputshape'])

        self.getInputLen()
        self.getOuputKeys()

        # initialization of data and verificationdata dict
        self.Data = dict()
        self.Data['Input'] = list()
        self.Data['Output'] = dict()
        for colume in self.OutputKeys:
            self.Data['Output'][colume] = list()
        self.VerificationData = dict()
        self.VerificationData['Input'] = list()
        self.VerificationData['Output'] = dict()
        for colume in self.OutputKeys:
            self.VerificationData['Output'][colume] = list()

        #Set magnitude data augmentation
        self.SetScalingFactor(ScalingFactor)
        self.SetVerificationScalingFactor(VerificationScalingFactor)

        #attempt to preload data (does not work)
        #
        #self.ReadyForDataSwap = False
        #self.ThreadGenerateNextDataSet = multiprocessing.Process(target = self.GenerateNextDataSet)
        #self.GenerateNextDataSet()
        #self.CoppyDataForUse()

        #Load general info()
        self.Info = Load(self.StoreLocation, 'Info')

        #initialization of the LastNumber variable. 
        #The LastNumber variable saves the number of the last file used
        self.LastNumber = int(random.random() * self.Info['NumberOfDataFiles'])


    def LoadData(self, FileName):
        return Load(self.StoreLocation, FileName)

    def SetScalingFactor(self, Factor):
        self.ScalingFactor =  math.log10(Factor) 

    def SetVerificationScalingFactor(self, Factor):
        self.VerificationScalingFactor =  math.log10(Factor) 

    def getOuputKeys(self):
        #This function initializes a list with al the keys for the output lists in the dict with the rawdata.
        self.RawData = self.LoadData('DataSet0Train')
        self.OutputKeys = list()
        for key in self.RawData[0]['eod'].keys():
            if not key in self.InputKeys['eod'].keys():
                self.OutputKeys.append(key)
    
    
    def getInputLen(self):
        #There is a singel input vector. This function calculate the length of the vector
        #the length of the EOD part
        NumberOfeodColums = len(self.InputKeys['eod']) - 1
        NumberOfeodInputsPerColum = len(self.DataSetConfig['eod']['open']['Inputshape'])
        LengthEod = NumberOfeodColums * NumberOfeodInputsPerColum

        #the length of the Fundamentals part
        funInputLen = 0
        for key in self.DataSetConfig['fundamentals']['Financials'].keys():
            funInputLen += len(self.DataSetConfig['fundamentals']['Financials'][key]['quarterly'])*2
        LengthFund = ((funInputLen + 1) * self.DataSetConfig['fundamentals']['len']) + 1

        #sum of the 2 parts
        self.InputLeng = LengthEod + LengthFund


    def GenerateDataSet(self, number):
        #Loading data
        if self.training:
            RawData = self.LoadData('DataSet' + str(number) + 'Train')
        else:
            RawData = self.LoadData('DataSet' + str(number) + 'Verification')
        RawVerificationData = self.LoadData('DataSet' + str(number) + 'Verification')

        #Loops thourgh al block of data 
        for block in RawData:
            self.AddBlock(block)
        
        #Loops thourgh al block of Verificationdata 
        for block in RawVerificationData:
            self.AddVerificationBlock(block)

        
        #attempt to preload data (does not work)
        #    
        #self.ReadyForDataSwap = True 


    def InputCreate1(self, block, i):
        #This function makes the input for datapoint i in block
        
        tempInput = list()

        #Putting the data of EOD in a list
        for ColumeForInput in self.InputKeys['eod'].keys():
            tempitem = block['eod'][ColumeForInput][i: i + self.InputKeys['eod'][ColumeForInput]]
            if len(tempitem) != self.InputKeys['eod'][ColumeForInput]:
                return [], True
            tempInput.extend(np.log10(tempitem))

        #Graping the current Date of position i in block
        temp = list()
        try:
            currentDate = dt.date.fromisoformat(block['eod']['date'][i + self.InputKeys['eod']['open'] - 1])
        except:
            currentDate = block['eod']['date'][i + self.InputKeys['eod']['open'] - 1]

        DateFundamentals = currentDate - relativedelta(months=4)
        try:
            for key in block['fundamentals']['Financials'].keys():
                
                #Making a list for al dates needed from fundamentals data
                listdate = list(block['fundamentals']['Financials'][key]['quarterly'].keys())
                listdate.sort(reverse=True)
                if len(listdate) == 0:
                    return [], True
                count = 0
                for date in listdate:
                    
                    #Here the dates in de listdate are compared with the currentDate to check if the dates are the lastdate before the currentDate
                    if dt.date.fromisoformat(date) <= currentDate:
                        if count < self.DataSetConfig['fundamentals']['len']:
                            if not dt.date.fromisoformat(date).timetuple().tm_yday == 31:
                                if DateFundamentals < dt.date.fromisoformat(date):
                                    DateFundamentals = dt.date.fromisoformat(date)
                                count += 1
                                for ItemsInList in self.DataSetConfig['fundamentals']['Financials'][key]['quarterly']:
                                    
                                    #Due to de use of a log function for normalization the values can not be zero or lower
                                    try:
                                        if block['fundamentals']['Financials'][key]['quarterly'][date][ItemsInList] == None:
                                            temp.append(0.001)
                                            temp.append(0.001)
                                        else:
                                            if float(block['fundamentals']['Financials'][key]['quarterly'][date][ItemsInList]) > 0:
                                                temp.append(0.001 + float(block['fundamentals']['Financials'][key]['quarterly'][date][ItemsInList]))
                                                temp.append(0.001)
                                            else:
                                                temp.append(0.001)
                                                temp.append(0.001 - float(block['fundamentals']['Financials'][key]['quarterly'][date][ItemsInList]))

                                    except Exception as e:
                                        return [], True
                        else:
                            break
                
                if count < self.DataSetConfig['fundamentals']['len']:
                    return [], True
        except:
            return [], True
        
        count = 0
        for date in listdate:
            if dt.date.fromisoformat(date) <= currentDate and count < self.DataSetConfig['fundamentals']['len'] and not dt.date.fromisoformat(date).timetuple().tm_yday == 31:
                count += 1
                try:
                    if block['fundamentals']['outstandingShares']['quarterly'][date] == None:
                        temp.append(0.1)
                    else:
                        temp.append(float(block['fundamentals']['outstandingShares']['quarterly'][date]))
                except Exception as e:
                    temp.append(0.1)
        if count < self.DataSetConfig['fundamentals']['len']:
            return [], True
        
        #Normalization
        temp = list(np.log10(temp))

        #Fraction to indicate how far along the currentDate is in the quarter
        fractionBetweenFundamentals = (currentDate - DateFundamentals)/dt.timedelta(days=90)
        temp.append(fractionBetweenFundamentals)

        tempInput.extend(temp)
        
        #Check if the list has the expeced length
        if len(tempInput) != self.InputLeng:
            return [], True
        
        #Value errors?
        ISNAN = any(np.isnan(tempInput))
        ISINF = any(np.isinf(tempInput))
        return tempInput, ISNAN or ISINF
    
    def AddBlock(self, block):
        #This function add a block of data to the Data dict other the Input key

        #if block is empty finish
        if len(block['eod']['open']) == 0:
            return
        
        #loop though all elements of list
        for i in range(len(block['eod'][self.OutputKeys[0]][0])):
            tempInput, ISNAN = self.InputCreate1(block, i)
            if ISNAN:
                continue

            #Add data point to data dict
            self.Data['Input'].append(tempInput)
            for colume in self.OutputKeys:
                self.Data['Output'][colume].append(block['eod'][colume][0][i])

    def AddVerificationBlock(self, block):
        #This function add a block of data to the VerificationData dict onther the Input key

        #loop though all elements of list
        for i in range(len(block['eod'][self.OutputKeys[0]][0])):
            tempInput, ISNAN = self.InputCreate1(block, i)
            if ISNAN:
                break
            #Add data point to verificationdata dict
            self.VerificationData['Input'].append(tempInput)
            for colume in self.OutputKeys:
                self.VerificationData['Output'][colume].append(block['eod'][colume][0][i])


    def GenerateNextDataSet(self):
        #Load 1 file and process the data inside it
        #LastNumber is there to save de last file used
        self.LastNumber = self.LastNumber + 1
        if self.LastNumber >= self.Info['NumberOfDataFiles']:
            self.LastNumber = 0
        self.GenerateDataSet(self.LastNumber)

    def GenerateNextDataSets(self, NumberOfDataSets):
        #this function loads (NumberOfDataSets) and process them in to data set that can be used to train model
        for Number in range(NumberOfDataSets):
            self.GenerateNextDataSet()

