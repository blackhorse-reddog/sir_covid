# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:09:39 2020


"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


class SIR(object):
    def __init__(self, province, country, N, modulator=5, startdate='', enddate='', length=14, incubation=5):
        self.province = province
        self.country = country
        self.N=N
        #base datas: length of infection period, length of incubation period
        self.length=length
        self.incubation=incubation
        self.modulator=modulator
        
        #the only variable in sir while length is known - R=beta/gamma
        self.beta=0.1

        #data sets
        #confirmed: all confirmed infection - terms in sir it is recovered+infected
        #recovered+death: recovered for sir (sorry for the inhumanity of math)
        df_confirmed = pd.read_csv('time_series_2019-ncov-Confirmed.csv')
        df_recovered = pd.read_csv('time_series_2019-ncov-Recovered.csv')
        df_deaths = pd.read_csv('time_series_2019-ncov-Deaths.csv')
        #country_data = df[df['Country/Region'] == country]
        province_used=True
        country_data = df_confirmed.loc[(df_confirmed['Country/Region'] == country) & (df_confirmed['Province/State'] == province)]
        rows=country_data.shape[0]
        if rows==0:
            province_used=False
            country_data = df_confirmed.loc[df_confirmed['Country/Region'] == country]
        
        #search the first and last key of the data
        index=0
        startkey=''
        endkey=''
        
        for key, value in country_data.iteritems(): 
            if index>3: #first 4 columns are: Province/State	Country/Region	Lat	Long
                if key==startdate:
                    startkey=key
                if int(value)>0 and startkey=='':
                    startkey=key
                if key==enddate:
                    endkey=key
            index+=1
        if endkey=='':
            original_confirmed=country_data.loc[:,startkey:]
            original_recovered=df_recovered.loc[(df_recovered['Country/Region'] == country) & 
                                                     (df_recovered['Province/State'] == province) 
                                                     if province_used else df_recovered['Country/Region'] == country,startkey:]
            original_deaths=df_deaths.loc[(df_deaths['Country/Region'] == country) & 
                                                     (df_deaths['Province/State'] == province) 
                                                     if province_used else df_deaths['Country/Region'] == country,startkey:]
        else:
            original_confirmed=country_data.loc[:,startkey:endkey]
            original_recovered=df_recovered.loc[(df_recovered['Country/Region'] == country) & 
                                                     (df_recovered['Province/State'] == province) 
                                                     if province_used else df_recovered['Country/Region'] == country,startkey:endkey]
            original_deaths=df_deaths.loc[(df_deaths['Country/Region'] == country) & 
                                                     (df_deaths['Province/State'] == province) 
                                                     if province_used else df_deaths['Country/Region'] == country,startkey:endkey]
        #modulated data
        self.mod_confirmed=[]
        self.mod_original=[]
        #modulated data+expected infected persons in incubation phase
        self.incubator=[]
        self.indexes=[]
        self.mod_recovered=[]

        #self.incubator=original_confirmed.values
        #dont want ndarray array
        for index in range(original_recovered.shape[1]):
            local_recovered=(original_recovered.iloc[0,index]+original_deaths.iloc[0,index])*self.modulator
            local_confirmed=original_confirmed.iloc[0,index]*self.modulator
            self.mod_recovered.append(local_recovered)
            self.mod_confirmed.append(local_confirmed)
            self.mod_original.append(local_confirmed-local_recovered)
            self.incubator.append(0)
            
        for key, value in original_confirmed.iteritems():
            self.indexes.append(key)
                    
        #temporary sir data for incubation estimation
        self.sir_data=[]
        
        self.ivp_method='RK45'
        self.minimize_method='nelder-mead' #while we have just a failure function nelder-mead or Powell are appropiate
        
        self.first_detection=''
        self.today=''
        self.today_num=len(self.mod_original)
        
        self.mod_infected=0
        self.mod_removed=0
        self.saved_predicted_removed=np.arange(0, len(self.mod_original),dtype=float)
        self.saved_predicted_infected=np.arange(0, len(self.mod_original),dtype=float)
        
        
        
    #TODO: should optimize
        
    def generate_incubator(self):
        lastvalue=self.sir_data[-1]
        randomdata = np.random.exponential(5, lastvalue)
        dataindex=0
        randomindex=0
        for i in range(1,lastvalue+1):
            if i>lastvalue:
                return
            if i>self.mod_confirmed[-1]:
                while dataindex<len(self.mod_confirmed) or i>self.sir_data[dataindex]:
                    dataindex+=1
            else:
                while i>self.mod_confirmed[dataindex]:
                    dataindex+=1
            randomindex=max([0,int(round(dataindex-randomdata[i-1]))])
                    
            for j in range (randomindex,len(self.incubator)):
                self.incubator[j]+=1
        for i in range(len(self.incubator)) :
            self.incubator[i]=self.incubator[i]-self.mod_recovered[i]
                        
    #originally for the x axis, is not used this functionality, but some usefull things remained here
    def generate_x_data(self, add_size):
        self.first_detection=self.indexes[0]
        loclist=self.indexes.copy()
            
        current = datetime.strptime(self.indexes[-1], '%m/%d/%y')
        today_value=current + timedelta(days=1)
        self.today="day {}., date: {}".format(len(loclist),datetime.strftime(today_value, '%m/%d/%y'))
        new_size=len(loclist)+add_size
        while len(loclist) < new_size:
            current = current + timedelta(days=1)
            loclist.append(datetime.strftime(current, '%m/%d/%y'))
        return loclist
    
    #solve SIR eqs
    def solve_eqs(self, size, data, beta, removed=0):
        def SIR(t, y):
            return [-beta*y[0]*y[1], beta*y[0]*y[1]-y[1]/self.length, y[1]/self.length]
        infected=data[0]/self.N
        
        return solve_ivp(SIR, [0, size], [1-infected-removed,infected,removed], t_eval=np.arange(0, size, 1),method=self.ivp_method)

    #general prediction
    def get_prediction(self, predict_len, data, removed, beta, full_len=True):
        x_data = self.generate_x_data(predict_len)
        new_size=predict_len
        if (full_len):
            new_size = len(x_data)
        prediction=self.solve_eqs(new_size, data, beta, removed[0]/self.N)
        
        return x_data,prediction.y[0],prediction.y[1],prediction.y[2]
        
    def get_raw_prediction(self, predict_len):
        
        return self.get_prediction(predict_len, self.mod_original, self.mod_recovered, self.beta)

    def get_incubation_prediction(self, predict_len):
        return self.get_prediction(predict_len, self.incubator, self.mod_recovered, self.beta)

    def set_incubation(self):
        
        search_len=len(self.mod_original)+self.incubation*3
        prediction=self.solve_eqs(search_len, self.mod_original, self.beta, self.mod_recovered[0]/self.N)
        #sir buffer (for incubation) contains the expected confirmed (ie infected+recovered)
        self.sir_data=np.round(prediction.y[1]*self.N+prediction.y[2]*self.N).astype(int)
        self.generate_incubator()
        
    #TODO: optimization may use rdata (mod_solution[2]) too
    def calculate_failure(self, beta, idata, rdata):
        solution = self.solve_eqs(len(idata), idata, beta, rdata[0]/self.N)
        mod_solution = solution.y*self.N
        rc=np.sqrt(np.mean((mod_solution[1] - idata)**2))
        return rc
    def optimize(self, idata, rdata):
        best_solution = minimize(self.calculate_failure, x0=self.beta, args=(idata,rdata),
            method=self.minimize_method, bounds=[(0.0000001, 0.9)])
        self.beta = best_solution.x
        
    def train(self):
        #generate sir with incubation
        self.optimize(self.mod_original,self.mod_recovered)
        self.set_incubation()
        self.optimize(self.incubator,self.mod_recovered)
    def raw_predict(self, predict_len):
            #'S': prediction.y[0]*self.N,
        self.optimize(self.mod_original,self.mod_recovered)
        x_dates,prediction0,prediction1,prediction2 = self.get_raw_prediction(predict_len)
        real_data = np.concatenate((self.mod_original, [None] * predict_len))
        end_of_infected=prediction2[len(prediction2)-1];
        df = pd.DataFrame({'Measured': real_data,'I': prediction1*self.N,'R': prediction2*self.N})
    
        figure, axes = plt.subplots(figsize=(12, 10))
        axes.set_title("Country: {}, population: {}\nbeta: {}, gamma: {}, R: {}, infected rate: {}".format
                     (self.country,self.N,self.beta,1/self.length,self.beta*self.length,end_of_infected))
        axes.set_xlabel("time, first detection (day 0): {}, today: {}, expected infected: {}".format(self.first_detection, self.today, self.mod_original[-1]))
        df.plot(ax=axes)
        
        #ax.plot()
        
        figure.savefig("{}_{}_{}_{}_{}_raw.png".
                       format(self.province,self.country,self.N,self.modulator,predict_len))
    def mod_predict(self, predict_len, connection_divider=0):
        mod_beta=self.beta
        if connection_divider>0:
            mod_beta=self.beta/connection_divider
            
        data=[]
        data.append(self.mod_infected*self.N)
        rdata=[]
        rdata.append(self.mod_removed*self.N)
        x_dates, prediction0,prediction1,prediction2 = self.get_prediction(predict_len, data, rdata, mod_beta,False)
        end_of_infected=prediction2[len(prediction2)-1];
        real_data = np.concatenate((self.incubator, [None] * predict_len))
        
        #real_prediction0 = np.concatenate(self.incubator, prediction0)
        real_prediction1 = np.concatenate((self.saved_predicted_infected, prediction1))
        real_prediction2 = np.concatenate((self.saved_predicted_removed, prediction2))
        
        df = pd.DataFrame({'Measured': real_data,'I': real_prediction1*self.N,'R': real_prediction2*self.N})
        
        figure, axes = plt.subplots(figsize=(12, 10))
        axes.set_title("Country: {}, population: {}, connections divided by {}:\nbeta: {}, gamma: {}, R: {}, infected rate: {}".format
                     (self.country,self.N,connection_divider,mod_beta,1/self.length,mod_beta*self.length,end_of_infected))
        axes.set_xlabel("time, first detection (day 0): {}, today: {}, expected infected: {}".format(self.first_detection, self.today, self.incubator[-1]))
        df.plot(ax=axes)
        
        figure.savefig("{}_{}_{}_{}_{}_div{}.png".
                       format(self.province,self.country,self.N,self.modulator,predict_len,connection_divider))
        
    def predict(self, predict_len):
            #'S': prediction.y[0]*self.N,
        
        
        x_dates,prediction0,prediction1,prediction2 = self.get_prediction(predict_len, self.incubator, self.mod_recovered, self.beta)
        
        real_data = np.concatenate((self.incubator, [None] * predict_len))
        end_of_infected=prediction2[len(prediction2)-1]; #removed end
        
        
        self.mod_infected=prediction1[self.today_num-1]
        self.mod_removed=prediction2[self.today_num-1]
        for index in range(0,len(self.saved_predicted_removed)):
            self.saved_predicted_removed[index]=prediction2[index]
            self.saved_predicted_infected[index]=float(self.incubator[index])/self.N
            #self.saved_predicted_susceptible[index]=1-self.saved_predicted_removed[index]-self.incubator[index]
            
        df = pd.DataFrame({'Measured': real_data,'I': prediction1*self.N,'R': prediction2*self.N})
        
        figure, axes = plt.subplots(figsize=(12, 10))
        axes.set_title("Country: {}, population: {}\nbeta: {}, gamma: {}, R: {}, infected rate: {}".format
                     (self.country,self.N,self.beta,1/self.length,self.beta*self.length,end_of_infected))
        axes.set_xlabel("time, first detection (day 0): {}, today: {}, expected infected: {}".format(self.first_detection, self.today, self.incubator[-1]))
        df.plot(ax=axes)
        
        figure.savefig("{}_{}_{}_{}_{}.png".
                       format(self.province,self.country,self.N,self.modulator,predict_len))

#model=SIR('Hubei','China', 10000000, 1, '2/22/20') #-> r~0,237
#model=SIR('Hubei','China', 10000000, 1, '','2/22/20') #->r~2,87
#model=SIR('','Italy', 60000000, 1) #->r~3,56
#☺model=SIR('','Hungary', 10000000, 1) #->r~3,38
model=SIR('','Hungary', 10000000, 5) #->r~3,51
model.raw_predict(5)
model.train()

model.predict(5)
model.predict(200)
model.mod_predict(400,2)
model.mod_predict(400,3)
model.mod_predict(400,4)