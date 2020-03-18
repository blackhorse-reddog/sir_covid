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
    def __init__(self, country, N, modulator=5, length=14, incubation=5):
        self.country = country
        self.N=N
        #base datas: length of infection period, length of incubation period
        self.length=length
        self.incubation=incubation
        self.modulator=modulator
        
        #the only variable in sir while length is known - R=beta/gamma
        self.beta=0.1

        #data sets
        df = pd.read_csv('time_series_2019-ncov-Confirmed.csv')
        country_data = df[df['Country/Region'] == country]
        index=0
        for key, value in country_data.iteritems(): 
            if index>3: #first 4 columns are: Province/State	Country/Region	Lat	Long
                if int(value)>0:
                    break
            index+=1

        self.original_data=country_data.iloc[0].loc[key:]
        #modulated data
        self.mod_original=[]
        #modulated data+expected infected persons in incubation phase
        self.incubator=[]

        #self.incubator=self.original_data.values
        #dont want ndarray array
        for existing in self.original_data.values:
            self.mod_original.append(existing*self.modulator)
            self.incubator.append(existing*self.modulator)
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
        
        
        
        #self.generate_incubator()
        
    def generate_incubator(self):
        lastvalue=self.sir_data[-1]
        randomdata = np.random.exponential(5, lastvalue)
        dataindex=0
        randomindex=0
        for i in range(1,lastvalue+1):
            if i>lastvalue:
                return
            if i>self.mod_original[-1]:
                while dataindex<len(self.mod_original) or i>self.sir_data[dataindex]:
                    dataindex+=1
            else:
                while i>self.mod_original[dataindex]:
                    dataindex+=1
            randomindex=max([0,int(round(dataindex-randomdata[i-1]))])
                    
            for j in range (randomindex,min(len(self.incubator),dataindex)):
                self.incubator[j]+=1
        #for existing in self.original_data:
            
        
                        
    #originally for the x axis, is not used this functionality, but some usefull things remained here
    def generate_x_data(self, add_size):
        list=[]
        self.first_detection=self.original_data.index[0]
        for existing in self.original_data.index.values:
            current = datetime.strptime(existing, '%m/%d/%y')
            list.append(datetime.strftime(current, '%m/%d/%y'))
            
        current = datetime.strptime(self.original_data.index[-1], '%m/%d/%y')
        today_value=current + timedelta(days=1)
        self.today="day {}., date: {}".format(len(list),datetime.strftime(today_value, '%m/%d/%y'))
        new_size=len(list)+add_size
        while len(list) < new_size:
            current = current + timedelta(days=1)
            list.append(datetime.strftime(current, '%m/%d/%y'))
        return list
    #solve SIR eqs
    def solve_eqs(self, size, data, beta, removed=0):
        def SIR(t, y):
            return [-beta*y[0]*y[1], beta*y[0]*y[1]-y[1]/self.length, y[1]/self.length]
        infected=data[0]/self.N
        
        return solve_ivp(SIR, [0, size], [1-infected-removed,infected,removed], t_eval=np.arange(0, size, 1),method=self.ivp_method)

    #general prediction
    def get_prediction(self, predict_len, data, beta, removed=0, full_len=True):
        x_data = self.generate_x_data(predict_len)
        new_size=predict_len
        if (full_len):
            new_size = len(x_data)
        prediction=self.solve_eqs(new_size, data, beta, removed)
        
        return x_data,prediction.y[0],prediction.y[1],prediction.y[2]
        
    def get_raw_prediction(self, predict_len):
        
        return self.get_prediction(predict_len, self.mod_original, self.beta)

    def get_incubation_prediction(self, predict_len):
        return self.get_prediction(predict_len, self.incubator, self.beta)

    def set_incubation(self):
        
        search_len=len(self.mod_original)+self.incubation*3
        prediction=self.solve_eqs(search_len, self.mod_original, self.beta)
        self.sir_data=np.round(prediction.y[1]*self.N).astype(int)
        self.generate_incubator()
        
    def calculate_failure(self, point, argdata):
        solution = self.solve_eqs(len(argdata), argdata, point)
        mod_solution = solution.y*self.N
        rc=np.sqrt(np.mean((mod_solution[1] - argdata)**2))
        return rc
    def optimize(self, data):
        best_solution = minimize(self.calculate_failure, x0=self.beta, args=data,
            method=self.minimize_method, bounds=[(0.0000001, 0.9)])
        self.beta = best_solution.x
        
    def train(self):
        #generate sir with incubation
        self.optimize(self.mod_original)
        self.set_incubation()
        self.optimize(self.incubator)
    def raw_predict(self, predict_len):
            #'S': prediction.y[0]*self.N,
        self.optimize(self.mod_original)
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
        
        figure.savefig("{}_{}_{}_{}_raw.png".
                       format(self.country,self.N,self.modulator,predict_len))
    def mod_predict(self, predict_len, connection_divider=0):
        mod_beta=self.beta
        if connection_divider>0:
            mod_beta=self.beta/connection_divider
            
        data=[]
        data.append(self.mod_infected*self.N)
        x_dates, prediction0,prediction1,prediction2 = self.get_prediction(predict_len, data, mod_beta, self.mod_removed,False)
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
        
        figure.savefig("{}_{}_{}_{}_div{}.png".
                       format(self.country,self.N,self.modulator,predict_len,connection_divider))
        
    def predict(self, predict_len):
            #'S': prediction.y[0]*self.N,
        
        
        x_dates,prediction0,prediction1,prediction2 = self.get_prediction(predict_len, self.incubator, self.beta)
        
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
        
        figure.savefig(f"{self.country}_{self.N}_{self.modulator}_{predict_len}.png")
        figure.savefig("{}_{}_{}_{}.png".
                       format(self.country,self.N,self.modulator,predict_len))

model=SIR('Italy', 60000000, 5)
model.raw_predict(5)
model.train()

model.predict(5)
model.predict(100)
model.mod_predict(300,4)