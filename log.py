# -*- coding: utf-8 -*-
import time

def clear(name='log'):
    open(name+'.txt', 'w').close()

def log(txt,save=True,error=False,name='log'):
    dateTimeStamp = time.strftime('%Y-%m-%d_%H:%M:%S | ')
    string = dateTimeStamp +  (" --ERROR-- " if error else "") + txt
    print(string)
    if(save):
        f=open(name+'.txt','a+')
        f.write(string+"\n")
        f.close()
        
if __name__ == "__main__":
    log("This is not an error!")
    log("This is an error!",error=True)