import numpy as np
import pandas as pd

import pyodbc
import sqlalchemy as sa


from retry import retry


#####################################################

def GetConnStr(ServerName: str, DBname: str, Driver: str,\
               Authenication: str = 'trusted',\
               UserName: str = 'UserName', Password: str= 'Password') -> str:
    
    # Authenication =  'MFA' or'trusted'  or  'password'  or  'trusted-azure'
    
    
    BaiscConnStr = 'Driver=' + Driver +\
                   ';Server=' + ServerName +\
                   ';PORT=1443' +\
                   ';Database=' + DBname                
    
    ConnStr = 'xxx'
          
    if Authenication =='trusted':
          ConnStr = BaiscConnStr +\
                     ";Trusted_Connection=yes"
                     
    elif  Authenication == 'MFA':
         
          ConnStr = BaiscConnStr +\
                     ";Authentication=ActiveDirectoryInteractive" +\
                     ";UID=" + UserName 
                     
    elif  Authenication == 'trusted-azure':
          ConnStr = BaiscConnStr +\
                     ";Authentication=ActiveDirectoryMsi"

    elif  Authenication == 'password':
          ConnStr = BaiscConnStr +\
                     ";UID=" + UserName +\
                     ";PWD=" + Password
    
    return ConnStr
 

##############################################################################################################

@retry(tries=2, delay=60)
def Download_Data_From_AzureDB(querySQL: str = '', Authenication: str = 'trusted') -> pd.DataFrame:
    
    
    ServerName = 'xxx'
    DBname = 'xxx'
    Driver= '{ODBC Driver 17 for SQL Server}'
    UserName = 'XXX'
    Password = 'XXX'

    
    
    # Authenication =  'MFA' or 'trusted'  or  'password'  or  'trusted-azure'
    connStr = GetConnStr( ServerName, DBname, Driver, Authenication, UserName, Password)
    conn = pyodbc.connect( connStr )

    with conn:
        data = pd.read_sql_query( querySQL,  conn)     
      
    
    return data




##############################################################################################################

#@retry(tries=1, delay=60)
def Download_Data_From_AzureDB_Alchemy(querySQL: str = '', Authenication: str = 'trusted', UserName = 'login') -> pd.DataFrame:
    
    
    ServerName = 'xxx'
    DBname = 'xxx'
    Driver= '{ODBC Driver 17 for SQL Server}'
    Password = 'XXX'
    UserName = UserName

    # Authenication = 'MFA' or 'trusted'  or  'password'  or  'trusted-azure'
    ConnStr = 'xxx'
    connStr = GetConnStr( ServerName, DBname, Driver, Authenication, UserName, Password)
    
    engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(connStr),
                              fast_executemany=True,\
                              connect_args={'connect_timeout': 10},\
                              echo=False)
        
    data = pd.read_sql_query( querySQL,  engine)     
      
    
    return data

    

    
    