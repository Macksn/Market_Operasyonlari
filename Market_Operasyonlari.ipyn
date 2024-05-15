
import pandas as pd
import numpy as np
import datetime
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
print(pd.__version__)
class ürün:
    def __init__(self, ürün_id,isim, fiyat, miktar):
        self.ürün_id = ürün_id
        self.ismi = isim
        self.fiyat = fiyat
        self.miktarı = miktar
        

class mağaza:
    def __init__(self):
        self.journal = pd.DataFrame(columns=['Tarih','Ürün İsmi','Ürün ID','İşlem','Miktar','Fiyat'])
    def alım (self,ürün,tarih):
        self.ürün = ürün
        self.tarih=tarih
        if ürün.ürün_id not in self.journal['Ürün ID'].unique():
            self.journal.loc[len(self.journal)] = [tarih, ürün.ismi, ürün.ürün_id, 'Alış', ürün.miktarı, ürün.fiyat]
        else:
            #DEBUG YES TO ALL
            #self.journal.loc[len(self.journal)] = [tarih, ürün.ismi, ürün.ürün_id, 'Alış', ürün.miktarı, ürün.fiyat]
            
            
            cevap=input('{} ürün ID`li ürün stoklarda yer almaktadır. \
            Eğer genede alım yapmak istiyorsanız lütfen Evet yazın,eğer alımdan \
            vazgeçmek isterseniz Hayır yazınız'.format(ürün.ürün_id))
            if cevap=='Evet':
                self.journal.loc[len(self.journal)] = [tarih, ürün.ismi, ürün.ürün_id, 'Alış', ürün.miktarı, ürün.fiyat]

    def iade (self,ürün_id,tarih,miktar,fiyat,isim):
        self.ürün_id=ürün_id
        self.tarih=tarih
        self.miktar=miktar
        self.fiyat=fiyat
        self.isim=isim
        alış=self.journal[(self.journal['Ürün ID']==ürün_id) &\
                          (self.journal['İşlem']=='Alış')].sum()['Miktar']
        satış=self.journal[(self.journal['Ürün ID']==ürün_id) & \
                           (self.journal['İşlem']=='Satış')].sum()['Miktar']
        iade=self.journal[(self.journal['Ürün ID']==ürün_id) & \
                          (self.journal['İşlem']=='İade')].sum()['Miktar']
        stok=alış-(satış+iade)
        if stok>miktar:
            self.journal.loc[len(self.journal)] = [tarih, isim, ürün_id, 'İade', miktar, fiyat]
            print('İade yapıldı.')
        else:
            print('İade etmek istediğiniz miktar stoktan büyük olduğu için iade yapılmadı')
        
           
    
    
    def satış(self, ürün_id, tarih, miktar, fiyat, isim):
        self.ürün_id = ürün_id
        self.tarih = tarih
        self.miktar = miktar
        self.fiyat = fiyat
        self.isim = isim
        stok_miktar = self.stok(ürün_id)
        
        # Stok kontrolü yapılıyor.
        if stok_miktar >= miktar:
            self.journal.loc[len(self.journal)] = [tarih, isim, ürün_id, 'Satış', miktar, fiyat]

#             print(f"{miktar} adet {isim} satıldı.")
        else:
            print("Stokta yeterli miktarda ürün bulunmamaktadır.")
            if stok_miktar == 0:
                print('{} id nolu ürün mevcut olmadığı için satış yapılmadı'.format(ürün_id))
            else:
                cevap=input('{} id nolu üründen talep edilen miktarda yoktur, eğer {} kadar \
                almak isterseniz Evet yazın, veya istemiyorsanız hayır yazınız'.format(ürün_id,stok_miktar))
                if cevap=='Evet':
                    self.journal.loc[len(self.journal)] = [tarih, isim, ürün_id, 'Satış', miktar, fiyat]
                else:
                    print('Satış işlemi yapılmadı')
        
    def stok(self, ürün_id):
        self.ürün_id = ürün_id
        alış = self.journal[(self.journal['Ürün ID'] == ürün_id) & (self.journal['İşlem'] == 'Alış')]['Miktar'].sum()
        satış = self.journal[(self.journal['Ürün ID'] == ürün_id) & (self.journal['İşlem'] == 'Satış')]['Miktar'].sum()
        iade = self.journal[(self.journal['Ürün ID'] == ürün_id) & (self.journal['İşlem'] == 'İade')]['Miktar'].sum()
        return (alış + iade) - satış

    def yıl_ortalama_satış(self,ürün_id,yıl):
        self.ürün_id=ürün_id
        self.yıl=yıl
        aa=self.journal.copy()
        aa['Tarih']=pd.to_datetime(aa['Tarih'])
        aa.set_index('Tarih',inplace=True)
        bb=aa.loc[str(yıl)].copy()
        ortalama= bb[(bb['Ürün ID']==ürün_id) & (bb['İşlem']=='Satış')][['Miktar','Fiyat']].mean()
        return ortalama
    def journal_göster(self):
        return self.journal
    
    def ay_ortalama_satış(self, ürün_id, yıl, ay):
        self.ürün_id=ürün_id
        self.yıl=yıl
        self.ay = ay
        
        aa = self.journal.copy()
        aa['Tarih'] = pd.to_datetime(aa['Tarih'])
        bb = aa[(aa['Tarih'].dt.year == yıl) & (aa['Tarih'].dt.month == ay) & (aa['Ürün ID'] == ürün_id)]
        average_purchase = bb[['Miktar', 'Fiyat']].mean()
        return average_purchase
    
    def train_forecasting_model(self, ürün_id):
        historical_data = []
        for year in range(2013, 2017):
            for month in range(1, 13):
                monthly_purchase = self.ay_ortalama_satış(ürün_id, year, month)
                historical_data.append([year, month, monthly_purchase['Miktar'], monthly_purchase['Fiyat']])

        df = pd.DataFrame(historical_data, columns=['Year', 'Month', 'Sales', 'Price'])
        

        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

        reference_date = pd.to_datetime('2019-01-01')
        df['DaysSince'] = (df['Date'] - reference_date).dt.days
        
        df.drop(['Year', 'Month', 'Date'], axis=1, inplace=True)
        
        X = df[['DaysSince', 'Price']]
        y = df['Sales']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model


    def make_forecast(self, model, periods):
        latest_date = self.journal['Tarih'].max()

        future_dates = pd.date_range(start=latest_date + pd.DateOffset(months=1), periods=periods, freq='M')

        reference_date = pd.to_datetime('2019-01-01')
        future_days_since = (future_dates - reference_date).days

        future_data = pd.DataFrame({'DaysSince': future_days_since, 'Price': self.journal['Fiyat'].mean()})

        forecast = model.predict(future_data)

        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast Sales': forecast})

        return forecast_df
aa=pd.read_excel('orders.xlsx')
aa.drop('Unnamed: 0',axis=1,inplace=True)
aa
list=[]

for i in aa.index:
    globals()['A11%s' % i] = ürün(aa.loc[i,'ürün_id'],aa.loc[i,'isim'],aa.loc[i,'fiyat'],aa.loc[i,'miktar'])
    list.append(globals()['A11%s' % i])
A110.ismi
A110.miktarı
A110.ürün_id
store=mağaza()
for i,j in zip(list,aa['tarih'].values):
    store.alım(i,j)
bb=pd.read_excel('item_28.xlsx')
bb
bb.drop('Unnamed: 0',axis=1,inplace=True)
for i in bb.index:
    store.satış(bb.loc[i,'ürün_id'],bb.loc[i,'tarih'],bb.loc[i,'miktar'],bb.loc[i,'fiyat'],bb.loc[i,'isim'])
store.yıl_ortalama_satış('A11',2014)
store.yıl_ortalama_satış('A11',2016)
store.ay_ortalama_satış('A11',2013,3)
model = store.train_forecasting_model("A11")

#6 aylık ileriye dönük forecast
forecast = store.make_forecast(model, periods=6)
print(forecast)











