import pandas as pd
import matplotlib.pyplot as plt

dict = {'DATE':[31-1-2020,29-2-2020,20-3-2020,31-4-2020,30-5-2020],
        'PRICE':[10000,7000,20000,8000,9000],
        'PRODUCT_ID':[901,902,903,904,905],
        'QUANTITY_PURCHASED':[34,37,56,60,50],
        'SERIAL_NO':[101,102,103,104,105],
        'USR_ID':[1001,1002,1003,1004,1005],
        'USR_TYPE':['A','B','C','A','B'],
        'USR_CLASS':['UPR','UPR','MDL','LWR','MDL'],
        'PUR_WEEK':['MON','TUE','WED','MON','WED']
}

df = pd.DataFrame(dict)
df.to_csv("df.csv")
print(df.head())

#GITITG

print(df)

#Prints all statistical values describe method
stats = df.describe(include='all')
print(stats)
df.plot(x = "USR_ID", y="PRICE", kind="line")
plt.show()
df[['QUANTITY_PURCHASED','PUR_WEEK']].plot.box()
plt.title('Quantity and Week value distribution')
plt.show()