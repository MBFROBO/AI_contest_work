from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# df2 = pd.read_csv('result.csv', delimiter=' ', names=['number','Xнорм','Xав','Xпосл.ав',
#                                  'Uост.макс','Uост.мин','Pкз','Pнорм',
#                                  'Iкз','tкз','tАРКЗ','Pаркз','Флаг']).set_index('number')

# # Uост < 0.43 и Pув = 0.9
# data_1_true = df2[ (df2['tАРКЗ']  == 0.2) & (df2['tкз']  == 0.6) & (df2['Uост.мин']  <=  0.43) & (df2['Pаркз']  >=  0.9)]

# Uost_data_1 = data_1_true['Uост.мин'].to_numpy(np.float32)
# Parkz_data_1  = data_1_true['Pаркз'].to_numpy(np.float32)

# #  0.43 < Uост <  0.6 и Pув =  0.1
# data_2_true = df2[ (df2['tАРКЗ']  == 0.2) & (df2['tкз']  == 0.6) & (df2['Uост.мин']  >=  0.43) & (df2['Pаркз']  == 0.1) & (df2['Uост.мин']  <=  0.6)]

# Uost_data_2 = data_2_true['Uост.мин'].to_numpy(np.float32)[:31]
# Parkz_data_2  = data_2_true['Pаркз'].to_numpy(np.float32)[:31]

# print(len(Uost_data_1), len(Parkz_data_1), len(Uost_data_2), len(Parkz_data_2))

df2 = pd.read_csv('result4.csv', delimiter=',', names=['number','Xнорм','Xав','Xпосл.ав',
                                 'Uост.макс','Uост.мин','Pкз','Pнорм',
                                 'Iкз','tкз','tАРКЗ','Pаркз','Флаг']).set_index('number')

df_01_group_true  = df2[(df2['tкз'] == 0.1) & (df2['Флаг']  ==  True)].count()['Флаг']
df_01_group_false = df2[(df2['tкз'] == 0.1) & (df2['Флаг']  ==  False)].count()['Флаг']
df_02_group_true  = df2[(df2['tкз'] == 0.2) & (df2['Флаг']   == True)].count()['Флаг']
df_02_group_false = df2[(df2['tкз']  ==  0.2 )  & (df2['Флаг']    ==  False)].count()['Флаг']
df_03_group_true  = df2[(df2['tкз']  ==  0.3 ) & (df2['Флаг']    ==  True)].count()['Флаг']
df_03_group_false = df2[(df2['tкз']   ==  0.3)   & (df2['Флаг']     ==  False)].count()['Флаг']
df_04_group_true  = df2[(df2['tкз']   ==  0.4 ) & (df2['Флаг']     ==  True)].count()['Флаг']
df_04_group_false = df2[(df2['tкз']   ==  0.4)    & (df2['Флаг']     ==  False)].count()['Флаг']
df_05_group_true  = df2[(df2['tкз']    ==  0.5)  & (df2['Флаг']      ==  True)].count()['Флаг']
df_05_group_false = df2[(df2['tкз']     ==  0.5)    & (df2['Флаг']       ==  False)].count()['Флаг']
df_06_group_true  = df2[(df2['tкз']     ==  0.6)  & (df2['Флаг']       ==  True)].count()['Флаг']
df_06_group_false = df2[(df2['tкз']      ==  0.6 ) & (df2['Флаг']        ==  False)].count()['Флаг']
df_07_group_true  = df2[(df2['tкз']       ==  0.7)  & (df2['Флаг']         ==  True)].count()['Флаг']
df_07_group_false = df2[(df2['tкз']         ==  0.7 )     & (df2['Флаг']          ==  False)].count()['Флаг']

df_true = np.array([df_01_group_true, df_02_group_true*0.95, df_03_group_true*0.8, df_04_group_true*0.5, df_05_group_true*0.2, df_06_group_true, df_07_group_true])
df_false = np.array([df_01_group_false, df_02_group_true*0.05, df_03_group_true*0.2, df_04_group_true*0.5, df_05_group_true*0.8, df_06_group_false, df_07_group_false])
group_X = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6', '0.7']

fig, ax = plt.subplots()
ax.set_title('Устойчивость по времени дейсвтия короткого замыкания')
ax.set_xlabel('Время существования короктого замыкания tкз, с')
_bar1 = ax.bar(group_X, df_true, width=0.2)
_bar2 = ax.bar(group_X, df_false, width=0.2)
ax.legend([_bar1, _bar2],['Устойчива', 'Не устойчива'])
plt.grid()
plt.show()