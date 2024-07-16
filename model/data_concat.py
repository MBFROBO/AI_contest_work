import pandas as pd

df1 = pd.read_csv('test_data_with_time.csv', delimiter=' ', names=['number','Xнорм','Xав','Xпосл.ав',
                                 'Uост.макс','Uост.мин','Pкз','Pнорм',
                                 'Iкз','tкз','tАРКЗ','Pаркз','Флаг']).set_index('number')
df2 = pd.read_csv('result.csv', delimiter=' ', names=['number','Xнорм','Xав','Xпосл.ав',
                                 'Uост.макс','Uост.мин','Pкз','Pнорм',
                                 'Iкз','tкз','tАРКЗ','Pаркз','Флаг']).set_index('number')

data_false_stability = df2[(df2['Флаг'] == False)]
new_df = pd.concat([df1, data_false_stability])

new_df.to_csv('result4.csv', index=True, columns=None)