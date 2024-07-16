from keras.models import load_model
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

# загружаем обученную модель
model = load_model('model5.h5')
"""
Ikz = (3.056-1.141)/(3.684 - 1.141) - ток короткого замыкания
Ukz = (0.277 - 0)/(1.1)             - напряжение КЗ
Tkz = 0.6                           - длительность КЗ
tARKZ = 0.1                         - время дейсвтия АРКЗ
yst = 1                             - флаг устойчивости (1 - устойчива)
x_test= np.array([[
    Ikz,
    Ukz,
    Tkz,
    tARKZ,
    yst
]]).astype(np.float32).reshape(1,-1)

"""
def graph(data_x1, data_x2, data_y, label_x, label_y):
    """
        Построение графика
    """
    fig, ax1 = plt.subplots(1)
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    line1 = ax1.plot(data_x1, data_y, label='Реальные данные')
    line2 = ax1.plot(data_x2, data_y, label = 'Предсказанные данные')
    ax1.legend()
    ax1.grid()
    plt.show()
    
data = pd.read_csv('test_data.csv', delimiter=' ', names=['number','Xнорм','Xав','Xпосл.ав',
                                 'Uост.макс','Uост.мин','Pкз','Pнорм',
                                 'Iкз','tкз','tАРКЗ','Pаркз','Флаг']).set_index('number')

data_true_stability = data[data['Флаг']==1]
data_true_stability_U_unic = data_true_stability['Uост.мин'].unique()

sorted_data_frame = pd.DataFrame(columns=['Xнорм','Xав','Xпосл.ав',
                                 'Uост.макс','Uост.мин','Pкз','Pнорм',
                                 'Iкз','tкз','tАРКЗ','Pаркз','Флаг'])

for i in range(len(data_true_stability_U_unic)):
    new_row = data_true_stability[data_true_stability['Uост.мин'].isin([data_true_stability_U_unic[i]])].head(1).values[0]
    sorted_data_frame.loc[-1] = new_row
    sorted_data_frame.index = sorted_data_frame.index+1

# делим сортированный массив на входные и выходные данные
input_df = sorted_data_frame[['Iкз', 'Uост.мин', 'tкз', 'tАРКЗ', 'Флаг']].replace([[True]], 1)
result_df = sorted_data_frame[['Pаркз']]

# приводим входные данные к типу float32
_input = input_df.to_numpy().astype(np.float32)

predicted_result = []
for i in _input:
    test= np.array([[
        (i[1] - 0)/(1.1),
        i[2],
        i[3],
        i[4]
        ]]).astype(np.float32).reshape(1,-1)
    predicted_result.append(model.predict(test)[0][0])

_result = result_df.to_numpy().astype(np.float32).reshape(1,-1)[0]
Uost = input_df['Uост.мин'].to_numpy().astype(np.float32).reshape(1,-1)[0]

graph(_result, predicted_result, Uost, 'УВ','Остаточное напряжение')