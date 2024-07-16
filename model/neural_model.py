import tensorflow as tf
import keras
from keras import layers

import pandas as pd
import numpy as np
import csv, ctypes
from pathlib import Path
from dataclasses import dataclass, field
from matplotlib import pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# Используем вставки динмаических библиотек С для ускорения работы программы
testc = ctypes.CDLL("./libtest.so")
testc.bubbleSort.restype = ctypes.c_float
testc.bubbleSort.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]


data = []
result = []
with open('result4.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='|')
    for row in csvreader:
        if row[-1] == 'False': row[-1] = 0
        if row[-1] == 'True': row[-1] = 1
        if 'nan' not in row and 'inf' not in row:
            data.append(row[:-1]+[row[-1]])
            result.append(row[-2])

np_result   = np.array(result).astype(np.float32)
np_data     = np.array(data).astype(np.float32)


@dataclass
class data_analysys:
    """
        Анализ данных с целью поиска корреляций и выбора параметров обучения. 
        Фильтрация выбросов.
    """
    data: np.ndarray
    result: np.ndarray

    labels: np.ndarray          = field(default_factory=lambda:np.array([i[0] for i in data]).astype(np.float32))  # привязанные лейблы
    resistance_norm: np.ndarray = field(default_factory=lambda:np.array([i[1] for i in data]).astype(np.float32))  # сопротивление в нормальном режиме
    resistance_avar: np.ndarray = field(default_factory=lambda:np.array([i[2] for i in data]).astype(np.float32))  # сопротивление в аварийном режиме
    resistance_post: np.ndarray = field(default_factory=lambda:np.array([i[3] for i in data]).astype(np.float32))  # сопротивление в послеаварийном режиме

    voltage_norm: np.ndarray    = field(default_factory=lambda:np.array([i[4] for i in data]).astype(np.float32))  # напряжение в нормальном режиме
    voltage_avar: np.ndarray    = field(default_factory=lambda:np.array([i[5] for i in data]).astype(np.float32))  # напряжение в аварийном режиме   

    power_avar: np.ndarray      = field(default_factory=lambda:np.array([i[6] for i in data]).astype(np.float32))  # мощность в нормальном режиме
    power_pred: np.ndarray      = field(default_factory=lambda:np.array([i[7] for i in data]).astype(np.float32))  # мощность в предшествующем режиме

    current_avar: np.ndarray    = field(default_factory=lambda:np.array([i[8] for i in data]).astype(np.float32))  # ток короткого замыкания
    time_avar: np.ndarray       = field(default_factory=lambda:np.array([i[9] for i in data]).astype(np.float32))  # время короткого замыкания
    time_ARKZ: np.ndarray       = field(default_factory=lambda:np.array([i[10] for i in data]).astype(np.float32))  # время дейсвия АРКЗ
    Pyst:np.ndarray             = field(default_factory=lambda:np.array([i[11] for i in data]).astype(np.float32)) # мощность в уставки АРКЗ
    flag: np.ndarray            = field(default_factory=lambda:np.array([i[-1] for i in data]).astype(np.float32)) # флаг устойчивости (1/0)

    
    def visualize(self):
        """
            Визулазиация данных в различных форматах.
        """

        v_data = self.data_sort(self.time_ARKZ, self.flag, self.labels)
        
        fig, ((ax,ax1),(ax2,ax3)) = plt.subplots(nrows=2, ncols=2)
        grouped_v_data_X = list(self.divide_array(v_data[1], int(len(v_data[1])/10)))
        grouped_v_data_Y = list(self.divide_array(v_data[2], int(len(v_data[2])/10)))
        dataX = ['%.1f - %.1f'%(round(min(i),2),round(max(i),2)) for i in grouped_v_data_X]
        dataY_1 = [self.count_elements(i, 1) for i in grouped_v_data_Y]
        dataY_2 = [self.count_elements(i, 0) for i in grouped_v_data_Y]
        ax.set_title('Устойчивость по времени дейсвтия АРКЗ')
        ax.set_xlabel('Время дейсвия АРКЗ tАРКЗ, с')
        _bar1 = ax.bar(dataX, dataY_1, width=0.2)
        _bar2 = ax.bar(dataX, dataY_2, width=0.2)
        ax.legend([_bar1, _bar2],['Устойчива', 'Не устойчива'])

        v_data = self.data_sort(self.time_avar, self.flag, self.labels)

        grouped_v_data_X = list(self.divide_array(v_data[1], int(len(v_data[1])/11)))
        grouped_v_data_Y = list(self.divide_array(v_data[2], int(len(v_data[2])/11)))
        dataX = ['%.1f - %.1f'%(round(min(i),2),round(max(i),2)) for i in grouped_v_data_X]
        dataY_1 = [self.count_elements(i, 1) for i in grouped_v_data_Y]
        dataY_2 = [self.count_elements(i, 0) for i in grouped_v_data_Y]
        ax1.set_title('Устойчивость по времени дейсвтия короткого замыкания')
        ax1.set_xlabel('Время существования короктого замыкания tкз, с')
        _bar1 = ax1.bar(dataX, dataY_1, width=0.2)
        _bar2 = ax1.bar(dataX, dataY_2, width=0.2)
        ax1.legend([_bar1, _bar2],['Устойчива', 'Не устойчива'])

        v_data = self.data_sort(self.resistance_avar, self.flag, self.labels)

        grouped_v_data_X = list(self.divide_array(v_data[1], int(len(v_data[1])/10)))
        grouped_v_data_Y = list(self.divide_array(v_data[2], int(len(v_data[2])/10)))
        dataX = ['%.1f - %.1f'%(round(min(i),2),round(max(i),2)) for i in grouped_v_data_X]
        dataY_1 = [self.count_elements(i, 1) for i in grouped_v_data_Y]
        dataY_2 = [self.count_elements(i, 0) for i in grouped_v_data_Y]
        ax2.set_title('Устойчивость по сопротивлению.')
        ax2.set_xlabel('Сопротивление сечения, Хав о.е')
        _bar1 = ax2.bar(dataX, dataY_1, width=0.2)
        _bar2 = ax2.bar(dataX, dataY_2, width=0.2)
        ax2.legend([_bar1, _bar2],['Устойчива', 'Не устойчива'])
        v_data = self.data_sort(self.current_avar, self.flag, self.labels)

        grouped_v_data_X = list(self.divide_array(v_data[1], int(len(v_data[1])/10)))
        grouped_v_data_Y = list(self.divide_array(v_data[2], int(len(v_data[2])/10)))
        dataX = ['%.1f - %.1f'%(round(min(i),2),round(max(i),2)) for i in grouped_v_data_X]
        dataY_1 = [self.count_elements(i, 1) for i in grouped_v_data_Y]
        dataY_2 = [self.count_elements(i, 0) for i in grouped_v_data_Y]
        ax3.set_title('Устойчивость по току КЗ.')
        ax3.set_xlabel('Ток короткого замыкания у шин ПС, Iкз о.е')
        _bar1 =ax3.bar(dataX, dataY_1, width=0.2)
        _bar2 =ax3.bar(dataX, dataY_2, width=0.2)
        ax3.legend([_bar1, _bar2],['Устойчива', 'Не устойчива'])

        ax.grid()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.show()

    @staticmethod
    def loss_visual(data, epoch):
        hist = pd.DataFrame(data.history)
        hist['epoch'] = epoch
        
        fig, ax1 = plt.subplots(1)
        ax1.set_xlabel('Количество эпох (итераций обучения)')
        ax1.set_ylabel('Погрешность')
        ax1.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Валидационная ошибка')
        ax1.plot(hist['epoch'], hist['loss'], label = 'Loss')
        
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod   
    def divide_array(list, n):   
        for i in range(0, len(list), n):  
            yield list[i:i + n] 

    @staticmethod
    def count_elements(list, elem):
        count = 0
        for i in list:
            if i == elem:
                count += 1
        return count
    
    @staticmethod
    def data_sort(data_x:np.ndarray, data_y:np.ndarray = None, labels:np.ndarray = None):
        """
            Сортировка данных c обязательой привязкой к значениям лейблов
        """
        
        local_data_x    = data_x
        if data_y  is not None and labels  is not None:
            local_data_y    = data_y
            local_labels    = labels
        n = len(local_data_x)
        for i in range(n):
            for j in range(n-i-1):
                if local_data_x[j] > local_data_x[j+1]:
                    local_data_x[j], local_data_x[j+1] =local_data_x[j+1], local_data_x[j]
                    if data_y  is not None and labels  is not None:
                        local_data_y[j], local_data_y[j+1] =local_data_y[j+1], local_data_y[j]
                        local_labels[j], local_labels[j+1] =local_labels[j+1], local_labels[j]
        if data_y  is not None and labels  is not None:
            stacked = np.row_stack((local_labels,local_data_x, local_data_y))
        else:
            stacked = local_data_x
        return stacked
        
    def data_train_package(self):
        """Структура данных для тренировки"""
        Xpred = self.resistance_norm[:int(len(self.resistance_norm)*1)]   # сопротивление предшествующего режима
        Xpost = self.resistance_post[:int(len(self.resistance_post)*1)]   # сопротивление поставарийного режима
        Ikz =  self.current_avar[:int(len(self.current_avar)*1)]          # ток короткого замыкания 
        Ukz = self.voltage_avar[:int(len(self.voltage_avar)*1)]           # напряжение коротокого замыкания
        Tkz = self.time_avar[:int(len(self.time_avar)*1)]                 # время короткого замыкания
        tARKZ = self.time_ARKZ[:int(len(self.time_ARKZ)*1)]               # время работы АРКЗ
        yst = self.flag[:int(len(self.flag)*1)]                           # результат работы АРКЗ (устойчивость системы)

        normilized_Xpred = Xpred
        normilized_Xpost  = Xpost
        normilized_Ikz  = self.normalize(Ikz)
        normilized_Ukz   = self.normalize(Ukz)

        return np.array([normilized_Ukz,Tkz,tARKZ, yst]).T
        
    def data_test_package(self):
        """Структура данных для тестирования"""
        # Xpred = self.resistance_norm[int(len(self.resistance_norm)*0.8):]   # сопротивление предшествующего режима
        # Xpost = self.resistance_post[int(len(self.resistance_post)*0.8):]   # сопротивление поставарийного режима
        # Ikz =  self.current_avar[int(len(self.current_avar)*0.8):]          # ток короткого замыкания 
        # Ukz = self.voltage_avar[int(len(self.voltage_avar)*0.8):]           # напряжение коротокого замыкания
        # Tkz = self.time_avar[int(len(self.time_avar)*0.8):]                 # время короткого замыкания
        # tARKZ = self.time_ARKZ[int(len(self.time_ARKZ)*0.8):]               # время работы АРКЗ
        # yst = self.flag[int(len(self.flag)*0.8):]      

        data = {
            "resistance_norm":  self.resistance_norm,
            "resistance_post":  self.resistance_post,
            "current_avar":  self.current_avar,
            "voltage_avar": self.voltage_avar,
            "time_avar": self.time_avar,
            "time_ARKZ":self.time_ARKZ,
            'flag': self.flag
            }
    
        
        df = pd.DataFrame(data)
        print(df)
        # normilized_Xpred = Xpred
        # normilized_Xpost  = Xpost
        # normilized_Ikz  = self.normalize(Ikz)
        # normilized_Ukz   = self.normalize(Ukz)

        # return np.array([normilized_Ikz, normilized_Ukz, Tkz, tARKZ, yst]).T
    
    def data_result_train_package(self):
        """Структура данных учителя для тренировки"""
        return self.Pyst[:int(len(self.Pyst)*1)]

    def data_result_test_package(self):
        """Структура данных учителя тестирования"""  
        return self.Pyst[int(len(self.Pyst)*0.8):]
    

    
    @staticmethod
    def normalize(arr):
        """
            Используем униполярную нормализацию данных по линейной функции.
        """
        arr = np.array(arr)
        data_norm = (arr - arr.min())/(arr.max() - arr.min())
        print(data_norm)
        return data_norm
        

    @staticmethod
    def model():
        """
            Нейросетевая модель
        """

        model = keras.models.Sequential([
            layers.Dense(4, activation='relu', input_shape=(4,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(1),
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, 
                    loss= keras.losses.mean_absolute_error, 
                    metrics=[
                        keras.metrics.MeanAbsoluteError(),
                        keras.metrics.MeanSquaredError(),
        ])

        return model
    

analysys = data_analysys(data=np_data, result=np_result)
model = analysys.model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto')
analysys.data_test_package()
analysys.visualize()
# history  = model.fit(analysys.data_train_package(), analysys.data_result_train_package(), 
#                      epochs=500, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# analysys.loss_visual(history, np.linspace(0,500,500))
# model.save('model5.h5')