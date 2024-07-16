import numpy as np
import test_model as tm
import csv

"""
    Гоняем кучу режимов и регерируем массив данных.
"""
distance_array = np.linspace(0,0.99,100)                         # массив расстояний до короткого замыкания
ARKZ_time_array = [0.1, 0.2]                                     # массив времён дейсвтия АРКЗ   
ARKZ_power_array  =  np.linspace(0,0.99,25)     # массив мощностей УВ АРКЗ
kz_type_array = [3]                                              # массив типов короткого замыкания
kz_off_array  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0 , 1.1, 1.15, 1.2]

result_array = []
regim_num = 0
local_l_flag = 0
for i in kz_type_array:
    for j in distance_array:
        for k in ARKZ_time_array:
            for m in kz_off_array:
                local_l_flag = 0
                for l in ARKZ_power_array:
                    regim = tm.main(regim_num, i, 0.5, m, j, l, k)
                    if regim[-1] == True and local_l_flag == 0:
                        result_array.append(regim)
                        print(regim)
                        local_l_flag = 1
                        regim_num+=1
                    

with open('test_data_with_time.csv', 'w',  newline='') as f:
    spamwriter = csv.writer(f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in result_array:
        spamwriter.writerow(i)