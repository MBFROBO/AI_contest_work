import numpy as np

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from keras.models import load_model
    arkz_neuro = load_model("model5.h5")
# исходные данные #

TJ = 7
W0 = 314
D = 0.08
delta_array_for_diagram = np.linspace(0,180,30000)
freq_array= [50, 50]
delta_0 = 19.116*np.pi/180 # град
delta = [delta_0,delta_0]
Xt, xt = 0.11, 0.11
Xl, xl = 0.08, 0.08
Xd, xd = 0.278, 0.278
U = 1
Xkz = 0
Xpost = 0
Ug = 242/220
Eg = 1.43
Pt_start = 1.0
Pt = 1.0
xt0 = 0.096
x0 = 0.18
xg2 = 0.234
Ikz = 0
Xce = 0
Pt_array = [Pt_start, Pt]
Ib = 100/(np.sqrt(3)*220)
Uost = [Ug, Ug]
Xem = Xt+Xl+Xd
power_array = [np.around(U*Eg/(Xt+Xl+Xd)*np.sin(delta_0), 1), np.around(U*Eg/(Xt+Xl+Xd)*np.sin(delta_0), 1)]
time_sim = 3

h = 0.0001
time_array = [0, h]

def system_model(iter = 2):
    """
        Программная модель ЭС
    """
    global Pt
    global power_array
    global h
    global delta_0
    global D
    global Xem
    _P = np.around(U*Eg/(Xem)*np.sin(delta_0), 5)
    freq_new = (Pt - _P)/TJ*2*h*50 + freq_array[iter-1]
    delta_new =  ((Pt - _P)*h**2 +(2*TJ/W0*delta[iter-1])-(TJ/W0*delta[iter-2]) + (0.5*D*h*delta[iter-1]))/(TJ/W0 + 0.5*D*h)
    freq_0 = freq_new
    delta_0 = delta_new

    return freq_0, _P, delta_0

def ARKZ_model(time,iter,Pyst, time_t_arkz, kz_on, kz_off, Uyst = 0.55):
    """
        Определение близких / дальних КЗ по напряжению на шинах
        Избрание управляющего воздейсвтия. 
        Уставку по напряженю примем 0.55Uном
    """
    global Uost, Pt, arkz_neuro
    if Uost[iter] < Uyst:
        if (time - time_t_arkz < time):
            time_t_arkz = time_t_arkz - h
            return False, time_t_arkz
        else:
            test= np.array([[
                (Uost[iter] - 0)/(1.1),
                kz_off - kz_on,
                time_t_arkz,
                1
                ]]).astype(np.float32).reshape(1,-1)
            Pyst = arkz_neuro.predict(test)[0][0]
            Pt = Pt - Pyst
            if __name__ == '__main__':
                print('Мощность после УВ ', Pt," Напряжение при КЗ ", Uost[iter])
            return True, time_t_arkz
      
    return False, time_t_arkz

def KPR_model(Post, Ppred):
    """
        Функция кортоля предшествующего режима. Выполняет роль выбора управляющего воздейсвтия (АДВ)
    """
    pass

def KZ_model_Xost(time, length:int, kz_type:int, kz_time_on:float, kz_time_off:float):
    """
        Модель сопротивления до короткого замыкания (пока только до трёхфазного)
    """
    global Xl, xd, Xt, Xem, Xkz, Xpost, xt0, xg2, x0, Xce
    if kz_type == 3 and time >= kz_time_on and time < kz_time_off and length != 0 and length!= 0.0:
        Xshunt = Xl*2*length
        Xa = xd+Xt
        xb = Xl*2
        Xab = Xa + xb + Xa*xb/Xshunt
        Xem = Xab
        Xkz = Xab
    if kz_type == 3 and (length == 0 or length == 0.0):
        Xem = np.inf
        Xkz = np.inf
    if kz_type == 1.1 and time >= kz_time_on and time < kz_time_off:
        x2_shunt = (xg2 + xt + length*xl)*((1-length)*xl)/((xg2+ xt + length*xl)+((1-length)*xl))
        x0_shunt = (xt0 + length*x0)*((1-length)*x0)/((xt0 + length*x0)+((1-length)*x0))
        x_shunt = x2_shunt * x0_shunt / (x2_shunt + x0_shunt)
        Xa = xd+Xt+Xl*length
        xb = Xl*(1- length)
        Xab = Xa + xb + Xa*xb/x_shunt
        Xem = Xab
        Xkz = Xab
    if kz_type == 2 and time >= kz_time_on and time < kz_time_off:
        x2_shunt = (xg2 + xt + length*xl)*((1-length)*xl)/((xg2+ xt + length*xl)+((1-length)*xl))
        x_shunt = x2_shunt
        Xa = xd+Xt+Xl*length
        xb = Xl*(1- length)
        Xab = Xa+ xb + Xa*xb/x_shunt
        Xem = Xab
        Xkz = Xab

    if time < kz_time_on:
        Xem = (xl+xt+xd)
    if time > kz_time_off:
        Xem = (2*xl+xt+xd)
        Xpost = (2*xl+xt+xd)
    

def KZ_model_Uost(time, length:float, kz_type:float, kz_time_on:float, kz_time_off:float):
    """
        Определяем остаточное напряжение на шинах при различных видах КЗ
    """
    global xd, Xl, xt, Eg, xt0, xg2, x0, Ikz, Xce
    if kz_type == 3 and time >= kz_time_on and time < kz_time_off:
        # Расчёт остаточного напряжения на шинах
        x11 = 2*Xl*2*length*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления слева от шунта
        x22 = 2*Xl*2*(1- length)*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления справа от шунта
        xkz = 2*length*Xl*2*(1-length)*Xl / (2*Xl + 2*Xl*length + 2*Xl * (1-length)) # расчёт сопротивления шунта
        xekv = (xd + xt + x11)*x22 / ((xd + xt + x11)+x22) # эквивалентирование сопротивлений
        Eekv = ((Eg*x22) + U*(xd + xt + x11))/(xd+xt+x11+x22) # эквивалентное эдс
        Ce = xekv/(xd+Xt+x11)
        Cu = xekv/x22
        Xce = (xekv+xkz)/Ce
        Ikz = Eg/Xce
        Uost = Eg - Ikz*(xd+xt)
        return Uost
    
    elif kz_type == 1.1 and time >= kz_time_on and time < kz_time_off:
        x11 = 2*Xl*2*length*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления слева от шунта
        x22 = 2*Xl*2*(1- length)*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления справа от шунта
        x0_shunt = (xt0 + length*x0)*((1-length)*x0)/((xt0 + length*x0)+((1-length)*x0))
        x2_shunt = (xg2 + xt + length*xl)*((1-length)*xl)/((xg2+ xt + length*xl)+((1-length)*xl))
        xkz = 2*length*Xl*2*(1-length)*Xl / (2*Xl + 2*Xl*length + 2*Xl * (1-length)) +  x0_shunt*x2_shunt/(x2_shunt+x0_shunt)# расчёт сопротивления шунта
        xekv = (xd + xt + x11)*x22 / ((xd + xt + x11)+x22) # эквивалентирование сопротивлений
        Eekv = ((Eg*x22) + U*(xd + xt + x11))/(xd+xt+x11+x22) # эквивалентное эдс
        Ce = xekv/(xd+Xt+x11)
        Cu = xekv/x22
        Xce = (xekv+xkz)/Ce
        Ikz = Eg/Xce
        Uost = Eg - Ikz*(xd+xt)
        return Uost
    
    elif kz_type == 2 and time >= kz_time_on and time < kz_time_off:
        x11 = 2*Xl*2*length*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления слева от шунта
        x22 = 2*Xl*2*(1- length)*Xl/(2*Xl + 2*Xl*length + 2*Xl * (1-length)) #расчёт сопротивления справа от шунта
        x2_shunt = (xg2 + xt + length*xl)*((1-length)*xl)/((xg2+ xt + length*xl)+((1-length)*xl))
        xkz = 2*length*Xl*2*(1-length)*Xl / (2*Xl + 2*Xl*length + 2*Xl * (1-length)) + x2_shunt# расчёт сопротивления шунта
        xekv = (xd + xt + x11)*x22 / ((xd + xt + x11)+x22) # эквивалентирование сопротивлений
        Eekv = ((Eg*x22) + U*(xd + xt + x11))/(xd+xt+x11+x22) # эквивалентное эдс
        Ce = xekv/(xd+Xt+x11)
        Cu = xekv/x22
        Xce = (xekv+xkz)/Ce
        Ikz = Eg/Xce
        Uost = Eg - Ikz*(xd+xt)
        return Uost
    
    else:
        return Ug
    
    
def graphis(kz_flag):
    """
        Построение графиков
    """
    global power_array
    global time_array
    global delta
    global Uost
    global Pt_array
    
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('P, o.e')
    ax1.set_title('График мощности во времени')
    ax1.plot(time_array, power_array)

    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
    ax2.set_xlabel('t, с')
    ax2.set_ylabel('\u03B4, град')
    ax2.set_title('График угла \u03B4 во времени')
    ax2.plot(time_array, [i*180/np.pi for i in delta])

    ax3 = plt.subplot2grid((3,3), (2,0), colspan=2)
    ax3.set_xlabel('t, с')
    ax3.set_ylabel('U, о.е')
    ax3.set_title('График напряжения во времени')
    ax3.set(ylim = (0,1.5))
    ax3.plot(time_array, Uost)

    ax4 = plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    ax4.plot(delta_array_for_diagram, U*Eg/(Xt+Xl+Xd)*np.sin([i*np.pi/180 for i in delta_array_for_diagram]))
    ax4.set_title('Диаграмма мощности от угла \u03B4')
    if kz_flag == True:
        ax4.plot(delta_array_for_diagram, U*Eg/Xkz*np.sin([i*np.pi/180 for i in delta_array_for_diagram]))
        ax4.plot(delta_array_for_diagram, U*Eg/Xpost*np.sin([i*np.pi/180 for i in delta_array_for_diagram]))
        ax4.plot(delta_array_for_diagram, Pt_array)

    ax4.set_xlabel('\u03B4, град')
    ax4.set_ylabel('P, о.е')
    plt.tight_layout()
    plt.grid()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()

def clean_data():
    """
        Очистка данных для новой итерации цикла
    """
    global delta_array_for_diagram, freq_array, delta_0, delta, Pt_start, Pt, Pt_array, Xem, power_array, time_array, Uost
    delta_array_for_diagram = np.linspace(0,180,30000)
    freq_array= [50, 50]
    delta_0 = 19.116*np.pi/180 # град
    delta = [delta_0,delta_0]

    Pt_start = 1.0
    Pt = 1.0
    Pt_array = [Pt_start, Pt]
    Xem = Xt+Xl+Xd
    power_array = [np.around(U*Eg/(Xt+Xl+Xd)*np.sin(delta_0), 1), np.around(U*Eg/(Xt+Xl+Xd)*np.sin(delta_0), 1)]
    time_array = [0, h]
    Uost = [Ug, Ug]

def data_marking(number, kz_on, kz_off, time_t_arkz, Pyst):
    """
        Пометка данных. По токам и напряжениям будем оценивать аварийный это режим или нет, а так же 
        устойчивость системы после КЗ.
        Маркировка осуществляется по времени симуляции.
        Из режима выбирается одно значение
        Структура данных

        [
            Маркировка режима

            сопротивление системы в нормальном режиме, Хсумм    о.е
            сопротивление системы в аварийном режиме, Хав       о.е
            сопротивление послеаварийного режима, Xпост         о.е

            напряжение системы в норальном режиме, U            о.е
            напряжние системы в аварийном режиме, Uав           о.е

            активная мощность аварийного режиме, Pав            о.е
            активная мощность предшествующего режима, Pт        о.е
            
            ток короткого замыкания, Ikz                        о.е
            время существования кз, tkz                         с
            время действия АРКЗ (с уч. УВ) tАРКЗ                с
            
            активная мощность УВ, Pyst                          о.е
            флаг устойчивости системы, flag                     bool (True - устойчивая / false - неустойчивая)

        ]
    """

    
    global Uost
    global delta
    global Xd, xl, Xt, Xl, Xkz, Xpost, Xce
    global Ikz

    flag = lambda delta: True if max(delta) * 180 / np.pi < 180 else False
    
    try:
        _pow_kz = U*Eg/Xkz
    except ZeroDivisionError:
        _pow_kz  = np.inf

    return [
        number,                             #0
        (Xd + Xt + Xl),                     #1
        np.round(Xce, 3),                   #2
        np.round(Xpost, 3),                 #3
        np.round(max(Uost), 3),             #4
        np.round(min(Uost), 3),             #5
        np.round(_pow_kz, 3),               #6
        np.round(U*Eg/(Xd + Xt + Xl), 3),   #7
        np.round(Ikz, 3),                   #8
        np.round(kz_off - kz_on, 3),        #9
        time_t_arkz,                        #10
        np.round(Pyst,  3),                 #11
        flag(delta)                         #12
    ]
    


def main(regim_number, kz_type, kz_on, kz_off, distance, Pyst, time_t_arkz):
    global time_sim
    global Pt_array
    t_arkz = time_t_arkz
    curr_time = h+h
    iter = 2
    kz_on, kz_off = kz_on, kz_off
    kz_flag = True
    distance = distance
    kz_type = kz_type
    Pyst = Pyst # мощность уставки АРКЗ
    ARKZ_srb = False
    while curr_time < time_sim:
        f, p, delta_local = system_model(iter)
        KZ_model_Xost(curr_time, distance, kz_type, kz_on, kz_off)
        Uost.append(KZ_model_Uost(curr_time, distance, kz_type, kz_on, kz_off))
        if  ARKZ_srb == False:
            ARKZ_srb, time_t_arkz = ARKZ_model(curr_time,iter, Pyst, time_t_arkz, kz_on, kz_off)
        else: pass
        freq_array.append(f)
        power_array.append(p)
        delta.append(delta_local)
        Pt_array.append(Pt)
        iter += 1 
        curr_time += h
        time_array.append(curr_time)

    marked_data = data_marking(regim_number, kz_on, kz_off, t_arkz, Pyst)

    if __name__ != '__main__':
        clean_data()

    return marked_data

if __name__ == '__main__':
    regim_data = main(1, 3, 0.5, 0.9, 0.3, 1, 0.1)
    graphis(True)
    