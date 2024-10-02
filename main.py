from random import randint
from numpy.linalg import solve as gauss
import numpy as np
from math import exp
from dsmltf import scale, mult_r_squared, mult_predict, dot

# стоит ли идти на пару
# практика(1) или лекция(0)/ профильный ли предмет(1) или нет(0/ обязательны конспекты(1) или нет(0/ хочется ли кушать(1) или нет(0)/третья+ пара(1) или нет(0)?/
# есть ли пары кроме этой(1) или нет(0)/ сложен ли предмет для понимания от 1 до 5/ пойду ли я на пару
def make_data() -> list:
    data=[]
    for i in range(30):
        arr = [randint(0,1),randint(0,1),randint(0,1),randint(0,1),randint(0,1),randint(0,1),randint(1,5)]
        print("Практика"*arr[0]+"Лекция"*(not arr[0]))
        print("Предмет -", "профильный"*arr[1]+"не профильный"*(not arr[1]))
        print("Конспекты -","обязательны"*arr[2]+"не обязательны"*(not arr[2]))
        print("Кушать","хочется"*arr[3]+"не хочется"*(not arr[3]))
        print("Пара уже 3+", "Да"*arr[4]+"Нет"*(not arr[4]))
        print("Есть ли еще пары?", "Да"*arr[5]+"Нет"*(not arr[5]))
        print(f"предмет сложен на {arr[6]}")
        arr.append(int(input("Пойдешь ли на пару? ")))
        data.append(arr)
    return data

def regression(X,y): # X – это список m штук векторов
    n = len(y)
    M = []
    b = []
    M.append([sum(x) for x in X]+[n])
    b.append(sum(y))
    for _,xl in enumerate(X):
        M.append([dot(x,xl) for x in X]+[sum(xl)])
        b.append(dot(y,xl))
    beta = gauss(np.array(M,dtype="float64"),np.array(b,dtype="float64"))
    return beta # свободный – в конце, т.е. beta[-1]есть alpha

def main():
    # Первоначальные данные
    data= [[1, 1, 0, 1, 0, 0, 3, 1], [1, 0, 1, 1, 1, 0, 5, 1], [1, 1, 1, 0, 0, 0, 3, 1], [1, 1, 0, 1, 1, 0, 4, 1], [1, 1, 1, 0, 1, 0, 4, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 5, 1], [0, 1, 0, 0, 0, 0, 3, 0], [1, 0, 0, 0, 1, 0, 2, 0], [0, 1, 0, 0, 1, 1, 3, 1], [1, 0, 1, 1, 1, 1, 4, 1], [0, 1, 0, 1, 0, 0, 5, 0], [1, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 2, 0], [0, 1, 0, 1, 1, 0, 2, 0], [1, 1, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 5, 1], [1, 0, 0, 0, 1, 1, 3, 1], [1, 0, 1, 0, 0, 0, 4, 1], [0, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 5, 0], [0, 1, 1, 0, 1, 1, 4, 1], [1, 1, 1, 1, 1, 1, 5, 1], [0, 1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 5, 1], [0, 1, 0, 1, 0, 1, 3, 1], [0, 1, 1, 0, 0, 1, 4, 1], [0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 3, 1], [0, 0, 0, 0, 0, 0, 5, 0]]
    # Будем переводжить их в нужный формат
    X = [[],[],[],[],[],[],[]]
    y = []
    # Но сначала прошкалируем
    dat = scale(data)
    # Переводим в нужный вид, что бы передать в функцию
    for i in range(len(dat[:-10])):
        for j in range(7):
            X[j].append(dat[i][j])
        y.append(dat[i][-1])
    # Вычисляем коофициенты
    beta = regression(X,y)
    # Вычисляем ошибку по квадратам
    R = mult_r_squared([i[:-1] for i in dat[-10:]], y, beta)
    # Проводим тест на тестовой выборке
    for i in range(20,30):
        Y = mult_predict(dat[i][:-1],beta)
        answer = round(exp(Y)/(1+exp(Y)))
        if data[i][-1]!=answer:
            print(f"Ошибка предсказания, должно быть {data[i][-1]}, а получилось {answer}, data:{data[i]}")
    print(f"Ошибка по квадратам {R}")

if __name__ == "__main__":
    main()
