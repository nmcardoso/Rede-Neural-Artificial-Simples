#!/usr/bin/python3

from numpy import array
import RedeNeural

if __name__ == "__main__":
    # Iniciando uma rede neural artificial
    rede_neural = RedeNeural.RedeNeural()

    print("-- Calculando a função booleana OU --\n")
    print("Pesos (aleatórios) antes do treinamento:")
    print(rede_neural.pesos_sinapse)

    # Conjunto de treinamento: 4 exemplos com 3 valores
    # de entrada e 1 de saída para função OU.
    conjunto_entrada = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    conjunto_saida = array([[0, 1, 1, 1]]).T

    # Treinamento da função OU para função sigmóide
    print("\n--- Funcao de ativacao: SIGMOIDE")
    print("Pesos depois do treinamento:")
    rede_neural.treinamento(conjunto_entrada, conjunto_saida)
    print(rede_neural.pesos_sinapse)
    print("Resposta para 1 OU 0 (bias = 0):")
    print(rede_neural.pensamento(array([1, 0, 0])))

    # Treinamento da função OU para função tangente hiperbólica
    print("\n--- Funcao de ativacao: TANGENTE HIPERBOLICA")
    print("Pesos depois do treinamento:")
    rede_neural.reinicia()
    rede_neural.tipo_funcao = rede_neural.FUNCAO_TANH
    rede_neural.treinamento(conjunto_entrada, conjunto_saida, 20000)
    print(rede_neural.pesos_sinapse)
    print("Resposta para 1 OU 0 (bias = 0):")
    print(rede_neural.pensamento(array([1, 0, 0])))

    # Treinamento da função OU para função arco tangente
    print("\n--- Funcao de ativacao: ARCO TANGENTE")
    print("Pesos depois do treinamento:")
    rede_neural.reinicia()
    rede_neural.tipo_funcao = rede_neural.FUNCAO_ARCTAN
    rede_neural.treinamento(conjunto_entrada, conjunto_saida)
    print(rede_neural.pesos_sinapse)
    print("Resposta para 1 OU 0 (bias = 0):")
    print(rede_neural.pensamento(array([1, 0, 0])))
