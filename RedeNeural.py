import numpy as np

class RedeNeural:
    FUNCAO_SGIMOIDE = 0
    FUNCAO_TANH = 1
    FUNCAO_ARCTAN = 2

    def __init__(self):
        # Semente do gerador pseudo-aleatório.
        # Irá gerar sempre os mesmos números.
        np.random.seed(1)

        # Modelando um neurônio com três conexões de entradas
        # e uma de saída.
        # Atribui-se valores aleatórios (pesos) à uma matriz 3x1
        # entre -1 e 1 e próximos de zero.
        self.pesos_sinapse = 2 * np.random.random((3, 1)) - 1

        # Tipo de função de ativação a ser usada
        self.tipo_funcao = self.FUNCAO_SGIMOIDE

    def funcao_ativacao(self, x, derivada=False):
        if self.tipo_funcao == self.FUNCAO_SGIMOIDE:
            if derivada:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))
        elif self.tipo_funcao == self.FUNCAO_ARCTAN:
            if derivada:
                return 1 / (1 + x ** 2)
            return np.arctan(x)
        elif self.tipo_funcao == self.FUNCAO_TANH:
            if derivada:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)

    # Função de treinamento por uma processo de tentativa e erro
    # A cada iteração são ajustados os pesos das sinápses
    def treinamento(self, conjunto_entrada, conjunto_saida, iteracoes=10000):
        for i in iter(range(iteracoes)):
            saida = self.pensamento(conjunto_entrada)

            # Cálculo do ero: diferença entre a saída prevista e calculada
            erro = conjunto_saida - saida

            ajuste = np.dot(conjunto_entrada.T, erro * self.funcao_ativacao(saida, True))

            self.pesos_sinapse += ajuste

    # Pensamento da rede neural
    def pensamento(self, entradas):
        return self.funcao_ativacao(np.dot(entradas, self.pesos_sinapse))

    # Define valores aleatórios para os pesos da rede neural
    def reinicia(self):
        self.pesos_sinapse = 2 * np.random.random((3, 1)) - 1