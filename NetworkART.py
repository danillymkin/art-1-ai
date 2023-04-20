from Neuron import Neuron
import numpy as np


class NetworkART:
    def __init__(self, first_vector, P, L):
        self.L = L
        self.P = P

        self.vector_length = len(first_vector)
        self.neurons = [Neuron(np.ones((self.vector_length,), dtype=int), self.L), Neuron(first_vector, self.L)]

    def recognition(self, input_vectors):
        print("\nРаспознание вектора...")
        print("Текущий образец: ", input_vectors)
        S = []

        for neuron in self.neurons:
            S.append(neuron.exit_S(input_vectors))

        for i in range(len(self.neurons)):
            print(f"Выходные значения нейрона {i}: S = {S}")

        winner = S.index(max(S))

        if winner == 0:
            print("Победил нераспределенный нейрон")
            print("\nОбучаем нейронную сеть...")
            self.learning_memorization(input_vectors)
            print("В сеть был добавлен новый нейрон!")
        else:
            print(f"Победил {winner} нейрон")

            C = self.neurons[winner].T * input_vectors
            comparison_result = (self.vector_length - np.sum(np.logical_xor(C, input_vectors))) / self.vector_length

            if comparison_result > self.P:
                print(f"\nСравниваем уровни: {comparison_result} (фактический) > {self.P} (заданный)")
                print("Переобучение {w}-ого нейрона...".format(w=winner))
                self.neurons[winner].T = C
                self.neurons[winner].B = self.L * C / (self.L - 1 + np.sum(C > 0))
            else:
                print(f"\nСравниваем уровни: {comparison_result} (фактический) <= {self.P} (заданный)")
                self.learning_memorization(input_vectors)
                print("В сеть был добавлен новый нейрон!")

        self.print_results()
        print("-" * 100)

    def learning_memorization(self, input_vectors):
        self.neurons.append(Neuron(input_vectors, self.L))

    def print_results(self):
        print("\nВесовые коэфиценты B:")
        for i, neuron in enumerate(self.neurons):
            if i == 0:
                print('Нераспределенный нейрон: ', neuron.B)
            else:
                print(f'Нейрон {i}:', neuron.B)

        print("\nВесовые коэфиценты T:")
        for i, neuron in enumerate(self.neurons):
            if i == 0:
                print('Нераспределенный нейрон: ', neuron.T)
            else:
                print(f'Нейрон {i}:', neuron.T)
