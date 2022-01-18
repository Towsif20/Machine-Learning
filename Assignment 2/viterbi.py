import numpy as np
from scipy.stats import norm
import scipy as sp

observations = []
N = 0
states = []
transition_matrix = []
emission_matrix = []
means = []
standard_deviations = []

with open('Sample input and output for HMM/Input/data.txt') as file:
    for readline in file:
        observations.append(float(readline.strip()))

observations = np.array(observations)

# print(observations)

with open('Sample input and output for HMM/Input/parameters.txt') as file:
    N = int(file.readline())

    for i in range(N):
        states.append(i)

    for i in range(N):
        line = file.readline()
        temp = []
        for w in line.split():
            temp.append(float(w.strip()))
        
        transition_matrix.append(temp)

    line = file.readline()
    for w in line.split():
        means.append(float(w.strip()))

    line = file.readline()
    for w in line.split():
        standard_deviations.append(float(w.strip()))

means = np.array(means)
standard_deviations = np.array(standard_deviations)

# pi = np.linalg.eig(transition_matrix)[0]

def initial_prob(transition_matrix):
    lam, vec = sp.linalg.eig(transition_matrix, left=True, right=False)
    # print(lam, vec)
    evec1 = vec[:,np.isclose(lam, 1)]
    evec1 = evec1[:,0]
    # print(evec1)
    pi = evec1/evec1.sum()
    return pi

pi = initial_prob(transition_matrix)

# pi /= pi.sum()

# print(pi)

# print(states)
# print(transition_matrix)
# print(means)
# print(standard_deviations)

def gaussian_distribution(x, mean, std):
    p = norm.pdf(x, loc=mean, scale=np.sqrt(std))
    if p == 0.0:
        p = 1e-323
    return p


for st in states:
    temp = []
    for o in observations:
        p = gaussian_distribution(o, means[st], standard_deviations[st])
        temp.append(p)
    
    emission_matrix.append(temp)



# pi = np.linalg.eig(transition_matrix)[0]

# observations = [2,0,2]
# transition_matrix = [[0.7, 0.3],[0.4, 0.6]]
# emission_matrix = [[0.2, 0.4, 0.4],[0.5, 0.4, 0.1]]


def viterbi(observations, states, pi, transition_matrix, emission_matrix):
    length = len(observations)
    # print(length)
    probability_table = np.array([[float(0)]*length]*N)
    previous_table = np.array([[-1]*length]*N)

    # first entry from stationary probability, without transition
    for st in states:
        # probability_table[st][0] = np.log(pi[st] * gaussian_distribution(observations[0], means[st], standard_deviations[st]))
        probability_table[st][0] = np.log(pi[st] * emission_matrix[st][0])

    
    for i in range(1, length):
        
        for current in states:
            max_incoming_prob = float('-inf')
            prev_selected = -1

            for prev in states:
                incoming_prob = probability_table[prev][i-1] + np.log(transition_matrix[prev][current])
                
                if incoming_prob > max_incoming_prob:
                    max_incoming_prob = incoming_prob
                    prev_selected = prev

            
            # max_prob = max_incoming_prob + np.log(gaussian_distribution(observations[i], means[current], standard_deviations[current]))
            max_prob = max_incoming_prob + np.log(emission_matrix[current][i])

            probability_table[current][i] = max_prob
            previous_table[current][i] = prev_selected

    # print(probability_table)
    # print(previous_table)
    # print(np.argmax(probability_table))

    max = float('-inf')
    best = -1

    most_likely_seuquence = []
    prev = -1

    for i in range(N):
        if probability_table[i][length-1] > max:
            max = probability_table[i][length-1]
            best = i

    most_likely_seuquence.append(best)
    prev = best

    # print(probability_table.shape)
    for i in range(length-2, -1, -1):
        most_likely_seuquence.insert(0, previous_table[prev][i+1])
        prev = previous_table[prev][i+1]

    # print((most_likely_seuquence))

    return most_likely_seuquence


# pi = np.array([0.5, 0.5])

# print(pi)




sequence = viterbi(observations, states, pi, transition_matrix, emission_matrix)





c_t = np.array([0.0]*len(observations))

def forward_algorithm(transition_matrix, emission_matrix, pi):
    length = len(observations)
    alpha = np.array([[0.0]*length]*N)

    for i in range(N):
        alpha[i][0] = pi[i] * emission_matrix[i][0]
        # c_t[0] += alpha[i][0]

    c_t[0] = 1.0/np.sum(alpha[:, 0])
    for i in range(N):
        alpha[i][0] *= c_t[0]
            

    for j in range(1, length):
        
        for i in range(N):
            for k in range(N):
                temp = alpha[k][j-1] * transition_matrix[k][i] * emission_matrix[i][j]
                alpha[i][j] += temp
            
            # c_t[j] += alpha[i][j]

        c_t[j] = 1.0/np.sum(alpha[:, j])
        for i in range(N):
            alpha[i][j] *= c_t[j]

        # dekho thik ache naki
    
    return alpha


def backward_algorithm(transition_matrix, emission_matrix):
    length = len(observations)
    beta = np.array([[0.0]*length]*N)

    for i in range(N):
        beta[i][length-1] = 1.0

    for j in range(length-2, -1, -1):
        
        for i in range(N):
            for k in range(N):
                temp = beta[k][j+1] * transition_matrix[i][k] * emission_matrix[k][j+1]
                beta[i][j] += temp
            
            beta[i][j] *= c_t[j]
            # c_t[j] += alpha[i][j]

        # dekho thik ache naki
    
    return beta




transition_matrix_updated = []
emission_matrix_updated = []
mu_updated = np.array([0.0]*N)
sigma_updated = np.array([0.0]*N)

for i in range(N):
  one_row = []
  for j in range(N):
    one_row.append(transition_matrix[i][j])
  transition_matrix_updated.append(one_row)

transition_matrix_updated = np.array(transition_matrix_updated)
# print(transition_matrix_updated)



for i in range(N):
  one_row = []
  for j in range(len(observations)):
    one_row.append(emission_matrix[i][j])
  emission_matrix_updated.append(one_row)

emission_matrix_updated = np.array(emission_matrix_updated)
# print(emission_matrix_updated)

def maximization_step(alpha, beta):
    length = len(observations)
    global transition_matrix_updated
    global emission_matrix_updated
    global mu_updated
    global sigma_updated

    for i in states:
        for j in states:
            denominator = 0.0
            numerator = 0.0
            
            for k in range(length-1):

                numerator += alpha[i][k] * beta[j][k+1] * transition_matrix_updated[i][j] * emission_matrix_updated[j][k+1]

                denominator += alpha[i][k] * beta[i][k] / c_t[k]

            transition_matrix_updated[i][j] = numerator / denominator

    # print(transition_matrix_updated)

    joint_alpha_beta = alpha * beta

    # print(joint_alpha_beta)

    for col in range(length):
        joint_alpha_beta[:, col] /= joint_alpha_beta[:, col].sum()

    # print(joint_alpha_beta)

    for st in states:
        temp = joint_alpha_beta[st] * observations
        mu_updated[st] = np.sum(temp) / np.sum(joint_alpha_beta[st])

    # print(mu_updated)

    for st in states:
        diff = observations - mu_updated[st]
        temp = np.sum(joint_alpha_beta[st] * diff * diff) / np.sum(joint_alpha_beta[st])
        sigma_updated[st] = np.sqrt(temp)

    # print(sigma_updated)

    
    for st in states:
        for j in range(len(observations)):
            emission_matrix_updated[st][j]  = gaussian_distribution(observations[j], mu_updated[st], sigma_updated[st])

    


def baum_welch():
    global transition_matrix_updated
    global emission_matrix_updated
    global sigma_updated
    global mu_updated
    global pi

    for _ in range(20):
        pi = initial_prob(transition_matrix_updated)

        alpha = forward_algorithm(transition_matrix_updated, emission_matrix_updated, pi)
        beta = backward_algorithm(transition_matrix_updated, emission_matrix_updated)

        maximization_step(alpha, beta)

    sigma_updated = np.power(sigma_updated, 2)

    return




# alpha = forward_algorithm(transition_matrix, emission_matrix, pi)
# beta = backward_algorithm(transition_matrix, emission_matrix)

# maximization_step(alpha, beta)


sequence = viterbi(observations, states, pi, transition_matrix, emission_matrix)

filename = 'output/viterbi_wo_learning.txt'

f = open(filename, "w")

one, zero = 0, 0
for s in sequence:
    if(s == 0):
        # print('\"El Nino\"')
        f.write('\"El Nino\"\n')
        zero += 1
    else:
        # print('\"La Nina\"')
        f.write('\"La Nina\"\n')
        one += 1

print(one, zero)

f.close()


baum_welch()


print(transition_matrix_updated)
print(mu_updated)
print(sigma_updated)
print(pi)

filename = 'output/parameters_learned.txt'
f = open(filename, "w")
f.write(str(N) + '\n')
for i in states:
    for j in states:
        f.write(str(transition_matrix_updated[i][j]) + "\t")
    
    f.write("\n")

for i in states:
    f.write(str(mu_updated[i]) + "\t")

f.write("\n")

for i in states:
    f.write(str(sigma_updated[i]) + "\t")

f.write("\n")

for i in states:
    f.write(str(pi[i]) + "\t")

f.write("\n")

f.close()

filename = "output/viterbi_after_learning.txt"

f = open(filename, "w")

sequence = viterbi(observations, states, pi, transition_matrix_updated, emission_matrix_updated)
one, zero = 0, 0
for s in sequence:
    if(s == 0):
        # print('\"El Nino\"')
        f.write('\"El Nino\"\n')
        zero += 1
    else:
        # print('\"La Nina\"')
        f.write('\"La Nina\"\n')
        one += 1

f.close()
print(one, zero)
