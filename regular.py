import os
import math

os.system('cls')

#regulazation
def compute_gradient(A, b, x, lambda_reg):
    gradient = [0.0, 0.0]
    
    for i in range(len(A)):
        prediction = sum(A[i][j] * x[j] for j in range(len(x)))
        error = prediction - b[i]
        for j in range(len(x)):
            gradient[j] += error * A[i][j]

    for j in range(len(x)):
        gradient[j] += lambda_reg * x[j]
    
    return gradient

#given's rotation
def givens_rotation(a, b):
    r = math.sqrt(pow(a,2) + pow(b,2))
    if r == 0:
        return 0, 0
    c = a / r
    s = -b / r
    return c, s

def apply_givens_rotation(A, b, i, j):
    c, s = givens_rotation(A[i][j], A[j][j])
    
    for k in range(len(A)):
        temp = c * A[i][k] + s * A[j][k]
        A[j][k] = -s * A[i][k] + c * A[j][k]
        A[i][k] = temp

    temp = c * b[i] + s * b[j]
    b[j] = -s * b[i] + c * b[j]
    b[i] = temp

def solve_using_givens(A, b):
    n = len(A)
    
    for j in range(n - 1):
        for i in range(j + 1, n):
            if A[i][j] != 0:
                apply_givens_rotation(A, b, i, j)

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x


matrix = [[1.03, 0.991],
     [0.991, 0.943]]

vector = [2.59, 2.47]

lambda_reg = 0.01

x = [0.0, 0.0]
learning_rate = 0.01
num_iterations = 1000

for iteration in range(num_iterations):
    gradient = compute_gradient(matrix, vector, x, lambda_reg)
    for j in range(len(x)):
        x[j] -= learning_rate * gradient[j]

solution = solve_using_givens(matrix, vector)

print("Regulazation solved: ", x)
print("Given's rotation: ", solution)