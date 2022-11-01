
def zeros1d(x): # 1d zero matrix
    z = [0 for i in range(len(x))]
    return z

def add1d(x, y): 
    """ Add one dimensional arrays together """
    if len(x) != len(y):
        print("Dimension mismatch")
        exit()
    else:
        z = [x[i] + y[i] for i in range(len(x))]
        return z

def dense(nunit, x, w, b, activation): # define a single dense layer followed by activation

    # this is ripe for a list comprehension
    res = []
    for i in range(nunit):
        z = neuron(x, w[i], b[i], activation)
        # print(z)
        res.append(z)
    return res

def neuron(x, w, b, activation): # perform operatoin on a single neuron and return a 1d array

    tmp = zeros1d(x[0])

    for i in range(len(x)):
        tmp = add1d(tmp, [(float(w[i]) * float(x[i][j])) for j in range(len(x[0]))])

    if activation == "sigmoid":
        yp = sigmoid([tmp[i] + b for i in range(len(tmp))])
    elif activation == 'relu':
        yp = relu([tmp[i]+ b for i in range(len(tmp))])
    else:
        print("Invalid activation function--->")
    return yp

def relu(x): 
    """ Relu activation function """
    # print(x)
    y = []
    for i in range(len(x)):
        if x[i] >= 0:
            y.append(x[i])
        else:
            y.append(0)
    
    # print(y)
    return y

def sigmoid(x):
    """ Sigmoid function """
    import math
    z = [1/(1 + math.exp(-x[kk])) for kk in range(len(x))]
    return z

def dot(A, B):
    """
    Returns the product of the matrix A * B where A is m an n and B is n by 1 matrix
        :param A: The first matrix - ORDER MATTERS! 
        :param B: The second matrix

        :return: the peoduct of the two matrices    
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = 1
    if colsA != rowsB:
        raise ArithmeticError("Number of A columns must equal number of B rows.")

    # Section 2: Store matrix mulitplication in a new matrix
    C = zeros(rowsA, colsB)
    for i in range(rowsA):
        total = 0
        for ii in range(colsA):
            total += A[i][ii] * B[ii]
            C[i] = total

    return C

def zeros(rows, cols):
    """ 
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have

        :return list of lists that form the matrix
    """
    matrix = []
    while len(matrix) < rows:
        while len(matrix[-1]) < cols:
            matrix[-1].append(0.0)

    return matrix

def transpose(matrix):
    """
    Returns a transpose of a matrix.
        :param matrix: The matrix to be transposed
        
        :return: The transpose of the given matrix
    """

    # Section 1: if a 1D array, convert to a 2d array = matrix
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    # Section 2: Get dimensions
    rows = len(matrix)
    cols = len(matrix[0])

    # Section 3: transposed_matrix is zeros matrix with transposed dimensions
    transposed_matrix = zeros(cols, rows)

    # Section 4: Copy values from matrix to its transpose MT
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: the matrix to be printed
    """

    for row in M:
        print([round(x, decimals) + 0 for x in row])

def classification_report(ytrue, ypred):
    """ Print prediction results in terms of metrics and confusion matrix """
    tmp = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytrue)):
        if ytrue[i] == 1 and ypred[i] == 1: # find true positive
            TP += 1
        if ytrue[i] == 0 and ypred[i] == 0: # find true negative
            TN += 1
        if ytrue[i] == 0 and ypred[i] == 1: # find false positive
            FP += 1
        if ytrue[i] == 1 and ypred[i] == 0: # find false negative
            FN += 1
    accuracy = tmp / len(ytrue)
    conf_matrix = [[TN, FP], [FN, TP]]
    # print(TP, FP, FN, TN)

    print("Accuracy: " + str(accuracy))
    print("Confusion Matrix:")
    print(print_matrix(conf_matrix))