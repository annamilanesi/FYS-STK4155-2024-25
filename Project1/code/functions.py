#Franke function definition
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# calculate MSE
def calculateMSE(f, fpred):
  mse=np.mean((f-fpred)**2)
  return mse

#calculate R2
def R2(f_data, f_model): #calculate R
    return 1 - (np.sum((f_data - f_model) ** 2) / np.sum((f_data - np.mean(f_data)) ** 2))


#def function to calculate the design matrix
def design_matrix(x, y, degree):
    # x and y are expected to be 1-dimensional arrays
    x = x.ravel()  # Flatten x
    y = y.ravel()  # Flatten y
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)  # Number of polynomial terms
    X = np.ones((N, l))

    idx = 0
    for i in range(degree + 1):
        for j in range(i + 1):
            X[:, idx] = (x ** (i - j)) * (y ** j)
            idx += 1
    return X

# beta OLS
def calcuateB(X, f):
  beta = np.linalg.pinv(X.T @ X) @ X.T @ f
  return beta


# beta Ridge
def betaRIDGE(X, f, lm):
  I = np.identity(X.shape[1])  # Identity matrix
  beta_RIDGE = np.linalg.inv(X.T @ X + lm * I) @ X.T @ f
  return beta_RIDGE

#function to divide teh data into train and test and to scale them
def split_and_scale(X, f, test_size=0.2, with_std=True):
    X_train, X_test, f_train, f_test = train_test_split(X, f, test_size=test_size)
    scaler_X = StandardScaler(with_std=with_std)
    scaler_X.fit(X_train)
    scaler_f = StandardScaler(with_std=with_std)
    scaler_f.fit(f_train.reshape(-1, 1))

    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    f_train_scaled = scaler_f.transform(f_train.reshape(-1, 1)).ravel()
    f_test_scaled = scaler_f.transform(f_test.reshape(-1, 1)).ravel()

    return X_train_scaled, X_test_scaled, f_train_scaled, f_test_scaled
