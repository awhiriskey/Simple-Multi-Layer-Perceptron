import numpy as np 
import matplotlib.pyplot as plt

class MLP(object):

    
    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
    
        #Variable initialization
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
    
    def randomise(self):
        #weight and bias initialization
        self.w1 = np.random.uniform(size=(self.n_input,self.n_hidden))
        self.b1 = np.random.uniform(size=(1,self.n_hidden))
        
        self.w2 = np.random.uniform(size=(self.n_hidden,self.n_output))
        self.b2 = np.random.uniform(size=(1,self.n_output))
        
    def forward(self, X):
        #Forward Propogation
        z1_no_b = np.dot(X,self.w1)
        z1 = z1_no_b + self.b1
        h = self.sigmoid(z1)
        z2_no_b=np.dot(h,self.w2)
        z2 = z2_no_b+ self.b2
        o = self.sigmoid(z2)
        return z1, h, z2, o
        
    def backwards(self, z1, h, z2, o, X, y):
        E = y-o
        slope_o = self.derivatives_sigmoid(o)
        dz2 = E * slope_o
        dw2 = np.dot(h.T,dz2)
        db2 = dz2
        
        da1 = np.dot(dz2, self.w2.T)
        slope_h = self.derivatives_sigmoid(h)
        dz1 = da1 * slope_h
        dw1 = np.dot(X.T,dz1)
        db1 = dz1

        return  dz2, dz1, dw1, dw2, E
    
    def update_weights(self, h, db2, db1,  dw1, dw2):
        self.w2 += dw2 *self.lr
        self.b2 += np.sum(db2, axis=0,keepdims=True) *self.lr
        self.w1 += dw1 *self.lr
        self.b1 += np.sum(db1, axis=0,keepdims=True) *self.lr
        
    def predict(self, X_test):
        z1, h, z2, o =self.forward(X_test)
        return o
    
    def get_error(self, y, o):
        error = 1/2*(np.sum((o - y)**2))
        return error
    
    #Sigmoid Function
    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

    #Derivative of Sigmoid Function
    def derivatives_sigmoid(self, x):
        return x * (1 - x)    
    
    
    def fit(self, X, y, filename, epoch=50000):
        self.error_list = []
        path = open(filename, 'w')
        
        for e in range(epoch):
            if e == 0:
                print("Starting, Creating weights")
                self.randomise()
                print("Beginning forward pass")
            z1, h, z2, o =self.forward(X)
            error = self.get_error(y, o)
            path.write("%.10f\n" % error.mean())
            self.error_list.append(error)
            dz2, dz1, dw1, dw2, E = self.backwards(z1, h, z2, o, X, y)
            self.update_weights(h, dz2, dz1, dw1, dw2)
            if (e%10000 == 0):
                print("At stage:%d" %e)
                #print("error:", error)
        path.close()
        


        
        
if __name__ == "__main__":
    
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    
    
    print("############ PART 1 ###################")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    nn = MLP(2, 2, 1)
    nn.fit(X, y, "part1_error.txt")
    
    print("############ PART 2 ###################")
    print("Predicting output")
    o = nn.predict(X)
    print(o)

    print("Error at end of training",nn.error_list[-1].mean())
    error = nn.get_error(y, o)
    print("Error on test set",error.mean())
    
    print("############ PART 3 ###################")
    X = np.random.uniform(low=-1, high=1, size=(50,4))
    #X = scale_range(X, 0, 1)
    y_list = []
    for m in X:
        output = np.sin(m[0] - m[1] + m[2] - m[3])
        y_list.append(output)

    y = np.vstack(y_list)
    y = scale_range(y, 0, 1)
    
    X_train = X[0:40]
    y_train = y[0:40]
    X_test = X[40:50]
    y_test = y[40:50]
    
    nn = MLP(4, 10, 1)
    nn.fit(X_train, y_train, "part2_error.txt")
    o = nn.predict(X_test)
    print("predicted output:", o)
    print("Actual output:", y_test)
    
    print("############ PART 4 ###################")
    
#     print("Loss at end of training",nn.loss_list[-1])
#     loss = nn.get_loss(y_test, o)
#     print("Loss on test set",loss)
    print("Error at end of training",nn.error_list[-1].mean())
    error = nn.get_error(y_test, o)
    print("Error on test set",error.mean())
    
