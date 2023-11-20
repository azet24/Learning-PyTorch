#This is a project of implementing polynomial regression using expended space of features of diffrent monomials and then implementing linear regression

import torch
import matplotlib.pyplot as plt

def gen_monoms(deg, x, y):
    List = []
    for i in range(deg + 1):  # Chanche for deg + 1
        for j in range(i + 1):
            List.append(x ** (i - j) * y ** j)
    List = torch.stack(List, dim=2)
    return List  # Return list of tensors

class RegressionModel(torch.nn.Module):
    def __init__(self, n=2):
        super(RegressionModel, self).__init__()
        self.poly = torch.nn.Linear(int(((n + 2) * (n + 1)) / 2), 1)
        self.n = n

    def forward(self, X):
        x, y = X[:, :, 0], X[:, :, 1]
        w = gen_monoms(self.n, x, y)
        output = self.poly(w)
        return output

dtype = torch.float
device = 'cuda:0'  # Use GPU
PAR = 9


# Creating 2D tensor
x = torch.linspace(-1.4, 1.4, steps=64, device=device)
y = torch.linspace(-1.4, 1.4, steps=64, device=device)
xx, yy = torch.meshgrid(x, y, indexing='ij')  # Creating grid of coordinates

# Definiowanie funkcji
w = torch.FloatTensor(xx.shape).uniform_(-0.1, 0.1).to(device)  # Creating random noise
f = torch.sin(xx**2 + yy**2)**2 + w  # Creating data for approximation

torch.manual_seed(42)  # Setting random seed
X = torch.stack((xx, yy), dim=2)
X = X.to(device) 
model = RegressionModel(PAR)
model.to(device)  
loss_fn = torch.nn.MSELoss()  
optimizer = torch.optim.SGD(params=model.parameters(), lr=.01)  
epochs = range(400)
for epoch in epochs:  
    model.train()  
    y_pred = model(X).squeeze(-1)
    loss = loss_fn(y_pred, f)  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    if epoch % 50 == 49:  
        print(f"Epoch {epoch} | Loss: {loss}")  

# Model predictions
model.eval()  
with torch.no_grad():  
    predictions = model(X)  

# Preparing data for plot
x = X[:, :, 0].cpu().numpy()
y = X[:, :, 1].cpu().numpy()
z_real = f.cpu().numpy()
z_pred = predictions.squeeze(-1).cpu().numpy()  # Squeeze last dimension of tensor

# Creating plot
fig = plt.figure(figsize=(12, 6))  
ax = fig.add_subplot(111, projection='3d')  

# Plot for input data
scatter = ax.scatter(x, y, z_real, color='red', s=.7, alpha=0.3)  

# Plot for predictions
surface = ax.plot_surface(x, y, z_pred, color='blue', alpha=0.4)  

ax.set_title('Dane wejściowe i przewidywania modelu') 
ax.set_xlabel('X') 
ax.set_ylabel('Y') 
ax.set_zlabel('Z') 

plt.show()

