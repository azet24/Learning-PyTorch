import torch
import matplotlib.pyplot as plt

def gen_monoms(deg, x, y):
    List = []
    for i in range(deg + 1):  # Zmieniamy na deg + 1
        for j in range(i + 1):
            List.append(x ** (i - j) * y ** j)
    List = torch.stack(List, dim=2)
    return List  # Zwracamy listę tensorów

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
device = 'cuda:0'  # Używamy GPU do przyspieszenia obliczeń
PAR = 9


# Tworzenie dwuwymiarowego tensora
x = torch.linspace(-1.4, 1.4, steps=64, device=device)  # Tworzymy liniową przestrzeń od -1 do 1 z 64 krokami
y = torch.linspace(-1.4, 1.4, steps=64, device=device)
xx, yy = torch.meshgrid(x, y, indexing='ij')  # Tworzymy siatkę współrzędnych

# Definiowanie funkcji
w = torch.FloatTensor(xx.shape).uniform_(-0.1, 0.1).to(device)  # Losowe szumy
f = torch.sin(xx**2 + yy**2)**2 + w  # Funkcja, którą chcemy aproksymować

torch.manual_seed(42)  # Ustawiamy ziarno losowości dla powtarzalności wyników
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

# Przewidywanie modelu
model.eval()  
with torch.no_grad():  
    predictions = model(X)  

# Przygotowanie danych do wykresu
x = X[:, :, 0].cpu().numpy()
y = X[:, :, 1].cpu().numpy()
z_real = f.cpu().numpy()
z_pred = predictions.squeeze(-1).cpu().numpy()  # Spłaszczamy ostatni wymiar

# Tworzenie wykresu
fig = plt.figure(figsize=(12, 6))  
ax = fig.add_subplot(111, projection='3d')  

# Wykres dla oryginalnych danych
scatter = ax.scatter(x, y, z_real, color='red', s=.7, alpha=0.3)  

# Wykres dla przewidywań modelu
surface = ax.plot_surface(x, y, z_pred, color='blue', alpha=0.4)  

ax.set_title('Dane wejściowe i przewidywania modelu') 
ax.set_xlabel('X') 
ax.set_ylabel('Y') 
ax.set_zlabel('Z') 

plt.show()

