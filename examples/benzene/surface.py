import numpy as np
import matplotlib.pyplot as plt

# Definim el rang de valors de x i y
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)

# Creem una malla de coordenades
X, Y = np.meshgrid(x, y)

# Calculem la funció sin(x) * sin(y)
Z = np.sin(X) * np.sin(Y*0.8+2)

# Creem el mapa de contorn
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
contour = plt.contour(X, Y, Z, levels=8, colors="black")  # Només línies negres

plt.colorbar(contour, label="sin(x) * sin(y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour map sin(x) * sin(y)")
plt.show()
