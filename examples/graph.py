import numpy as np
import matplotlib.pyplot as plt

gradient_1 = [0.40004185934, 0.388614998, 0.2888692991, 0.218979389, 0.146492577, 0.11279497, 0.0106752036]
e_1 = -2.16620900180717

gradient_2 = [0.20763983208, 0.1719688594440, 0.1374934176896, 0.09899132135]
e_2 = -2.1812277696601

e_full = -2.2183708118566
e_hf = -2.15380950

plt.plot(gradient_1, '--o', label='small basis')
plt.plot(gradient_2, '--o', label='large basis')
plt.axhline(0.1, linestyle='--', linewidth=0.5, color='black')
plt.xlabel('step')
plt.ylabel('gradient (H)')
plt.legend()

plt.figure()

plt.ylabel('Energy (H)')
plt.axhline(e_hf, linestyle='-', linewidth=2.5, color='green')
plt.axhline(e_1, linestyle='-', linewidth=2.5, color='red')
plt.axhline(e_2, linestyle='-', linewidth=2.5, color='red')
plt.axhline(e_full, linestyle='-', linewidth=2.5, color='blue')


plt.show()