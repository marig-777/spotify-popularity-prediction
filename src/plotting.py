import numpy as np
import matplotlib.pyplot as plt

def graficar_circulo_correlacion(x, y, inercia_x, inercia_y,
                                 etiquetas=None,
                                 comp_x=1, comp_y=2,
                                 titulo_x="Componente", titulo_y="Componente"):
    """
    Dibuja el círculo de correlaciones para dos componentes principales.

    Parámetros
    ----------
    x, y : array-like
        Coordenadas (correlaciones) de las variables en las dos componentes elegidas.
    inercia_x, inercia_y : float
        Porcentajes de inercia (ya en %) para cada eje.
    etiquetas : list-like, opcional
        Etiquetas para cada variable. Si es None, usa 0..n-1.
    comp_x, comp_y : int
        Números de componente mostrados en los ejes (solo para el rótulo).
    titulo_x, titulo_y : str
        Texto base de cada eje.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if etiquetas is None:
        etiquetas = [str(i) for i in range(n)]

    fig, ax = plt.subplots()

    # Vectores (flechas) y etiquetas
    for i in range(n):
        ax.arrow(0, 0, x[i]*0.95, y[i]*0.95,
                 color='steelblue', alpha=0.5,
                 head_width=0.05, head_length=0.05, length_includes_head=True)
        ax.text(x[i]*1.05, y[i]*1.05, etiquetas[i],
                color='steelblue', ha='center', va='center')

    # Círculo unidad
    circulo = plt.Circle((0, 0), radius=1, color='steelblue', fill=False)
    ax.add_artist(circulo)

    # Formato del plano
    ax.axis('scaled')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='dimgrey', linestyle='--')
    ax.axvline(x=0, color='dimgrey', linestyle='--')

    # Ejes con inercia
    ax.set_xlabel(f"{titulo_x} {comp_x} ({round(inercia_x, 2)}%)")
    ax.set_ylabel(f"{titulo_y} {comp_y} ({round(inercia_y, 2)}%)")

    plt.show()