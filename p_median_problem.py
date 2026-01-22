import numpy as np
from poblacion_combinaciones import Combinaciones

class PMedian:
    _pmed_data = {
        1:  (100, 5, 5819),
        2:  (100, 10, 4093),
        3:  (100, 10, 4250),
        4:  (100, 20, 3034),
        5:  (100, 33, 1355),
        6:  (200, 5, 7824),
        7:  (200, 10, 5631),
        8:  (200, 20, 4445),
        9:  (200, 40, 2734),
        10: (200, 67, 1255),
        11: (300, 5, 7696),
        12: (300, 10, 6634),
        13: (300, 30, 4374),
        14: (300, 60, 2968),
        15: (300, 100, 1729),
        16: (400, 5, 8162),
        17: (400, 10, 6999),
        18: (400, 40, 4809),
        19: (400, 80, 2845),
        20: (400, 133, 1789),
        21: (500, 5, 9138),
        22: (500, 10, 8579),
        23: (500, 50, 4619),
        24: (500, 100, 2961),
        25: (500, 167, 1828),
        26: (600, 5, 9917),
        27: (600, 10, 8307),
        28: (600, 60, 4498),
        29: (600, 120, 3033),
        30: (600, 200, 1989),
        31: (700, 5, 10086),
        32: (700, 10, 9297),
        33: (700, 70, 4700),
        34: (700, 140, 3013),
        35: (800, 5, 10400),
        36: (800, 10, 9934),
        37: (800, 80, 5057),
        38: (900, 5, 11060),
        39: (900, 10, 9423),
        40: (900, 90, 5128),
    }

    def __init__(self, pmed_index: int):
        if pmed_index not in self._pmed_data:
            raise ValueError(f"Dataset {pmed_index} no encontrado.")
        
        self._n, self._p, self._optimo = self._pmed_data[pmed_index]
        self._cost_matrix = np.load(f"datasets/pmed{pmed_index}.npy")
        self._poblacion = Combinaciones()

    def generar_poblacion(self, tam: int) -> set:
        self._poblacion.generar(self._n, self._p, tam)
        return self._poblacion.get_datos()

    def evaluar_poblacion(self, poblacion) -> np.ndarray:
        """
        Cálcula el fitness de cada individuo en una población 
        Args:
        ----------
        poblacion : set o list
            Conjunto o lista de tuplas con los índices de las instalaciones seleccionadas.
            IMPORTANTE: Si se pasa una lista, el orden se preserva en el resultado.
                        Si se pasa un set, el orden NO está garantizado.

        Returns:
        -------
        costos : np.ndarray
            Vector de tamaño (pop_size,) con el costo total de cada individuo.
            (Menor costo = mejor solución en el pMP).
        """
        # Convertir a array (preserva orden si es lista)
        poblacion_arr = np.array(list(poblacion))
        
        # Validación de entrada
        if poblacion_arr.size == 0:
            return np.array([], dtype=float)

        if poblacion_arr.ndim == 1:
             # Si es 1D (un solo individuo pero sin dimensión de batch?), lo ajustamos o lanzamos error.
             # Esperamos shape (pop_size, p). Si paso [ind], es (1, p).
             # Si paso [] -> size 0 handled above.
             pass

        # cost_matrix tiene shape (n, n)
        # poblacion_arr tiene shape (pop_size, p)
        
        # Esto genera una matriz de shape (n, pop_size, p)
        # Para cada cliente (n),
        # para cada individuo (pop_size),
        # para cada mediana del individuo (p),
        try:
             dist_clientes_a_inst = self._cost_matrix[:, poblacion_arr]   # (n, pop_size, p)
        except IndexError as e:
             # Puede pasar si los indices en poblacion_arr no son enteros validos o exceden n
             raise ValueError(f"Error al acceder a cost_matrix con poblacion: {e}")

        # Para cada cliente y cada individuo, buscamos la distancia más pequeña entre todas sus medianas
        # Resultado: (n, pop_size)
        if dist_clientes_a_inst.shape[2] == 0:
             # Si p=0 (individuos vacíos)
             return np.full((dist_clientes_a_inst.shape[1],), np.inf)

        dist_min_por_cliente = np.min(dist_clientes_a_inst, axis=2)

        # Para cada individuo ind, sumamos los costos de todos los clientes
        # Resultado: (pop_size,)
        costos = np.sum(dist_min_por_cliente, axis=0)
        
        return costos.astype(float)


    # Se incluyen los metodos de la clase
    def __str__(self):
        return f"PMedian(n={self._n}, p={self._p}, optimal_value={self._optimo})"

    def __eq__(self, other):
        return self._n == other.n and self._p == other.p and self._optimo == other.optimo

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def optimo(self):
        return self._optimo

    @property
    def cost_matrix(self):
        return self._cost_matrix

    @property
    def poblacion(self):
        return self._poblacion.get_datos()


if __name__ == "__main__":
    # Bloque de prueba
    
    n_problem = 25
    try:
        pmed = PMedian(n_problem)
        print(f"PMedian inicializado con pmed=1. n={pmed.n}, p={pmed.p}")
        
        pob = pmed.generar_poblacion(1000)
        print(f"Población generada. Tamaño: {len(pob)}")
        
        fitness = pmed.evaluar_poblacion(pob)
        print("Fitness calculado:", fitness)

    except Exception as e:
        print(f"Error durante la prueba: {e}")

