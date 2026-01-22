import random, math
import numpy as np
from itertools import chain
from collections import Counter

class Combinaciones:
    """
    Enfoque usando estructuras nativas de Python (set de tuplas).
    Ideal para: Velocidad de inserción, eliminación y búsquedas.
    """
    def __init__(self):
        # Usamos un set para garantizar unicidad y búsquedas O(1)
        self.datos = set()

    def _normalizar(self, combinacion):
        """Convierte [6, 4, 2] en (2, 4, 6) para ignorar el orden."""
        return tuple(sorted(combinacion))

    def agregar(self, datos):
        """
        Método unificado inteligente.
        - Si recibe [1, 5, 3] -> Agrega una sola combinación.
        - Si recibe [[1, 5, 3], [2, 4, 6]] -> Itera y agrega ambas.
        """
        if not datos or len(datos) == 0:
            return

        # Verificamos el primer elemento para inferir la estructura
        primer_elemento = datos[0]

        # Si el primer elemento es una lista, tupla o array, asumimos que 'datos' es un LOTE (lista de listas)
        if isinstance(primer_elemento, (list, tuple, np.ndarray)):
            for comb in datos:
                self.datos.add(self._normalizar(comb))
        
        # Si el primer elemento es un número, asumimos que 'datos' es un INDIVIDUO único
        elif isinstance(primer_elemento, (int, float, np.number)):
            self.datos.add(self._normalizar(datos))
            
        else:
            # Fallback: Si no reconocemos el tipo, intentamos iterar (asumiendo lote genérico)
            try:
                for comb in datos:
                    self.datos.add(self._normalizar(comb))
            except Exception as e:
                print(f"Error al agregar datos: {e}")

    def generar(self, n, m, tam):
        """
        Genera una población de 'tam' individuos de manera eficiente usando NumPy.
        
        Args:
            n (int): Límite superior del rango (0 a n-1).
            m (int): Tamaño de cada individuo (número de elementos).
            tam (int): Cantidad de individuos a generar.
        """
        if m > n:
            raise ValueError(f"El tamaño de la muestra (m={m}) no puede ser mayor que la población (n={n}).")

        max_combinaciones = math.comb(n, m)
        if tam > max_combinaciones:
            raise ValueError(f"La cantidad solicitada ({tam}) excede el número máximo de combinaciones posibles ({max_combinaciones}).")

        # Estrategia vectorizada:
        # Generar candidatos en lote. Para m pequeño, es eficiente generar
        # indices aleatorios y descartar los que tienen repetidos.
        # Pero si m ~ n/2, es mejor shuffle.
        
        # Para evitar complejidad, usamos una estrategia híbrida robusta:
        # Generamos un exceso de candidatos y filtramos.
        
        needed = tam - len(self.datos)
        if needed <= 0:
            return

        # Factor de seguridad para generar de más y compensar colisiones/repetidos
        batch_size = int(needed * 1.5) + 10 
        
        while len(self.datos) < tam:
            current_need = tam - len(self.datos)
            # Generamos lote
            # Matriz (batch, n) con noise para sortear argsort es lento O(n log n).
            # Matriz (batch, m) con randint puede tener repetidos internos.
            
            # Enfoque robusto y rápido para m << n:
            # Floyd selection o rejection sampling.
            # Aquí usamos rejection sampling masivo con numpy
            
            # Generamos batch * m números
            pool = np.random.randint(0, n, (current_need * 2, m))
            point = 0
            
            added_this_round = 0
            
            for i in range(len(pool)):
                if added_this_round >= current_need:
                    break
                    
                cand = pool[i]
                # Verificar unicidad interna (genes distintos)
                # np.unique es costoso por fila.
                # Si m es pequeño (<50), sort y check vecinos es rápido?
                # O simplemente set en python.
                cand_set = frozenset(cand)
                if len(cand_set) == m:
                    # Es válido (sin repetidos internos)
                    cand_tuple = tuple(sorted(cand_set))
                    if cand_tuple not in self.datos:
                        self.datos.add(cand_tuple)
                        added_this_round += 1
            
            # Si rejection sampling falla mucho (denso), fallback a random.sample (lento pero seguro)
            if added_this_round == 0:
                # Fallback para completar
                while len(self.datos) < tam:
                     comb = tuple(sorted(random.sample(range(n), m)))
                     self.datos.add(comb)
                break

    def existe(self, combinacion):
        return self._normalizar(combinacion) in self.datos

    def eliminar(self, combinacion):
        norm = self._normalizar(combinacion)
        self.datos.discard(norm)  # discard no lanza error si no existe

    def muestra(self, n):
        """Devuelve n elementos aleatorios."""
        if n > len(self.datos):
            raise ValueError("El tamaño de la muestra excede el total de datos.")
        return random.sample(list(self.datos), n)

    def total_elementos(self):
        return len(self.datos)
    
    def frecuencia(self):
        """Ejemplo de cálculo de totales: Frecuencia de cada número individual."""
        # Usamos chain.from_iterable para evitar crear una lista intermedia
        return Counter(chain.from_iterable(self.datos))

    # Se agregan los getters y setters
    def get_n(self):
        return self.n
    def get_m(self):
        return self.m
    def get_tam(self):
        return self.tam
    def get_datos(self):
        return self.datos   


if __name__ == "__main__":
    print("--- Tests ---")
    poblacion = Combinaciones()
    
    # EJEMPLO 1: Agregar un LOTE (lista de listas) usando el mismo método

    # lote = [
    #     [1, 5, 3], 
    #     [3, 1, 5], # Duplicado
    #     [2, 4, 6],
    #     [12, 20, 30],
    #     [1, 21, 3], 
    #     [11, 2, 3],
    #     [1, 8, 27],
    # ]

    poblacion.generar(n=30, m=3, tam=30)

    # EJEMPLO 2: Agregar un INDIVIDUO único usando el mismo método
    unico = [7, 8, 9]
    print(f"Agregando individuo único: {unico}")
    poblacion.agregar(unico)
    
    print(f"Total elementos únicos: {poblacion.total_elementos()}") # Debería ser 4
    print(f"¿Existe [6, 4, 2]? {poblacion.existe([6, 4, 2])}")       
    
    print("Muestra aleatoria:", poblacion.muestra(5))
    print("Frecuencias:", poblacion.frecuencia())
