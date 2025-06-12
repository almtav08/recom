#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para contar las aristas en los grafos prev_graph y repeat_graph para cada curso.
"""

import os
import json
from pathlib import Path

# Lista de cursos disponibles (obtenida de graph_info.py)
courses = ["ecourse", "fakecourse", "vcourse"]

# Directorio base donde se encuentran los datos
BASE_DIR = Path("/home/alex/recom/database/data")


def count_edges(graph_data):
    """
    Cuenta el número de aristas en un grafo representado como un diccionario.

    Args:
        graph_data: Diccionario que representa el grafo. Las claves son nodos origen
                   y los valores son listas de nodos destino.

    Returns:
        Número total de aristas.
    """
    edge_count = 0

    # Para cada nodo origen en el grafo
    for source_node, target_nodes in graph_data.items():
        # Si el valor es una lista, contar todos los enlaces
        if isinstance(target_nodes, list):
            edge_count += len(target_nodes)
        # Si el valor es un solo número (no una lista), contar como 1 enlace
        else:
            edge_count += 1

    return edge_count


def main():
    """Función principal que procesa todos los cursos."""
    print(f"{'Curso':<15} {'Prev Edges':<15} {'Repeat Edges':<15} {'Total Edges':<15}")
    print("-" * 60)

    total_prev_edges = 0
    total_repeat_edges = 0

    for course in courses:
        # Rutas a los archivos JSON
        prev_graph_path = BASE_DIR / course / "prev_graph.json"
        repeat_graph_path = BASE_DIR / course / "repeat_graph.json"

        try:
            # Cargar prev_graph.json
            with open(prev_graph_path, "r") as f:
                prev_graph = json.load(f)

            # Contar aristas en prev_graph
            prev_edges = count_edges(prev_graph)
            total_prev_edges += prev_edges

            # Cargar repeat_graph.json
            with open(repeat_graph_path, "r") as f:
                repeat_graph = json.load(f)

            # Contar aristas en repeat_graph
            repeat_edges = count_edges(repeat_graph)
            total_repeat_edges += repeat_edges

            # Calcular el total de aristas para este curso
            course_total = prev_edges + repeat_edges

            print(
                f"{course:<15} {prev_edges:<15} {repeat_edges:<15} {course_total:<15}"
            )

        except FileNotFoundError as e:
            print(f"Error: No se pudo encontrar el archivo para {course}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error: Problema al decodificar JSON para {course}: {e}")

    # Mostrar totales
    print("-" * 60)
    grand_total = total_prev_edges + total_repeat_edges
    print(
        f"{'TOTAL':<15} {total_prev_edges:<15} {total_repeat_edges:<15} {grand_total:<15}"
    )


if __name__ == "__main__":
    main()
