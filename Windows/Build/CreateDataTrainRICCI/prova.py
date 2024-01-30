import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def calcola_curvatura_ricci(graph):
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    print("Grafo del club di karate: La curvatura di Ollivier-Ricci dell'arco (0,1) è %f" % orc.G[0][1]["ricciCurvature"])

if __name__ == '__main__':
    try:
        print("\n- Importa un esempio di grafo del club di karate da NetworkX")
        G = nx.karate_club_graph()

        print("\n===== Calcola la curvatura di Ollivier-Ricci del grafo dato G =====")
        calcola_curvatura_ricci(G)
    except Exception as e:
        print(f"Errore: {e}")
