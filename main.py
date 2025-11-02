import numpy as np
from sklearn.cluster import KMeans
import heapq
import pandas as pd
import matplotlib.pyplot as plt # Importação adicionada para visualização

# -------------------------------------------------------------
# 1. SIMULAÇÃO DE DADOS E MODELAGEM DO GRAFO
# -------------------------------------------------------------

# Criação de dados de entrega simulados (latitude, longitude, ID)
def gerar_dados_entregas(num_entregas):
    """Gera coordenadas de entrega simuladas."""
    np.random.seed(42) # Para reprodutibilidade
    data = {
        'ID': [f'P{i+1}' for i in range(num_entregas)],
        'Latitude': np.random.uniform(0, 10, num_entregas),
        'Longitude': np.random.uniform(0, 10, num_entregas)
    }
    return pd.DataFrame(data)

# Função de Distância (Heurística e Custo do Grafo)
def distancia_euclidiana(ponto1, ponto2):
    """Calcula a distância Euclidiana entre dois pontos (coordenadas)."""
    # Garante que os pontos são tratáveis como tuplas/arrays para a operação
    p1 = np.array(ponto1)
    p2 = np.array(ponto2)
    return np.sqrt(np.sum((p1 - p2)**2))

# O local de partida (Depósito da Sabor Express)
DEPOSITO = (5.0, 5.0)

# -------------------------------------------------------------
# 2. ALGORITMO DE CLUSTERING (K-MEANS)
# -------------------------------------------------------------

def agrupar_entregas(df_entregas, num_entregadores):
    """
    Agrupa as entregas em 'num_entregadores' grupos usando K-Means.
    O K-Means é usado para criar zonas de entrega eficientes.
    """
    coordenadas = df_entregas[['Latitude', 'Longitude']].values
    
    # Se houver menos entregas que o número de entregadores, ajusta K
    K = min(num_entregadores, len(coordenadas)) 
    
    if K == 0:
        return {}

    print(f"\n--- Aplicando K-Means para agrupar {len(coordenadas)} pedidos em {K} zonas ---")
    
    # n_init='auto' é o padrão moderno para evitar warnings
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
    df_entregas['Cluster'] = kmeans.fit_predict(coordenadas)
    
    # Organiza os clusters para o roteamento
    clusters = {}
    for cluster_id in range(K):
        # Pontos de entrega no cluster (como TUPLAS para consistência)
        pontos_do_cluster = [DEPOSITO] + [tuple(p) for p in df_entregas[df_entregas['Cluster'] == cluster_id][['Latitude', 'Longitude']].values.tolist()]
        clusters[cluster_id] = pontos_do_cluster
        
    return clusters

# -------------------------------------------------------------
# 3. ALGORITMO DE BUSCA DE MENOR CAMINHO (A*)
# -------------------------------------------------------------

def a_star_search(start_node, goal_node, nodes):
    """
    Implementação básica do algoritmo A* para encontrar o menor caminho entre dois pontos
    em um grafo completo (qualquer nó se conecta a qualquer outro).
    """
    
    # Mapeia as coordenadas (tuplas) para IDs simples (índices na lista 'nodes')
    node_map = {node: i for i, node in enumerate(nodes)}
    
    # Verifica se os pontos inicial e final estão no conjunto de nós
    if start_node not in node_map or goal_node not in node_map:
        return None, float('inf')

    start_index = node_map[start_node]
    goal_index = node_map[goal_node]
    
    # (f_score, g_score, index_do_no_atual, path_percorrido)
    priority_queue = [(0, 0, start_index, [start_node])]
    
    # g_score_map armazena o menor g_score encontrado para cada nó
    g_score_map = {start_index: 0}
    
    while priority_queue:
        f_score_current, g_score_current, current_index, path_current = heapq.heappop(priority_queue)
        current_node = nodes[current_index]

        if current_index == goal_index:
            return path_current, g_score_current # Rota encontrada!

        # Iterar sobre todos os outros nós como 'vizinhos' (grafo completo)
        for next_index, next_node in enumerate(nodes):
            if current_index == next_index:
                continue

            # Custo entre o nó atual e o próximo nó (aresta do grafo)
            cost = distancia_euclidiana(current_node, next_node)
            
            # Novo custo do caminho (g_score)
            g_score_new = g_score_current + cost

            if g_score_new < g_score_map.get(next_index, float('inf')):
                # Este é um caminho melhor. Atualiza g_score e calcula f_score
                g_score_map[next_index] = g_score_new
                
                # Heurística: distância Euclidiana do próximo nó até o destino
                h_score = distancia_euclidiana(next_node, goal_node)
                
                f_score_new = g_score_new + h_score
                
                new_path = path_current + [next_node]
                
                # Adiciona à fila de prioridade
                heapq.heappush(priority_queue, (f_score_new, g_score_new, next_index, new_path))
                
    return None, float('inf') # Nenhum caminho encontrado

# -------------------------------------------------------------
# 4. FUNÇÃO PRINCIPAL DE ROTEAMENTO (TSP simplificado por A*)
# -------------------------------------------------------------

def encontrar_melhor_rota_no_cluster(pontos_entrega_cluster):
    """
    Calcula uma rota que passa por todos os pontos de entrega em um cluster,
    começando no Depósito.
    """
    
    if len(pontos_entrega_cluster) <= 1:
        return pontos_entrega_cluster, 0

    # O depósito é sempre o ponto de partida (primeiro item na lista)
    rota_final = [pontos_entrega_cluster[0]]
    custo_total = 0
    
    # CORREÇÃO: Converte as coordenadas para TUPLAS para usar em um set
    pontos_a_visitar = set(tuple(ponto) for ponto in pontos_entrega_cluster[1:])
    ponto_atual = pontos_entrega_cluster[0]

    # Enquanto houver pontos a visitar...
    while pontos_a_visitar:
        melhor_custo = float('inf')
        melhor_proximo_ponto = None
        melhor_caminho = None

        # Encontra o próximo ponto de entrega mais eficiente
        for proximo_ponto in pontos_a_visitar:
            # A* para encontrar o menor caminho entre o ponto_atual e o proximo_ponto
            caminho, custo = a_star_search(ponto_atual, proximo_ponto, pontos_entrega_cluster)
            
            if custo < melhor_custo:
                melhor_custo = custo
                melhor_proximo_ponto = proximo_ponto
                melhor_caminho = caminho

        # Adiciona o melhor caminho e ponto à rota final
        if melhor_caminho and melhor_proximo_ponto:
            # Adiciona o caminho percorrido, exceto o ponto inicial (que já está na rota)
            rota_final.extend(melhor_caminho[1:]) 
            custo_total += melhor_custo
            pontos_a_visitar.remove(melhor_proximo_ponto)
            ponto_atual = melhor_proximo_ponto
        else:
            # Se não encontrou caminho, encerra a rota para este entregador
            break 
            
    return rota_final, custo_total

# -------------------------------------------------------------
# 5. FUNÇÃO DE VISUALIZAÇÃO DO GRAFO (Diagrama)
# -------------------------------------------------------------

def visualizar_rotas(df_entregas, resultados_finais, deposito):
    """Plota o resultado do clustering e das rotas otimizadas."""
    plt.figure(figsize=(10, 8))
    
    # 1. Plotar os Pontos de Entrega (Clusters)
    
    # Mapeia os IDs dos entregadores/clusters para cores
    num_clusters = df_entregas['Cluster'].nunique()
    cores = plt.cm.get_cmap('viridis', num_clusters)

    for i in range(num_clusters):
        cluster_data = df_entregas[df_entregas['Cluster'] == i]
        plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                    color=cores(i), label=f'Cluster {i+1} ({len(cluster_data)} Entregas)', s=100, alpha=0.7)

    # 2. Plotar o Depósito (Ponto de Partida)
    # Lembre-se: O eixo X é Longitude e o eixo Y é Latitude no Matplotlib
    plt.scatter(deposito[1], deposito[0], color='red', marker='s', s=300, label='Depósito (Sabor Express)', edgecolors='black')
    
    # 3. Desenhar as Rotas Otimizadas
    
    # Iterar sobre os resultados e desenhar as linhas da rota
    for resultado in resultados_finais:
        # Extrai a string de coordenadas e remove o texto em excesso
        rota_coords_str = resultado['Rota_Otimizada']
        
        # Parseia a string para uma lista de tuplas de coordenadas
        # Ex: "(5.00, 5.00) -> (1.83, 8.08)" vira [ (5.0, 5.0), (1.83, 8.08) ]
        
        # Simplificando o parsing, confiando no formato:
        coords_list = []
        for ponto_str in rota_coords_str.split(' -> '):
             # Remove parênteses e quebras, e separa por vírgula
             try:
                 lat, lon = map(float, ponto_str.strip('()').split(','))
                 coords_list.append((lat, lon))
             except ValueError:
                 # Ignora se o ponto não puder ser parseado corretamente
                 continue 

        # Se houver coordenadas válidas
        if coords_list:
            latitudes = [c[0] for c in coords_list]
            longitudes = [c[1] for c in coords_list]
        
            entregador_id = resultado['Entregador']
            
            # Desenha as linhas da rota (aresta do grafo)
            plt.plot(longitudes, latitudes, color=cores(entregador_id - 1), 
                     linestyle='-', linewidth=2, alpha=0.7) # Cor da linha baseada no cluster/entregador

    plt.title('Diagrama do Grafo: Otimização de Rotas (K-Means + A*)', fontsize=14)
    plt.xlabel('Longitude (Eixo X)')
    plt.ylabel('Latitude (Eixo Y)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta layout para a legenda
    
    # Salva o arquivo (Imagem estática para o README)
    plt.savefig('diagrama_grafo_rotas.png')
    print("\n[INFO] Diagrama do grafo salvo como 'diagrama_grafo_rotas.png'.")
    plt.show() # Mostra o gráfico na tela

# -------------------------------------------------------------
# 6. EXECUÇÃO DO SISTEMA
# -------------------------------------------------------------

def sistema_rota_inteligente(num_pedidos=15, num_entregadores=3):
    """Função principal que integra Clustering e Roteamento."""
    print(f"--- Simulação 'Sabor Express' ---")
    print(f"Número de pedidos: {num_pedidos}")
    print(f"Número de entregadores disponíveis: {num_entregadores}")
    print(f"Local do Depósito: {DEPOSITO}\n")

    # 1. Geração e Agrupamento dos Dados
    df_entregas = gerar_dados_entregas(num_pedidos)
    clusters = agrupar_entregas(df_entregas, num_entregadores)
    
    if not clusters:
        print("Não há pedidos para agrupar.")
        return

    resultados_finais = []
    
    # 2. Roteamento para cada Cluster
    for cluster_id, pontos in clusters.items():
        pontos_para_roteamento = [ponto for ponto in pontos if ponto is not None]

        print(f"\n--- Processando Rota para o Entregador {cluster_id + 1} (Cluster {cluster_id}) ---")
        
        # Encontra a melhor ordem de visita (rota) dentro deste cluster
        rota, custo = encontrar_melhor_rota_no_cluster(pontos_para_roteamento)
        
        # Formata os resultados
        rota_str = " -> ".join([f"({p[0]:.2f}, {p[1]:.2f})" for p in rota])
        
        resultado = {
            'Entregador': cluster_id + 1,
            'Num_Entregas': len(pontos_para_roteamento) - 1, # Menos o depósito
            'Rota_Otimizada': rota_str,
            'Custo_Total_Estimado': f'{custo:.2f}'
        }
        resultados_finais.append(resultado)
        
        print(f"  > Rota (Coords): {rota_str}")
        print(f"  > Custo/Distância Total: {custo:.2f}")

    print("\n" + "="*50)
    print("RESUMO DOS RESULTADOS DA OTIMIZAÇÃO:")
    print("="*50)
    
    # 3. Análise e Visualização
    df_resultados = pd.DataFrame(resultados_finais)
    print(df_resultados)
    
    print(f"\nDistância total percorrida (sum): {df_resultados['Custo_Total_Estimado'].astype(float).sum():.2f}")
    print("A otimização de rota e o agrupamento são projetados para reduzir a distância total e o tempo de entrega.")
    
    # CHAMADA PARA A FUNÇÃO DE VISUALIZAÇÃO
    visualizar_rotas(df_entregas, resultados_finais, DEPOSITO)


if __name__ == "__main__":
    # Exemplo: 20 pedidos e 4 entregadores
    sistema_rota_inteligente(num_pedidos=20, num_entregadores=4)
    