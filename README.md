# 🤖 Rota Inteligente: Otimização de Entregas com Algoritmos de IA

## 1. Descrição do Desafio (Sabor Express)

Este projeto foi desenvolvido para atender à necessidade da empresa de delivery "Sabor Express", que enfrenta problemas de ineficiência logística, alto custo de combustível e atrasos nas entregas devido ao planejamento manual e não otimizado das rotas.

**O objetivo central** é criar uma solução inteligente que combine a distribuição de tarefas (agrupamento de pedidos) com a otimização do percurso (menor caminho) para garantir entregas mais rápidas e econômicas.

**Modelagem do Problema:**
O cenário da cidade e das entregas é modelado como um **Grafo Completo**, onde:
* **Nós (Nodes):** O depósito (ponto de partida) e todos os pontos de entrega (pedidos).
* **Arestas (Edges):** As conexões entre quaisquer dois pontos.
* **Pesos (Weights):** A distância Euclidiana entre os nós, representando o custo (tempo/combustível) de deslocamento.

---

## 2. Abordagem da Solução

Para resolver o problema complexo de roteamento e distribuição de carga, a solução foi dividida em duas etapas principais, utilizando algoritmos clássicos de Inteligência Artificial:

### Etapa A: Agrupamento de Entregas (Clustering)

* **Algoritmo:** **K-Means (Aprendizado Não Supervisionado)**
* **Finalidade:** Balancear a carga de trabalho e agrupar geograficamente os pedidos, definindo as zonas de entrega de cada entregador.
* **Métrica:** O K-Means divide o conjunto de pedidos ($N$) no número de entregadores ($K$), minimizando a distância entre cada pedido e o centroide do seu cluster. Isso garante que cada entregador atenda a uma região próxima do depósito.

### Etapa B: Otimização do Menor Caminho (Roteamento)

* **Algoritmo:** **A\* (A-Star Search)**
* **Finalidade:** Encontrar o caminho mais eficiente (menor distância) para visitar todos os pontos de entrega dentro de um cluster específico, começando no depósito.
* **Função de Avaliação ($f(n) = g(n) + h(n)$):**
    * **$g(n)$ (Custo Real):** Distância total percorrida do depósito até o nó atual.
    * **$h(n)$ (Heurística):** Distância Euclidiana (linha reta) do nó atual até o destino final. Por ser admissível (nunca superestima o custo real), o $A^{*}$ garante um caminho ótimo entre dois pontos.
* **Estratégia de Roteamento:** O $A^{*}$ é aplicado de forma **Guloso-Iterativa**. A rota é construída sequencialmente, onde o algoritmo escolhe o próximo ponto de entrega mais eficiente, garantindo que o entregador minimize o custo a cada etapa.

---

## 3. Análise dos Resultados

A simulação foi executada com **20 pedidos** e **4 entregadores**, demonstrando a capacidade da solução de otimizar a distribuição e o percurso.

### 3.1. Resumo da Otimização

| Entregador | Cluster | Nº de Entregas | Custo/Distância Otimizada |
| :--- | :--- | :--- | :--- |
| 1 | 0 | 5 | 10.79 |
| 2 | 1 | 6 | 13.11 |
| 3 | 2 | 4 | 9.94 |
| 4 | 3 | 5 | 9.75 |
| **TOTAL** | | **20** | **43.59** |

**Conclusão:**
O sistema conseguiu balancear a carga de trabalho de forma eficiente (entre 4 e 6 entregas por entregador) e calculou uma rota total percorrida de **43.59 unidades**. Esta otimização resulta em:
1.  **Redução de Custo:** Menor quilometragem total reduz diretamente o consumo de combustível.
2.  **Melhora na Produtividade:** O roteamento lógico (A\*) evita rotas aleatórias e minimiza o tempo ocioso do entregador.

### 3.2. Diagrama do Grafo e Rotas Otimizadas

*<img width="1000" height="800" alt="diagrama_grafo_rotas" src="https://github.com/user-attachments/assets/2207a19f-db8a-4545-b62e-8cdd4fe9cbfb" />
*
**Análise do Diagrama:**
* Cada cor representa um cluster (zona de entrega) definido pelo K-Means, indicando a distribuição geográfica dos pedidos.
* O quadrado vermelho central representa o Depósito (ponto de partida).
* As linhas (arestas) mostram a sequência otimizada da rota calculada pelo algoritmo A\* dentro de cada cluster, partindo do depósito e retornando ao ponto final.

---

## 4. Limitações e Próximos Passos (Crítica e Melhorias)

| Tipo | Limitação da Solução Atual | Sugestões de Melhoria |
| :--- | :--- | :--- |
| **Heurística de Roteamento** | O uso do A\* iterativo é uma aproximação gulosa do Problema do Caixeiro Viajante (TSP) e não garante a ordem *globalmente* ótima entre todos os pontos do cluster. | Implementar Algoritmos Genéticos ou Programação Linear Inteira Mista (MILP) para encontrar a ordem de visita (a sequência) verdadeiramente ótima do TSP. |
| **Modelagem do Grafo** | O custo da aresta é baseado apenas na Distância Euclidiana, ignorando o tráfego, semáforos e vias urbanas reais. | Integrar dados de APIs de geolocalização e tráfego em tempo real para utilizar o **tempo real de viagem** como peso da aresta, tornando a solução mais robusta (ex: sistemas como UPS ORION). |
| **Clustering** | O número de entregadores ($K$) é fixo. | Adotar técnicas para determinar o $K$ ideal (como o Método do Cotovelo) ou usar algoritmos de clustering que não exijam $K$ pré-definido (ex: DBSCAN), dependendo da densidade dos pedidos. |

---

## 5. Instruções de Execução

Para replicar os resultados e gerar o diagrama, siga os passos abaixo:

1.  **Pré-requisitos:** Python 3.x e as bibliotecas listadas.
2.  **Instalação de Bibliotecas:**
    ```bash
    pip install numpy scikit-learn pandas matplotlib
    ```
3.  **Execução do Código:** Salve o código Python como `rota_inteligente.py` e execute:
    ```bash
    python rota_inteligente.py
    ```
4.  **Output:** O script gerará a análise textual no console, mostrará o diagrama do grafo e salvará a imagem **`diagrama_grafo_rotas.png`** no mesmo diretório.
