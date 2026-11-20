"""
Trabalho prático: Resolução de problemas com Cadeias de Markov
---------------------------------------------------------------
Problema 1 - Previsão do clima (cadeia regular)
Problema 2 - Jogo Cobras e Escadas (cadeia absorvente)

Requisitos: Python 3.9+, numpy, matplotlib
Execução:   python src/markov_trabalho.py   (a partir da raiz do projeto)

Saídas:
  graphics/  -> gráficos gerados (.png)
  docs/      -> saida_execucao.txt com todos os resultados impressos
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # gera os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt

# Pastas do projeto, calculadas a partir da localização deste arquivo (src/).
# Assim os arquivos vão sempre para o lugar certo, não importa de onde o
# programa seja executado.
RAIZ = Path(__file__).resolve().parent.parent
PASTA_GRAFICOS = RAIZ / "graphics"
PASTA_DOCS = RAIZ / "docs"
ARQUIVO_SAIDA = PASTA_DOCS / "saida_execucao.txt"

SEED = 42
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=4, suppress=True)


# =====================================================================
# Funções genéricas de Cadeias de Markov
# =====================================================================
def validar_matriz(P):
    """Confere se P é quadrada, não negativa e com linhas somando 1."""
    P = np.asarray(P, dtype=float)
    assert P.shape[0] == P.shape[1], "A matriz deve ser quadrada"
    assert np.all(P >= 0), "Probabilidades não podem ser negativas"
    assert np.allclose(P.sum(axis=1), 1), "Cada linha deve somar 1"
    return P


def distribuicao_n_passos(pi0, P, n):
    """Chapman-Kolmogorov: pi_n = pi_0 * P^n."""
    return pi0 @ np.linalg.matrix_power(P, n)


def distribuicao_estacionaria(P):
    """Resolve pi P = pi com soma 1 (autovetor à esquerda do autovalor 1)."""
    autovalores, autovetores = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(autovalores - 1))
    pi = np.real(autovetores[:, idx])
    return pi / pi.sum()


def simular_cadeia(P, estado_inicial, n_passos):
    """Gera uma trajetória da cadeia sorteando cada próximo estado."""
    estados = np.empty(n_passos + 1, dtype=int)
    estados[0] = estado_inicial
    for t in range(n_passos):
        estados[t + 1] = rng.choice(len(P), p=P[estados[t]])
    return estados


def estimar_matriz(sequencia, n_estados):
    """Estima P a partir de dados observados contando as transições i -> j."""
    contagem = np.zeros((n_estados, n_estados))
    for a, b in zip(sequencia[:-1], sequencia[1:]):
        contagem[a, b] += 1
    return contagem / contagem.sum(axis=1, keepdims=True)


# =====================================================================
# PROBLEMA 1 - PREVISÃO DO CLIMA
# =====================================================================
def problema_clima():
    print("=" * 66)
    print("PROBLEMA 1 - PREVISÃO DO CLIMA (cadeia regular)")
    print("=" * 66)

    estados = ["Ensolarado", "Nublado", "Chuvoso"]
    #               Sol   Nub   Chuva     <- amanhã
    P = validar_matriz([[0.60, 0.30, 0.10],   # hoje: Ensolarado
                        [0.30, 0.40, 0.30],   # hoje: Nublado
                        [0.20, 0.30, 0.50]])  # hoje: Chuvoso
    print("\nMatriz de transição P (linha = hoje, coluna = amanhã):")
    print(P)

    # (a) Previsão para os próximos dias partindo de um dia chuvoso
    pi0 = np.array([0, 0, 1.0])
    print("\n(a) Hoje está chovendo. Previsão para os próximos dias:")
    print(f"{'Dia':>4} | {'Sol':>7} | {'Nublado':>7} | {'Chuva':>7}")
    historico = [pi0]
    for n in range(1, 11):
        pin = distribuicao_n_passos(pi0, P, n)
        historico.append(pin)
        if n in (1, 2, 3, 5, 7, 10):
            print(f"{n:>4} | {pin[0]:7.2%} | {pin[1]:7.2%} | {pin[2]:7.2%}")

    # (b) Comportamento de longo prazo
    pi = distribuicao_estacionaria(P)
    print("\n(b) Distribuição estacionária (longo prazo):")
    for e, p in zip(estados, pi):
        print(f"    {e:<11}: {p:.4f}  -> cerca de {p * 365:5.1f} dias por ano")

    # (c) Tempo médio de retorno a cada estado = 1 / pi_i
    print("\n(c) Tempo médio de retorno (1/pi_i):")
    for e, p in zip(estados, pi):
        print(f"    {e:<11}: {1 / p:.2f} dias")

    # (d) Verificação por simulação: 10 anos de clima simulado
    dias = 3650
    traj = simular_cadeia(P, estado_inicial=2, n_passos=dias)
    freq = np.bincount(traj, minlength=3) / len(traj)
    print(f"\n(d) Simulação de {dias} dias - frequência observada x teórica:")
    for e, f, p in zip(estados, freq, pi):
        print(f"    {e:<11}: simulado {f:.4f} | teórico {p:.4f} | erro {abs(f - p):.4f}")

    # (e) Problema inverso: estimar P a partir dos dados simulados
    P_est = estimar_matriz(traj, 3)
    print("\n(e) Matriz estimada a partir dos 10 anos de 'dados observados':")
    print(P_est)
    print(f"    Maior diferença em relação à P verdadeira: {np.abs(P_est - P).max():.4f}")

    # Gráfico: convergência para a distribuição estacionária
    hist = np.array(historico)
    plt.figure(figsize=(8, 4.5))
    for k, (e, cor) in enumerate(zip(estados, ["#f2a900", "#8a8a8a", "#1f77b4"])):
        plt.plot(range(len(hist)), hist[:, k], marker="o", color=cor, label=e)
        plt.axhline(pi[k], color=cor, linestyle="--", linewidth=0.8)
    plt.title("Clima: convergência de π₀Pⁿ para a distribuição estacionária")
    plt.xlabel("Dias a partir de hoje (hoje = chuva)")
    plt.ylabel("Probabilidade")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "grafico_clima_convergencia.png", dpi=120)
    plt.close()

    return pi


# =====================================================================
# PROBLEMA 2 - COBRAS E ESCADAS
# =====================================================================
# Tabuleiro clássico de 100 casas (Milton Bradley, 1943)
ESCADAS = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
COBRAS = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60,
          87: 24, 93: 73, 95: 75, 98: 78}
SALTOS = {**ESCADAS, **COBRAS}
FINAL = 100


def mover(casa, dado):
    """Regra: se o dado passar da casa 100, o jogador fica onde está."""
    destino = casa + dado
    if destino > FINAL:
        return casa
    return SALTOS.get(destino, destino)


def montar_matriz_tabuleiro():
    """Estados 0..100 (0 = fora do tabuleiro). Cada face do dado tem prob. 1/6."""
    P = np.zeros((FINAL + 1, FINAL + 1))
    for casa in range(FINAL + 1):
        if casa == FINAL:
            P[casa, casa] = 1.0          # estado absorvente
            continue
        for dado in range(1, 7):
            P[casa, mover(casa, dado)] += 1 / 6
    return validar_matriz(P)


def jogar_uma_partida():
    casa, jogadas = 0, 0
    while casa != FINAL:
        casa = mover(casa, rng.integers(1, 7))
        jogadas += 1
    return jogadas


def problema_cobras_escadas():
    print("\n" + "=" * 66)
    print("PROBLEMA 2 - COBRAS E ESCADAS (cadeia absorvente)")
    print("=" * 66)

    P = montar_matriz_tabuleiro()
    transitorios = list(range(FINAL))      # estados 0..99
    Q = P[np.ix_(transitorios, transitorios)]

    # Matriz fundamental N = (I - Q)^-1 e tempo esperado até a absorção
    N = np.linalg.inv(np.eye(len(Q)) - Q)
    t = N.sum(axis=1)
    print(f"\nMatriz P: {P.shape[0]}x{P.shape[1]} | Q (transitórios): {Q.shape[0]}x{Q.shape[1]}")
    print(f"(a) Número esperado de jogadas para terminar (início na casa 0): {t[0]:.2f}")

    # Casas mais visitadas em média numa partida
    top = np.argsort(N[0])[::-1][:5]
    print("\n(b) Casas mais visitadas, em média, por partida:")
    for c in top:
        print(f"    casa {c:>3}: {N[0, c]:.3f} visitas")

    # Distribuição exata do número de jogadas: P(terminar até n) = (P^n)[0, 100]
    max_n = 200
    dist = np.zeros(FINAL + 1)
    dist[0] = 1.0
    acumulada = []
    for _ in range(max_n):
        dist = dist @ P
        acumulada.append(dist[FINAL])
    acumulada = np.array(acumulada)
    prob_exata = np.diff(np.concatenate([[0], acumulada]))
    mediana = int(np.argmax(acumulada >= 0.5)) + 1
    print("\n(c) Probabilidade de terminar em até n jogadas (cálculo exato):")
    for n in (10, 20, 30, 50, 100):
        print(f"    até {n:>3} jogadas: {acumulada[n - 1]:.2%}")
    print(f"    Mínimo de jogadas possível: {int(np.argmax(prob_exata > 0)) + 1}")
    print(f"    Mediana: {mediana} jogadas")

    # Verificação por Monte Carlo
    n_partidas = 20000
    amostras = np.array([jogar_uma_partida() for _ in range(n_partidas)])
    ic = 1.96 * amostras.std(ddof=1) / np.sqrt(n_partidas)
    print(f"\n(d) Monte Carlo com {n_partidas} partidas:")
    print(f"    média simulada = {amostras.mean():.2f} ± {ic:.2f} (IC 95%) | teórico = {t[0]:.2f}")
    print(f"    P(até 20 jogadas) simulada = {(amostras <= 20).mean():.2%} | "
          f"teórica = {acumulada[19]:.2%}")

    # Gráfico: distribuição do número de jogadas (exata x simulada)
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(1, 151)
    plt.hist(amostras, bins=bins, density=True, alpha=0.5,
             color="#1f77b4", label=f"Simulação ({n_partidas} partidas)")
    plt.plot(np.arange(1, max_n + 1), prob_exata, color="#d62728",
             linewidth=2, label="Exato (cadeia de Markov)")
    plt.axvline(t[0], color="black", linestyle="--", linewidth=1,
                label=f"Média teórica = {t[0]:.1f}")
    plt.xlim(0, 150)
    plt.title("Cobras e Escadas: distribuição do número de jogadas")
    plt.xlabel("Jogadas até chegar à casa 100")
    plt.ylabel("Probabilidade")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "grafico_cobras_escadas.png", dpi=120)
    plt.close()

    return t[0], amostras


# =====================================================================
# Saída simultânea no terminal e em arquivo
# =====================================================================
class Tee:
    """Escreve tudo o que for impresso no terminal e também em um arquivo."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for d in self.destinos:
            d.write(texto)

    def flush(self):
        for d in self.destinos:
            d.flush()


def main():
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    PASTA_DOCS.mkdir(exist_ok=True)

    terminal = sys.stdout
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as arquivo:
        sys.stdout = Tee(terminal, arquivo)
        try:
            print(f"Execução em {datetime.now():%d/%m/%Y %H:%M:%S}\n")
            problema_clima()
            problema_cobras_escadas()
            print("\nGráficos salvos em:")
            print(f"    {PASTA_GRAFICOS / 'grafico_clima_convergencia.png'}")
            print(f"    {PASTA_GRAFICOS / 'grafico_cobras_escadas.png'}")
            print(f"Resultados salvos em:\n    {ARQUIVO_SAIDA}")
        finally:
            sys.stdout = terminal


if __name__ == "__main__":
    main()
