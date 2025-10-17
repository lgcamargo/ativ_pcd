# kmeans_1d_omp.c

Versão paralela (OpenMP) do K-means 1D.

## Compilar

```bash
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
```

## Uso

```bash
./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
```

Parâmetros:
- dados.csv — arquivo de entrada com os pontos (uma coluna).
- centroides_iniciais.csv — centróides iniciais (uma coluna).
- max_iter — máximo de iterações (padrão: 50).
- eps — tolerância de convergência (padrão: 1e-4).
- assign.csv — (opcional) arquivo de saída com atribuições de cada ponto.
- centroids.csv — (opcional) arquivo de saída com centróides finais.

Exemplo:
LINUX
```bash
./kmeans_1d_omp dados.csv centroides.csv 100 1e-5 assign_out.csv centroids_out.csv
```
WINDOWS
```bash
kmeans_1d_omp.exe dados.csv centroides.csv 100 1e-5 assign_out.csv centroids_out.csv
```
