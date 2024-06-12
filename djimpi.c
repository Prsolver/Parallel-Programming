#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX

void serial_dijkstra(int *graph, int n, int src, int *dist) {
    int *visited = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) {
        int min = INF, min_index;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }
        }

        int u = min_index;
        visited[u] = 1;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && graph[u * n + v] && dist[u] != INF && dist[u] + graph[u * n + v] < dist[v]) {
                dist[v] = dist[u] + graph[u * n + v];
            }
        }
    }
    free(visited);
}

void parallel_dijkstra(int *graph, int n, int src, int rank, int size, MPI_Comm comm, int *dist) {
    int *visited = (int *)malloc(n * sizeof(int));
    int *local_dist = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) {
        int min = INF, min_index = -1;

        for (int v = rank; v < n; v += size) {
            if (!visited[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }
        }

        struct {
            int value;
            int rank;
        } local_data, global_data;

        local_data.value = min;
        local_data.rank = min_index;

        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MINLOC, comm);

        int u = global_data.rank;
        if (u == -1) break;
        visited[u] = 1;

        for (int v = rank; v < n; v += size) {
            if (!visited[v] && graph[u * n + v] && dist[u] != INF && dist[u] + graph[u * n + v] < dist[v]) {
                dist[v] = dist[u] + graph[u * n + v];
            }
        }
        MPI_Allgather(dist, n, MPI_INT, local_dist, n, MPI_INT, comm);
        for (int i = 0; i < n; i++) {
            dist[i] = local_dist[i];
        }
    }

    free(visited);
    free(local_dist);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 6;
    int graph[6][6] = {
        {0, 10, 20, 0, 0, 0},
        {10, 0, 5, 16, 0, 0},
        {20, 5, 0, 20, 1, 0},
        {0, 16, 20, 0, 2, 10},
        {0, 0, 1, 2, 0, 3},
        {0, 0, 0, 10, 3, 0}
    };

    int *flat_graph = NULL;
    if (rank == 0) {
        flat_graph = (int *)malloc(n * n * sizeof(int));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_graph[i * n + j] = graph[i][j];
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        flat_graph = (int *)malloc(n * n * sizeof(int));
    }
    MPI_Bcast(flat_graph, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    int src = 0;

    double start_time, end_time, serial_time, parallel_time;

    if (rank == 0) {
        int *serial_dist = (int *)malloc(n * sizeof(int));
        start_time = MPI_Wtime();
        serial_dijkstra(flat_graph, n, src, serial_dist);
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;
        free(serial_dist);
        printf("Serial execution time: %f seconds\n", serial_time);
    }

    for (int p = 1; p <= 64; p++) {
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start at the same time

        int *parallel_dist = (int *)malloc(n * sizeof(int));
        start_time = MPI_Wtime();
        parallel_dijkstra(flat_graph, n, src, rank, p, MPI_COMM_WORLD, parallel_dist);
        end_time = MPI_Wtime();
        parallel_time = end_time - start_time;

        if (rank == 0) {
            printf("Parallel execution time with %d processors: %f seconds\n", p, parallel_time);
            double speedup = serial_time / parallel_time;
            printf("Speedup with %d processors: %f\n", p, speedup);
        }

        free(parallel_dist);
    }

    free(flat_graph);
    MPI_Finalize();
    return 0;
}
