#include <stdio.h>
#include <mpi.h>


int main(int argc, char *argv[]){
  int rank;
  int i, j;
  int rec = 0;
  int rec1 = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm graph_comm, graph_comm1;
  
  int nnodes = 16; /* number of nodes */
  /*Initialize the graph and matrix for the 16k3 graph */
  int index[16] = {3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48};
  int edges[48] = {1,8,15,0,2,11,1,3,7,2,4,13,3,5,10,4,6,15,5,7,12,6,8,2,7,9,0,8,10,14,9,11,4,10,12,1,11,13,6,12,14,3,13,15,9,14,0,5};
  int steps = 5;
  int m[nnodes][steps];
  for(i = 0; i < nnodes; i++){
    for(j = 0; j < steps; j++){
      m[i][j] = 0;
    }
  }
  m[0][0] = 1;
  m[0][1] = 8;
  m[0][2] = 15;
  m[1][1] = 2;
  m[1][2] = 11;
  m[2][2] = 3;
  m[3][3] = 4;
  m[3][4] = 13;
  m[4][4] = 5;
  m[6][4] = 12;
  m[7][3] = 6;
  m[8][2] = 7;
  m[8][3] = 9;
  m[11][3] = 10;
  m[15][4] = 14;
  
  /* Initialize the graph and matrix for the torus */
  int index1[16] = {4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64};
  int edges1[64] = {1,3,4,12,0,2,5,13,1,3,6,14,0,2,7,15,0,5,7,12,1,4,6,9,2,5,7,10,3,4,6,11,4,9,11,12,5,8,10,13,6,9,11,14,7,8,10,15,0,8,13,15,1,9,12,14,2,10,13,15,3,11,12,14};
  int steps1 = 4;
  int m1[nnodes][steps1];
  for(i = 0; i < nnodes; i++){
    for(j = 0; j < steps1; j++){
      m1[i][j] = 0;
    }
  }
  m1[0][0] = 1;
  m1[0][1] = 3;
  m1[0][2] = 4;
  m1[0][3] = 12;
  m1[1][1] = 2;
  m1[1][2] = 5;
  m1[1][3] = 13;
  m1[2][2] = 6;
  m1[2][3] = 14;
  m1[3][2] = 7;
  m1[3][3] = 15;
  m1[4][3] = 8;
  m1[5][3] = 9;
  m1[6][3] = 10;
  m1[7][3] = 11;
  
  /*Command to create the two graphs */
  int reorder = 0;
  MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, reorder, &graph_comm);
  MPI_Graph_create(MPI_COMM_WORLD, nnodes, index1, edges1, reorder, &graph_comm1);
  
  /*Create two distinct 16 floating point number arrays for each graph to pass*/
  int number;
  static const float src[16] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0};
  float array[16];
  float array1[16];
  if(rank == 0){
    for(i = 0; i < 16; i++){
      array[i] = src[i];
      array1[i] = 2*src[i];
    }
    printf("Process %d Initialized arrays: ", rank);
    for(j = 0; j < 16; j++)
      printf("[%f] ", array[j]);
    printf("\n");
    for(j = 0; j <16; j++)
      printf("[%f] ", array1[j]);
    printf("\n");
    rec = 1;
    rec1 = 1;
  }
  
  /*Run through steps needed to broadcast message to each node */
  for(i = 0; i < steps; i++){
    number = m[rank][i]; /*holds non-zero when message is to be passed */
    if(number == 0 && rec == 0){ /*Only receive a message if haven't received already, and not sending at this step */
      for(j = 0; j < 16; j++){
        MPI_Recv(&array[j], 16, MPI_FLOAT, MPI_ANY_SOURCE, 0, graph_comm, MPI_STATUS_IGNORE); /* Wait to receive message from any at each step if don't have message*/
      }
      printf("Process %d: \n", rank);
      for(j = 0; j < 16; j++)
        printf("[%.1f] ", array[j]);
      rec = 1;
    }
    if(number != 0){ /*Send to correct adjacent node */
      for(j = 0; j < 16; j++){
       MPI_Send(&array[j], 16, MPI_FLOAT, number, 0, graph_comm); /* Send message */
      }
    }
    
  }
  /* Check that all nodes received the broadcast */
  int global_sum;
  MPI_Reduce(&rec, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(global_sum == 16){
    printf("\nAll processes successfully received broadcast for 16k3 graph \n");
  }
  /*Repeat for torus using correct references for graph */
  for(i = 0; i < steps1; i++){
    number = m1[rank][i];
    if(number == 0 && rec1 == 0){
      for(j = 0; j < 16; j++){
        MPI_Recv(&array1[j], 16, MPI_FLOAT, MPI_ANY_SOURCE, 0, graph_comm1, MPI_STATUS_IGNORE); /* Wait to receive message from any at each step if don't have message*/
      }
      printf("Process %d: \n", rank);
      for(j = 0; j < 16; j++)
        printf("[%.1f] ", array1[j]);
      rec1 = 1;
    }
    if(number != 0){
      for(j = 0; j < 16; j++){
        MPI_Send(&array1[j], 16, MPI_FLOAT, number, 0, graph_comm1); /* Send message */
      }
    }
    
  }
  
  int global_sum1;
  MPI_Reduce(&rec1, &global_sum1, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(global_sum1 == 16){
    printf("\nAll processes successfully received broadcast for torus graph \n");
  }
  
  
  MPI_Finalize();
  
}

