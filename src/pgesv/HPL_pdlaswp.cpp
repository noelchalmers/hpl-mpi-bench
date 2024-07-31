/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <chrono>

using timePoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

void HPL_pdlaswp_exchange(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswp_exchange applies the  NB  row interchanges to  NN columns of
   * the trailing submatrix and broadcast a column panel.
   *
   * A "Spread then roll" algorithm performs  the swap :: broadcast  of the
   * row panel U at once,  resulting in a minimal communication volume  and
   * a "very good"  use of the connectivity if available.  With  P  process
   * rows  and  assuming  bi-directional links,  the  running time  of this
   * function can be approximated by:
   *
   *    (log_2(P)+(P-1)) * lat +   K * NB * LocQ(N) / bdwth
   *
   * where  NB  is the number of rows of the row panel U,  N is the global
   * number of columns being updated,  lat and bdwth  are the latency  and
   * bandwidth  of  the  network  for  double  precision real words.  K is
   * a constant in (2,3] that depends on the achieved bandwidth  during  a
   * simultaneous  message exchange  between two processes.  An  empirical
   * optimistic value of K is typically 2.4.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  double *U, *W;
  double *dA, *dU, *dW;
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork,
      *lindxU = NULL, *lindxA = NULL, *lindxAU, *permU;
  int *dlindxU = NULL, *dlindxA = NULL, *dlindxAU, *dpermU, *dpermU_ex;
  int  icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW;

  /* ..
   * .. Executable Statements ..
   */
  n  = PANEL->n;
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  iflag = PANEL->IWORK;

  MPI_Comm comm = PANEL->grid->col_comm;

  // quick return if we're 1xQ
  if(nprow == 1) return;

  icurrow = PANEL->prow;

  if(UPD == HPL_LOOK_AHEAD) {
    U   = PANEL->U;
    W   = PANEL->W;
    dU  = PANEL->dU;
    dW  = PANEL->dW;
    LDU = PANEL->ldu0;
    LDW = PANEL->ldu0;
    n   = PANEL->nu0;
  } else if(UPD == HPL_UPD_1) {
    U   = PANEL->U1;
    W   = PANEL->W1;
    dU  = PANEL->dU1;
    dW  = PANEL->dW1;
    LDU = PANEL->ldu1;
    LDW = PANEL->ldu1;
    n   = PANEL->nu1;
  } else if(UPD == HPL_UPD_2) {
    U   = PANEL->U2;
    W   = PANEL->W2;
    dU  = PANEL->dU2;
    dW  = PANEL->dW2;
    LDU = PANEL->ldu2;
    LDW = PANEL->ldu2;
    n   = PANEL->nu2;
  }

  /*
   * Quick return if there is nothing to do
   */
  if((jb <= 0)) return;
  // if((n <= 0) || (jb <= 0)) return;

  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k         = (int)((unsigned int)(jb) << 1);
  ipl       = iflag + 1;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  /* Set MPI message counts and offsets */
  ipcounts[0]  = (iplen[1] - iplen[0]) * LDU;
  ipoffsets[0] = 0;

  for(int i = 1; i < nprow; ++i) {
    ipcounts[i]  = (iplen[i + 1] - iplen[i]) * LDU;
    ipoffsets[i] = ipcounts[i - 1] + ipoffsets[i - 1];
  }
  ipoffsets[nprow] = ipcounts[nprow - 1] + ipoffsets[nprow - 1];

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */

  MPI_Barrier(MPI_COMM_WORLD);

  double scatter_time=0., gather_time=0.;

  if (n>0) {
    if(myrow == icurrow) {
      // send rows to other ranks
      timePoint_t scatter_start = std::chrono::high_resolution_clock::now();
      HPL_scatterv(dU, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);
      timePoint_t scatter_end = std::chrono::high_resolution_clock::now();

      scatter_time = std::chrono::duration_cast<std::chrono::microseconds>(scatter_end - scatter_start).count()/1000.0;

      MPI_Barrier(comm);

      // All gather dU
      timePoint_t gather_start = std::chrono::high_resolution_clock::now();
      HPL_allgatherv(dU, ipcounts[myrow], ipcounts, ipoffsets, comm);
      timePoint_t gather_end = std::chrono::high_resolution_clock::now();

      gather_time = std::chrono::duration_cast<std::chrono::microseconds>(gather_end - gather_start).count()/1000.0;

    } else {

      // receive rows from icurrow into dW
      timePoint_t scatter_start = std::chrono::high_resolution_clock::now();
      HPL_scatterv(dW, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);
      timePoint_t scatter_end = std::chrono::high_resolution_clock::now();

      scatter_time = std::chrono::duration_cast<std::chrono::microseconds>(scatter_end - scatter_start).count()/1000.0;

      MPI_Barrier(comm);

      // All gather dU
      timePoint_t gather_start = std::chrono::high_resolution_clock::now();
      HPL_allgatherv(dU, ipcounts[myrow], ipcounts, ipoffsets, comm);
      timePoint_t gather_end = std::chrono::high_resolution_clock::now();

      gather_time = std::chrono::duration_cast<std::chrono::microseconds>(gather_end - gather_start).count()/1000.0;
    }
  }

  if (UPD == HPL_LOOK_AHEAD) {
    MPI_Request request[3];
    double scatterTimeRoot=0.;
    double gatherTimeRoot=0.;
    int nroot = 0;

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
      MPI_Irecv(&scatterTimeRoot, 1, MPI_DOUBLE, MModAdd1(PANEL->pcol, PANEL->grid->npcol), 0, PANEL->grid->row_comm, request+0);
      MPI_Irecv(&gatherTimeRoot,  1, MPI_DOUBLE, MModAdd1(PANEL->pcol, PANEL->grid->npcol), 1, PANEL->grid->row_comm, request+1);
      MPI_Irecv(&nroot,  1, MPI_INT, MModAdd1(PANEL->pcol, PANEL->grid->npcol), 2, PANEL->grid->row_comm, request+2);
    }
    if (PANEL->grid->mycol==MModAdd1(PANEL->pcol, PANEL->grid->npcol) && PANEL->grid->myrow==0) {
      MPI_Send(&scatter_time, 1, MPI_DOUBLE, 0, 0, PANEL->grid->row_comm);
      MPI_Send(&gather_time,  1, MPI_DOUBLE, 0, 1, PANEL->grid->row_comm);
      MPI_Send(&n,  1, MPI_INT, 0, 2, PANEL->grid->row_comm);
    }

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
      MPI_Waitall(3, request, MPI_STATUSES_IGNORE);

      if (scatterTimeRoot>0.0) {
        printf("Plaswp: Size (%5d x %4d), LookAhead Column %3d,                    Scatter Time (ms) =            %8.3f,                        BW Est (GB/s) = %6.2f\n",
                nroot, nroot,
                MModAdd1(PANEL->pcol, PANEL->grid->npcol),
                scatterTimeRoot,
                nroot*nroot*8/(1.0E6*scatterTimeRoot));
      }
      if (gatherTimeRoot>0.0) {
        printf("Plaswp: Size (%5d x %4d), LookAhead Column %3d,                     Gather Time (ms) =            %8.3f,                        BW Est (GB/s) = %6.2f\n",
                nroot, nroot,
                MModAdd1(PANEL->pcol, PANEL->grid->npcol),
                gatherTimeRoot,
                nroot*nroot*8/(1.0E6*gatherTimeRoot));
      }
    }

    return;
  }

  int nq_max=0;
  MPI_Reduce(&n, &nq_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

  double scatter_time_avg=0.;
  double scatter_time_min=0.;
  double scatter_time_max=0.;
  double scatter_time_stddev=0.;

  MPI_Reduce(&scatter_time, &scatter_time_avg, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
  MPI_Reduce(&scatter_time, &scatter_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->row_comm);
  MPI_Reduce(&scatter_time, &scatter_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
  scatter_time_avg /= PANEL->grid->npcol;

  double scatter_var = (scatter_time - scatter_time_avg)*(scatter_time - scatter_time_avg);
  MPI_Reduce(&scatter_var, &scatter_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
  scatter_time_stddev /= PANEL->grid->npcol;
  scatter_time_stddev = sqrt(scatter_time_stddev);

  double gather_time_avg=0.;
  double gather_time_min=0.;
  double gather_time_max=0.;
  double gather_time_stddev=0.;

  MPI_Reduce(&gather_time, &gather_time_avg, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
  MPI_Reduce(&gather_time, &gather_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->row_comm);
  MPI_Reduce(&gather_time, &gather_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
  gather_time_avg /= PANEL->grid->npcol;

  double gather_var = (gather_time - gather_time_avg)*(gather_time - gather_time_avg);
  MPI_Reduce(&gather_var, &gather_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
  gather_time_stddev /= PANEL->grid->npcol;
  gather_time_stddev = sqrt(gather_time_stddev);

  MPI_Request request[4];
  double scatterTimeAvgRoot=0.;
  double scatterTimeMinRoot=0.;
  double scatterTimeMaxRoot=0.;
  double scatterTimeStdDevRoot=0.;

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Irecv(&scatterTimeAvgRoot,    1, MPI_DOUBLE, PANEL->prow, 0, PANEL->grid->col_comm, request+0);
    MPI_Irecv(&scatterTimeMinRoot,    1, MPI_DOUBLE, PANEL->prow, 1, PANEL->grid->col_comm, request+1);
    MPI_Irecv(&scatterTimeMaxRoot,    1, MPI_DOUBLE, PANEL->prow, 2, PANEL->grid->col_comm, request+2);
    MPI_Irecv(&scatterTimeStdDevRoot, 1, MPI_DOUBLE, PANEL->prow, 3, PANEL->grid->col_comm, request+3);
  }
  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==PANEL->prow) {
    MPI_Send(&scatter_time_avg,    1, MPI_DOUBLE, 0, 0, PANEL->grid->col_comm);
    MPI_Send(&scatter_time_min,    1, MPI_DOUBLE, 0, 1, PANEL->grid->col_comm);
    MPI_Send(&scatter_time_max,    1, MPI_DOUBLE, 0, 2, PANEL->grid->col_comm);
    MPI_Send(&scatter_time_stddev, 1, MPI_DOUBLE, 0, 3, PANEL->grid->col_comm);
  }

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    if (scatterTimeMaxRoot>0.0) {
      printf("Plaswp: Size (%5d x %4d), RootRow    %3d, Scatter Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              nq_max, PANEL->jb, PANEL->prow, scatterTimeMinRoot, scatterTimeAvgRoot, scatterTimeStdDevRoot, scatterTimeMaxRoot,
              nq_max*PANEL->jb*8/(1.0E6*scatterTimeAvgRoot));
    }
  }

  double gatherTimeAvgRoot=0.;
  double gatherTimeMinRoot=0.;
  double gatherTimeMaxRoot=0.;
  double gatherTimeStdDevRoot=0.;

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Irecv(&gatherTimeAvgRoot,    1, MPI_DOUBLE, PANEL->prow, 0, PANEL->grid->col_comm, request+0);
    MPI_Irecv(&gatherTimeMinRoot,    1, MPI_DOUBLE, PANEL->prow, 1, PANEL->grid->col_comm, request+1);
    MPI_Irecv(&gatherTimeMaxRoot,    1, MPI_DOUBLE, PANEL->prow, 2, PANEL->grid->col_comm, request+2);
    MPI_Irecv(&gatherTimeStdDevRoot, 1, MPI_DOUBLE, PANEL->prow, 3, PANEL->grid->col_comm, request+3);
  }
  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==PANEL->prow) {
    MPI_Send(&gather_time_avg,    1, MPI_DOUBLE, 0, 0, PANEL->grid->col_comm);
    MPI_Send(&gather_time_min,    1, MPI_DOUBLE, 0, 1, PANEL->grid->col_comm);
    MPI_Send(&gather_time_max,    1, MPI_DOUBLE, 0, 2, PANEL->grid->col_comm);
    MPI_Send(&gather_time_stddev, 1, MPI_DOUBLE, 0, 3, PANEL->grid->col_comm);
  }

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    if (gatherTimeMaxRoot>0.0) {
      printf("Plaswp: Size (%5d x %4d), RootRow    %3d,  Gather Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              nq_max, PANEL->jb, PANEL->prow, gatherTimeMinRoot, gatherTimeAvgRoot, gatherTimeStdDevRoot, gatherTimeMaxRoot,
              nq_max*PANEL->jb*8/(1.0E6*gatherTimeAvgRoot));
    }
  }

  /*
   * End of HPL_pdlaswp_exchange
   */
}
