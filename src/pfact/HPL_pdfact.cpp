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
#include <assert.h>
#include <cstdlib>
#include <limits>
#include <chrono>

using timePoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

void HPL_pdfact(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdfact recursively factorizes a  1-dimensional  panel of columns.
   * The  RPFACT  function pointer specifies the recursive algorithm to be
   * used, either Crout, Left- or Right looking.  NBMIN allows to vary the
   * recursive stopping criterium in terms of the number of columns in the
   * panel, and  NDIV allows to specify the number of subpanels each panel
   * should be divided into. Usuallly a value of 2 will be chosen. Finally
   * PFACT is a function pointer specifying the non-recursive algorithm to
   * to be used on at most NBMIN columns. One can also choose here between
   * Crout, Left- or Right looking.  Empirical tests seem to indicate that
   * values of 4 or 8 for NBMIN give the best results.
   *
   * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
   * operations  at once  for one column in the panel.  This  results in a
   * lower number of slightly larger  messages than usual.  On P processes
   * and assuming bi-directional links,  the running time of this function
   * can be approximated by (when N is equal to N0):
   *
   *    N0 * log_2( P ) * ( lat + ( 2*N0 + 4 ) / bdwth ) +
   *    N0^2 * ( M - N0/3 ) * gam2-3
   *
   * where M is the local number of rows of  the panel, lat and bdwth  are
   * the latency and bandwidth of the network for  double  precision  real
   * words, and  gam2-3  is  an estimate of the  Level 2 and Level 3  BLAS
   * rate of execution. The  recursive  algorithm  allows indeed to almost
   * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
   * large  number of modern machines,  this  operation is however latency
   * bound,  meaning  that its cost can  be estimated  by only the latency
   * portion N0 * log_2(P) * lat.  Mono-directional links will double this
   * communication cost.
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

  int jb, i;

  jb = PANEL->jb;
  PANEL->n -= jb;
  PANEL->ja += jb;

  double pdfact_time;

  MPI_Barrier(MPI_COMM_WORLD);

  if(PANEL->grid->mycol == PANEL->pcol) {

    timePoint_t pdfact_start = std::chrono::high_resolution_clock::now();

    /*
     * Factor the panel - Update the panel pointers
     */
    double max_value[512];
    int    max_index[512];

    double* WORK = PANEL->fWORK;

    roctxRangePush("pdfact");

    #pragma omp parallel shared(max_value, max_index)
    {
      const int thread_rank = omp_get_thread_num();
      const int thread_size = omp_get_num_threads();
      assert(thread_size <= 512);

      for (int i=0;i<jb;++i) {
        if(PANEL->mp > 0) {

          int mp = PANEL->mp;
          int nb = PANEL->nb;
          int myrow = PANEL->grid->myrow;
          int nprow = PANEL->grid->nprow;

          if(thread_rank == 0) {

            int ilindx = rand()%mp;
            int igindx = 0;
            Mindxl2g(igindx, ilindx, nb, nb, myrow, 0, nprow);
            /*
             * WORK[0] := local maximum absolute value scalar,
             * WORK[1] := corresponding local  row index,
             * WORK[2] := corresponding global row index,
             * WORK[3] := coordinate of process owning this max.
             */
            WORK[0] = drand48() - 0.5;
            WORK[1] = (double)(ilindx);
            WORK[2] = (double)(igindx);
            WORK[3] = (double)(myrow);
          }

        } else {
          /*
           * If I do not have any row of A, then set the coordinate of the process
           * (WORK[3]) owning this "ghost" row,  such that it  will never be used,
           * even if there are only zeros in the current column of A.
           */
          if(thread_rank == 0) {
            WORK[0] = WORK[1] = WORK[2] = HPL_rzero;
            WORK[3]                     = (double)(PANEL->grid->nprow);
          }
        }

        #pragma omp barrier

        if (thread_rank==0) {
          MPI_Comm comm    = PANEL->grid->col_comm;
          int icurrow = PANEL->prow;

          int cnt0 = 4 + 2*PANEL->nb;
          double* Wwork = WORK + cnt0;

          /* Perform swap-broadcast */
          timePoint_t swap_start = std::chrono::high_resolution_clock::now();
          HPL_all_reduce_dmxswp(WORK, cnt0, icurrow, comm, Wwork);
          timePoint_t swap_end = std::chrono::high_resolution_clock::now();
          PANEL->timers[i] = std::chrono::duration_cast<std::chrono::microseconds>(swap_end - swap_start).count();

          PANEL->ipiv[i] = (int)WORK[2];
        }

        #pragma omp barrier
      }
    }

    roctxRangePop();

    timePoint_t pdfact_end = std::chrono::high_resolution_clock::now();

    pdfact_time = std::chrono::duration_cast<std::chrono::microseconds>(pdfact_end - pdfact_start).count()/1000.0;

    // PANEL->A   = Mptr( PANEL->A, 0, jb, PANEL->lda );
    // PANEL->dA = Mptr(PANEL->dA, 0, jb, PANEL->dlda);
    PANEL->nq -= jb;
    PANEL->jj += jb;

    int* ipiv     = PANEL->ipiv;
    int* dipiv     = PANEL->dipiv;

    // send the ipivs along with L2 in the Bcast
    CHECK_HIP_ERROR(hipMemcpy(dipiv,
                              ipiv,
                              jb * sizeof(int),
                              hipMemcpyHostToDevice));\
  }

  // Compute stats on swaps
  double swap_avg=0.;
  double swap_min=std::numeric_limits<double>::max();
  double swap_max=std::numeric_limits<double>::min();

  for (int i=0;i<jb;++i) {
    swap_avg += PANEL->timers[i];
    swap_min = std::min(swap_min, PANEL->timers[i]);
    swap_max = std::max(swap_max, PANEL->timers[i]);
  }
  swap_avg /= jb;

  double swap_stddev=0.0;
  for (int i=0;i<jb;++i) {
    swap_stddev += (PANEL->timers[i]-swap_avg)*(PANEL->timers[i]-swap_avg);
  }
  swap_stddev /= jb;
  swap_stddev = sqrt(swap_stddev);

  int mp_max=0;
  MPI_Reduce(&PANEL->mp, &mp_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

  MPI_Request request[5];
  double pdfactTimeRoot=0.;
  double swapAvgRoot=0.;
  double swapMinRoot=0.;
  double swapMaxRoot=0.;
  double swapStdDevRoot=0.;

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Irecv(&pdfactTimeRoot, 1, MPI_DOUBLE, PANEL->pcol, 0, PANEL->grid->row_comm, request+0);
    MPI_Irecv(&swapAvgRoot,    1, MPI_DOUBLE, PANEL->pcol, 1, PANEL->grid->row_comm, request+1);
    MPI_Irecv(&swapMinRoot,    1, MPI_DOUBLE, PANEL->pcol, 2, PANEL->grid->row_comm, request+2);
    MPI_Irecv(&swapMaxRoot,    1, MPI_DOUBLE, PANEL->pcol, 3, PANEL->grid->row_comm, request+3);
    MPI_Irecv(&swapStdDevRoot, 1, MPI_DOUBLE, PANEL->pcol, 4, PANEL->grid->row_comm, request+4);
  }
  if (PANEL->grid->mycol==PANEL->pcol && PANEL->grid->myrow==0) {
    MPI_Send(&pdfact_time, 1, MPI_DOUBLE, 0, 0, PANEL->grid->row_comm);
    MPI_Send(&swap_avg,    1, MPI_DOUBLE, 0, 1, PANEL->grid->row_comm);
    MPI_Send(&swap_min,    1, MPI_DOUBLE, 0, 2, PANEL->grid->row_comm);
    MPI_Send(&swap_max,    1, MPI_DOUBLE, 0, 3, PANEL->grid->row_comm);
    MPI_Send(&swap_stddev, 1, MPI_DOUBLE, 0, 4, PANEL->grid->row_comm);
  }

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Waitall(5, request, MPI_STATUSES_IGNORE);

    printf("Pdfact: Size (%5d x %4d), RootColumn %3d,    Swap Time (us) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), Total Pdfact Time (ms) = %8.3f, \n",
            mp_max, PANEL->jb, PANEL->pcol, swapMinRoot, swapAvgRoot, swapStdDevRoot, swapMaxRoot, pdfactTimeRoot);
  }
}
