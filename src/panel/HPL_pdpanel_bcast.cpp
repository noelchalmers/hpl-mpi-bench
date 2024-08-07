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

int HPL_pdpanel_bcast(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_bcast broadcasts  the  current  panel.  Successful  completion
   * is indicated by a return code of HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * PANEL   (input/output)                HPL_T_panel *
   *         On entry,  PANEL  points to the  current panel data structure
   *         being broadcast.
   *
   * ---------------------------------------------------------------------
   */

  if(PANEL == NULL) { return HPL_SUCCESS; }

  int err = HPL_SUCCESS;

  MPI_Barrier(MPI_COMM_WORLD);

  if(PANEL->grid->npcol > 1) {

    MPI_Comm comm = PANEL->grid->row_comm;
    int      root = PANEL->pcol;

    /*
     * Single Bcast call
     */
    timePoint_t bcast_start = std::chrono::high_resolution_clock::now();
    err = HPL_bcast(PANEL->dL2, PANEL->len, root, comm, PANEL->algo->btopo);
    timePoint_t bcast_end = std::chrono::high_resolution_clock::now();

    double bcast_time = std::chrono::duration_cast<std::chrono::microseconds>(bcast_end - bcast_start).count()/1000.0;

    // Total time for this row's bcast is the max over the row
    double bcast_time_total=0.;
    MPI_Reduce(&bcast_time, &bcast_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);

    // Min max avg over all rows' bcast
    double bcast_time_avg=0.;
    double bcast_time_min=0.;
    double bcast_time_max=0.;
    double bcast_time_stddev=0.;

    MPI_Allreduce(&bcast_time_total, &bcast_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
    MPI_Reduce(&bcast_time_total, &bcast_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
    MPI_Reduce(&bcast_time_total, &bcast_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
    bcast_time_avg /= PANEL->grid->nprow;

    double bcast_var = (bcast_time_total - bcast_time_avg)*(bcast_time_total - bcast_time_avg);
    MPI_Reduce(&bcast_var, &bcast_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
    bcast_time_stddev /= PANEL->grid->nprow;
    bcast_time_stddev = sqrt(bcast_time_stddev);

    int mp_max=0;
    MPI_Reduce(&PANEL->mp, &mp_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
      printf("LBcast: Size (%5d x %4d), RootColumn %3d,   Bcast Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              mp_max, PANEL->jb, PANEL->pcol, bcast_time_min, bcast_time_avg, bcast_time_stddev, bcast_time_max,
              mp_max*PANEL->jb*8/(1.0E6*bcast_time_avg));
    }
  }

  if (err != HPL_SUCCESS) return err;


  int* ipiv     = PANEL->ipiv;
  int* dipiv    = PANEL->dipiv;


  CHECK_HIP_ERROR(hipMemcpy(ipiv,
                            dipiv,
                            PANEL->jb * sizeof(int),
                            hipMemcpyDeviceToHost));

  int  jb        = PANEL->jb;
  int  k         = (int)((unsigned int)(jb) << 1);
  int* iflag     = PANEL->IWORK;
  int* ipl       = iflag + 1;
  int* ipID      = ipl + 1;
  int* ipA       = ipID + ((unsigned int)(k) << 1);
  int* iplen     = ipA + 1;
  int* ipcounts  = iplen + PANEL->grid->nprow + 1;
  int* ipoffsets = ipcounts + PANEL->grid->nprow;
  int* iwork     = ipoffsets + PANEL->grid->nprow;

  // compute spreading info
  HPL_pipid(PANEL, ipl, ipID);
  HPL_piplen(PANEL, *ipl, ipID, iplen, iwork);

  return HPL_SUCCESS;
}
