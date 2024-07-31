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

#include <limits>
#include "hpl.hpp"

void HPL_pdtest(HPL_T_test* TEST,
                HPL_T_grid* GRID,
                HPL_T_palg* ALGO,
                const int   N,
                const int   NB) {
/*
 * Purpose
 * =======
 *
 * HPL_pdtest performs  one  test  given a set of parameters such as the
 * process grid, the  problem size, the distribution blocking factor ...
 * This function generates  the data, calls  and times the linear system
 * solver,  checks  the  accuracy  of the  obtained vector solution  and
 * writes this information to the file pointed to by TEST->outfp.
 *
 * Arguments
 * =========
 *
 * TEST    (global input)                HPL_T_test *
 *         On entry,  TEST  points  to a testing data structure:  outfp
 *         specifies the output file where the results will be printed.
 *         It is only defined and used by the process  0  of the  grid.
 *         thrsh  specifies  the  threshhold value  for the test ratio.
 *         Concretely, a test is declared "PASSED"  if and only if the
 *         following inequality is satisfied:
 *         ||Ax-b||_oo / ( epsil *
 *                         ( || x ||_oo * || A ||_oo + || b ||_oo ) *
 *                          N )  < thrsh.
 *         epsil  is the  relative machine precision of the distributed
 *         computer. Finally the test counters, kfail, kpass, kskip and
 *         ktest are updated as follows:  if the test passes,  kpass is
 *         incremented by one;  if the test fails, kfail is incremented
 *         by one; if the test is skipped, kskip is incremented by one.
 *         ktest is left unchanged.
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters to be used for this test.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the order of the coefficient matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
#ifdef HPL_DETAILED_TIMING
  double HPL_w[HPL_TIMING_N];
#endif
  HPL_T_pmat mat;
  double     wtime[1];
  int        ierr;
  double     Anorm1, AnormI, Gflops, Xnorm1, XnormI, BnormI, resid0, resid1;
  double*    Bptr;
  double*    dBptr;
  static int first = 1;
  int        ii, ip2, mycol, myrow, npcol, nprow, nq;
  char       ctop, cpfact, crfact;
  time_t     current_time_start, current_time_end;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  /*
   * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
   * N by N+1.  One column is added in every process column for the solve.
   * The  result  however  is stored in a 1 x N vector replicated in every
   * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
   * workspace is mp.
   */
  ierr = HPL_pdmatgen(TEST, GRID, ALGO, &mat, N, NB);

  if(ierr != HPL_SUCCESS) {
    (TEST->kskip)++;
    HPL_pdmatfree(&mat);
    return;
  }

  /* Create row-swapping data type */
  MPI_Type_contiguous(NB + 4, MPI_DOUBLE, &PDFACT_ROW);
  MPI_Type_commit(&PDFACT_ROW);

  /*
   * generate matrix and right-hand-side, [ A | b ] which is N by N+1.
   */
  // HPL_pdrandmat(GRID, N, N + 1, NB, mat.dA, mat.ld, HPL_ISEED);

  /*
   * Solve linear system
   */
  HPL_ptimer_boot();
  (void)HPL_barrier(GRID->all_comm);
  time(&current_time_start);
  HPL_ptimer(0);
  HPL_pdgesv(GRID, ALGO, &mat);
  HPL_ptimer(0);
  time(&current_time_end);


  /* Release row swapping datatype */
  MPI_Type_free(&PDFACT_ROW);

  HPL_pdmatfree(&mat);
}
