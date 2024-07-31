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

void HPL_pdgesv(HPL_T_grid* GRID, HPL_T_palg* ALGO, HPL_T_pmat* A) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdgesv factors a N+1-by-N matrix using LU factorization with row
   * partial pivoting.  The main algorithm  is the "right looking" variant
   * with  or  without look-ahead.  The  lower  triangular  factor is left
   * unpivoted and the pivots are not returned. The right hand side is the
   * N+1 column of the coefficient matrix.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  if(A->n <= 0) return;

  A->info = 0;

  HPL_T_panel * p, **panel = NULL;
  HPL_T_UPD_FUN HPL_pdupdate;
  int N, icurcol = 0, j, jb, jj = 0, jstart, k, mycol, n, nb, nn, npcol, nq,
         tag = MSGID_BEGIN_FACT, test;

  const int depth = 1; // NC: Hardcoded now

  mycol        = GRID->mycol;
  npcol        = GRID->npcol;
  HPL_pdupdate = ALGO->upfun;
  N            = A->n;
  nb           = A->nb;

  if(N <= 0) return;

  /*
   * Allocate a panel list of length depth + 1 (depth >= 1)
   */
  panel = (HPL_T_panel**)malloc((size_t)(depth + 1) * sizeof(HPL_T_panel*));
  if(panel == NULL) {
    HPL_pabort(__LINE__, "HPL_pdgesvK2", "Memory allocation failed");
  }
  /*
   * Create and initialize the first panel
   */
  nq     = HPL_numroc(N + 1, nb, nb, mycol, 0, npcol);
  nn     = N;
  jstart = 0;

  jb = Mmin(nn, nb);
  HPL_pdpanel_new(
      GRID, ALGO, nn, nn + 1, jb, A, jstart, jstart, tag, &panel[0]);
  nn -= jb;
  jstart += jb;
  if(mycol == icurcol) {
    jj += jb;
    nq -= jb;
  }
  icurcol = MModAdd1(icurcol, npcol);
  tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Create second panel
   */
  HPL_pdpanel_new(
      GRID, ALGO, nn, nn + 1, Mmin(nn, nb), A, jstart, jstart, tag, &panel[1]);
  tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Initialize the lookahead - Factor jstart columns: panel[0]
   */
  jb = jstart;
  jb = Mmin(jb, nb);
  /*
   * Factor and broadcast 0-th panel
   */

  if(GRID->myrow == 0 && mycol == 0) {
      printf("-----------------------------------\n");
      printf("Iteration: %5.1f%%, Column: %09d \n", 0.0, 0);
      printf("-----------------------------------\n");
    }

  HPL_pdfact(panel[0]);

  HPL_pdpanel_bcast(panel[0]);

  // Ubcast+row swaps for second part of A
  HPL_pdlaswp_exchange(panel[0], HPL_UPD_2);

  // Ubcast+row swaps for look ahead
  // nn = HPL_numrocI(jb, j, nb, nb, mycol, 0, npcol);
  HPL_pdlaswp_exchange(panel[0], HPL_LOOK_AHEAD);

  double stepStart, stepEnd;

  /*
   * Main loop over the remaining columns of A
   */
  for(j = jstart; j < N; j += nb) {
    HPL_ptimer_stepReset(HPL_TIMING_N, HPL_TIMING_BEG);

    stepStart = MPI_Wtime();
    n         = N - j;
    jb        = Mmin(n, nb);
    /*
     * Initialize current panel - Finish latest update, Factor and broadcast
     * current panel
     */
    (void)HPL_pdpanel_free(panel[1]);
    HPL_pdpanel_init(GRID, ALGO, n, n + 1, jb, A, j, j, tag, panel[1]);

    if(GRID->myrow == 0 && mycol == 0) {
      printf("-----------------------------------\n");
      printf("Iteration: %5.1f%%, Column: %09d \n", j * 100.0 / N, j);
      printf("-----------------------------------\n");
    }


    // HPL_pdupdate(panel[0], HPL_UPD_2);

    /*Panel factorization FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB +
     * (N-i*NB)(NB^2-NB)*/
    HPL_pdfact(panel[1]); /* factor current panel */

    /* broadcast current panel */
    HPL_pdpanel_bcast(panel[1]);

    // while the second section is updating, exchange the rows from the first
    // section
    HPL_pdlaswp_exchange(panel[0], HPL_UPD_1);

    /* Queue up finishing the first section */
    // HPL_pdupdate(panel[0], HPL_UPD_1);

    if(mycol == icurcol) {
      jj += jb;
      nq -= jb;
    }
    icurcol = MModAdd1(icurcol, npcol);
    tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

    HPL_pdlaswp_exchange(panel[1], HPL_UPD_2);

    // prep the row swaps for the next look ahead
    HPL_pdlaswp_exchange(panel[1], HPL_LOOK_AHEAD);

    // wait here for the updates to compete

    /*
     * Circular  of the panel pointers:
     * xtmp = x[0]; for( k=0; k < 1; k++ ) x[k] = x[k+1]; x[d] = xtmp;
     *
     * Go to next process row and column - update the message ids for broadcast
     */
    p        = panel[0];
    panel[0] = panel[1];
    panel[1] = p;
  }

  HPL_pdpanel_disp(&panel[0]);
  HPL_pdpanel_disp(&panel[1]);
  if(panel) free(panel);

  /*
   * Solve upper triangular system
   */
  // if(A->info == 0) HPL_pdtrsv(GRID, A);
}
