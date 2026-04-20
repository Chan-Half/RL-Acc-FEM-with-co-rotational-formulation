#include "kkt.h"

#ifndef EMBEDDED


csc* form_KKT(const csc  *P,
              const  csc *A,
              c_int       format,
              c_float    *param1,
              c_float    *param2,
              c_int      *PtoKKT,
              c_int      *AtoKKT,
              c_int     **Pdiag_idx,
              c_int      *Pdiag_n,
              c_int      *param2toKKT,
              c_int      *sigmatoKKT) {
  c_int  nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros
                          // in KKT matrix
  csc   *KKT_trip, *KKT;  // KKT matrix in triplet format and CSC format
  c_int  ptr, i, j;       // Counters for elements (i,j) and index pointer
  c_int  zKKT = 0;        // Counter for total number of elements in P and in
                          // KKT
  c_int *KKT_TtoC;        // Pointer to vector mapping from KKT in triplet form
                          // to CSC

  // Get matrix dimensions
  nKKT = P->m + A->m;

  // Get maximum number of nonzero elements (now for a FULL matrix)
  nnzKKTmax = P->p[P->n] +   // Number of elements in P
              P->m +         // Number of elements in param1 * I
              2 * A->p[A->n] + // Number of nonzeros in A AND A^T (top-right & bottom-left)
              A->m;          // Number of elements in - diag(param2)

  // Preallocate KKT matrix in triplet format
  KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

  if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

  // Allocate vector of indices on the diagonal. Worst case it has m elements
  if (Pdiag_idx != OSQP_NULL) {
    (*Pdiag_idx) = (c_int *)c_malloc(P->m * sizeof(c_int));
    *Pdiag_n     = 0; // Set 0 diagonal elements to start
  }

  // Allocate Triplet matrices
  // -------------------------
  // 1. P + param1 I (Top-Left block)
  // -------------------------
  for (j = 0; j < P->n; j++) { // cycle over columns
    c_int has_diag = 0; // Flag to check if diagonal element exists in this column

    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // cycle over rows
      // Get current row
      i = P->i[ptr];

      // Add element of P
      KKT_trip->i[zKKT] = i;
      KKT_trip->p[zKKT] = j;
      KKT_trip->x[zKKT] = P->x[ptr];

      if (PtoKKT != OSQP_NULL) PtoKKT[ptr] = zKKT;  // Update index from P to KKTtrip

      if (i == j) {                                 // P has a diagonal element,
                                                    // add param1
        KKT_trip->x[zKKT] += param1[j];
        if (sigmatoKKT != OSQP_NULL) sigmatoKKT[j] = zKKT;
        // If index vector pointer supplied -> Store the index
        if (Pdiag_idx != OSQP_NULL) {
          (*Pdiag_idx)[*Pdiag_n] = ptr;
          (*Pdiag_n)++;
        }
        has_diag = 1; // Mark that diagonal was found
      }
      zKKT++;
    }

    // If no diagonal element was found in this column, explicitly add it
    if (!has_diag) {
      KKT_trip->i[zKKT] = j;
      KKT_trip->p[zKKT] = j;
      KKT_trip->x[zKKT] = param1[j];
      if (sigmatoKKT != OSQP_NULL) sigmatoKKT[j] = zKKT;
      zKKT++;
    }
  }

  if (Pdiag_idx != OSQP_NULL) {
    // Realloc Pdiag_idx so that it contains exactly *Pdiag_n diagonal elements
    (*Pdiag_idx) = (c_int *)c_realloc((*Pdiag_idx), (*Pdiag_n) * sizeof(c_int));
  }


  // -------------------------
  // 2. A' at top right block
  // -------------------------
  for (j = 0; j < A->n; j++) {                      // Cycle over columns of A
    for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
      KKT_trip->p[zKKT] = P->m + A->i[ptr];         // Assign column index from row index of A
      KKT_trip->i[zKKT] = j;                        // Assign row index from column index of A
      KKT_trip->x[zKKT] = A->x[ptr];                // Assign A value element

      if (AtoKKT != OSQP_NULL) AtoKKT[ptr] = zKKT;  // Update index from A to KKTtrip
                                                    // Note: AtoKKT only tracks the top-right block
      zKKT++;
    }
  }

  // -------------------------
  // 3. A at bottom left block (NEW BLOCK FOR FULL KKT)
  // -------------------------
  for (j = 0; j < A->n; j++) {                      // Cycle over columns of A
    for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
      KKT_trip->i[zKKT] = P->m + A->i[ptr];         // Row index: below P block
      KKT_trip->p[zKKT] = j;                        // Column index: same as A
      KKT_trip->x[zKKT] = A->x[ptr];                // Value
      
      // We don't track AtoKKT here since OSQP's AtoKKT vector assumes only one mapping per element of A.
      // (See Warning below regarding update_matrices)
      zKKT++;
    }
  }

  // -------------------------
  // 4. - diag(param2) at bottom right
  // -------------------------
  for (j = 0; j < A->m; j++) {
    KKT_trip->i[zKKT] = j + P->n;
    KKT_trip->p[zKKT] = j + P->n;
    KKT_trip->x[zKKT] = -param2[j];

    if (param2toKKT != OSQP_NULL) param2toKKT[j] = zKKT;  // Update index from param2 to KKTtrip
    zKKT++;
  }

  // Allocate number of nonzeros
  KKT_trip->nz = zKKT;

  // Convert triplet matrix to csc format
  if (!PtoKKT && !AtoKKT && !param2toKKT && !sigmatoKKT) {
    if (format == 0) KKT = triplet_to_csc(KKT_trip, OSQP_NULL);
    else KKT = triplet_to_csr(KKT_trip, OSQP_NULL);
  }
  else {
    // Allocate vector of indices from triplet to csc
    KKT_TtoC = (c_int *)c_malloc((zKKT) * sizeof(c_int));

    if (!KKT_TtoC) {
      csc_spfree(KKT_trip);
      if(Pdiag_idx) c_free(*Pdiag_idx);
      return OSQP_NULL;
    }

    if (format == 0)
      KKT = triplet_to_csc(KKT_trip, KKT_TtoC);
    else
      KKT = triplet_to_csr(KKT_trip, KKT_TtoC);

    // Update vectors of indices from P, A, param2 to KKT (now in CSC format)
    if (PtoKKT != OSQP_NULL) {
      for (i = 0; i < P->p[P->n]; i++) {
        PtoKKT[i] = KKT_TtoC[PtoKKT[i]];
      }
    }

    if (AtoKKT != OSQP_NULL) {
      for (i = 0; i < A->p[A->n]; i++) {
        AtoKKT[i] = KKT_TtoC[AtoKKT[i]]; 
      }
    }

    if (param2toKKT != OSQP_NULL) {
      for (i = 0; i < A->m; i++) {
        param2toKKT[i] = KKT_TtoC[param2toKKT[i]];
      }
    }
    
    if (sigmatoKKT != OSQP_NULL) {
      for (i = 0; i < P->n; i++) {
        sigmatoKKT[i] = KKT_TtoC[sigmatoKKT[i]];
      }
    }

    c_free(KKT_TtoC);
  }

  csc_spfree(KKT_trip);

  return KKT;
}

#endif /* ifndef EMBEDDED */


#if EMBEDDED != 1

void update_KKT_P(csc           *KKT,
                  const csc     *P,
                  const c_int   *PtoKKT,
                  const c_float *param1,
                  const c_int   *Pdiag_idx,
                  const c_int    Pdiag_n) {
  c_int i, j; // Iterations

  // Update elements of KKT using P
  for (i = 0; i < P->p[P->n]; i++) {
    KKT->x[PtoKKT[i]] = P->x[i];
  }

  // Update diagonal elements of KKT by adding sigma
  for (i = 0; i < Pdiag_n; i++) {
    j                  = Pdiag_idx[i]; // Extract index of the element on the
                                       // diagonal
    KKT->x[PtoKKT[j]] += param1[i];
  }
}

void update_KKT_A(csc *KKT, const csc *A, const c_int *AtoKKT) {
  c_int i; // Iterations

  // Update elements of KKT using A
  for (i = 0; i < A->p[A->n]; i++) {
    KKT->x[AtoKKT[i]] = A->x[i];
  }
}

void update_KKT_param2(csc *KKT, const c_float *param2,
                       const c_int *param2toKKT, const c_int m) {
  c_int i; // Iterations

  // Update elements of KKT using param2
  for (i = 0; i < m; i++) {
    KKT->x[param2toKKT[i]] = -param2[i];
  }
}


// update_KKT_sigma_vec(s->KKT, s->PtoKKT, sigma_vec, s->sigma_vec, s->n);
void update_KKT_sigma_vec(csc *KKT, const c_int *sigmatoKKT, const c_float *new_sigma_vec, const c_float *sigma_vec, const c_int n) {
  c_int i; // Iterations

  // Update elements of KKT using param2
  for (i = 0; i < n; i++) {
    // c_print("sigmavec: %f, %f, %f\n", new_sigma_vec[i], sigma_vec[i], KKT->x[sigmatoKKT[i]]);
    KKT->x[sigmatoKKT[i]] += - sigma_vec[i] + new_sigma_vec[i];
    // c_print("sigmavec: %f, %f, %f\n", new_sigma_vec[i], sigma_vec[i], KKT->x[sigmatoKKT[i]]);
  }
}

#endif // EMBEDDED != 1
