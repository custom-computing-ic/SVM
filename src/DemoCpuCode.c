#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


////////////////////// Definitions //////////////////////

// type of input data X
typedef double DataType;

// type of input data Y and internal calculations
typedef double CalcType;

// SVR Parameters
typedef struct {
	// Input File Name
	char * InFile;
	// Output File Name
	char * OutFile;
	// Length of the input data file
	size_t DataSize;
	// dimension of input data X - number of features
	size_t DataDim;
	// Window Size
	size_t WinSize;
	// Allocated Size of Matrix R - actual size SSize+1
	size_t RSize;
	// epsilon - for SVR
	CalcType ep;
	// C - for SVR
	CalcType C;
	// sigma^2 - for RBF Kernel
	CalcType sigma_sq;
	// eps - for numerical stability
	CalcType eps;
} Param;



////////////////////// Utility Functions //////////////////////

// RBF Kernel
static inline CalcType Kernel(DataType *X1, DataType*X2, const size_t DataDim, const CalcType sigma_sq) {
	CalcType sum = 0;
	for (size_t i=0; i<DataDim; ++i) sum -= (X1[i] - X2[i]) * (X1[i] - X2[i]);
	return exp(sum/2/sigma_sq);
}


// Calculating h(Xi) - Assuming Q is full matrix
CalcType hCalc(	const size_t ID,
				DataType *dataY,
				CalcType *Q,
				CalcType *theta,
				CalcType *b,
				const size_t WinSize) {

	// Initialise f(Xi)
	CalcType fXi = *b;

	// Iterating over the whole window - optimisable
	for (size_t i=0; i<WinSize; ++i) {
		fXi += theta[i] * Q[ID*WinSize+i];
	}

	// Compute h(Xi) = f(Xi) - Yi
	return fXi - dataY[ID];
}

CalcType objCalc(	CalcType *dataY,
					char *Group,
					CalcType *theta,
					CalcType *Q,
					CalcType *b,
					const CalcType C,
					const CalcType ep,
					const size_t WinSize){

	// Calculate the first term
	CalcType temp1 = 0;
	for (size_t i=0; i<WinSize; ++i) {
		// Calculate the first term
		for (size_t j=0; j<WinSize; ++j) {
			temp1 += theta[i] * theta[j] * Q[i*WinSize+j] / 2;
		}
	}

	// Calculate the second term - only Set E matters
	CalcType temp2 = 0;
	for (size_t i=0; i<WinSize; ++i) {
		if (Group[i]=='E'){
			CalcType hXi = hCalc(i, dataY, Q, theta, b, WinSize);
			temp2 += (fabs(hXi)>ep) ? C*(fabs(hXi)-ep) : 0;
		}
	}

	return temp1+temp2;
}


////////////////////// SVM Functions //////////////////////

// SVM regression
CalcType regSVM(	DataType *X_IN,
				DataType *dataX,
				CalcType *theta,
				CalcType *b,
				const size_t WinSize,
				const size_t DataDim,
				const CalcType sigma_sq) {

	CalcType f = *b;
	for (size_t i=0; i<WinSize; ++i) {
		f += theta[i] * Kernel(X_IN, &(dataX[i*DataDim]), DataDim, sigma_sq);
	}

	return f;
}

// SVM initialisation
void initSVM(		DataType *X1,
				CalcType Y1,
				DataType *X2,
				CalcType Y2,
				DataType *dataX,
				CalcType *dataY,
				char *Group,
				size_t *SMask,
				size_t *NMask,
				CalcType *Q,
				CalcType *R,
				CalcType *beta,
				CalcType *gamma,
				CalcType *theta,
				CalcType *b,
				CalcType *hXi,
				size_t *_CurSize,
				size_t *_SSize,
				size_t *_NSize,
				const Param param) {

	///////////////////////////// Read Parameters /////////////////////////////

	const size_t DataSize	= param.DataSize;
	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;
	const CalcType eps		= param.eps;


	///////////////////////////// Adding items to data set /////////////////////////////

	if (Y1>=Y2) {
		for (size_t i=0; i<DataDim; ++i) {
			dataX[0*DataDim+i] = X1[i];
			dataX[1*DataDim+i] = X2[i];
		}
		dataY[0] = Y1;
		dataY[1] = Y2;
	}
	else {
		for (size_t i=0; i<DataDim; ++i) {
			dataX[1*DataDim+i] = X1[i];
			dataX[0*DataDim+i] = X2[i];
		}
		dataY[1] = Y1;
		dataY[0] = Y2;
	}

	// Update Q
	Q[0*WinSize+0] = Kernel(&(dataX[0*DataDim]), &(dataX[0*DataDim]), DataDim, sigma_sq);
	Q[0*WinSize+1] = Kernel(&(dataX[0*DataDim]), &(dataX[1*DataDim]), DataDim, sigma_sq);
	Q[1*WinSize+1] = Kernel(&(dataX[1*DataDim]), &(dataX[1*DataDim]), DataDim, sigma_sq);
	Q[1*WinSize+0] = Q[0*WinSize+1];

	// Update theta
	CalcType temp = (dataY[0]-dataY[1]-2*ep)/2/(Q[0*WinSize+0]-Q[0*WinSize+1]);
	CalcType temp1 = (temp < C) ? temp : C;
	theta[0] = (temp1 > 0) ? temp1 : 0;
	theta[1] = -theta[0];

	// Update b
	*b = (dataY[0]+dataY[1])/2;

	// Update S, E, R
	// theta[0]=C, theta[1] = -C, both join E
	if (theta[0]==C) {
		Group[0] = 'E';
		Group[1] = 'E';
		NMask[0] = 0;
		NMask[1] = 1;
		hXi[0] = hCalc(0, dataY, Q, theta, b, WinSize);
		hXi[1] = hCalc(1, dataY, Q, theta, b, WinSize);
		*_NSize = 2;
		*_CurSize = 2;
	}
	// theta[0]=theta[1]=0, both join R
	else if (theta[0]==0) {
		Group[0] = 'R';
		Group[1] = 'R';
		NMask[0] = 0;
		NMask[1] = 1;
		hXi[0] = hCalc(0, dataY, Q, theta, b, WinSize);
		hXi[1] = hCalc(1, dataY, Q, theta, b, WinSize);
		*_NSize = 2;
		*_CurSize = 2;
	}
	// |theta[0]| and |theta[1]| within (0, C), both join S
	else {
		Group[0] = 'S';
		Group[1] = 'S';
		SMask[0] = 0;
		SMask[1] = 1;
		*_SSize = 2;
		*_CurSize = 2;

		// initialise R
		CalcType Q00 = Q[0*WinSize+0];
		CalcType Q01 = Q[0*WinSize+1];
		CalcType Q11 = Q[1*WinSize+1];
		CalcType div = Q00 + Q11 - 2*Q01;
		R[0*RSize+0] = (Q01*Q01 - Q00*Q11) / div;
		R[0*RSize+1] = (Q11-Q01) / div;
		R[0*RSize+2] = (Q00-Q01) / div;
		R[1*RSize+0] = R[0*RSize+1];
		R[1*RSize+1] = 1 / div;
		R[1*RSize+2] = -R[1*RSize+1];
		R[2*RSize+0] = R[0*RSize+2];
		R[2*RSize+1] = R[1*RSize+2];
		R[2*RSize+2] = R[1*RSize+1];
	}

}

// incremental learning
int incSVM(		size_t ID,
				DataType *Xc,
				CalcType Yc,
				DataType *dataX,
				CalcType *dataY,
				char *Group,
				size_t *SMask,
				size_t *NMask,
				CalcType *Q,
				CalcType *R,
				CalcType *beta,
				CalcType *gamma,
				CalcType *theta,
				CalcType *b,
				CalcType *hXi,
				size_t *_CurSize,
				size_t *_SSize,
				size_t *_NSize,
				const Param param) {

	///////////////////////////// Read Parameters /////////////////////////////

	const size_t DataSize	= param.DataSize;
	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;
	const CalcType eps		= param.eps;

	size_t CurSize = *_CurSize;
	size_t SSize = *_SSize;
	size_t NSize = *_NSize;


	///////////////////////////// Adding (Xc, Yc) to data set /////////////////////////////

	for (size_t i=0; i<DataDim; ++i) {
		dataX[ID*DataDim+i] = Xc[i];
	}
	dataY[ID] = Yc;


	///////////////////////////// Check (Xc, Yc) /////////////////////////////

	// Initialise theta_c
	theta[ID] = 0;

	// Update Q - optimisable
	for (size_t i=0; i<WinSize; ++i) Q[ID*WinSize+i] = Kernel(Xc, &(dataX[i*DataDim]), DataDim, sigma_sq);
	for (size_t i=0; i<WinSize; ++i) Q[i*WinSize+ID] = Q[ID*WinSize+i];

	// Compute h(Xc)
	hXi[ID] = hCalc(ID, dataY, Q, theta, b, WinSize);

	// non-support vector => Xc joins 'R' and terminate, without changing theta or b
	if (fabs(hXi[ID])<=ep) {
		Group[ID] = 'R';
		NMask[NSize] = ID;
		(*_NSize)++;
		(*_CurSize)++;
		return 0;
	}


	///////////////////////////// Prepare for bookkeeping /////////////////////////////

	// Label the new comer as Xc
	Group[ID] = 'C';

	// Assign q
	CalcType q = hXi[ID] > 0 ? -1 : 1;


	///////////////////////////// Main Loop /////////////////////////////

	while(1) {

		if(SSize==0){
			// Calculate bC - Case 1
			CalcType bC = fabs(hXi[ID] + q*ep);

			// Calculate bE,bR - Case 2&3
			CalcType bE = INFINITY;
			CalcType bR = INFINITY;
			size_t keyE = 0;
			size_t keyR = 0;
			for (size_t i=0; i<NSize; ++i) {
				CalcType hX = hXi[NMask[i]];
				if (Group[NMask[i]]=='E') {
					CalcType curbE = INFINITY;
					if ((q>0 && hX<-ep)||(q<0 && hX>ep)) curbE = fabs(hX + q*ep);
					if (curbE<bE) {
						bE = curbE;
						keyE = i;
					}
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curbR = fabs(-hX + q*ep);
					if (curbR<bR) {
						bR = curbR;
						keyR = i;
					}
				}
			}

			// Calculate delta_b
			CalcType L[3] = {bC, bE, bR};
			size_t key[3] = {ID, keyE, keyR};
			CalcType minL = bC;
			size_t minkey = ID;
			size_t flag = 0;
			for (size_t i=1; i<3; ++i) {
				if (L[i]<minL) {
					minL = L[i];
					minkey = key[i];
					flag = i;
				}
			}
			CalcType delta_b = q*minL;

			printf("[Adding item #%zu] <q=%1.0f> bC=%f, bE=%f, bR=%f. [flag=%zu, minkey=%zu] SSize=%zu\n", CurSize+1, q, bC, bE, bR, flag, minkey, SSize);

			///////////////////////////// Updating coefficients /////////////////////////////

			// Update b
			*b += delta_b;

			// Update hXi - for set E, set R
			for (size_t i=0; i<NSize; ++i) {
				hXi[NMask[i]] += delta_b;
			}

			// Update hXi - for Xc
			hXi[ID] += delta_b;


			///////////////////////////// Moving data items /////////////////////////////

			if (minL<INFINITY) {
				switch (flag) {
					// Xc joins R
					case 0: {
						// Xc joins R and terminate
						Group[ID] = 'R';
						NMask[NSize] = ID;
						(*_NSize)++;
						(*_CurSize)++;
						return 0;
					}
					// Xi moves from E to S
					case 1: {
						size_t k = NMask[minkey];
						// Update Matrix R
						R[0] = -Kernel(&(dataX[k*DataDim]), &(dataX[k*DataDim]), DataDim, sigma_sq);
						R[1] = 1;
						R[RSize] = 1;
						R[RSize+1] = 0;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					// Xi moves from R to S
					case 2: {
						size_t k = NMask[minkey];
						// Update Matrix R
						R[0] = -Kernel(&(dataX[k*DataDim]), &(dataX[k*DataDim]), DataDim, sigma_sq);
						R[1] = 1;
						R[RSize] = 1;
						R[RSize+1] = 0;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(stderr, "[ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag);
						return -1;
					}
				} // end of 'switch(flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(stderr, "[ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		} // end of 'if(SSize==0)'
		else {
			// Calculate beta - intensive, optimisable
			// this beta is the relation between Xc and theta of set S, offset b
			for (size_t i=0; i<SSize+1; ++i) {
				CalcType temp = -R[i*RSize];
				for (size_t j=0; j<SSize; ++j) {
					temp -= R[i*RSize+j+1] * Q[ID*WinSize+SMask[j]];
				}
				beta[i] = temp;
			}

			// Calculate gamma - intensive, optimisable
			// this gamma is the relation between Xc and hXi of set E, set R
			for (size_t i=0; i<NSize; ++i) {
				CalcType temp1 = Q[ID*WinSize+NMask[i]];
				CalcType temp2 = beta[0];
				for (size_t j=0; j<SSize; ++j) {
					temp2 += Q[NMask[i]*WinSize+SMask[j]] * beta[j+1];
				}
				gamma[i] = temp1 + temp2;
			}

			// Calculate gamma_c
			CalcType gamma_c = Q[ID*WinSize+ID] + beta[0];
			for (size_t i=0; i<SSize; ++i) {
				gamma_c += Q[ID*WinSize+SMask[i]] * beta[i+1];
			}

			///////////////////////////// Bookkeeping /////////////////////////////

			// Calculate Lc1 - Case 1
			CalcType Lc1 = INFINITY;
			CalcType qrc = q*gamma_c;
			if (qrc>0 && hXi[ID]<-ep) Lc1 = (-ep-hXi[ID])/gamma_c;
			else if (qrc<0 && hXi[ID]>ep) Lc1 = (ep-hXi[ID])/gamma_c;
			Lc1 = fabs(Lc1);

			// Calculate Lc2 - Case 2
			CalcType Lc2 = fabs(q*C - theta[ID]);

			// Calculate LiS - Case 3
			CalcType LiS = INFINITY;
			size_t keyS = 0;
			for (size_t i=0; i<SSize; ++i) {
				CalcType curLiS = INFINITY;
				CalcType qbi = q*beta[i+1];
				if (qbi>0 && theta[SMask[i]]>=0) curLiS = (C-theta[SMask[i]])/beta[i+1];
				else if (qbi>0 && theta[SMask[i]]<0) curLiS = theta[SMask[i]]/beta[i+1];
				else if (qbi<0 && theta[SMask[i]]>0) curLiS = theta[SMask[i]]/beta[i+1];
				else if (qbi<0 && theta[SMask[i]]<=0) curLiS = (-C-theta[SMask[i]])/beta[i+1];
				curLiS = fabs(curLiS);
				if (curLiS<LiS) {
					LiS = curLiS;
					keyS = i;
				}
			}

			// Calculate LiE - Case 4
			// Calculate LiR - Case 5
			CalcType LiE = INFINITY;
			size_t keyE = 0;
			CalcType LiR = INFINITY;
			size_t keyR = 0;
			for (size_t i=0; i<NSize; ++i) {
				CalcType hX = hXi[NMask[i]];
				CalcType qri = q*gamma[i];
				if (Group[NMask[i]]=='E') {
					CalcType curLiE = INFINITY;
					if (qri>0 && hX<-ep) curLiE = (-hX-ep)/gamma[i];
					else if (qri<0 && hX>ep) curLiE = (-hX+ep)/gamma[i];
					curLiE = fabs(curLiE);
					if (curLiE<LiE) {
						LiE = curLiE;
						keyE = i;
					}
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curLiR = INFINITY;
					if (qri>0) curLiR = (-hX+ep)/gamma[i];
					else if (qri<0) curLiR = (-hX-ep)/gamma[i];
					curLiR = fabs(curLiR);
					if (curLiR<LiR) {
						LiR = curLiR;
						keyR = i;
					}
				}
			}

			// Calculate delta theta_c
			CalcType L[5] = {Lc1, Lc2, LiS, LiE, LiR};
			size_t key[5] = {ID, ID, keyS, keyE, keyR};
			CalcType minL = Lc1;
			size_t minkey = ID;
			size_t flag = 0;
			for (size_t i=1; i<5; ++i) {
				if (L[i]<minL) {
					minL = L[i];
					minkey = key[i];
					flag = i;
				}
			}
			CalcType d_theta_c = q*minL;

			printf("[Adding item #%zu] <q=%1.0f> Lc1=%f, Lc2=%f, LiS=%f, LiE=%f, LiR=%f. [flag=%zu, minkey=%zu] SSize=%zu\n", CurSize+1, q, Lc1, Lc2, LiS, LiE, LiR, flag, minkey, SSize);


			///////////////////////////// Updating coefficients /////////////////////////////

			// Update theta_c
			theta[ID] += d_theta_c;

			// Update b
			*b += beta[0] * d_theta_c;

			// Update theta_S
			for (size_t i=0; i<SSize; ++i) {
				theta[SMask[i]] += beta[i+1] * d_theta_c;
			}

			// Update hXi - for set E, set R
			for (size_t i=0; i<NSize; ++i) {
				hXi[NMask[i]] += gamma[i] * d_theta_c;
			}

			// Update hXi - for Xc
			hXi[ID] += gamma_c * d_theta_c;


			///////////////////////////// Moving data items /////////////////////////////

			if (minL<INFINITY) {
				switch (flag) {
					// Xc joins S
					case 0: {
						// Update Matrix R - enlarge
						for (size_t i=0; i<SSize+1; ++i) {
							for (size_t j=0; j<SSize+1; ++j) {
								R[i*RSize+j] += beta[i] * beta[j] / gamma_c;
							}
							R[i*RSize+SSize+1] = beta[i] / gamma_c;
						}
						for (size_t j=0; j<SSize+1; ++j) {
							R[(SSize+1)*RSize+j] = beta[j] / gamma_c;
						}
						R[(SSize+1)*RSize+SSize+1] = 1.0 / gamma_c;

						// Xc joins S and terminate
						Group[ID] = 'S';
						SMask[SSize] = ID;
						(*_SSize)++;
						(*_CurSize)++;
						return 0;
					}
					// Xc joins E
					case 1: {
						// Xc joins E and terminate
						Group[ID] = 'E';
						NMask[NSize] = ID;
						(*_NSize)++;
						(*_CurSize)++;
						return 0;
					}
					// Xl moves from S to R or E
					case 2: {
						// Update Matrix R - shrink
						if (SSize>1) {
							size_t k = minkey + 1;
							for (size_t i=0; i<SSize+1; ++i) {
								for (size_t j=0; j<SSize+1; ++j) {
									if (i==k || j==k) continue;
									else R[i*RSize+j] -= R[i*RSize+k] * R[k*RSize+j] / R[k*RSize+k];
								}
							}
							for (size_t i=0; i<SSize+1; ++i) {
								for (size_t j=k; j<SSize; ++j) R[i*RSize+j] = R[i*RSize+j+1];
							}
							for (size_t i=k; i<SSize; ++i) {
								for (size_t j=0; j<SSize; ++j) R[i*RSize+j] = R[(i+1)*RSize+j];
							}
						}
						else {
							R[0] = 0;
						}
						// move Xl to R
						size_t k = SMask[minkey];
						if (fabs(theta[k])<eps) {
							Group[k] = 'R';
							NMask[NSize++] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) {
								SMask[i] = SMask[i+1];
							}
							SSize--;
							(*_SSize)--;
							(*_NSize)++;
							theta[k] = 0;  // for numerical stability
						}
						// move Xl to E
						else if (fabs(fabs(theta[k])-C)<eps){
							Group[k] = 'E';
							NMask[NSize++] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) {
								SMask[i] = SMask[i+1];
							}
							SSize--;
							(*_SSize)--;
							(*_NSize)++;
							theta[k] = theta[k]>0 ? C : -C;
						}
						break;
					}
					// Xl joins S
					case 3: {
						size_t k = NMask[minkey];
						// Update beta - intensive, optimisable
						// this beta is the relation between Xl and coeff b and set S
						for (size_t i=0; i<SSize+1; ++i) {
							CalcType temp = -R[i*RSize];
							for (size_t j=0; j<SSize; ++j) {
								temp -= R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
							}
							beta[i] = temp;
						}

						// Calculate gamma_k
						CalcType gamma_k = Q[k*WinSize+k] + beta[0];
						for (size_t i=0; i<SSize; ++i) {
							gamma_k += Q[k*WinSize+SMask[i]] * beta[i+1];
						}

						// Update Matrix R - enlarge
						for (size_t i=0; i<SSize+1; ++i) {
							for (size_t j=0; j<SSize+1; ++j) {
								R[i*RSize+j] += beta[i] * beta[j] / gamma_k;
							}
							R[i*RSize+SSize+1] = beta[i] / gamma_k;
						}
						for (size_t j=0; j<SSize+1; ++j) {
							R[(SSize+1)*RSize+j] = beta[j] / gamma_k;
						}
						R[(SSize+1)*RSize+SSize+1] = 1.0 / gamma_k;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update theta[k]
						CalcType temp = 0;
						for (size_t i=0; i<WinSize; ++i) {
							if (i!=k) temp += theta[i];
						}
						theta[k] = -temp;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					case 4: {
						size_t k = NMask[minkey];
						// Update beta - intensive, optimisable
						// this beta is the relation between Xl and coeff b and set S
						for (size_t i=0; i<SSize+1; ++i) {
							CalcType temp = -R[i*RSize];
							for (size_t j=0; j<SSize; ++j) {
								temp -= R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
							}
							beta[i] = temp;
						}
						// Calculate gamma_k
						CalcType gamma_k = Q[k*WinSize+k] + beta[0];
						for (size_t i=0; i<SSize; i++) {
							gamma_k += Q[k*WinSize+SMask[i]] * beta[i+1];
						}
						// Update Matrix R - enlarge
						for (size_t i=0; i<SSize+1; ++i) {
							for (size_t j=0; j<SSize+1; ++j) {
								R[i*RSize+j] += beta[i] * beta[j] / gamma_k;
							}
							R[i*RSize+SSize+1] = beta[i] / gamma_k;
						}
						for (size_t j=0; j<SSize+1; ++j) {
							R[(SSize+1)*RSize+j] = beta[j] / gamma_k;
						}
						R[(SSize+1)*RSize+SSize+1] = 1.0 / gamma_k;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update theta[k]
						CalcType temp = 0;
						for (size_t i=0; i<WinSize; ++i) {
							if (i!=k) temp += theta[i];
						}
						theta[k] = -temp;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(stderr, "[ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag);
						return -1;
					}
				} // end of 'switch (flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(stderr, "[ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		}
	} // end of 'while(1)'

	return -1;
}


// decremental learning
int decSVM(		size_t ID,
				DataType *dataX,
				CalcType *dataY,
				char *Group,
				size_t *SMask,
				size_t *NMask,
				CalcType *Q,
				CalcType *R,
				CalcType *beta,
				CalcType *gamma,
				CalcType *theta,
				CalcType *b,
				CalcType *hXi,
				size_t *_CurSize,
				size_t *_SSize,
				size_t *_NSize,
				const Param param) {

	///////////////////////////// Read Parameters /////////////////////////////

	const size_t DataSize	= param.DataSize;
	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;
	const CalcType eps		= param.eps;

	size_t SSize = *_SSize;
	size_t NSize = *_NSize;


	///////////////////////////// Remove ID from SMask and NMask /////////////////////////////

	if (Group[ID]=='R'||fabs(theta[ID])<eps) {
		// non-support vector => Xc can be directly removed
		// Update NMask
		size_t key;
		for (size_t i=0; i<NSize; ++i) {
			if (NMask[i]==ID) {key=i; break;}
		}
		for (size_t i=key; i<NSize-1; ++i) {
			NMask[i] = NMask[i+1];
		}
		// Update theta - for extra safety
		theta[ID] = 0;
		// Update Group
		Group[ID] = 'N';
		(*_NSize)--;
		(*_CurSize)--;
		return 0;
	}
	else if (Group[ID]=='E'){
		// Update NMask
		size_t key;
		for (size_t i=0; i<NSize; ++i) {
			if (NMask[i]==ID) {key=i; break;}
		}
		for (size_t i=key; i<NSize-1; ++i) {
			NMask[i] = NMask[i+1];
		}
		Group[ID] = 'C';
		(*_NSize)--;
		NSize--;
	}
	else if (Group[ID]=='S'){
		size_t key;
		for (size_t i=0; i<SSize; ++i) {
			if (SMask[i]==ID) {key=i; break;}
		}
		// Update Matrix R - shrink
		if (SSize>1) {
			size_t k = key + 1;
			for (size_t i=0; i<SSize+1; ++i) {
				for (size_t j=0; j<SSize+1; ++j) {
					if (i==k || j==k) continue;
					else R[i*RSize+j] -= R[i*RSize+k] * R[k*RSize+j] / R[k*RSize+k];
				}
			}
			for (size_t i=0; i<SSize+1; ++i) {
				for (size_t j=k; j<SSize; ++j) R[i*RSize+j] = R[i*RSize+j+1];
			}
			for (size_t i=k; i<SSize; ++i) {
				for (size_t j=0; j<SSize; ++j) R[i*RSize+j] = R[(i+1)*RSize+j];
			}
		}
		else {
			R[0] = 0;
		}
		// Update SMask
		for (size_t i=key; i<SSize-1; ++i) {
			SMask[i] = SMask[i+1];
		}
		Group[ID] = 'C';
		(*_SSize)--;
		SSize--;
	}
	else {
		// ERROR!
		fprintf(stderr, "[ERROR] Group[%zu]='%c' \n", ID, Group[ID]);
		return -1;
	}


	///////////////////////////// Main Loop /////////////////////////////

	while(1) {

		if(SSize==0){

			// Assign q
			CalcType q = theta[ID] > 0 ? 1 : -1;

			// Calculate bE,bR - Case 2&3
			CalcType bE = INFINITY;
			CalcType bR = INFINITY;
			size_t keyE = 0;
			size_t keyR = 0;
			for (size_t i=0; i<NSize; ++i) {
				CalcType hX = hXi[NMask[i]];
				if (Group[NMask[i]]=='E') {
					CalcType curbE = INFINITY;
					if ((q>0 && hX<-ep)||(q<0 && hX>ep)) curbE = fabs(hX + q*ep);
					if (curbE<bE) {
						bE = curbE;
						keyE = i;
					}
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curbR = fabs(-hX + q*ep);
					if (curbR<bR) {
						bR = curbR;
						keyR = i;
					}
				}
			}

			// Calculate delta_b
			CalcType minL = (bE<bR) ? bE : bR;
			CalcType delta_b = q*minL;
			size_t minkey = (bE<bR) ? keyE : keyR;
			size_t flag   = (bE<bR) ? 1 : 2;

			printf("[Removing item #%zu] <q=%1.0f> bE=%f, bR=%f. [flag=%zu, minkey=%zu] SSize=%zu\n", ID+1, q, bE, bR, flag, minkey, SSize);

			///////////////////////////// Updating coefficients /////////////////////////////

			// Update b
			*b += delta_b;

			// Update hXi - for set E, set R
			for (size_t i=0; i<NSize; ++i) {
				hXi[NMask[i]] += delta_b;
			}

			// Update hXi - for Xc
			hXi[ID] += delta_b;


			///////////////////////////// Moving data items /////////////////////////////

			if (minL<INFINITY) {
				switch (flag) {

					// Xi moves from E to S
					case 1: {
						size_t k = NMask[minkey];
						// Update Matrix R
						R[0] = -Kernel(&(dataX[k*DataDim]), &(dataX[k*DataDim]), DataDim, sigma_sq);
						R[1] = 1;
						R[RSize] = 1;
						R[RSize+1] = 0;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					// Xi moves from R to S
					case 2: {
						size_t k = NMask[minkey];
						// Update Matrix R
						R[0] = -Kernel(&(dataX[k*DataDim]), &(dataX[k*DataDim]), DataDim, sigma_sq);
						R[1] = 1;
						R[RSize] = 1;
						R[RSize+1] = 0;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(stderr, "[ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag);
						return -1;
					}
				} // end of 'switch(flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(stderr, "[ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		} // end of 'if(SSize==0)'
		else {
			// Assign q
			CalcType q = theta[ID] > 0 ? -1 : 1;

			// Calculate beta - intensive, optimisable
			// this beta is the relation between Xc and theta of set S, offset b
			for (size_t i=0; i<SSize+1; ++i) {
				CalcType temp = -R[i*RSize];
				for (size_t j=0; j<SSize; ++j) {
					temp -= R[i*RSize+j+1] * Q[ID*WinSize+SMask[j]];
				}
				beta[i] = temp;
			}

			// Calculate gamma - intensive, optimisable
			// this gamma is the relation between Xc and hXi of set E, set R
			for (size_t i=0; i<NSize; ++i) {
				CalcType temp1 = Q[ID*WinSize+NMask[i]];
				CalcType temp2 = beta[0];
				for (size_t j=0; j<SSize; ++j) {
					temp2 += Q[NMask[i]*WinSize+SMask[j]] * beta[j+1];
				}
				gamma[i] = temp1 + temp2;
			}

			// Calculate gamma_c
			CalcType gamma_c = Q[ID*WinSize+ID] + beta[0];
			for (size_t i=0; i<SSize; ++i) {
				gamma_c += Q[ID*WinSize+SMask[i]] * beta[i+1];
			}

			///////////////////////////// Bookkeeping /////////////////////////////

			// Calculate Lc2 - Case 1
			CalcType Lc2 = fabs(theta[ID]);

			// Calculate LiS - Case 2
			CalcType LiS = INFINITY;
			size_t keyS = 0;
			for (size_t i=0; i<SSize; ++i) {
				CalcType curLiS = INFINITY;
				CalcType qbi = q*beta[i+1];
				if (qbi>0 && theta[SMask[i]]>=0) curLiS = (C-theta[SMask[i]])/beta[i+1];
				else if (qbi>0 && theta[SMask[i]]<0) curLiS = theta[SMask[i]]/beta[i+1];
				else if (qbi<0 && theta[SMask[i]]>0) curLiS = theta[SMask[i]]/beta[i+1];
				else if (qbi<0 && theta[SMask[i]]<=0) curLiS = (-C-theta[SMask[i]])/beta[i+1];
				curLiS = fabs(curLiS);
				if (curLiS<LiS) {
					LiS = curLiS;
					keyS = i;
				}
			}

			// Calculate LiE - Case 3
			// Calculate LiR - Case 4
			CalcType LiE = INFINITY;
			size_t keyE = 0;
			CalcType LiR = INFINITY;
			size_t keyR = 0;
			for (size_t i=0; i<NSize; ++i) {
				CalcType hX = hXi[NMask[i]];
				CalcType qri = q*gamma[i];
				if (Group[NMask[i]]=='E') {
					CalcType curLiE = INFINITY;
					if (qri>0 && hX<-ep) curLiE = (-hX-ep)/gamma[i];
					else if (qri<0 && hX>ep) curLiE = (-hX+ep)/gamma[i];
					curLiE = fabs(curLiE);
					if (curLiE<LiE) {
						LiE = curLiE;
						keyE = i;
					}
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curLiR = INFINITY;
					if (qri>0) curLiR = (-hX+ep)/gamma[i];
					else if (qri<0) curLiR = (-hX-ep)/gamma[i];
					curLiR = fabs(curLiR);
					if (curLiR<LiR) {
						LiR = curLiR;
						keyR = i;
					}
				}
			}

			// Calculate delta theta_c
			CalcType L[4] = {Lc2, LiS, LiE, LiR};
			size_t key[4] = {ID, keyS, keyE, keyR};
			CalcType minL = Lc2;
			size_t minkey = ID;
			size_t flag = 0;
			for (size_t i=1; i<4; ++i) {
				if (L[i]<minL) {
					minL = L[i];
					minkey = key[i];
					flag = i;
				}
			}
			CalcType d_theta_c = q*minL;

			printf("[Removing item #%zu] <q=%1.0f> Lc2=%f, LiS=%f, LiE=%f, LiR=%f. [flag=%zu, minkey=%zu] SSize=%zu\n", ID+1, q, Lc2, LiS, LiE, LiR, flag, minkey, SSize);


			///////////////////////////// Updating coefficients /////////////////////////////

			// Update theta_c
			theta[ID] += d_theta_c;

			// Update b
			*b += beta[0] * d_theta_c;

			// Update theta_S
			for (size_t i=0; i<SSize; ++i) {
				theta[SMask[i]] += beta[i+1] * d_theta_c;
			}

			// Update hXi - for set E, set R
			for (size_t i=0; i<NSize; ++i) {
				hXi[NMask[i]] += gamma[i] * d_theta_c;
			}

			// Update hXi - for Xc
			hXi[ID] += gamma_c * d_theta_c;


			///////////////////////////// Moving data items /////////////////////////////

			if (minL<INFINITY) {
				switch (flag) {
					// Xc joins R and is removed
					case 0: {
						Group[ID] = 'N';
						(*_CurSize)--;
						return 0;
					}
					// Xl moves from S to R or E
					case 1: {
						// Update Matrix R - shrink
						if (SSize>1) {
							size_t k = minkey + 1;
							for (size_t i=0; i<SSize+1; ++i) {
								for (size_t j=0; j<SSize+1; ++j) {
									if (i==k || j==k) continue;
									else R[i*RSize+j] -= R[i*RSize+k] * R[k*RSize+j] / R[k*RSize+k];
								}
							}
							for (size_t i=0; i<SSize+1; ++i) {
								for (size_t j=k; j<SSize; ++j) R[i*RSize+j] = R[i*RSize+j+1];
							}
							for (size_t i=k; i<SSize; ++i) {
								for (size_t j=0; j<SSize; ++j) R[i*RSize+j] = R[(i+1)*RSize+j];
							}
						}
						else {
							R[0] = 0;
						}
						// move Xl to R
						size_t k = SMask[minkey];
						if (fabs(theta[k])<eps) {
							Group[k] = 'R';
							NMask[NSize++] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) {
								SMask[i] = SMask[i+1];
							}
							SSize--;
							(*_SSize)--;
							(*_NSize)++;
							theta[k] = 0;  // for numerical stability
						}
						// move Xl to E
						else if (fabs(fabs(theta[k])-C)<eps){
							Group[k] = 'E';
							NMask[NSize++] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) {
								SMask[i] = SMask[i+1];
							}
							SSize--;
							(*_SSize)--;
							(*_NSize)++;
							theta[k] = theta[k]>0 ? C : -C;
						}
						break;
					}
					// Xl joins S
					case 2: {
						size_t k = NMask[minkey];
						// Update beta - intensive, optimisable
						// this beta is the relation between Xl and coeff b and set S
						for (size_t i=0; i<SSize+1; ++i) {
							CalcType temp = -R[i*RSize];
							for (size_t j=0; j<SSize; ++j) {
								temp -= R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
							}
							beta[i] = temp;
						}

						// Calculate gamma_k
						CalcType gamma_k = Q[k*WinSize+k] + beta[0];
						for (size_t i=0; i<SSize; ++i) {
							gamma_k += Q[k*WinSize+SMask[i]] * beta[i+1];
						}

						// Update Matrix R - enlarge
						for (size_t i=0; i<SSize+1; ++i) {
							for (size_t j=0; j<SSize+1; ++j) {
								R[i*RSize+j] += beta[i] * beta[j] / gamma_k;
							}
							R[i*RSize+SSize+1] = beta[i] / gamma_k;
						}
						for (size_t j=0; j<SSize+1; ++j) {
							R[(SSize+1)*RSize+j] = beta[j] / gamma_k;
						}
						R[(SSize+1)*RSize+SSize+1] = 1.0 / gamma_k;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update theta[k]
						CalcType temp = 0;
						for (size_t i=0; i<WinSize; ++i) {
							if (i!=k) temp += theta[i];
						}
						theta[k] = -temp;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					case 3: {
						size_t k = NMask[minkey];
						// Update beta - intensive, optimisable
						// this beta is the relation between Xl and coeff b and set S
						for (size_t i=0; i<SSize+1; ++i) {
							CalcType temp = -R[i*RSize];
							for (size_t j=0; j<SSize; ++j) {
								temp -= R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
							}
							beta[i] = temp;
						}
						// Calculate gamma_k
						CalcType gamma_k = Q[k*WinSize+k] + beta[0];
						for (size_t i=0; i<SSize; i++) {
							gamma_k += Q[k*WinSize+SMask[i]] * beta[i+1];
						}
						// Update Matrix R - enlarge
						for (size_t i=0; i<SSize+1; ++i) {
							for (size_t j=0; j<SSize+1; ++j) {
								R[i*RSize+j] += beta[i] * beta[j] / gamma_k;
							}
							R[i*RSize+SSize+1] = beta[i] / gamma_k;
						}
						for (size_t j=0; j<SSize+1; ++j) {
							R[(SSize+1)*RSize+j] = beta[j] / gamma_k;
						}
						R[(SSize+1)*RSize+SSize+1] = 1.0 / gamma_k;

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize++] = k;

						// Update theta[k]
						CalcType temp = 0;
						for (size_t i=0; i<WinSize; ++i) {
							if (i!=k) temp += theta[i];
						}
						theta[k] = -temp;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(stderr, "[ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag);
						return -1;
					}
				} // end of 'switch (flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(stderr, "[ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		}
	} // end of 'while(1)'

	return -1;
}


/*
int DemoKernel() {

	const int inSize = 384;

	int *a = malloc(sizeof(int) * inSize);
	int *b = malloc(sizeof(int) * inSize);
	int *expected = malloc(sizeof(int) * inSize);
	int *out = malloc(sizeof(int) * inSize);

	memset(out, 0, sizeof(int) * inSize);

	for(int i = 0; i < inSize; ++i) {
		a[i] = i + 1;
		b[i] = i - 1;
		expected[i] = 2 * i;
	}

	printf("Running on DFE.\n");
	Demo(inSize, a, b, out);

	for (int i = 0; i < inSize; i++)
		if (out[i] != expected[i]) {
			fprintf(stderr, "Output from DFE did not match CPU: %d : %d != %d\n", i, out[i], expected[i]);
			return 1;
		}

	free(a); free(b);
	free(expected); free(out);

	printf("Test passed!\n");
	return 0;
}
*/

int SimpleDataSet(const Param param){

	/////////////////////////// SVM Parameters ///////////////////////////

	const size_t DataSize = param.DataSize;
	const size_t DataDim = param.DataDim;
	const size_t WinSize = param.WinSize;
	const size_t RSize = param.RSize;
	const CalcType ep = param.ep;
	const CalcType C  = param.C;
	const CalcType sigma_sq = param.sigma_sq;


	/////////////////////////// Read data file ///////////////////////////

	// Open input file
	char *inFile = param.InFile;
	FILE *infp = fopen(inFile, "r");
	if (infp==NULL) {
		fprintf(stderr, "[ERROR] Cannot open input file [%s]. \n", inFile);
  		exit(1);
	}

	// Allocating Memory for input data
	DataType *X_IN = calloc(DataSize * DataDim, sizeof(DataType));
	CalcType *Y_IN = calloc(DataSize, sizeof(CalcType));

	// read file into memory
	// We use LibSVM data format
	// Assuming both X and Y are double
	size_t ActualDataSize = 0;
	while(!feof(infp) && ActualDataSize<DataSize) {
		fscanf(infp, "%lf", &Y_IN[ActualDataSize]);
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%*d:%lf", &X_IN[ActualDataSize*DataDim+j]);
		}
		++ActualDataSize;
	}
	fclose(infp);

	// At least we need 2 data points to initialise SVM
	if (ActualDataSize<2) {
		free(X_IN); free(Y_IN);
		fprintf(stderr, "[ERROR] At least we need 2 data points, but there are only %zu. \n", ActualDataSize);
  		exit(1);
	}

	/////////////////////////// Allocating Memory for SVM ///////////////////////////

	// Current Number of Elements
	size_t CurSize = 0;

	// Record of Input Data (X, Y)
	DataType *dataX = calloc(WinSize * DataDim, sizeof(DataType));
	CalcType *dataY = calloc(WinSize, sizeof(CalcType));

	// Prediction of Y from SVM
	CalcType *Ypredict = calloc(DataSize, sizeof(CalcType));

	// Group: 'S', 'E', 'R', 'C', 'N'
	char *Group = malloc(sizeof(char) * WinSize);
	for (size_t i=0; i<WinSize; ++i) Group[i] = 'N';

	// Position of each S vector
	size_t SSize = 0;
	size_t *SMask = calloc(WinSize, sizeof(size_t));

	// Position of each E or R vector
	size_t NSize = 0;
	size_t *NMask = calloc(WinSize, sizeof(size_t));

	// Matrix Q
	CalcType *Q = calloc(WinSize * WinSize, sizeof(CalcType));

	// Matrix R : should be (1+S)-by-(1+S)
	CalcType *R = calloc(RSize * RSize, sizeof(CalcType));

	// Coeff beta
	CalcType *beta = calloc(WinSize, sizeof(CalcType));

	// Coeff gamma
	CalcType *gamma = calloc(WinSize, sizeof(CalcType));

	// Coeff theta
	CalcType *theta = calloc(WinSize, sizeof(CalcType));

	// hXi
	CalcType *hXi = calloc(WinSize, sizeof(CalcType));

	// Coeff b
	CalcType b = 0;


	/////////////////////////// Initialise SVM ///////////////////////////

	// initialise SVM using 2 data points
	initSVM(&X_IN[0*DataDim], Y_IN[0], &X_IN[1*DataDim], Y_IN[1], dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, param);

	// Calculate Objective Function Value
	CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
	printf("[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);

	/////////////////////////// Incremental Training ///////////////////////////

	DataType Xc[DataDim];
	CalcType Yc;

	for (size_t i=2; i<ActualDataSize; ++i) {

		// read current data point
		for (size_t j=0; j<DataDim; ++j) {
			Xc[j] = X_IN[i*DataDim+j];
		}
		Yc = Y_IN[i];

		// calculate SVM prediction of Yc
		Ypredict[i] = regSVM(Xc, dataX, theta, &b, WinSize, DataDim, sigma_sq);

		// Assign the place for Xc
		size_t ID = i;

		// train SVM incrementally
		int isTrainingSuccessful = incSVM(ID, Xc, Yc, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, param);

		if (isTrainingSuccessful!=0) {
			free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
			free(Group); free(SMask); free(NMask);
			free(Q); free(R);
			free(beta); free(gamma); free(theta);
			free(hXi);
			fprintf(stderr, "[ERROR] Incremental SVM training failed at data[%zu]. \n", i);
		}

		// Calculate Objective Function Value
		CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
		printf("[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");

	}

	/////////////////////////// Decremental Training ///////////////////////////

	for (size_t ID=9; ID>1; ID--) {

		// Decremental Training
		int isTrainingSuccessful = decSVM(ID, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, param);

		// Calculate Objective Function Value
		CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
		printf("[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");
	}



	/////////////////////////// Checking Results ///////////////////////////

	// Open output file
	char *outFile = param.OutFile;
	FILE *outfp = fopen(outFile, "w");
	if (outfp==NULL) {
		fprintf(stderr, "[ERROR] Cannot open output file [%s]. \n", outFile);
  		exit(1);
	}

	// write results
	for (size_t i=2; i<ActualDataSize; ++i) {
		fprintf(outfp, "%f %f\n", Y_IN[i], Ypredict[i]);
	}
	fclose(outfp);


	/////////////////////////// Clean up ///////////////////////////

	free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
	free(Group); free(SMask); free(NMask);
	free(Q); free(R);
	free(beta); free(gamma); free(theta);
	free(hXi);

	return 0;
}


int main(){

	///////////// Simple Data Set /////////////

	Param ParamSimple;
	ParamSimple.InFile 		= "SimpleData.txt";
	ParamSimple.OutFile 	= "result.txt";
	ParamSimple.DataSize 	= 10;
	ParamSimple.DataDim 	= 1;
	ParamSimple.WinSize 	= 10;
	ParamSimple.RSize 		= 5;
	ParamSimple.ep 			= 0.01;
	ParamSimple.C 			= 1000;
	ParamSimple.sigma_sq 	= 50;
	ParamSimple.eps 		= 1e-6;

	SimpleDataSet(ParamSimple);

	// Demo Kernel
//	int Result = DemoKernel();

	return 0;
}
