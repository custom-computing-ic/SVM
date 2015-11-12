#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


////////////////////// Definitions //////////////////////

// Maximum input data size allowed
#define DataSize 	10

// dimension of input data X
#define DataDim 	1

// dimension of matrix R
#define RSize 		5

// precision
#define eps		1e-6

// type of input data X
typedef double DataType;

// type of input data Y and internal calculations
typedef double CalcType;


////////////////////// Utility Functions //////////////////////

// RBF Kernel
static inline CalcType Kernel(DataType *X1, DataType*X2, const CalcType sigma_sq) {
	CalcType sum = 0;
	for (size_t i=0; i<DataDim; ++i) sum -= (X1[i] - X2[i]) * (X1[i] - X2[i]);
	return exp(sum/2/sigma_sq);
}


// Calculating h(Xi) - Assuming Q is full matrix
CalcType hCalc(	const size_t ID,
				const size_t CurSize,
				char *Group,
				DataType *dataY,
				CalcType *Q,
				CalcType *theta,
				CalcType *b) {

	// Compute f(Xi)
	CalcType fXi = *b;
	for (size_t i=0; i<CurSize; ++i) {
		fXi += theta[i] * Q[ID*DataSize+i];
	}

	// Compute h(Xi) = f(Xi) - Yi
	return fXi - dataY[ID];
}

CalcType objCalc(	const size_t CurSize,
					CalcType *dataY,
					char *Group,
					CalcType *theta,
					CalcType *Q,
					CalcType *b,
					const CalcType C,
					const CalcType ep){

	// Calculate the first term
	CalcType temp1 = 0;
	for (size_t i=0; i<CurSize; ++i) {
		// Calculate the first term
		for (size_t j=0; j<CurSize; ++j) {
			temp1 += theta[i] * theta[j] * Q[i*DataSize+j] / 2;
		}
	}

	// Calculate the second term
	CalcType temp2 = 0;
	for (size_t i=0; i<CurSize; ++i) {
		CalcType hXi = hCalc(i, CurSize, Group, dataY, Q, theta, b);
		temp2 += (fabs(hXi)>ep) ? C*(fabs(hXi)-ep) : 0;
	}

	return temp1+temp2;
}


////////////////////// SVM Functions //////////////////////

// SVM regression
CalcType regSVM(	DataType *X_IN,
				DataType *dataX,
				const size_t CurSize,
				char *Group,
				CalcType *theta,
				CalcType *b,
				const CalcType sigma_sq) {

	CalcType f = *b;
	for (size_t i=0; i<CurSize; ++i) {
		//if (Group[i]=='E' || Group[i]=='S') f += theta[i] * Kernel(X_IN, &(dataX[i*DataDim]), sigma_sq);
		f += theta[i] * Kernel(X_IN, &(dataX[i*DataDim]), sigma_sq);
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
				const CalcType ep,
				const CalcType C,
				const CalcType sigma_sq) {


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
	Q[0*DataSize+0] = Kernel(&(dataX[0*DataDim]), &(dataX[0*DataDim]), sigma_sq);
	Q[0*DataSize+1] = Kernel(&(dataX[0*DataDim]), &(dataX[1*DataDim]), sigma_sq);
	Q[1*DataSize+1] = Kernel(&(dataX[1*DataDim]), &(dataX[1*DataDim]), sigma_sq);
	Q[1*DataSize+0] = Q[0*DataSize+1];

	// Update theta
	CalcType temp = (dataY[0]-dataY[1]-2*ep)/2/(Q[0*DataSize+0]-Q[0*DataSize+1]);
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
		hXi[0] = hCalc(0, 2, Group, dataY, Q, theta, b);
		hXi[1] = hCalc(1, 2, Group, dataY, Q, theta, b);
		*_NSize = 2;
		*_CurSize = 2;
	}
	// theta[0]=theta[1]=0, both join R
	else if (theta[0]==0) {
		Group[0] = 'R';
		Group[1] = 'R';
		NMask[0] = 0;
		NMask[1] = 1;
		hXi[0] = hCalc(0, 2, Group, dataY, Q, theta, b);
		hXi[1] = hCalc(1, 2, Group, dataY, Q, theta, b);
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
		CalcType Q00 = Q[0*DataSize+0];
		CalcType Q01 = Q[0*DataSize+1];
		CalcType Q11 = Q[1*DataSize+1];
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
int incSVM(		DataType *Xc,
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
				const CalcType ep,
				const CalcType C,
				const CalcType sigma_sq) {

	///////////////////////////// Reading params /////////////////////////////

	size_t CurSize = *_CurSize;
	size_t SSize = *_SSize;
	size_t NSize = *_NSize;


	///////////////////////////// Adding (Xc, Yc) to data set /////////////////////////////

	for (size_t i=0; i<DataDim; ++i) {
		dataX[CurSize*DataDim+i] = Xc[i];
	}
	dataY[CurSize] = Yc;


	///////////////////////////// Check (Xc, Yc) /////////////////////////////

	// Initialise theta_c
	theta[CurSize] = 0;

	// Update Q - optimisable
	for (size_t i=0; i<CurSize; ++i) Q[CurSize*DataSize+i] = Kernel(Xc, &(dataX[i*DataDim]), sigma_sq);
	for (size_t i=0; i<CurSize; ++i) Q[i*DataSize+CurSize] = Q[CurSize*DataSize+i];
	Q[CurSize*DataSize+CurSize] = Kernel(Xc, Xc, sigma_sq);

	// Compute h(Xc)
	hXi[CurSize] = hCalc(CurSize, CurSize, Group, dataY, Q, theta, b);

	// non-support vector => Xc joins 'R' and terminate, without changing b
	if (fabs(hXi[CurSize])<=ep) {
		Group[CurSize] = 'R';
		NMask[NSize] = CurSize;
		(*_NSize)++;
		(*_CurSize)++;
		return 0;
	}

	// non-support vector => Xc joins 'R' after changing b
	// This could only happen when set S is empty
	if (SSize==0) {
		//CalcType delta_b = (hXi[CurSize]>ep)? ep - hXi[CurSize] : -ep - hXi[CurSize];
		CalcType delta_b = - hXi[CurSize];
		int isValid = 1;
		for (size_t i=0; i<NSize; ++i) {
			size_t cur = NMask[i];
			if (Group[cur]=='R' && fabs(hXi[cur]+delta_b)>ep) {isValid = 0; break;}
			else if (Group[cur]=='E' && theta[cur]>0 && (hXi[cur]+delta_b)>-ep) {isValid = 0; break;}
			else if (Group[cur]=='E' && theta[cur]<0 && (hXi[cur]+delta_b)<ep) {isValid = 0; break;}
		}
		if (isValid==1) {
			// Update b
			*b += delta_b;
			// Update Group
			Group[CurSize] = 'R';
			NMask[NSize] = CurSize;
			(*_NSize)++;
			(*_CurSize)++;
			// Update hXi
			for (size_t i=0; i<NSize+1; ++i) hXi[NMask[i]] += delta_b;
			// Return
			return 0;
		}
	}


	///////////////////////////// Prepare for bookkeeping /////////////////////////////

	// Label the new comer as Xc
	Group[CurSize] = 'C';

	// Assign q
	CalcType q = hXi[CurSize] > 0 ? -1 : 1;


	///////////////////////////// Main Loop /////////////////////////////

	while(1) {

	// Calculate beta - intensive, optimisable
	// this beta is the relation between Xc and theta of set S, offset b
	for (size_t i=0; i<SSize+1; ++i) {
		CalcType temp = -R[i*RSize];
		for (size_t j=0; j<SSize; ++j) {
			temp -= R[i*RSize+j+1] * Q[CurSize*DataSize+SMask[j]];
		}
		beta[i] = temp;
	}

	// Calculate gamma - intensive, optimisable
	// this gamma is the relation between Xc and hXi of set E, set R
	for (size_t i=0; i<NSize; ++i) {
		CalcType temp1 = Q[CurSize*DataSize+NMask[i]];
		CalcType temp2 = beta[0];
		for (size_t j=0; j<SSize; ++j) {
			temp2 += Q[NMask[i]*DataSize+SMask[j]] * beta[j+1];
		}
		gamma[i] = temp1 + temp2;
	}

	// Calculate gamma_c
	CalcType gamma_c = Q[CurSize*DataSize+CurSize] + beta[0];
	for (size_t i=0; i<SSize; ++i) {
		gamma_c += Q[CurSize*DataSize+SMask[i]] * beta[i+1];
	}

	///////////////////////////// Bookkeeping /////////////////////////////

	// Calculate Lc1 - Case 1
	CalcType Lc1 = INFINITY;
	CalcType qrc = q*gamma_c;
	if (qrc>0 && hXi[CurSize]<-ep) Lc1 = (-ep-hXi[CurSize])/gamma_c;
	else if (qrc<0 && hXi[CurSize]>ep) Lc1 = (ep-hXi[CurSize])/gamma_c;
	Lc1 = fabs(Lc1);

	// Calculate Lc2 - Case 2
	CalcType Lc2 = fabs(q*C - theta[CurSize]);

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
	size_t key[5] = {CurSize, CurSize, keyS, keyE, keyR};
	CalcType minL = Lc1;
	size_t minkey = CurSize;
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
	theta[CurSize] += d_theta_c;

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
	hXi[CurSize] += gamma_c * d_theta_c;


	///////////////////////////// Moving data items /////////////////////////////

	if (minL<INFINITY) {
		switch (flag) {
			// Xc joins S
			case 0: {
				// Update Matrix R - enlarge
				if (SSize!=0) {
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
				}
				else {
					R[0] = -Kernel(Xc, Xc, sigma_sq);
					R[1] = 1;
					R[RSize] = 1;
					R[RSize+1] = 0;
				}
				// Xc joins S and terminate
				Group[CurSize] = 'S';
				SMask[SSize] = CurSize;
				(*_SSize)++;
				(*_CurSize)++;
				return 0;
			}
			// Xc joins E
			case 1: {
				// Xc joins E and terminate
				Group[CurSize] = 'E';
				NMask[NSize] = CurSize;
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
					hXi[k] = hCalc(k, CurSize+1, Group, dataY, Q, theta, b);
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
					hXi[k] = hCalc(k, CurSize+1, Group, dataY, Q, theta, b);
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
						temp -= R[i*RSize+j+1] * Q[k*DataSize+SMask[j]];
					}
					beta[i] = temp;
				}
				// Calculate gamma_k
				CalcType gamma_k = Q[k*DataSize+k] + beta[0];
				for (size_t i=0; i<SSize; ++i) {
					gamma_k += Q[k*DataSize+SMask[i]] * beta[i+1];
				}
				// Update Matrix R - enlarge
				if (SSize!=0) {
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
				}
				else {
					R[0] = -1;
					R[1] = 1;
					R[RSize] = 1;
					R[RSize+1] = 0;
				}
				// move Xl to S
				Group[k] = 'S';
				SMask[SSize++] = k;
				// TODO: Update theta[k]
				CalcType temp = 0;
				for (size_t i=0; i<=CurSize; ++i) {
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
						temp -= R[i*RSize+j+1] * Q[k*DataSize+SMask[j]];
					}
					beta[i] = temp;
				}
				// Calculate gamma_k
				CalcType gamma_k = Q[k*DataSize+k] + beta[0];
				for (size_t i=0; i<SSize; i++) {
					gamma_k += Q[k*DataSize+SMask[i]] * beta[i+1];
				}
				// Update Matrix R - enlarge
				if (SSize!=0) {
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
				}
				else {
					R[0] = -1;
					R[1] = 1;
					R[RSize] = 1;
					R[RSize+1] = 0;
				}
				// move Xl to S
				Group[k] = 'S';
				SMask[SSize++] = k;
				// TODO: Update theta[k]
				CalcType temp = 0;
				for (size_t i=0; i<=CurSize; ++i) {
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
		}

	}
	else {
		// ERROR!
		fprintf(stderr, "[ERROR] unable to make any move because minL=%f. \n", minL);
		return -1;
	}

	}

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

int main(){

	/////////////////////////// SVM Parameters ///////////////////////////

	const CalcType ep = 0.01;
	const CalcType C = 1000;
	const CalcType sigma_sq = 50;


	/////////////////////////// Read data file ///////////////////////////

	// Open input file
	char *inFile = "data.txt";
	FILE *infp = fopen(inFile, "r");
	if (infp==NULL) {
		fprintf(stderr, "[ERROR] Cannot open input file [%s]. \n", inFile);
  		exit(1);
	}

	// Allocating Memory for input data
	DataType *X_IN = calloc(DataSize * DataDim, sizeof(DataType));
	CalcType *Y_IN = calloc(DataSize, sizeof(CalcType));

	// read file into memory
	size_t ActualDataSize = 0;
	while(!feof(infp) && ActualDataSize<DataSize) {
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%lf", &X_IN[ActualDataSize*DataDim+j]);
		}
		fscanf(infp, "%lf", &Y_IN[ActualDataSize]);
		++ActualDataSize;
	}
	fclose(infp);

	// Atleast we need 2 data points to initialise SVM
	if (ActualDataSize<2) {
		free(X_IN); free(Y_IN);
		fprintf(stderr, "[ERROR] Atleast we need 2 data points, but there are only %zu. \n", ActualDataSize);
  		exit(1);
	}

	/////////////////////////// Allocating Memory for SVM ///////////////////////////

	// Current Number of Elements
	size_t CurSize = 0;

	// Record of Input Data (X, Y)
	DataType *dataX = calloc(DataSize * DataDim, sizeof(DataType));
	CalcType *dataY = calloc(DataSize, sizeof(CalcType));

	// Prediction of Y from SVM
	CalcType *Ypredict = calloc(DataSize, sizeof(CalcType));

	// Group: 'S', 'E', 'R', 'C', 'N'
	char *Group = malloc(sizeof(char) * DataSize);
	for (size_t i=0; i<DataSize; ++i) Group[i] = 'N';

	// Position of each S vector
	size_t SSize = 0;
	size_t *SMask = calloc(DataSize, sizeof(size_t));

	// Position of each E or R vector
	size_t NSize = 0;
	size_t *NMask = calloc(DataSize, sizeof(size_t));

	// Matrix Q
	CalcType *Q = calloc(DataSize * DataSize, sizeof(CalcType));

	// Matrix R : should be (1+S)-by-(1+S)
	CalcType *R = calloc(RSize * RSize, sizeof(CalcType));

	// Coeff beta
	CalcType *beta = calloc(DataSize, sizeof(CalcType));

	// Coeff gamma
	CalcType *gamma = calloc(DataSize, sizeof(CalcType));

	// Coeff theta
	CalcType *theta = calloc(DataSize, sizeof(CalcType));

	// hXi
	CalcType *hXi = calloc(DataSize, sizeof(CalcType));

	// Coeff b
	CalcType b = 0;


	/////////////////////////// Initialise SVM ///////////////////////////

	// initialise SVM using 2 data points
	initSVM(&X_IN[0*DataDim], Y_IN[0], &X_IN[1*DataDim], Y_IN[1], dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, ep, C, sigma_sq);

	// Calculate Objective Function Value
	CalcType init_obj = objCalc(CurSize, dataY, Group, theta, Q, &b, C, ep);
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
		Ypredict[i] = regSVM(Xc, dataX, CurSize, Group, theta, &b, sigma_sq);

		// train SVM incrementally
		int isTrainingSuccessful = incSVM(Xc, Yc, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, ep, C, sigma_sq);

		if (isTrainingSuccessful!=0) {
			free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
			free(Group); free(SMask); free(NMask);
			free(Q); free(R);
			free(beta); free(gamma); free(theta);
			free(hXi);
			fprintf(stderr, "[ERROR] Incremental SVM training failed at data[%zu]. \n", i);
		}

		// Calculate Objective Function Value
		CalcType init_obj = objCalc(CurSize, dataY, Group, theta, Q, &b, C, ep);
		printf("[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");

	}

	/////////////////////////// Checking Results ///////////////////////////

	// Open output file
	char *outFile = "result.txt";
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


	// Demo Kernel
//	int Result = DemoKernel();

	return 0;
}
