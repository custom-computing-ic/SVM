#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


////////////////////// Definitions //////////////////////

// Output Statistics or not
#define EN_STAT 1
// Use double or float
#define EN_DOUBLE 1
// Generate log or not
#define EN_LOG 1

// Types
#ifdef EN_DOUBLE
	typedef double DataType;
	typedef double CalcType;
#else
	typedef float DataType;
	typedef float CalcType;
#endif

// SVR Parameters
typedef struct {
	char * InFile;				// Input File Name
	char * OutFile;				// Output File Name
	char * LogFile;				// Log File Name
	FILE * LogFileHandle; 		// Log File Pointer
	size_t DataSize; 			// SVR: Number of samples
	size_t DataDim; 			// SVR: Number of features
	size_t WinSize;				// SVR: Window Size
	size_t RSize; 				// SVR: Size of R (RSize<=WinSize+1)
	CalcType ep; 				// SVR: Epsilon
	CalcType C; 				// SVR: C
	CalcType sigma_sq; 			// SVR: sigma^2 (RBF Kernel)
	CalcType eps; 				// SVR: eps (detect ties)
} Param;

#ifdef EN_STAT
	CalcType MaxKernelProduct = 0;
	CalcType MaxKernelSum = 0;
	CalcType MaxhCalcProduct = 0;
	CalcType MaxhCalcSum = 0;
	CalcType MaxbetaProduct = 0;
	CalcType MaxbetaSum = 0;
	CalcType MaxgammaProduct = 0;
	CalcType MaxgammaSum = 0;
	CalcType MaxR = 0;
	CalcType MaxREgammak = 0;
	CalcType MaxREgammaInv = 0;
	CalcType MaxRECalc = 0;
	CalcType MaxRSKelem = 0;
	CalcType MaxRSKelemInv = 0;
	CalcType MaxRSCalc = 0;
	CalcType MaxminL = 0;
	int 	 isTie = -1;

static inline CalcType max(CalcType record, CalcType cur) {
	CalcType result = (record<fabs(cur)) ? fabs(cur) : record;
	return result;
}

static inline void stat_report() {
	FILE *fp = fopen("StatReport.txt", "w");
	fprintf(fp, "[Statistics] MaxKernelProduct=%f => %d digits.\n", MaxKernelProduct, (int)ceil(log2(MaxKernelProduct)));
	fprintf(fp, "[Statistics] MaxKernelSum=%f => %d digits.\n", MaxKernelSum, (int)ceil(log2(MaxKernelSum)));
	fprintf(fp, "[Statistics] MaxhCalcProduct=%f => %d digits.\n", MaxhCalcProduct, (int)ceil(log2(MaxhCalcProduct)));
	fprintf(fp, "[Statistics] MaxhCalcSum=%f => %d digits.\n", MaxhCalcSum, (int)ceil(log2(MaxhCalcSum)));
	fprintf(fp, "[Statistics] MaxbetaProduct=%f => %d digits.\n", MaxbetaProduct, (int)ceil(log2(MaxbetaProduct)));
	fprintf(fp, "[Statistics] MaxbetaSum=%f => %d digits.\n", MaxbetaSum, (int)ceil(log2(MaxbetaSum)));
	fprintf(fp, "[Statistics] MaxgammaProduct=%f => %d digits.\n", MaxgammaProduct, (int)ceil(log2(MaxgammaProduct)));
	fprintf(fp, "[Statistics] MaxgammaSum=%f => %d digits.\n", MaxgammaSum, (int)ceil(log2(MaxgammaSum)));
	fprintf(fp, "[Statistics] MaxR=%f => %d digits.\n", MaxR, (int)ceil(log2(MaxR)));
	fprintf(fp, "[Statistics] MaxREgammak=%f => %d digits.\n", MaxREgammak, (int)ceil(log2(MaxREgammak)));
	fprintf(fp, "[Statistics] MaxREgammaInv=%f => %d digits.\n", MaxREgammaInv, (int)ceil(log2(MaxREgammaInv)));
	fprintf(fp, "[Statistics] MaxRECalc=%f => %d digits.\n", MaxRECalc, (int)ceil(log2(MaxRECalc)));
	fprintf(fp, "[Statistics] MaxRSKelem=%f => %d digits.\n", MaxRSKelem, (int)ceil(log2(MaxRSKelem)));
	fprintf(fp, "[Statistics] MaxRSKelemInv=%f => %d digits.\n", MaxRSKelemInv, (int)ceil(log2(MaxRSKelemInv)));
	fprintf(fp, "[Statistics] MaxRSCalc=%f => %d digits.\n", MaxRSCalc, (int)ceil(log2(MaxRSCalc)));
	fprintf(fp, "[Statistics] MaxminL=%f => %d digits.\n", MaxminL, (int)ceil(log2(MaxminL)));
	fprintf(fp, "[Statistics] isTie=%d.\n", isTie);
	fclose(fp);
}

#endif


////////////////////// Utility Functions //////////////////////

// RBF Kernel
static inline CalcType Kernel(DataType *X1, DataType*X2, const size_t DataDim, const CalcType sigma_sq) {
	CalcType sum = 0;
#ifdef EN_STAT
	for (size_t i=0; i<DataDim; ++i) {
		CalcType temp = (X1[i] - X2[i]) * (X1[i] - X2[i]);
		MaxKernelProduct = max(MaxKernelProduct, temp);
		sum += temp;
	}
	MaxKernelSum = max(MaxKernelSum, sum);
#else
	for (size_t i=0; i<DataDim; ++i) sum += (X1[i]-X2[i])*(X1[i]-X2[i]);
#endif
#ifdef EN_DOUBLE
	return exp(sum*(-0.5/sigma_sq));
#else
	return expf(sum*(-0.5/sigma_sq));
#endif
}

// Calculating h(Xi)
CalcType hCalc(const size_t ID, DataType *dataY, CalcType *Q, CalcType *theta, CalcType *b, const size_t WinSize) {
#ifdef EN_STAT
	CalcType offset = *b - dataY[ID];
	MaxhCalcSum = max(MaxhCalcSum, offset);
	CalcType sum = 0;
	for (size_t i=0; i<WinSize; ++i) {
		CalcType temp = theta[i] * Q[ID*WinSize+i];
		MaxhCalcProduct = max(MaxhCalcProduct, temp);
		sum += temp;
		MaxhCalcSum = max(MaxhCalcSum, sum);
	}
	sum += offset;
	MaxhCalcSum = max(MaxhCalcSum, sum);
	return sum;
#else
	CalcType offset = *b - dataY[ID];
	CalcType sum = 0;
	for (size_t i=0; i<WinSize; ++i) sum += theta[i] * Q[ID*WinSize+i];
	return sum + offset;
#endif
}


// [USED FREQUENTLY] Calculate beta for X[k] - eq. 21
void betaCalc(CalcType *R, CalcType *Q, CalcType *beta, size_t *SMask, size_t k, size_t SSize, size_t RSize, size_t WinSize){
#ifdef EN_STAT
	for (size_t i=0; i<SSize+1; ++i) {
		CalcType temp = -R[i*RSize];
		MaxbetaSum = max(MaxbetaSum, temp);
		for (size_t j=0; j<SSize; ++j) {
			CalcType cur = R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
			MaxbetaProduct = max(MaxbetaProduct, cur);
			temp -= cur;
			MaxbetaSum = max(MaxbetaSum, temp);
		}
		beta[i] = temp;
	}
#else
	for (size_t i=0; i<SSize+1; ++i) {
		CalcType temp = -R[i*RSize];
		for (size_t j=0; j<SSize; ++j) {
			temp -= R[i*RSize+j+1] * Q[k*WinSize+SMask[j]];
		}
		beta[i] = temp;
	}
#endif
}


// [USED FREQUENTLY] Calculate gamma[k] - eq. 22
CalcType gammaCalc(CalcType *Q, CalcType *beta, size_t *SMask, size_t k, size_t SSize, size_t WinSize){
#ifdef EN_STAT
	CalcType gamma_k = Q[k*WinSize+k] + beta[0];
	MaxgammaSum = max(MaxgammaSum, gamma_k);
	for (size_t i=0; i<SSize; ++i) {
		CalcType cur = Q[k*WinSize+SMask[i]] * beta[i+1];
		MaxgammaProduct = max(MaxgammaProduct, cur);
		gamma_k += cur;
		MaxgammaSum = max(MaxgammaSum, gamma_k);
	}
	return gamma_k;
#else
	CalcType gamma_k = Q[k*WinSize+k] + beta[0];
	for (size_t i=0; i<SSize; ++i) {
		gamma_k += Q[k*WinSize+SMask[i]] * beta[i+1];
	}
	return gamma_k;
#endif
}


// Initialise R for X[k] - eq. 24
void RInit(CalcType *R, DataType *dataX, size_t k, size_t DataDim, size_t RSize, CalcType sigma_sq){
	R[0] = -Kernel(&(dataX[k*DataDim]), &(dataX[k*DataDim]), DataDim, sigma_sq);
	R[1] = 1;
	R[RSize] = 1;
	R[RSize+1] = 0;
}


// [USED FREQUENTLY] Enlarge Matrix R - eq. 20
void REnlarge(CalcType *R, CalcType *beta, CalcType gamma_k, size_t SSize, size_t RSize){
#ifdef EN_STAT
	MaxREgammak = max(MaxREgammak, gamma_k);
	CalcType gammaInv = 1.0/gamma_k;
	MaxREgammaInv = max(MaxREgammaInv, gammaInv);
	for (size_t i=0; i<SSize+1; ++i) {
		for (size_t j=0; j<SSize+1; ++j) {
			R[i*RSize+j] += beta[i] * beta[j] * gammaInv;
			MaxRECalc = max(MaxRECalc, beta[i] * beta[j] * gammaInv);
			MaxRECalc = max(MaxRECalc, R[i*RSize+j]);
			MaxR = max(MaxR, R[i*RSize+j]);
		}
		CalcType temp = beta[i] * gammaInv;
		MaxRECalc = max(MaxRECalc, temp);
		R[i*RSize+SSize+1] = temp;
		MaxR = max(MaxR, temp);
	}
	for (size_t j=0; j<SSize+1; ++j) {
		CalcType temp = beta[j] * gammaInv;
		MaxRECalc = max(MaxRECalc, temp);
		R[(SSize+1)*RSize+j] = temp;
		MaxR = max(MaxR, temp);
	}
	R[(SSize+1)*RSize+SSize+1] = gammaInv;
	MaxR = max(MaxR, gammaInv);
#else
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
#endif
}


// [USED FREQUENTLY] Shrink Matrix R - eq. 23
void RShrink(CalcType *R, size_t p, size_t SSize, size_t RSize) {
#ifdef EN_STAT
	if (SSize>1) {
		size_t k = p+1;
		CalcType Kelem = R[k*RSize+k];
		MaxRSKelem = max(MaxRSKelem, Kelem);
		CalcType KelemInv = 1.0 / Kelem;
		MaxRSKelemInv = max(MaxRSKelemInv, KelemInv);
		// Shrink
		for (size_t i=0; i<SSize+1; ++i) {
			for (size_t j=0; j<SSize+1; ++j) {
				if (i==k||j==k) continue;
				else R[i*RSize+j] -= R[i*RSize+k] * R[k*RSize+j] * KelemInv;
				MaxRSCalc = max(MaxRSCalc, R[k*RSize+j] * KelemInv);
				MaxRSCalc = max(MaxRSCalc, R[i*RSize+k] * R[k*RSize+j] * KelemInv);
				MaxRSCalc = max(MaxRSCalc, R[i*RSize+j]);
				MaxR = max(MaxR, R[i*RSize+j]);
			}
		}
		// Left Shift
		for (size_t i=0; i<SSize+1; ++i) {
			for (size_t j=k; j<SSize; ++j) R[i*RSize+j] = R[i*RSize+j+1];
		}
		// Up Shift
		for (size_t i=k; i<SSize; ++i) {
			for (size_t j=0; j<SSize+1; ++j) R[i*RSize+j] = R[(i+1)*RSize+j];
		}
		// Write Zero
		for (size_t i=0; i<SSize+1; ++i) R[i*RSize+SSize] = 0;
		for (size_t j=0; j<SSize+1; ++j) R[SSize*RSize+j] = 0;
	}
	else {
		for (size_t i=0; i<RSize*RSize; ++i) R[i] = 0;
	}
#else
	if (SSize>1) {
		size_t k = p+1;
		// Shrink
		for (size_t i=0; i<SSize+1; ++i) {
			for (size_t j=0; j<SSize+1; ++j) {
				if (i==k||j==k) continue;
				else R[i*RSize+j] -= R[i*RSize+k] * R[k*RSize+j] / R[k*RSize+k];
			}
		}
		// Left Shift
		for (size_t i=0; i<SSize+1; ++i) {
			for (size_t j=k; j<SSize; ++j) R[i*RSize+j] = R[i*RSize+j+1];
		}
		// Up Shift
		for (size_t i=k; i<SSize; ++i) {
			for (size_t j=0; j<SSize+1; ++j) R[i*RSize+j] = R[(i+1)*RSize+j];
		}
		// Write Zero
		for (size_t i=0; i<SSize+1; ++i) R[i*RSize+SSize] = 0;
		for (size_t j=0; j<SSize+1; ++j) R[SSize*RSize+j] = 0;
	}
	else {
		for (size_t i=0; i<RSize*RSize; ++i) R[i] = 0;
	}
#endif
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
		if (Group[i]=='E') {
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

	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;


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
				CalcType *val,
				size_t *flag,
				size_t *_CurSize,
				size_t *_SSize,
				size_t *_NSize,
				const Param param) {

	///////////////////////////// Read Parameters /////////////////////////////

	FILE * LogFile			= param.LogFileHandle;
	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;
	CalcType eps			= param.eps;

	size_t SSize = *_SSize;
	size_t NSize = *_NSize;


	///////////////////////////// Adding (Xc, Yc) to data set /////////////////////////////

	for (size_t i=0; i<DataDim; ++i) dataX[ID*DataDim+i] = Xc[i];
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
		printf("[INC] Write to [%zu] Sample joins R, SSize=%zu\n", ID, SSize);
		fprintf(LogFile,"[INC] Write to [%zu] Sample joins R, SSize=%zu\n", ID, SSize);
		return 0;
	}


	///////////////////////////// Prepare for bookkeeping /////////////////////////////

	// Label the new comer as Xc
	Group[ID] = 'C';

	// Assign q
	CalcType q = hXi[ID] > 0 ? -1 : 1;

	// Cycle count - to stop infinite loop
	size_t count = 0;

	///////////////////////////// Main Loop /////////////////////////////

	while(Group[ID]!='S' && Group[ID]!='E' && Group[ID]!='R') {

		// cycle count - to stop infinite loop
		count++;
		if(count==500) {
			fprintf(stderr,"[ERROR] Maximum Cycle Count (%zu) reached. Force Quit.\n", count);
			fprintf(LogFile,"[ERROR] Maximum Cycle Count (%zu) reached. Force Quit.\n", count);
			return -1;
		}

		if(SSize==0){
			// Calculate bC - Case 1
			CalcType bC = fabs(hXi[ID] + q*ep);
			// handle ties
			val[ID] = bC;
			flag[ID] = 0;

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
					// handle ties
					val[NMask[i]] = curbE;
					flag[NMask[i]] = 1;
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curbR = fabs(-hX + q*ep);
					if (curbR<bR) {
						bR = curbR;
						keyR = i;
					}
					// handle ties
					val[NMask[i]] = curbR;
					flag[NMask[i]] = 2;
				}
			}

			// Calculate delta_b
			CalcType L_[3] = {bC, bE, bR};
			size_t key_[3] = {ID, keyE, keyR};
			CalcType minL = L_[0];
			size_t minkey = key_[0];
			size_t flag_ = 0;
			for (size_t i=1; i<3; ++i) {
				if (L_[i]<minL) {
					minL = L_[i];
					minkey = key_[i];
					flag_ = i;
				}
			}

			CalcType delta_b = q*minL;

#ifdef EN_STAT
			MaxminL = max(MaxminL, minL);
#endif

			printf("[INC] Write to [%zu] <q=%1.0f> bC=%e, bE=%e, bR=%e. [flag=%zu, minkey=%zu] SSize=%zu --- minL=%.12e, eps=%e\n", ID, q, bC, bE, bR, flag_, minkey, SSize, minL, eps);
			fprintf(LogFile,"[INC] Write to [%zu] <q=%1.0f> bC=%e, bE=%e, bR=%e. [flag=%zu, minkey=%zu] SSize=%zu \n minL=%.12e\n", ID, q, bC, bE, bR, flag_, minkey, SSize, minL);

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
				switch (flag_) {
					case 0: {
						// Xc joins R and TERMINATE
						Group[ID] = 'R';
						theta[ID] = 0;
						NMask[NSize] = ID;
						(*_NSize)++;
						(*_CurSize)++;
						break;
					}
					case 1: { // case 1 and case 2 are handled the same way
						// X[NMask[minkey]] moves from E to S
						// Update Matrix R
						size_t k = NMask[minkey];
						// Initialise Matrix R
						RInit(R, dataX, k, DataDim, RSize, sigma_sq);
						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					case 2: { // case 1 and case 2 are handled the same way
						// X[NMask[minkey]] moves from E to S
						// Update Matrix R
						size_t k = NMask[minkey];
						// Initialise Matrix R
						RInit(R, dataX, k, DataDim, RSize, sigma_sq);
						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) {
							NMask[i] = NMask[i+1];
						}
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(LogFile, "[INC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						fprintf(stderr, "[INC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						return -1;
					}
				} // end of 'switch(flag[i])'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(LogFile, "[INC][ERROR] unable to make any move because minL=%f. \n", minL);
				fprintf(stderr, "[INC][ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		}
		else { // if (SSize!=0)

			// Calc beta
			betaCalc(R, Q, beta, SMask, ID, SSize, RSize, WinSize);

			// Calculate gamma - intensive, optimisable
			// this gamma is the relation between Xc and hXi of set E, set R
			for (size_t i=0; i<NSize; ++i) {
				CalcType temp1 = Q[ID*WinSize+NMask[i]];
				CalcType temp2 = beta[0];
				for (size_t j=0; j<SSize; ++j) {
					temp2 += Q[NMask[i]*WinSize+SMask[j]] * beta[j+1];
#ifdef EN_STAT
					MaxgammaProduct = max(MaxgammaProduct, Q[NMask[i]*WinSize+SMask[j]] * beta[j+1]);
					MaxgammaSum = max(MaxgammaSum, temp2);
#endif

				}
				gamma[i] = temp1 + temp2;
#ifdef EN_STAT
				MaxgammaSum = max(MaxgammaSum, gamma[i]);
#endif
			}

			// Calc gamma_c
			CalcType gamma_c = gammaCalc(Q, beta, SMask, ID, SSize, WinSize);


			///////////////////////////// Bookkeeping /////////////////////////////

			// Calculate Lc1 - Case 1
			CalcType Lc1 = INFINITY;
			CalcType qrc = (q>0) ? gamma_c : -gamma_c;
			if (qrc>0 && hXi[ID]<-ep) Lc1 = (hXi[ID]+ep)/gamma_c;
			else if (qrc<0 && hXi[ID]>ep) Lc1 = (hXi[ID]-ep)/gamma_c;
			Lc1 = fabs(Lc1);

			// Calculate Lc2 - Case 2
			CalcType Lc2 = fabs(q*C - theta[ID]);

			// handle ties
			val[ID] = (Lc1<Lc2) ? Lc1 : Lc2;
			flag[ID] = (Lc1<Lc2) ? 0 : 1;

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
				// handle ties
				val[SMask[i]] = curLiS;
				flag[SMask[i]] = 2;
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
					if (qri>0 && hX<=-ep) curLiE = (-hX-ep)/gamma[i];
					else if (qri<0 && hX>=ep) curLiE = (-hX+ep)/gamma[i];
					curLiE = fabs(curLiE);
					if (curLiE<LiE) {
						LiE = curLiE;
						keyE = i;
					}
					// handle ties
					val[NMask[i]] = curLiE;
					flag[NMask[i]] = 3;
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
					// handle ties
					val[NMask[i]] = curLiR;
					flag[NMask[i]] = 4;
				}
			}

			// Calculate delta theta_c
			CalcType L_[5] = {Lc1, Lc2, LiS, LiE, LiR};
			size_t key_[5] = {ID, ID, keyS, keyE, keyR};
			CalcType minL = L_[0];
			size_t minkey = key_[0];
			size_t flag_ = 0;
			for (size_t i=1; i<5; ++i) {
				if (L_[i]<=minL) {
					minL = L_[i];
					minkey = key_[i];
					flag_ = i;
				}
			}

			CalcType d_theta_c = q*minL;

#ifdef EN_STAT
			MaxminL = max(MaxminL, minL);
#endif

			printf("[INC] Write to [%zu] <q=%1.0f> Lc1=%e, Lc2=%e, LiS=%e, LiE=%e, LiR=%e. [flag=%zu, minkey=%zu] SSize=%zu --- minL=%.12e, eps=%e \n", ID, q, Lc1, Lc2, LiS, LiE, LiR, flag_, minkey, SSize, minL, eps);
			fprintf(LogFile, "[INC] Write to [%zu] <q=%1.0f> Lc1=%e, Lc2=%e, LiS=%e, LiE=%e, LiR=%e. [flag=%zu, minkey=%zu] SSize=%zu \n minL=%.12e\n", ID, q, Lc1, Lc2, LiS, LiE, LiR, flag_, minkey, SSize, minL);


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
				switch (flag_) {
					case 0: {
						// Xc joins S and terminate
						// Update Matrix R - enlarge
						REnlarge(R, beta, gamma_c, SSize, RSize);
						// Xc joins S and terminate
						Group[ID] = 'S';
						SMask[SSize] = ID;
						(*_SSize)++;
						(*_CurSize)++;
						break;
					}
					case 1: {
						// Xc joins E and terminate
						Group[ID] = 'E';
						theta[ID] = (theta[ID]>0) ? C : -C;
						NMask[NSize] = ID;
						(*_NSize)++;
						(*_CurSize)++;
						break;
					}
					case 2: {
						// Xl moves from S to R or E
						// Update Matrix R - shrink
						RShrink(R, minkey, SSize, RSize);

						// move Xl to R
						size_t k = SMask[minkey];
						if (fabs(theta[k])<(C/2)) {
							Group[k] = 'R';
							theta[k] = 0;
							NMask[NSize] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) SMask[i] = SMask[i+1];
							SSize--;
							(*_SSize)--;
							NSize++;
							(*_NSize)++;
						}
						// move Xl to E
						else{
							Group[k] = 'E';
							theta[k] = (theta[k]>0) ? C : -C;
							NMask[NSize] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) SMask[i] = SMask[i+1];
							SSize--;
							(*_SSize)--;
							NSize++;
							(*_NSize)++;
						}
						break;
					}
					case 3: {
						// Xl joins S
						size_t k = NMask[minkey];
						// Calc beta
						betaCalc(R, Q, beta, SMask, k, SSize, RSize, WinSize);
						// Calc gamma_k
						CalcType gamma_k = gammaCalc(Q, beta, SMask, k, SSize, WinSize);
						// Enlarge Matrix R
						REnlarge(R, beta, gamma_k, SSize, RSize);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					case 4: {
						// Xl joins S
						size_t k = NMask[minkey];
						// Calc beta
						betaCalc(R, Q, beta, SMask, k, SSize, RSize, WinSize);
						// Calc gamma_k
						CalcType gamma_k = gammaCalc(Q, beta, SMask, k, SSize, WinSize);
						// Enlarge Matrix R
						REnlarge(R, beta, gamma_k, SSize, RSize);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(LogFile, "[INC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						fprintf(stderr, "[INC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						return -1;
					}
				} // end of 'switch (flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(LogFile, "[INC][ERROR] unable to make any move because minL=%f. \n", minL);
				fprintf(stderr, "[INC][ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		} // end of if(SSize!=0)
	} // end of 'while(1)'

	return 0;
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
				CalcType *val,
				size_t *flag,
				size_t *_CurSize,
				size_t *_SSize,
				size_t *_NSize,
				const Param param) {

	///////////////////////////// Read Parameters /////////////////////////////

	FILE * LogFile			= param.LogFileHandle;
	const size_t DataDim	= param.DataDim;
	const size_t WinSize	= param.WinSize;
	const size_t RSize		= param.RSize;
	const CalcType ep		= param.ep;
	const CalcType C		= param.C;
	const CalcType sigma_sq = param.sigma_sq;
	CalcType eps			= param.eps;

	size_t SSize = *_SSize;
	size_t NSize = *_NSize;


	///////////////////////////// Remove ID from SMask and NMask /////////////////////////////

	if (Group[ID]=='R') {
		// Vectors in R can be directly removed
		// Update NMask
		// Search for current location
		size_t p=0;
		while(NMask[p]!=ID) p++;
		for (size_t i=p; i<NSize-1; ++i) {
			NMask[i] = NMask[i+1];
		}
		// Update Group
		Group[ID] = 'N';
		theta[ID] = 0;
		(*_NSize)--;
		(*_CurSize)--;
		return 0;
	}
	else if (Group[ID]=='E'){
		// Update NMask
		// Search for current location
		size_t p=0;
		while(NMask[p]!=ID) p++;
		for (size_t i=p; i<NSize-1; ++i) {
			NMask[i] = NMask[i+1];
		}
		Group[ID] = 'C';
		(*_NSize)--;
		NSize--;
	}
	else if (Group[ID]=='S'){
		// Update SMask
		// Search for current location
		size_t p=0;
		while(SMask[p]!=ID) p++;
		// Update Matrix R - shrink
		RShrink(R, p, SSize, RSize);
		// Update SMask
		for (size_t i=p; i<SSize-1; ++i) {
			SMask[i] = SMask[i+1];
		}
		Group[ID] = 'C';
		(*_SSize)--;
		SSize--;
	}
	else {
		// ERROR!
		fprintf(LogFile, "[DEC][ERROR] Group[%zu]='%c' \n", ID, Group[ID]);
		fprintf(stderr, "[DEC][ERROR] Group[%zu]='%c' \n", ID, Group[ID]);
		return -1;
	}

	// cycle count
	size_t count = 0;

	///////////////////////////// Main Loop /////////////////////////////

	while(Group[ID]!='N') {

		// cycle count
		count++;
		if(count==500) {
			fprintf(stderr,"[ERROR] Maximum Cycle Count (%zu) reached. Force Quit.\n", count);
			fprintf(LogFile,"[ERROR] Maximum Cycle Count (%zu) reached. Force Quit.\n", count);
			return -1;
		}

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
					// handle ties
					val[NMask[i]] = curbE;
					flag[NMask[i]] = 1;
				}
				else if (Group[NMask[i]]=='R') {
					CalcType curbR = fabs(-hX + q*ep);
					if (curbR<bR) {
						bR = curbR;
						keyR = i;
					}
					// handle ties
					val[NMask[i]] = curbR;
					flag[NMask[i]] = 2;
				}
			}

			// Calculate delta_b
			CalcType minL = (bE<bR) ? bE : bR;
			size_t minkey = (bE<bR) ? keyE : keyR;
			size_t flag_  = (bE<bR) ? 1 : 2;

			CalcType delta_b = q*minL;

#ifdef EN_STAT
			MaxminL = max(MaxminL, minL);
#endif

			printf("[DEC] Remove [%zu] <q=%1.0f> bE=%e, bR=%e. [flag=%zu, minkey=%zu] SSize=%zu --- minL=%.12e, eps=%e\n", ID, q, bE, bR, flag_, minkey, SSize, minL, eps);
			fprintf(LogFile, "[DEC] Remove [%zu] <q=%1.0f> bE=%e, bR=%e. [flag=%zu, minkey=%zu] SSize=%zu \n minL=%.12e\n", ID, q, bE, bR, flag_, minkey, SSize, minL);

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
				switch (flag_) {
					// case 1 and case 2 are handled the same way
					case 1: {
						// Xi moves from E to S
						size_t k = NMask[minkey];
						// Initialise Matrix R
						RInit(R, dataX, k, DataDim, RSize, sigma_sq);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					case 2: {
						// Xi moves from R to S
						size_t k = NMask[minkey];
						// Initialise Matrix R
						RInit(R, dataX, k, DataDim, RSize, sigma_sq);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(LogFile, "[DEC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						fprintf(stderr, "[DEC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						return -1;
					}
				} // end of 'switch(flag[cur])'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(LogFile, "[DEC][ERROR] unable to make any move because minL=%f. \n", minL);
				fprintf(stderr, "[DEC][ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		}
		else { // if(SSize!=0)

			// Assign q
			CalcType q = theta[ID] > 0 ? -1 : 1;

			// Calc beta
			betaCalc(R, Q, beta, SMask, ID, SSize, RSize, WinSize);

			// Calculate gamma - intensive, optimisable
			// this gamma is the relation between Xc and hXi of set E, set R
			for (size_t i=0; i<NSize; ++i) {
				CalcType temp1 = Q[ID*WinSize+NMask[i]];
				CalcType temp2 = beta[0];
				for (size_t j=0; j<SSize; ++j) {
					temp2 += Q[NMask[i]*WinSize+SMask[j]] * beta[j+1];
#ifdef EN_STAT
					MaxgammaProduct = max(MaxgammaProduct, Q[NMask[i]*WinSize+SMask[j]] * beta[j+1]);
					MaxgammaSum = max(MaxgammaSum, temp2);
#endif
				}
				gamma[i] = temp1 + temp2;
#ifdef EN_STAT
				MaxgammaSum = max(MaxgammaSum, gamma[i]);
#endif
			}

			// Calculate gamma_c
			CalcType gamma_c = gammaCalc(Q, beta, SMask, ID, SSize, WinSize);

			///////////////////////////// Bookkeeping /////////////////////////////

			// Calculate Lc2 - Case 1
			CalcType Lc2 = fabs(theta[ID]);
			// handle ties
			val[ID] = Lc2;
			flag[ID] = 0;

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
				// handle ties
				val[SMask[i]] = curLiS;
				flag[SMask[i]] = 1;
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
					// handle ties
					val[NMask[i]] = curLiE;
					flag[NMask[i]] = 2;
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
					// handle ties
					val[NMask[i]] = curLiR;
					flag[NMask[i]] = 3;
				}
			}

			// Calculate delta theta_c
			CalcType L[4] = {Lc2, LiS, LiE, LiR};
			size_t key[4] = {ID, keyS, keyE, keyR};
			CalcType minL = Lc2;
			size_t minkey = ID;
			size_t flag_ = 0;
			for (size_t i=1; i<4; ++i) {
				if (L[i]<minL) {
					minL = L[i];
					minkey = key[i];
					flag_ = i;
				}
			}

			CalcType d_theta_c = q*minL;

#ifdef EN_STAT
			MaxminL = max(MaxminL, minL);
#endif

			printf("[DEC] Remove [%zu] <q=%1.0f> Lc2=%e, LiS=%e, LiE=%e, LiR=%e. [flag=%zu, minkey=%zu] SSize=%zu --- minL=%.12e, eps=%e\n", ID, q, Lc2, LiS, LiE, LiR, flag_, minkey, SSize, minL, eps);
			fprintf(LogFile, "[DEC] Remove [%zu] <q=%1.0f> Lc2=%e, LiS=%e, LiE=%e, LiR=%e. [flag=%zu, minkey=%zu] SSize=%zu \n minL=%.12e\n", ID, q, Lc2, LiS, LiE, LiR, flag_, minkey, SSize, minL);


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
				switch (flag_) {
					case 0: {
						// Xc joins R and is removed
						Group[ID] = 'N';
						theta[ID] = 0;
						(*_CurSize)--;
						break;
					}
					case 1: {
						// Xl moves from S to R or E
						// Update Matrix R - shrink
						RShrink(R, minkey, SSize, RSize);

						// move Xl to R
						size_t k = SMask[minkey];
						if (fabs(theta[k])<(C/2)) {
							Group[k] = 'R';
							theta[k] = 0;
							NMask[NSize] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) SMask[i] = SMask[i+1];
							SSize--;
							(*_SSize)--;
							NSize++;
							(*_NSize)++;
						}
						// move Xl to E
						else {
							Group[k] = 'E';
							theta[k] = theta[k]>0 ? C : -C;
							NMask[NSize] = k;
							hXi[k] = hCalc(k, dataY, Q, theta, b, WinSize);
							for (size_t i=minkey; i<SSize-1; ++i) SMask[i] = SMask[i+1];
							SSize--;
							(*_SSize)--;
							NSize++;
							(*_NSize)++;
						}
						break;
					}
					case 2: {
						// Xl joins S
						size_t k = NMask[minkey];
						// Calc beta
						betaCalc(R, Q, beta, SMask, k, SSize, RSize, WinSize);
						// Calc gamma_k
						CalcType gamma_k = gammaCalc(Q, beta, SMask, k, SSize, WinSize);
						// Enlarge Matrix R
						REnlarge(R, beta, gamma_k, SSize, RSize);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					case 3: {
						// Xl joins S
						size_t k = NMask[minkey];
						// Calc beta
						betaCalc(R, Q, beta, SMask, k, SSize, RSize, WinSize);
						// Calc gamma_k
						CalcType gamma_k = gammaCalc(Q, beta, SMask, k, SSize, WinSize);
						// Enlarge Matrix R
						REnlarge(R, beta, gamma_k, SSize, RSize);

						// move Xl to S
						Group[k] = 'S';
						SMask[SSize] = k;

						// Update NMask
						for (size_t i=minkey; i<NSize-1; ++i) NMask[i] = NMask[i+1];
						NSize--;
						(*_NSize)--;
						SSize++;
						(*_SSize)++;
						break;
					}
					default: {
						// ERROR!
						fprintf(LogFile, "[DEC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						fprintf(stderr, "[DEC][ERROR] minL(%f) is smaller than INF, but flag(%zu) is invalid. \n", minL, flag_);
						return -1;
					}
				} // end of 'switch (flag)'
			} // end of 'if(minL)<infinity'
			else {
				// ERROR!
				fprintf(LogFile, "[DEC][ERROR] unable to make any move because minL=%f. \n", minL);
				fprintf(stderr, "[DEC][ERROR] unable to make any move because minL=%f. \n", minL);
				return -1;
			}
		}// end of if(SSize!=0)
	} // end of 'while()'

	return 0;
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

int SimpleDataSet(Param param){


	/////////////////////////// SVM Parameters ///////////////////////////

	const size_t DataSize = param.DataSize;
	const size_t DataDim = param.DataDim;
	const size_t WinSize = param.WinSize;
	const size_t RSize = param.RSize;
	const CalcType ep = param.ep;
	const CalcType C  = param.C;
	const CalcType sigma_sq = param.sigma_sq;


	/////////////////////////// Open Log File ///////////////////////////

	// Open log file
	param.LogFileHandle = fopen(param.LogFile, "w");
	if (param.LogFileHandle==NULL) {
		fprintf(stderr, "[ERROR] Cannot open log file [%s]. \n", param.LogFile);
  		exit(1);
	}


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
	size_t ActualDataSize = 0;
#ifdef EN_DOUBLE
	while(!feof(infp) && ActualDataSize<DataSize) {
		fscanf(infp, "%lf", &Y_IN[ActualDataSize]);
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%*d:%lf", &X_IN[ActualDataSize*DataDim+j]);
		}
		++ActualDataSize;
	}
#else
	while(!feof(infp) && ActualDataSize<DataSize) {
		fscanf(infp, "%f", &Y_IN[ActualDataSize]);
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%*d:%f", &X_IN[ActualDataSize*DataDim+j]);
		}
		++ActualDataSize;
	}
#endif
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

	// val
	CalcType *val = calloc(WinSize, sizeof(CalcType));

	// flag
	size_t *flag = calloc(WinSize, sizeof(size_t));

	// Coeff b
	CalcType b = 0;


	/////////////////////////// Initialise SVM ///////////////////////////

	// initialise SVM using 2 data points
	initSVM(&X_IN[0*DataDim], Y_IN[0], &X_IN[1*DataDim], Y_IN[1], dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, param);

	// Calculate Objective Function Value
	CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
	printf("[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
	printf("----------------------------------------------------\n");
	fprintf(param.LogFileHandle, "[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
	fprintf(param.LogFileHandle, "----------------------------------------------------\n");

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

		// Display the new item
		printf("\n--------[NEW INPUT]-------(%f,%f)------b=%.9f------------------------\n", Xc[0], Yc, b);
		fprintf(param.LogFileHandle,"\n--------[NEW INPUT]-------(%f,%f)------b=%.9f------------------------\n", Xc[0], Yc, b);

		// train SVM incrementally
		int isTrainingSuccessful = incSVM(ID, Xc, Yc, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, val, flag, &CurSize, &SSize, &NSize, param);

		if (isTrainingSuccessful!=0) {
			free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
			free(Group); free(SMask); free(NMask);
			free(Q); free(R);
			free(beta); free(gamma); free(theta);
			free(hXi); free(val); free(flag);
			fprintf(stderr, "[ERROR] Incremental SVM training failed at data[%zu]. \n", i);
			fprintf(param.LogFileHandle, "[ERROR] Incremental SVM training failed at data[%zu]. \n", i);
			fclose(param.LogFileHandle);
			return -1;
		}

		// Calculate Objective Function Value
		CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
		printf("[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");
		fprintf(param.LogFileHandle, "[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		fprintf(param.LogFileHandle, "----------------------------------------------------\n");

	}

#ifdef EN_STAT
	stat_report();
#endif

	/////////////////////////// Decremental Training ///////////////////////////

	for (size_t ID=ActualDataSize-1; ID>1; ID--) {

		// Decremental Training
		int isTrainingSuccessful = decSVM(ID, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, val, flag, &CurSize, &SSize, &NSize, param);

		if (isTrainingSuccessful!=0) {
			free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
			free(Group); free(SMask); free(NMask);
			free(Q); free(R);
			free(beta); free(gamma); free(theta);
			free(hXi);  free(val); free(flag);
			fprintf(stderr, "[ERROR] Decremental SVM training failed at data[%zu]. \n", ID);
			fprintf(param.LogFileHandle, "[ERROR] Decremental SVM training failed at data[%zu]. \n", ID);
			fclose(param.LogFileHandle);
			return -1;
		}
		// Calculate Objective Function Value
		CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
		printf("[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");
		fprintf(param.LogFileHandle, "[SVM]CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		fprintf(param.LogFileHandle, "----------------------------------------------------\n");
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
	free(hXi);  free(val); free(flag);
	fclose(param.LogFileHandle);

	return 0;
}


int LIBSVMData(Param param){


	/////////////////////////// SVM Parameters ///////////////////////////

	const size_t DataSize = param.DataSize;
	const size_t DataDim = param.DataDim;
	const size_t WinSize = param.WinSize;
	const size_t RSize = param.RSize;
	const CalcType ep = param.ep;
	const CalcType C  = param.C;
	const CalcType sigma_sq = param.sigma_sq;


	/////////////////////////// Open Files ///////////////////////////

	// Open input file
	char *inFile = param.InFile;
	FILE *infp = fopen(inFile, "r");
	if (infp==NULL) {
		fprintf(stderr, "[ERROR] Cannot open input file [%s]. \n", inFile);
  		exit(1);
	}

	// Open output file
	FILE *outfp = fopen(param.OutFile, "w");
	if (outfp==NULL) {
		fprintf(stderr, "[ERROR] Cannot open output file [%s]. \n", param.OutFile);
  		exit(1);
	}

	// Open log file
	param.LogFileHandle = fopen(param.LogFile, "w");
	if (param.LogFileHandle==NULL) {
		fprintf(stderr, "[ERROR] Cannot open log file [%s]. \n", param.LogFile);
  		exit(1);
	}

	/////////////////////////// Read data file ///////////////////////////

	// Allocating Memory for input data
	DataType *X_IN = calloc(DataSize * DataDim, sizeof(DataType));
	CalcType *Y_IN = calloc(DataSize, sizeof(CalcType));

	// read file into memory
	// We use LibSVM data format
	size_t ActualDataSize = 0;
#ifdef EN_DOUBLE
	while(!feof(infp) && ActualDataSize<DataSize) {
		fscanf(infp, "%lf", &Y_IN[ActualDataSize]);
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%*d:%lf", &X_IN[ActualDataSize*DataDim+j]);
		}
		++ActualDataSize;
	}
#else
	while(!feof(infp) && ActualDataSize<DataSize) {
		fscanf(infp, "%f", &Y_IN[ActualDataSize]);
		for (size_t j=0; j<DataDim; ++j) {
			fscanf(infp, "%*d:%f", &X_IN[ActualDataSize*DataDim+j]);
		}
		++ActualDataSize;
	}
#endif
	fclose(infp);


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

	// val
	CalcType *val = calloc(WinSize, sizeof(CalcType));

	// flag
	size_t *flag = calloc(WinSize, sizeof(size_t));

	// Coeff b
	CalcType b = 0;


	/////////////////////////// Initialise SVM ///////////////////////////

	// initialise SVM using 2 data points
	initSVM(&X_IN[0*DataDim], Y_IN[0], &X_IN[1*DataDim], Y_IN[1], dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, &CurSize, &SSize, &NSize, param);

	// Calculate Objective Function Value
	CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
	fprintf(param.LogFileHandle, "[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
	fprintf(param.LogFileHandle, "----------------------------------------------------\n");

	/////////////////////////// Window Based Training ///////////////////////////

	DataType Xc[DataDim];
	CalcType Yc;

	// Position for the new item
	size_t AddPos = 2;

	// Position for the item to be removed
	size_t DelPos = 0;

	// Performance Statistics
	int numItems = 0;
	double AbsError = 0;
	double SqError = 0;
	double PerError = 0;

	for (size_t i=2; i<ActualDataSize; ++i) {

		// read current data point
		for (size_t j=0; j<DataDim; ++j) {
			Xc[j] = X_IN[i*DataDim+j];
		}
		Yc = Y_IN[i];

		// Performance Statistics
		CalcType YcPredict = regSVM(Xc, dataX, theta, &b, WinSize, DataDim, sigma_sq);
		AbsError += fabs(Yc-YcPredict);
		SqError += (Yc-YcPredict)*(Yc-YcPredict);
		PerError += fabs((Yc-YcPredict)/Yc);
		numItems++;

		// Training
		if (CurSize<WinSize) {
			// Window is not full - add without remove

			// Incremental Training
			int isIncSuccessful = incSVM(AddPos, Xc, Yc, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, val, flag, &CurSize, &SSize, &NSize, param);
			if (isIncSuccessful!=0) {
				free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
				free(Group); free(SMask); free(NMask); free(hXi); free(val); free(flag);
				free(Q); free(R);free(beta); free(gamma); free(theta);
				fprintf(param.LogFileHandle, "[INC][ERROR] Incremental SVM training failed at data[%zu]. \n", i);
				fprintf(stderr, "[INC][ERROR] Incremental SVM training failed at data[%zu]. \n", i);
				fclose(param.LogFileHandle);
				return -1;
			}
			else{
				printf("[INC] data[%zu] WRITE TO [%zu], new CurSize = %zu.\n", i, AddPos, CurSize);
				fprintf(param.LogFileHandle, "[INC] data[%zu] WRITE TO %zu, new CurSize = %zu.\n", i, AddPos, CurSize);
			}

			// Increment AddPos
			AddPos++;
		}
		else{
			// Window is full - remove then add

			// Decremental Training
			int isDecSuccessful = decSVM(DelPos, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, val, flag, &CurSize, &SSize, &NSize, param);
			if (isDecSuccessful!=0) {
				free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
				free(Group); free(SMask); free(NMask); free(hXi); free(val); free(flag);
				free(Q); free(R);free(beta); free(gamma); free(theta);
				fprintf(param.LogFileHandle, "[DEC][ERROR] Decremental SVM training failed at data[%zu]. \n", i);
				fprintf(stderr, "[DEC][ERROR] Decremental SVM training failed at data[%zu]. \n", i);
				fclose(param.LogFileHandle);
				return -1;
			}
			else {
				printf("[DEC] item[%zu] removed, new CurSize = %zu.\n", DelPos, CurSize);
				fprintf(param.LogFileHandle, "[DEC] item[%zu] removed, new CurSize = %zu.\n", DelPos, CurSize);
			}

			// Use the empty slot
			AddPos = DelPos;

			// Incremental Training
			int isIncSuccessful = incSVM(AddPos, Xc, Yc, dataX, dataY, Group, SMask, NMask, Q, R, beta, gamma, theta, &b, hXi, val, flag, &CurSize, &SSize, &NSize, param);
			if (isIncSuccessful!=0) {
				free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
				free(Group); free(SMask); free(NMask); free(hXi);  free(val); free(flag);
				free(Q); free(R);free(beta); free(gamma); free(theta);
				fprintf(param.LogFileHandle, "[INC][ERROR] Incremental SVM training failed at data[%zu]. \n", i);
				fprintf(stderr, "[INC][ERROR] Incremental SVM training failed at data[%zu]. \n", i);
				fclose(param.LogFileHandle);
				return -1;
			}
			else{
				printf("[INC] data[%zu] written to [%zu], new CurSize = %zu.\n", i, AddPos, CurSize);
				fprintf(param.LogFileHandle, "[INC] data[%zu] written to [%zu], new CurSize = %zu.\n", i, AddPos, CurSize);
			}

			// Increment DelPos - wrap at WinSize-1
			DelPos = (DelPos==WinSize-1) ? 0 : DelPos+1;
		}

		// Calculate Objective Function Value
		CalcType init_obj = objCalc(dataY, Group, theta, Q, &b, C, ep, WinSize);
		printf("[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		printf("----------------------------------------------------\n");
		fprintf(param.LogFileHandle, "[SVM] CurSize = %zu, Obj = %f, b = %f\n", CurSize, init_obj, b);
		fprintf(param.LogFileHandle, "----------------------------------------------------\n");

	}

#ifdef EN_STAT
	stat_report();
#endif

	// Performance Statistics
	double MSE = SqError / (double)numItems;
	double MAE = AbsError / (double)numItems;
	double MAPE = PerError / (double)numItems;
	printf("--------------------Performance Statistics--------------------\n");
	printf("MSE = %lf, MAE = %lf, MAPE = %lf\n", MSE, MAE, MAPE);
	printf("--------------------------------------------------------------\n");
	fprintf(param.LogFileHandle,"--------------------Performance Statistics--------------------\n");
	fprintf(param.LogFileHandle,"MSE = %lf, MAE = %lf\n", MSE, MAE);
	fprintf(param.LogFileHandle,"--------------------------------------------------------------\n");

	/////////////////////////// Clean up ///////////////////////////

	free(X_IN); free(Y_IN); free(dataX); free(dataY); free(Ypredict);
	free(Group); free(SMask); free(NMask); free(hXi);  free(val); free(flag);
	free(Q); free(R);free(beta); free(gamma); free(theta);
	fclose(outfp);
	fclose(param.LogFileHandle);
	return 0;
}

int main(){

	///////////// Simple Data Set /////////////

	Param ParamSimple10;
	ParamSimple10.InFile 	= "SimpleData10.txt";
	ParamSimple10.OutFile  	= "SimpleResult10.txt";
	ParamSimple10.LogFile	= "SimpleLog10.txt";
	ParamSimple10.DataSize  = 10;
	ParamSimple10.DataDim  	= 1;
	ParamSimple10.WinSize  	= 10;
	ParamSimple10.RSize 	= 5;
	ParamSimple10.ep 		= 0.01;
	ParamSimple10.C 		= 1000;
	ParamSimple10.sigma_sq  = 50;
	ParamSimple10.eps  		= 1e-6;

//	SimpleDataSet(ParamSimple10);

	Param ParamSimple40;
	ParamSimple40.InFile 	= "SimpleData40.txt";
	ParamSimple40.OutFile  	= "SimpleResult40.txt";
	ParamSimple40.LogFile	= "SimpleLog40.txt";
	ParamSimple40.DataSize  = 40;
	ParamSimple40.DataDim  	= 1;
	ParamSimple40.WinSize  	= 16;
	ParamSimple40.RSize 	= 10;
	ParamSimple40.ep 		= 0.01;
	ParamSimple40.C 		= 1000;
	ParamSimple40.sigma_sq  = 50;
	ParamSimple40.eps  		= 1e-6;

	LIBSVMData(ParamSimple40);

	///////////// Order Book Data /////////////

	Param ParamOrderBook;
	ParamOrderBook.InFile 		= "data9970.txt";
	ParamOrderBook.OutFile  	= "data9970result.txt";
	ParamOrderBook.LogFile		= "data9970log.txt";
	ParamOrderBook.DataSize  	= 10000;
	ParamOrderBook.DataDim  	= 16;
	ParamOrderBook.WinSize  	= 500;
	ParamOrderBook.RSize 		= ParamOrderBook.WinSize;
	ParamOrderBook.ep 			= 1500;
	ParamOrderBook.C 			= 5000;
	ParamOrderBook.sigma_sq  	= 0.0625;
	ParamOrderBook.eps  		= 1e-6;

//	LIBSVMData(ParamOrderBook);

	///////////// CPU_SMALL DATASET /////////////

	Param ParamCPUSmall;
	ParamCPUSmall.InFile 	= "cpusmall_scale.txt";
	ParamCPUSmall.OutFile  	= "cpusmall_scale_result.txt";
	ParamCPUSmall.LogFile	= "cpusmall_scale_log.txt";
	ParamCPUSmall.DataSize 	= 8192;
	ParamCPUSmall.DataDim  	= 12;
	ParamCPUSmall.WinSize  	= 480;
	ParamCPUSmall.RSize 	= 480;
	ParamCPUSmall.ep 		= 4;
	ParamCPUSmall.C 		= 32;
	ParamCPUSmall.sigma_sq 	= 0.25;
	ParamCPUSmall.eps  		= 1e-10;

//	LIBSVMData(ParamCPUSmall);

	///////////// Mackey-Glass Equation DATASET /////////////

	Param ParamMG;
	ParamMG.InFile 	= "mg_scale.txt";
	ParamMG.OutFile = "mg_scale_result.txt";
	ParamMG.LogFile	= "mg_scale_log.txt";
	ParamMG.DataSize = 1385;
	ParamMG.DataDim  = 6;
	ParamMG.WinSize  = 480;
	ParamMG.RSize 	= 480;
	ParamMG.ep 		= 0.2;
	ParamMG.C 		= 32;
	ParamMG.sigma_sq = 0.25;
	ParamMG.eps  	= 1e-10;

//	LIBSVMData(ParamMG);


	///////////// Stress Encoding DATASET /////////////

	Param ParamStress;
	ParamStress.InFile 	= "stress.txt";
	ParamStress.OutFile = "stress_result.txt";
	ParamStress.LogFile	= "stress_log.txt";
	ParamStress.DataSize = 1187;
	ParamStress.DataDim  = 2;
	ParamStress.WinSize  = 480;
	ParamStress.RSize 	= 480;
	ParamStress.ep 		= 2;
	ParamStress.C 		= 64;
	ParamStress.sigma_sq = 0.025;
	ParamStress.eps  	= 1e-10;

//	LIBSVMData(ParamStress);

	///////////// DFE /////////////

//	int Result = DemoKernel();

	printf("[System] Job Finished.\n");

	return 0;
}
