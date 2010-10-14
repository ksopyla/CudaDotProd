/*
	author: Krzysztof Sopyla (ksopyla@uwm.edu.pl)
	

*/


//computes two sparse matrix product in CRS format
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_naive(const float * AVals,
									   const int * AIdx, 
									   const int * APtrs,
									   const float * BVals,
									   const int * BIdx, 
									   const int * BPtrs,
									   float * result,
									   const int ARows,
									   const int BCols,
									   const int AElements,
									   const int BElements)
{

	const int row = blockIdx.y*blockDim.y+threadIdx.y;
	const int col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if( !(row<ARows && col<BCols) )
	{
		return;
	}

	//possible optimization, cache this in shared memory
	int AStart = APtrs[row];
	int AEnd = APtrs[row+1];
	int curPosA = AStart;

	int BStart = BPtrs[col];
	int BEnd = BPtrs[col+1];
	int curPosB = BStart;

	int AcurIdx = AIdx[AStart];
	int BcurIdx = BIdx[BStart];
	

	float sum=0;

	while(curPosA<AEnd && curPosB<BEnd)
	{
		AcurIdx = AIdx[curPosA];
		BcurIdx = BIdx[curPosB];

		if(AcurIdx == BcurIdx)
		{
			sum+=AVals[curPosA]*BVals[curPosB];
			curPosA++;
			curPosB++;
		}else if( AcurIdx< BcurIdx)
		{
			curPosA++;
		}else
		{
			curPosB++;
		}

		

	}

	result[row*BCols+col] = sum;


}


//computes two sparse matrix product in CRS format, use shared memory to cache  
//one column vector in second matrix
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_naive_shared_one(const float * AVals,
									   const int * AIdx, 
									   const int * APtrs,
									   const float * BVals,
									   const int * BIdx, 
									   const int * BPtrs,
									   float * result,
									   const int ARows,
									   const int BCols,
									   const int AElements,
									   const int BElements)
{
	//max size = 4081
	__shared__ float svIdx[1000];
	__shared__ float svVals[1000];

	
	const int row = blockIdx.y*blockDim.y+threadIdx.y;
	const int col = blockIdx.x*blockDim.x+threadIdx.x;
	
	//int BStart = BPtrs[col];
	int BEnd = BPtrs[col+1];
	int curPosB = BPtrs[col];
	
	int diff=BEnd-curPosB;
	
	int BcurIdx;

	for(int th=threadIdx.y; th<(BEnd-curPosB);th+=blockDim.y)
	{
		svVals[th]= BVals[curPosB+th];
		svIdx[th]=BIdx[curPosB+th];
	}
	__syncthreads();

	if( !(row<ARows && col<BCols) )
	{
		return;
	}

	//possible optimization, cache this in shared memory
	//int AStart = APtrs[row];
	int AEnd = APtrs[row+1];
	int curPosA = APtrs[row];
	int AcurIdx;

	float sum=0;

	//now B column is in shared mem, so it starts from 0
	curPosB=0;
	BEnd=diff;

	while(curPosA<AEnd && curPosB<BEnd)
	{
		AcurIdx = AIdx[curPosA];
		BcurIdx = svIdx[curPosB];

		if(AcurIdx == BcurIdx)
		{
			sum+=AVals[curPosA]*svVals[curPosB];
			curPosA++;
			curPosB++;
		}else if( AcurIdx< BcurIdx)
		{
			curPosA++;
		}else
		{
			curPosB++;
		}

	}

	result[row*BCols+col] = sum;


}
