

#define BLOCKSIZE 16



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

		/*
		if(curPosA<AElements)
			AcurIdx = AIdx[curPosA];
		
		if(curPosB<BElements)
			BcurIdx = BIdx[curPosB];
			*/

	}

	result[row*BCols+col] = sum;


}
