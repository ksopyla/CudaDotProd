/*
	author: Krzysztof Sopyla (ksopyla@uwm.edu.pl)
	

*/

//helper functions

//Use binary search algorith to find index and value in B matrix
__device__ float FindValForBIdx(const int* BIdx,
								const float* BVals,
								const int index,
								int col_start, int col_end)
{
	int low = col_start;
    int high = col_end;
	int mid=-1;
    while (low < high) {
		mid = low + ((high - low) / 2);
		if (BIdx[mid] < index)
			low = mid + 1;
        else
			//can't be high = mid-1: here A[mid] >= value,
            //so high can't be < mid if A[mid] == value
            high = mid;
       }
       // high == low, using high or low depends on taste
       if ((low < col_end) && (BIdx[low] == index))
           return BVals[low]; // found
       else
           return 0.0; // not found

}


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
	//int AStart = APtrs[row];
	
	int curPosA = APtrs[row];
	int AEnd = APtrs[row+1];
	

	//int BStart = BPtrs[col];
	int curPosB = BPtrs[col];
	int BEnd = BPtrs[col+1];
	
	int AcurIdx=-1;
	int BcurIdx=-1;

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
	__shared__ int svIdx[121];
	__shared__ float svVals[121];

	//barier[0]=BStart
	//barier[1]=BEnd
	__shared__ int barier[2];
	
	const int row = blockIdx.y*blockDim.y+threadIdx.y;
	const int col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if( !(row<ARows && col<BCols) )
	{
		return;
	}

	//int BStart = BPtrs[col];
	if(threadIdx.y<2){
		barier[threadIdx.y]=BPtrs[col+threadIdx.y]	;
	}
	//????
	__syncthreads();
	int curPosB = barier[0];
	int diff=barier[1]-barier[0];
	
	//int curPosB = BPtrs[col];
	//int diff = BPtrs[col+1] - curPosB;

	int BcurIdx;

	for(int th=threadIdx.y; th<diff;th+=blockDim.y)
	{
		svVals[th]= BVals[curPosB+th];
		svIdx[th]=BIdx[curPosB+th];
	}
	__syncthreads();

	int curPosA = APtrs[row];
	int AEnd = APtrs[row+1];
	int AcurIdx;
	float sum=0;
	//now B column is in shared mem, so it starts from 0
	curPosB=0;
	
	while(curPosA<AEnd && curPosB<diff)
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
	__syncthreads();
	result[row*BCols+col] = sum;
	//column major order
	//result[row+ARows*col] = sum;
}


#define BLOCK_SIZE 128
#define WARP_SIZE 32
//computes two sparse matrix product in CRS format, try to align memory access
//in warps
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_warp(const float * AVals,
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
	__shared__ float sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
    __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
    
	//stores "start" and "end" of column
	__shared__ int bShPtrs[2];
	// global thread index
    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  
	// thread index within the warp (0,31)
    const int thread_lane = threadIdx.x & (WARP_SIZE-1);            
	// global warp index
    const int warp_id     = thread_id   / WARP_SIZE;                
	// warp index within the CTA
    const int warp_lane   = threadIdx.x / WARP_SIZE;                
	// total number of active warps
    const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  

	//index of column in B matrix
	const int col = blockDim.y*blockIdx.y+threadIdx.y;

	/*if(threadIdx.y<2){
			bShPtrs[threadIdx.y]=BPtrs[col+threadIdx.y]	;
	}
	const int col_start = bShPtrs[0];
	const int col_end =	bShPtrs[1];*/

	const int col_start = BPtrs[col];
	const int col_end =	BPtrs[col+1];

    for(int row = warp_id; row < ARows; row += num_warps){
        // use two threads to fetch vecPointers[row] and vecPointers[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = APtrs[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];   //same as: row_start = vecPointers[row];
        const int row_end   = ptrs[warp_lane][1];   //same as: row_end   = vecPointers[row+1];

        // compute local sum
        float sum = 0;
		
		float bVal=0;

        for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
		{
			bVal=FindValForBIdx(BIdx,BVals,AIdx[jj],col_start,col_end);
            sum += AVals[jj] * bVal;
		}
        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x] = sum;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

        // first thread writes warp result
        if (thread_lane == 0)
            //results[row] += sdata[threadIdx.x];
			result[row] =sdata[threadIdx.x];
	}
			
}