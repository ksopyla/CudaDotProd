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
    int high = col_end-1;
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



//	int low = col_start;
//  int high=col_end-1;
//int mid=-1;
//int curIdx=-1;
//  do
//    mid = low + ((high-low)/ 2);//better bit shift
//	curIdx =BIdx[mid];
//	if( index > curIdx){
//      low= mid + 1;
//	}
//    else 
//      high= mid - 1;
//  while (curIdx = index) || (low > high);

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


#define BLOCK_SIZE 64
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
	//__shared__ int bShPtrs[BLOCK_SIZE/WARP_SIZE][2];

	// global thread index
    const int thread_id   = BLOCK_SIZE * blockIdx.y + threadIdx.y;  
	// thread index within the warp (0,31)
    const int thread_lane = threadIdx.y & (WARP_SIZE-1);            
	// global warp index
    const int warp_id     = thread_id   / WARP_SIZE;                
	// warp index within the CTA
    const int warp_lane   = threadIdx.y / WARP_SIZE;                
	// total number of active warps
    const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.y;  

	//index of column in B matrix
	const int col = blockDim.x*blockIdx.x+threadIdx.x;

	/*
	if(thread_lane<2){
			bShPtrs[warp_lane][thread_lane]=BPtrs[col+thread_lane]	;
	}
	
	const int col_start = bShPtrs[warp_lane][0];
	const int col_end =	bShPtrs[warp_lane][1];
	*/
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
        sdata[threadIdx.y] = sum;
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y + 16]; 
		__syncthreads(); 
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  8]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  4]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  2]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  1];
		__syncthreads();

        // first thread writes warp result
		if (thread_lane == 0){
            //results[row] += sdata[threadIdx.x];
			//result[row] =sdata[threadIdx.x];
			result[row*BCols+col] = sdata[threadIdx.y];
		}
	}
			
}

//computes two sparse matrix product in CRS format, try to align memory access
//in warps use shared memory to cache B column
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_warp_shared(const float * AVals,
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

	__shared__ int svIdx[121];
	__shared__ float svVals[121];
    
	//stores "start" and "end" of column
	__shared__ int bShPtrs[2];
	// global thread index
    const int thread_id   = BLOCK_SIZE * blockIdx.y + threadIdx.y;  
	// thread index within the warp (0,31)
    const int thread_lane = threadIdx.y & (WARP_SIZE-1);            
	// global warp index
    const int warp_id     = thread_id   / WARP_SIZE;                
	// warp index within the CTA
    const int warp_lane   = threadIdx.y / WARP_SIZE;                
	// total number of active warps
    const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.y;  

	//index of column in B matrix
	const int col = blockDim.x*blockIdx.x+threadIdx.x;

	
	const int col_start = BPtrs[col];
	const int col_end =	BPtrs[col+1];

	for(int th=threadIdx.y; th<(col_end - col_start);th+=blockDim.y)
	{
		svVals[th]= BVals[col_start+th];
		svIdx[th]=BIdx[col_start+th];
	}
	__syncthreads();

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
			bVal=FindValForBIdx(svIdx,svVals,AIdx[jj],0,(col_end-col_start));
            sum += AVals[jj] * bVal;
		}
        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.y] = sum;
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y + 16]; 
		__syncthreads(); 
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  8]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  4]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  2]; 
		__syncthreads();
        sdata[threadIdx.y] = sum = sum + sdata[threadIdx.y +  1];
		__syncthreads();

        // first thread writes warp result
		if (thread_lane == 0){
            //results[row] += sdata[threadIdx.x];
			//result[row] =sdata[threadIdx.x];
			result[row*BCols+col] = sdata[threadIdx.y];
		}
	}
			
}

//computes two sparse matrix product in CRS format, try to align memory access
//in warps use shared memory to cache B column
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_warp_shared_Y(const float * AVals,
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

	__shared__ int svIdx[121];
	__shared__ float svVals[121];
    
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

	
	const int col_start = BPtrs[col];
	const int col_end =	BPtrs[col+1];

	for(int th=threadIdx.x; th<(col_end - col_start);th+=blockDim.x)
	{
		svVals[th]= BVals[col_start+th];
		svIdx[th]=BIdx[col_start+th];
	}
	__syncthreads();

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
			bVal=FindValForBIdx(svIdx,svVals,AIdx[jj],0,(col_end-col_start));
            sum += AVals[jj] * bVal;
		}
        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x] = sum;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; 
		__syncthreads(); 
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; 
		__syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; 
		__syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; 
		__syncthreads();
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
		__syncthreads();

        // first thread writes warp result
		if (thread_lane == 0){
            //results[row] += sdata[threadIdx.x];
			//result[row] =sdata[threadIdx.x];
			result[row*BCols+col] = sdata[threadIdx.x];
		}
	}
			
}



#define BLOCK_XY  2*64
//computes two sparse matrix product in CRS format, try to align memory access
//in warps use shared memory to cache B column
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_warp_shared_doubled(const float * AVals,
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


//!!!!!! not ready many errors !!!!!
// do not use it !

	__shared__ float sdata[2][BLOCK_XY + 16];                          // padded to avoid reduction ifs
    __shared__ int ptrs[BLOCK_XY/WARP_SIZE][2];

	__shared__ int svIdx[2][121];
	__shared__ float svVals[2][121];
    

	//stores "start" and "end" of column
	__shared__ int bShPtrs[2][2];

	const int threadPart=2*threadIdx.y+threadIdx.x;
	// global thread index
    
	const int thread_id   =BLOCK_XY* blockIdx.y + threadPart;
	// thread index within the warp (0,31)
    const int thread_lane = threadPart & (WARP_SIZE-1);            
	// global warp index
    const int warp_id     = thread_id   / WARP_SIZE;                
	// warp index within the block (CTA)
    const int warp_lane   = threadPart / WARP_SIZE;                
	// total number of active warps
    
	const int num_warps   = (blockDim.y / WARP_SIZE) * gridDim.y;//*gridDim.x;  

	//index of first column in block in B matrix
	//assume that blockDim.x==2
	const int col = blockDim.x*blockIdx.x; //+threadIdx.x;

	//copy pointers to each column to shared memory
	if(threadIdx.y<2)
	{
		//blockDim.x must equal 2
		//bShPtrs[threadIdx.x][threadIdx.y]=BPtrs[col+threadIdx.y];
		
		//if col is computed without adding threadIdx.x then above
		//line has an error and pointers should be set this way
		bShPtrs[threadIdx.x][threadIdx.y]=BPtrs[col+threadIdx.y+threadIdx.x];
	}
	__syncthreads();

	//copy vals and indexes for two column to shared mem.
	for(int th=threadIdx.y; th<(bShPtrs[threadIdx.x][1]-bShPtrs[threadIdx.x][0]);th+=blockDim.y)
	{
		svVals[threadIdx.x][th]= BVals[bShPtrs[threadIdx.x][0]+th];
		svIdx[threadIdx.x][th]=BIdx[bShPtrs[threadIdx.x][0]+th];
	}
	__syncthreads();

	

    for(int row = warp_id; row < ARows; row += num_warps){
        // use two threads to fetch vecPointers[row] and vecPointers[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = APtrs[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];   //same as: row_start = vecPointers[row];
        const int row_end   = ptrs[warp_lane][1];   //same as: row_end   = vecPointers[row+1];

        // compute local sum for two row and two column
		float sum[2] = {0,0};
		
		float bVal1=0;
		float bVal2=0;

        for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
		{
			int aIdx=AIdx[jj];
			bVal1=FindValForBIdx(svIdx[0],svVals[0],aIdx,0,	bShPtrs[0][1]-bShPtrs[0][0]);
			bVal2=FindValForBIdx(svIdx[1],svVals[1],aIdx,0,	bShPtrs[1][1]-bShPtrs[1][0]);
			float aVals = AVals[jj];
            sum[0] += aVals * bVal1;
			sum[1] += aVals * bVal2;
		}
        // reduce local sums to row sum (ASSUME: warpsize 32)

		/*
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x];
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart + 16]; 
		__syncthreads(); 
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart + 8]; 
		__syncthreads();
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart + 4]; 
		__syncthreads();
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart + 2]; 
		__syncthreads();
		sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart + 1]; 
		__syncthreads();
		*/
		
        sdata[0][threadPart] = sum[0];
		sdata[1][threadPart] = sum[1];
        sdata[0][threadPart] = sum[0] = sum[0] + sdata[0][threadPart + 16]; 
		sdata[1][threadPart] = sum[1] = sum[1] + sdata[1][threadPart + 16]; 
		__syncthreads(); 
        //sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart +  8]; 
		sdata[0][threadPart] = sum[0] = sum[0] + sdata[0][threadPart + 8]; 
		sdata[1][threadPart] = sum[1] = sum[1] + sdata[1][threadPart + 8]; 
		__syncthreads();
       // sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart +  4]; 
		sdata[0][threadPart] = sum[0] = sum[0] + sdata[0][threadPart + 4]; 
		sdata[1][threadPart] = sum[1] = sum[1] + sdata[1][threadPart + 4]; 
		__syncthreads();
        //sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart +  2]; 
		sdata[0][threadPart] = sum[0] = sum[0] + sdata[0][threadPart + 2]; 
		sdata[1][threadPart] = sum[1] = sum[1] + sdata[1][threadPart + 2]; 
		__syncthreads();
        //sdata[threadIdx.x][threadPart] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadPart +  1];
		sdata[0][threadPart] = sum[0] = sum[0] + sdata[0][threadPart + 1]; 
		sdata[1][threadPart] = sum[1] = sum[1] + sdata[1][threadPart + 1]; 
		__syncthreads();


		/*sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x];
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y + 16]; 
		__syncthreads(); 
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  8]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  4]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  2]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  1];
		__syncthreads();*/

        // first thread writes warp result
		if (thread_lane <1){
            
			result[row*BCols+col] = sdata[0][threadPart];
			result[row*BCols+col+1] = sdata[1][threadPart];
		}
	}
			
}

//computes two sparse matrix product in CRS format, try to align memory access
//in warps use shared memory to cache B column
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//BVals - values for second matrix
//BIdx - indexes for second matrix
//BPtrs - pointers to next vectors 
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
extern "C" __global__ void spmm_csr_warp_shared_doubled_test(const float * AVals,
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


//!!!!!! not ready many errors !!!!!
// do not use it !

	__shared__ float sdata[2][BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
    __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	__shared__ int svIdx[2][121];
	__shared__ float svVals[2][121];
    
	//stores "start" and "end" of column
	__shared__ int bShPtrs[2][2];
	// global thread index
    const int thread_id   = BLOCK_SIZE * blockIdx.y + threadIdx.y;  
	// thread index within the warp (0,31)
    const int thread_lane = threadIdx.y & (WARP_SIZE-1);            
	// global warp index
    const int warp_id     = thread_id   / WARP_SIZE;                
	// warp index within the CTA
    const int warp_lane   = threadIdx.y / WARP_SIZE;                
	// total number of active warps
    const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.y;  

	//index of first column in block in B matrix
	//assume that blockDim.x==2
	const int col = blockDim.x*blockIdx.x; //+threadIdx.x;

	//copy pointers to each column to shared memory
	if(threadIdx.y<2)
	{
		//blockDim.x must equal 2
		//bShPtrs[threadIdx.x][threadIdx.y]=BPtrs[col+threadIdx.y];
		
		//if col is computed without adding threadIdx.x then above
		//line has an error and pointers should be set this way
		bShPtrs[threadIdx.x][threadIdx.y]=BPtrs[col+threadIdx.y+threadIdx.x];
	}
	__syncthreads();

	//copy vals and indexes for two column to shared mem.
	for(int th=threadIdx.y; th<(bShPtrs[threadIdx.x][1]-bShPtrs[threadIdx.x][0]);th+=blockDim.y)
	{
		svVals[threadIdx.x][th]= BVals[bShPtrs[threadIdx.x][0]+th];
		svIdx[threadIdx.x][th]=BIdx[bShPtrs[threadIdx.x][0]+th];
	}
	__syncthreads();

	

    for(int row = warp_id; row < ARows; row += num_warps){
        // use two threads to fetch vecPointers[row] and vecPointers[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2 && threadIdx.x==0)
            ptrs[warp_lane][thread_lane] = APtrs[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];   //same as: row_start = vecPointers[row];
        const int row_end   = ptrs[warp_lane][1];   //same as: row_end   = vecPointers[row+1];

        // compute local sum for two row and two column
		float sum[2] = {0,0};
		
		float bVal1=0;
		float bVal2=0;

        for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
		{
			int aIdx=AIdx[jj];
			bVal1=FindValForBIdx(svIdx[0],svVals[0],aIdx,0,	bShPtrs[0][1]-bShPtrs[0][0]);
			bVal2=FindValForBIdx(svIdx[1],svVals[1],aIdx,0,	bShPtrs[1][1]-bShPtrs[1][0]);
			float aVals = AVals[jj];
            sum[0] += aVals * bVal1;
			sum[1] += aVals * bVal2;
		}
        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x];
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y + 16]; 
		__syncthreads(); 
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  8]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  4]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  2]; 
		__syncthreads();
        sdata[threadIdx.x][threadIdx.y] = sum[threadIdx.x] = sum[threadIdx.x] + sdata[threadIdx.x][threadIdx.y +  1];
		__syncthreads();

        // first thread writes warp result
		if (thread_lane <2){
            //results[row] += sdata[threadIdx.x];
			//result[row] =sdata[threadIdx.x];
			//dupa tutaj 
			result[row*BCols+col+thread_lane] = sdata[threadIdx.x][threadIdx.y];
		}
	}
			
}