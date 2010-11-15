using System;
using System.Collections.Generic;
using System.Text;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace TestDotProduct
{
    /// <summary>
    /// class contains methods  for different version of 
    /// sparse matrix multiplication
    /// </summary>
    public class SparseMatrixMatrixProd
    {
        public const int Rows = 6*1024;
        public const int Cols = 1024;

        public const int displayCount=0;
        public const int maxVal = 1;

        public static int avgElements= 80;
        public static int stdElements = 20;


        /// <summary>
        /// implementation of sparese matrix product
        /// </summary>
        /// <param name="repetition">how many times kernel should be launch</param>
        /// <param name="moduleFunction">cuda kenrel name</param>
        /// <param name="blockSizeX">block size X</param>
        /// <param name="blockSizeY">block size Y</param>
        /// <param name="transposeGrid">indicate that grid dimensions should be 
        /// computed alternativly, if false than gridDimY- connected with rows
        /// else gridDim.Y conected with cols</param>
        /// <returns></returns>
        public static float[] CRSSparseMM(int repetition, string moduleFunction, 
            int blockSizeX,int blockSizeY, bool transposeGrid)
        {
            //int blockSizeX = 4;
            //int blockSizeY = 4;

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "matrixKernels.cubin"));

            CUfunction cuFunc = cuda.GetModuleFunction(moduleFunction);

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("------------------------------------");
            Console.WriteLine("init Matrix");
            Stopwatch t = Stopwatch.StartNew();

            //values in CRS format
            float[] AVals, BVals;
            //indexes in Crs format
            int[] AIdx, BIdx;
            //Lenght of each row in CRS format
            int[] ARowLen, BRowLen;
            int maxIndex = 0;
            MakeRandCrsSparseMatrix(Rows, maxRowSize, out AVals, out AIdx, out ARowLen,out maxIndex);

           // DisplayCrsMatrix(AVals, AIdx, ARowLen,maxIndex);
            MakeRandCrsSparseMatrix(Cols, maxRowSize, out BVals, out BIdx, out BRowLen,out maxIndex);
            //DisplayCrsMatrix(BVals, BIdx, BRowLen, maxIndex);


            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr AValsPtr = cuda.CopyHostToDevice(AVals);
            CUdeviceptr AIdxPtr = cuda.CopyHostToDevice(AIdx);
            CUdeviceptr ALenghtPtr = cuda.CopyHostToDevice(ARowLen);

            CUdeviceptr BValsPtr = cuda.CopyHostToDevice(BVals);
            CUdeviceptr BIdxPtr = cuda.CopyHostToDevice(BIdx);
            CUdeviceptr BLenghtPtr = cuda.CopyHostToDevice(BRowLen);

            int outputSize = Rows * Cols;
            float[] output = new float[outputSize];
            //CUdeviceptr dOutput = cuda.Allocate(output);

            IntPtr outputPtr2 = cuda.HostAllocate((uint)(outputSize * sizeof(float)), CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            CUdeviceptr dOutput = cuda.GetHostDevicePointer(outputPtr2, 0);


            Console.WriteLine("copy to device takes {0}", t.Elapsed);
            #region set cuda parameters

            
            int Aelements = AVals.Length;
            int Belements = BVals.Length;

            cuda.SetFunctionBlockShape(cuFunc,blockSizeX,blockSizeY, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, AValsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, AIdxPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, ALenghtPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, BValsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, BIdxPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, BLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)Rows);
            offset += sizeof(int);
            cuda.SetParameter(cuFunc, offset, (uint)Cols);
            offset += sizeof(int);

            cuda.SetParameter(cuFunc, offset, (uint)Aelements);
            offset += sizeof(int);
            cuda.SetParameter(cuFunc, offset, (uint)Belements);
            offset += sizeof(int);
            
            
            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion
            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            //CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            //cuda.SetTextureFlags(cuTexRef, 0);

            int gridDimX =(int) Math.Ceiling((Cols + 0.0) / (blockSizeX));
            int gridDimY = (int)Math.Ceiling((0.0+Rows)/blockSizeY);
            if (transposeGrid)
            {
                gridDimX = (int)Math.Ceiling((Rows + 0.0) / (blockSizeX));
                gridDimY = (int)Math.Ceiling((0.0 + Cols) / blockSizeY);
            }

            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);

            
            for (int k = 0; k < repetition; k++)
            {
                cuda.Launch(cuFunc, gridDimX, gridDimY);

                cuda.SynchronizeContext();
               //  cuda.CopyDeviceToHost(dOutput, output);
                Marshal.Copy(outputPtr2, output, 0, outputSize);
            }

            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            
            timer.Stop();
            float cudaTime = cuda.ElapsedTime(start, end);

            Console.WriteLine("Matrix products with kernel {0}",moduleFunction);
            Console.WriteLine("  takes {0} ms stopwatch time {1} ms", cudaTime, timer.Elapsed);


            int lenght = displayCount;// Math.Min(displayCount, Rows);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(AValsPtr);
            cuda.Free(AIdxPtr);
            cuda.Free(ALenghtPtr);

            cuda.Free(BValsPtr);
            cuda.Free(BIdxPtr);
            cuda.Free(BLenghtPtr);

            cuda.Free(dOutput);
          
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(end);

            return output;
        }


        public static float[] CRSSparseMMwithDenseVector(int repetition,
            string moduleFunction, int blockSizeX, int blockSizeY)
        {
            CUDA cuda = new CUDA(0, true);

            // load module

            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "matrixKernels.cubin"));

            CUfunction cuFunc = cuda.GetModuleFunction(moduleFunction);

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("------------------------------------");
            Console.WriteLine("init Matrix");
            Stopwatch t = Stopwatch.StartNew();

            //values in CRS format
            float[] AVals, BVals;
            //indexes in Crs format
            int[] AIdx, BIdx;
            //Lenght of each row in CRS format
            int[] ARowLen, BRowLen;
            
            int maxIndex = 0;
            MakeRandCrsSparseMatrix(Rows, maxRowSize, out AVals, out AIdx, out ARowLen, out maxIndex);

            // DisplayCrsMatrix(AVals, AIdx, ARowLen,maxIndex);
            MakeRandCrsSparseMatrix(Cols, maxRowSize, out BVals, out BIdx, out BRowLen, out maxIndex);
            //DisplayCrsMatrix(BVals, BIdx, BRowLen, maxIndex);


            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr AValsPtr = cuda.CopyHostToDevice(AVals);
            CUdeviceptr AIdxPtr = cuda.CopyHostToDevice(AIdx);
            CUdeviceptr ALenghtPtr = cuda.CopyHostToDevice(ARowLen);

            int outputSize = Rows * Cols;
            float[] output = new float[outputSize];
            
            //allocate memory for output
            IntPtr outputPtr2 = cuda.HostAllocate((uint)(outputSize * sizeof(float)), CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            CUdeviceptr dOutput = cuda.GetHostDevicePointer(outputPtr2, 0);

            //create dense vector for each column in B matrix
            float[] mainVec = new float[maxIndex + 1];
            
            uint memSize = (uint)((maxIndex + 1) * sizeof(float));
            
            CUstream stream0 =cuda.CreateStream();
                       
 
            IntPtr[] mainVecIntPtrs= new IntPtr[2];

            //write combined memory allocation
            //IntPtr mainVecIPtr = cuda.HostAllocate(memSize,CUDADriver.CU_MEMHOSTALLOC_WRITECOMBINED);
            //CUdeviceptr mainVecPtr=cuda.CopyHostToDeviceAsync(mainVecIPtr,memSize,stream0);

            //
            //mainVecIntPtrs[0] = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_WRITECOMBINED);
            //mainVecIntPtrs[1] = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_WRITECOMBINED);

            mainVecIntPtrs[0] = cuda.AllocateHost(memSize);
            mainVecIntPtrs[1] = cuda.AllocateHost(memSize);
            CUdeviceptr mainVecPtr = cuda.CopyHostToDeviceAsync(mainVecIntPtrs[0], memSize, stream0);

            //IntPtr mainVecIPtr = cuda.HostAllocate(memSize,CUDADriver.CU_MEMHOSTALLOC_PORTABLE);
            //CUdeviceptr mainVecPtr=cuda.CopyHostToDeviceAsync(mainVecIPtr,memSize,stream0);

            //mapped memory allocation
            //IntPtr mainVecIPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            //CUdeviceptr mainVecPtr = cuda.CopyHostToDevice(mainVecIPtr, memSize);

            //get texture reference
            CUtexref cuTexRef = cuda.GetModuleTexture(module, "vectorTexRef");
            cuda.SetTextureFlags(cuTexRef, 0);
            cuda.SetTextureAddress(cuTexRef, mainVecPtr, memSize);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);
            #region set cuda parameters

            int Aelements = AVals.Length;

            cuda.SetFunctionBlockShape(cuFunc, blockSizeX, blockSizeY, 1);
            
            int offset = 0;
            cuda.SetParameter(cuFunc, offset, AValsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, AIdxPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, ALenghtPtr.Pointer);
            offset += IntPtr.Size;
            
            cuda.SetParameter(cuFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)Rows);
            offset += sizeof(int);
            cuda.SetParameter(cuFunc, offset, (uint)Cols);
            offset += sizeof(int);
            
            int colIndexParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)0);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion
            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            
            int gridDimX = (int)Math.Ceiling((Rows + 0.0) / (blockSizeX));
            int gridDim= (Rows + blockSizeX - 1) / blockSizeX;

            

            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            for (int rep = 0; rep < repetition; rep++)
            {
                for (int k = 0; k < Cols; k++)
                {

                    Helpers.InitBuffer(BVals, BIdx, BRowLen, k, mainVecIntPtrs[k % 2]);

                    cuda.SynchronizeStream(stream0);
                    
                    cuda.CopyHostToDeviceAsync(mainVecPtr, mainVecIntPtrs[k % 2], memSize, stream0);
                    cuda.SetParameter(cuFunc, colIndexParamOffset,(uint) k);
                    cuda.LaunchAsync(cuFunc, gridDimX, 1, stream0);
                    //cuda.SynchronizeStream(stream0);
                    ////clear host buffer
                    Helpers.SetBufferIdx(BIdx, BRowLen, k-1, mainVecIntPtrs[(k+1) % 2], 0.0f);

                    //Helpers.InitBuffer(BVals, BIdx, BRowLen, k, mainVecIPtr);
                    ////make asynchronius copy and kernel lauch
                    //cuda.CopyHostToDeviceAsync(mainVecPtr, mainVecIPtr, memSize, stream0);
                    //cuda.SetParameter(cuFunc, colIndexParamOffset,(uint) k);
                    //cuda.LaunchAsync(cuFunc, gridDimX, 1, stream0);
                    //cuda.SynchronizeStream(stream0);
                    ////clear host buffer
                    //Helpers.SetBufferIdx(BIdx, BRowLen, k, mainVecIPtr, 0.0f);
                }
            }
            cuda.RecordEvent(end);
            cuda.SynchronizeContext();
            
            timer.Stop();
            float cudaTime = cuda.ElapsedTime(start, end);

            Marshal.Copy(outputPtr2, output, 0, outputSize);
            
            Console.WriteLine("Matrix products with kernel {0}", moduleFunction);
            Console.WriteLine("  takes {0} ms stopwatch time {1} ms", cudaTime, timer.Elapsed);


            int lenght = displayCount;// Math.Min(displayCount, Rows);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(AValsPtr);
            cuda.Free(AIdxPtr);
            cuda.Free(ALenghtPtr);
            cuda.Free(dOutput);
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(end);

            cuda.DestroyStream(stream0);
            cuda.Free(mainVecPtr);
            cuda.DestroyTexture(cuTexRef);
            
            
            return output;
        }

        
        
        
        
       


        /// <summary>
        /// Normal naive method for matrix multiplication
        /// </summary>
        /// <param name="repetition"></param>
        /// <returns></returns>
        public static float[] NormalCRSSparseMM(int repetition)
        {

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init Matrix");
            Stopwatch t = Stopwatch.StartNew();

            //values in CRS format
            float[] AVals, BVals;
            //indexes in Crs format
            int[] AIdx, BIdx;
            //Lenght of each row in CRS format
            int[] ARowLen, BRowLen;
            int maxIndex = 0;
            MakeRandCrsSparseMatrix(Rows, maxRowSize, out AVals, out AIdx, out ARowLen, out maxIndex);

            // DisplayCrsMatrix(AVals, AIdx, ARowLen,maxIndex);
            MakeRandCrsSparseMatrix(Cols, maxRowSize, out BVals, out BIdx, out BRowLen, out maxIndex);
            //DisplayCrsMatrix(BVals, BIdx, BRowLen, maxIndex);
            Console.WriteLine("Init takes {0}", t.Elapsed);

            float[] result = null;
            t.Reset();
            t.Start();
            for (int i = 0; i < repetition; i++)
            {
               result= Mul2SparseMatrix(AVals, AIdx, ARowLen, BVals, BIdx, BRowLen, Rows, Cols);
            }
            Console.WriteLine("computation on CPU takes {0}", t.Elapsed);

            return result;

        }

        private static float[] Mul2SparseMatrix(float[] AVals, int[] AIdx, int[] ARowLen, float[] BVals, int[] BIdx, int[] BRowLen, int Rows, int Cols)
        {
            float[] result = new float[Rows*Cols];

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {

                    int AStart = ARowLen[i];
                    int AEnd = ARowLen[i + 1];
                    int curPosA = AStart;

                    int BStart = BRowLen[j];
                    int BEnd = BRowLen[j + 1];
                    int curPosB = BStart;

                    int AcurIdx = AIdx[AStart];
                    int BcurIdx = BIdx[BStart];


                    float sum = 0;

                    while (curPosA < AEnd && curPosB < BEnd)
                    {
                        AcurIdx = AIdx[curPosA];
                        BcurIdx = BIdx[curPosB];

                        if (AcurIdx == BcurIdx)
                        {
                            sum += AVals[curPosA] * BVals[curPosB];
                            curPosA++;
                            curPosB++;
                        }
                        else if (AcurIdx < BcurIdx)
                        {
                            curPosA++;
                        }
                        else
                        {
                            curPosB++;
                        }

                    }

                    result[i*Cols+ j] = sum;


                }
            }

            return result;

        }

        private static void DisplayCrsMatrix(float[] AVals, int[] AIdx, int[] ARowLen, int maxCols)
        {
            int rows = ARowLen.Length - 1;
            int curRow = 0;
            Console.WriteLine("----------------------");

            for (int i = 0; i < rows; i++)
            {
                int rowStart = ARowLen[i];
                int rowEnd=ARowLen[i+1];

                int rowLenght = rowEnd-rowStart;
                
                int currPosition = ARowLen[i];
                int currIdx = AIdx[currPosition];

                for (int j = 0; j <= maxCols; j++)
                {
                    
                    if (currIdx == j)
                    {
                        Console.Write("{0:0.000} ", AVals[currPosition]);
                        currPosition++;
                        if(currPosition<AIdx.Length)
                            currIdx = AIdx[currPosition];
                        

                    }
                    else
                    {
                        Console.Write("{0:0.000} ", 0.0);
                    }

                }

                Console.WriteLine();
            }
            Console.WriteLine();
        }


        public static void MakeRandCrsSparseMatrix(int N, int maxRowSize, out float[] Vals, out int[] Idx, out int[] RowLen, out int maxIndex)
        {
            //temp lists for values, indices and vecotr lenght
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N+1);

            Helpers.step = 3;

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;
                
                Helpers.RandomSeed = i % 1000;
                
                float[] vals =Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
                vecIdxL.AddRange(index);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            Vals = vecValsL.ToArray();
            Idx = vecIdxL.ToArray();
            RowLen = vecLenghtL.ToArray();


        }

    }
}
