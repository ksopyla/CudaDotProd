using System;
using System.Collections.Generic;
using System.Text;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace TestDotProduct
{
    /// <summary>
    /// class contains methods  for different version of 
    /// sparse matrix multiplication
    /// </summary>
    public class SparseMatrixTests
    {
        public const int Rows = 4;
        public const int Cols = 4;

        public const int displayCount=5;
        public const int maxVal = 1;

        public static int avgElements= 40;
        public static int stdElements = 20;

        /// <summary>
        /// naive implementation of sparese matrix product
        /// </summary>
        /// <param name="repetition"></param>
        /// <param name="moduleFunction"></param>
        /// <returns></returns>
        public static float[] NaiveSpMM(int repetition, string moduleFunction)
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "matrixKernels.cubin"));

            CUfunction cuFunc = cuda.GetModuleFunction(moduleFunction);

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();

            //values in CRS format
            float[] AVals, BVals;
            //indexes in Crs format
            int[] AIdx, BIdx;
            //Lenght of each row in CRS format
            int[] ARowLen, BRowLen;

            MakeRandCrsSparseMatrix(Rows, maxRowSize, out AVals, out AIdx, out ARowLen);

            MakeRandCrsSparseMatrix(Cols, maxRowSize, out BVals, out BIdx, out BRowLen);


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

            int blockSizeX = 16;
            int blockSizeY = 16;
            
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
            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion
            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            //CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            //cuda.SetTextureFlags(cuTexRef, 0);

            int gridDimX= Cols/blockSizeX;
            int gridDimY= Rows/blockSizeY;
            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);

            for (int k = 0; k < repetition; k++)
            {
                cuda.Launch(cuFunc, gridDimX, gridDimY);

                cuda.SynchronizeContext();
                // cuda.CopyDeviceToHost(dOutput, output);
                Marshal.Copy(outputPtr2, output, 0, outputSize);
            }

            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            
            timer.Stop();
            float cudaTime = cuda.ElapsedTime(start, end);

            Console.Write("Matrix products with kernel {0}, takes {1} ms stopwatch time {2} ms", moduleFunction, cudaTime, timer.Elapsed);


            int lenght = Math.Min(displayCount, Rows);
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


        public static void MakeRandCrsSparseMatrix(int N, int maxRowSize, out float[] Vals, out int[] Idx, out int[] RowLen)
        {
            //temp lists for values, indices and vecotr lenght
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N+1);

           

            int maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

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
