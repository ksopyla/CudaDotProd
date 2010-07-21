using System;
using System.Runtime.InteropServices;
using System;
using System.Collections.Generic;

using System.Text;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace TestDotProduct
{
    class Program
    {


        [StructLayout(LayoutKind.Sequential)]
        public  struct SparseVec
        {
            public int size;

            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)]
            public float[] values;

            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)]
            public int[] indices;

        }

        [StructLayout(LayoutKind.Sequential)]
        public  struct SparseVecPtr
        {
            public int size;

            //[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)]
            public IntPtr values;

            //[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)]
            public IntPtr indices;

        }



       static int N = 512*256;
       static int mainIndex = 0;
       static int maxIndex = 0;
       static int maxVal = 10;

       static int avgElements = 120;
       static int stdElements = 40;
       static int displayCount = 5;

       static int threadsPerBlock = 128;
       static int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        static void Main(string[] args)
        {



            //CuAddVec();
            //CuStructPass();
            float[] good = NormalDotProd();
//
            Console.WriteLine("-----------------------------------");
            //float[] prod1 = CuDotProd();
            CuDotProd();

          //  Console.WriteLine("-----------------------------------");
            //float[] prod2=CuDotProdEllPack();
            CuDotProdEllPack();

           // Console.WriteLine("-----------------------------------");

           //float[] prod3= CuDotProdEllPackTexCached();
           CuDotProdEllPackTexCached();

            Console.WriteLine("-----------------------------------");

            //float[] prod4 = CuDotProdSegmentedTexCached();
           // SegmentedMulCached();
            // Console.WriteLine("-----------------------------------");

            DotProdSegmentedCached();

            Console.WriteLine("-----------------------------------");

            DotProdCSRCached();

            Console.WriteLine("-----------------------------------");
            
            Console.ReadKey();

        }

        private unsafe static float[] CuDotProd()
        {



            int sparseVecSize = sizeof(SparseVecPtr);

            uint size = (uint)(N * sizeof(SparseVecPtr));

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            //CUfunction structPassFunc = cuda.GetModuleFunction("DotProd");
            CUfunction structPassFunc = cuda.GetModuleFunction("DotProd2");


            SparseVecPtr[] vectors = new SparseVecPtr[N];
            Console.WriteLine("init and copy data");
            Stopwatch t = Stopwatch.StartNew();
            mainIndex = 0;
            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVecPtr();

                int vecSize = avgElements + i % stdElements;
                vectors[i].size = vecSize;
                float[] vals = InitValues(i, vecSize, maxVal, rnd);

                int[] index = InitIndices(i, vecSize,ref maxIndex, rnd);

                CUdeviceptr valsPtr = cuda.CopyHostToDevice(vals);
                CUdeviceptr idxPtr = cuda.CopyHostToDevice(index);

                vectors[i].indices = new IntPtr(idxPtr.Pointer);
                vectors[i].values = (IntPtr)valsPtr.Pointer;
            }

            GCHandle handle = GCHandle.Alloc(vectors, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();



            float[] output = new float[N];

            //CUdeviceptr dVectors = cuda.CopyHostToDevice(vectors);

            CUdeviceptr dVectors = cuda.CopyHostToDevice(ptr, size);
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy and init takes {0}",t.Elapsed);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, dVectors.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, (uint)mainIndex);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");
          
            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);

            
            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime,timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }


            cuda.Free(dVectors);
            cuda.Free(dOutput);

            return output;
        }

        private  static float[] CuDotProdEllPack()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            //CUfunction structPassFunc = cuda.GetModuleFunction("DotProd");
            CUfunction structPassFunc = cuda.GetModuleFunction("DotProdEllPack");

            int maxRowSize = avgElements+stdElements-1;
            float[] vecVals = new float[N*maxRowSize];
            int[] vecIdx = new int[N*maxRowSize];

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            maxIndex = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;
                
                float[] vals = InitValues(i, vecSize, maxVal, rnd);

                //values are column-major aligment
                for (int z = 0; z < vals.Length; z++)
                {
                    int m = z*N+i;
                    vecVals[m] = vals[z];
                }

                //Array.Copy(vals,0,vecVals,i*maxRowSize,vals.Length);

                int[] index = InitIndices(i, vecSize,ref maxIndex, rnd);
                //Array.Copy(index, 0, vecIdx, i * maxRowSize, index.Length);
                for (int z = 0; z < index.Length; z++)
                {
                    int m = z * N+i;
                    vecIdx[m] = index[z];
                }
                
            }

            

           

            //todo: dense X copy
            float[] mainVec = new float[maxIndex + 1];

            for (int j = 0; j < maxRowSize; j++)
            {
                int idx = vecIdx[mainIndex*maxRowSize+N*j];
                float val= vecVals[mainIndex*maxRowSize+N*j];
                mainVec[idx] = val;
            }

            Console.WriteLine("init arrays takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);

            float[] output = new float[N];
            CUdeviceptr mainVecPtr = cuda.CopyHostToDevice(mainVec);
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy array to device takes {0}",t.Elapsed);

            

            //error = cuFuncSetBlockShape(vecAdd, threadsPerBlock, 1, 1);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, mainVecPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)maxRowSize);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);


            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("EllPack Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(mainVecPtr);
            cuda.Free(dOutput);


            return output;
        }

        private  static float[] CuDotProdEllPackTexCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module= cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));



            CUfunction structPassFunc = cuda.GetModuleFunction("DotProdEllPackCached");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            float[] vecVals = new float[N * maxRowSize];
            int[] vecIdx = new int[N * maxRowSize];

           
            maxIndex = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = InitValues(i, vecSize, maxVal, rnd);

                //values are column-major aligment
                for (int z = 0; z < vals.Length; z++)
                {
                    int m = z * N + i;
                    vecVals[m] = vals[z];
                }

                //Array.Copy(vals,0,vecVals,i*maxRowSize,vals.Length);

                int[] index = InitIndices(i, vecSize, ref maxIndex, rnd);
                //Array.Copy(index, 0, vecIdx, i * maxRowSize, index.Length);
                for (int z = 0; z < index.Length; z++)
                {
                    int m = z * N + i;
                    vecIdx[m] = index[z];
                }

            }

            float[] mainVec = new float[maxIndex + 1];

            for (int j = 0; j < maxRowSize; j++)
            {
                int idx = vecIdx[mainIndex * maxRowSize + N * j];
                float val = vecVals[mainIndex * maxRowSize + N * j];
                mainVec[idx] = val;
            }
            Console.WriteLine("Init takes {0}",t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUarray cuArr = cuda.CreateArray(mainVec);

            cuda.CopyHostToArray(cuArr, mainVec, 0);


            //CUDAArrayDescriptor cuDesc = new CUDAArrayDescriptor();
            //cuDesc.Format = CUArrayFormat.Float;
            //cuDesc.NumChannels = 1;
            //cuDesc.Width = maxIndex+1;

            CUtexref cuTexRef = cuda.GetModuleTexture(module,"texRef");
            cuda.SetTextureFlags(cuTexRef, 0);

            cuda.SetTextureArray(cuTexRef, cuArr);


            float[] output = new float[N];
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            
            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)maxRowSize);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);


            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("EllPack Cached Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(dOutput);
            return output;
        }


        private  static float[] SegmentedMulCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));



            CUfunction structPassFunc = cuda.GetModuleFunction("SegmentedMulCached");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            List<float> vecValsL = new List<float>(N * maxRowSize/2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N);

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = InitValues(i, vecSize, maxVal, rnd);
                vecValsL.AddRange(vals);

                int[] index = InitIndices(i, vecSize, ref maxIndex, rnd);
                vecIdxL.AddRange(index);
                
                
                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();

            float[] mainVec = new float[maxIndex + 1];

            for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex+1]; j++)
            {
                int idx = vecIdx[j];
                float val = vecVals[j];
                mainVec[idx] = val;
            }
            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUarray cuArr = cuda.CreateArray(mainVec);

            cuda.CopyHostToArray(cuArr, mainVec, 0);


            //CUDAArrayDescriptor cuDesc = new CUDAArrayDescriptor();
            //cuDesc.Format = CUArrayFormat.Float;
            //cuDesc.NumChannels = 1;
            //cuDesc.Width = maxIndex+1;

            CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            cuda.SetTextureFlags(cuTexRef, 0);

            cuda.SetTextureArray(cuTexRef, cuArr);

            //!!!- now output has partial dot prods, so its lenght is equal to vals
            float[] output = new float[vecVals.Length];
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;
            
            cuda.SetParameter(structPassFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);


            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("segmented mull with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                float dot = 0;
                for (int k = vecLenght[i]; k < vecLenght[i+1]; k++)
                {
                    dot += output[k];
                }

                Console.WriteLine("{0}-{1}", i, dot);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(dOutput);
            return output;
        }


        private  static float[] DotProdSegmentedCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));



            CUfunction structPassFunc = cuda.GetModuleFunction("DotProdSegmentedCached");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N);

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = InitValues(i, vecSize, maxVal, rnd);
                vecValsL.AddRange(vals);

                int[] index = InitIndices(i, vecSize, ref maxIndex, rnd);
                vecIdxL.AddRange(index);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();

            float[] mainVec = new float[maxIndex + 1];

            for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
            {
                int idx = vecIdx[j];
                float val = vecVals[j];
                mainVec[idx] = val;
            }
            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUdeviceptr vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            //copy to texture
            CUarray cuArr = cuda.CreateArray(mainVec);
            cuda.CopyHostToArray(cuArr, mainVec, 0);
            CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            cuda.SetTextureFlags(cuTexRef, 0);
            cuda.SetTextureArray(cuTexRef, cuArr);

            //!!!- temp has partial dot prods, so its lenght is equal to vals
            float[] temp = new float[vecVals.Length];
            CUdeviceptr tempPtr = cuda.Allocate(temp);
            
            float[] output = new float[N];
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, tempPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);


            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("segmented Cached Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(dOutput);
            cuda.Free(vecLenghtPtr);
            cuda.Free(tempPtr);
            
            return output;
        }


        private  static float[] DotProdCSRCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));



            CUfunction structPassFunc = cuda.GetModuleFunction("spmv_csr_vector_kernel");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N);

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = InitValues(i, vecSize, maxVal, rnd);
                vecValsL.AddRange(vals);

                int[] index = InitIndices(i, vecSize, ref maxIndex, rnd);
                vecIdxL.AddRange(index);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();

            float[] mainVec = new float[maxIndex + 1];

            for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
            {
                int idx = vecIdx[j];
                float val = vecVals[j];
                mainVec[idx] = val;
            }
            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUdeviceptr vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            //copy to texture
            CUarray cuArr = cuda.CreateArray(mainVec);
            cuda.CopyHostToArray(cuArr, mainVec, 0);
            CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            cuda.SetTextureFlags(cuTexRef, 0);
            cuda.SetTextureArray(cuTexRef, cuArr);

           

            float[] output = new float[N];
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();


            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);


            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);
            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("csr vector Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(dOutput);
            cuda.Free(vecLenghtPtr);
            

            return output;
        }


        private static float[] NormalDotProd()
        {

            
                  //always the same values
            Random rnd = new Random(1);

            SparseVec[] vectors = new SparseVec[N];
            mainIndex = 0;

            Console.WriteLine("array init ");
            Stopwatch t = Stopwatch.StartNew();
            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVec();

                int vecSize = avgElements + i % stdElements;
                vectors[i].size = vecSize;
                float[] vals = InitValues(i, vecSize, maxVal, rnd);

                int[] index = InitIndices(i, vecSize,ref maxIndex, rnd);

                vectors[i].indices = index;
                vectors[i].values = vals;
            }
            Console.WriteLine("init takes {0}", t.Elapsed);
            float[] output = new float[N];

            
            Stopwatch timer = Stopwatch.StartNew();

            NormalSparseDotProduct(vectors, mainIndex,ref output);

            timer.Stop();

            Console.Write("Normal Dot products with mainIndex {0} and {1}-vectors takes {2}", mainIndex, N, timer.Elapsed);
            

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            return output;
        }

        private static void NormalSparseDotProduct(SparseVec[] vectors, int mainIndex,ref float[] output)
        {
            SparseVec vec1 = vectors[mainIndex];

            for (int i = 0; i < vectors.Length; i++)
            {
                output[i] = DotProd(vec1,vectors[i]);
            }

            
        }

        private static float DotProd(SparseVec vec1, SparseVec vec2)
        {
            int i1 = 0;
            int i2 = 0;
            float result = 0;
            
            
            while (i1 < vec1.size && i2 < vec2.size)
            {
                int index1 = vec1.indices[i1];
                int index2 = vec2.indices[i2];

                if (index1 == index2)
                {
                    result += vec1.values[i1] * vec2.values[i2];
                    i1++; i2++;
                }
                else if (index1 < index2)
                {
                    i1++;
                }
                else
                {
                    i2++;
                }
            }

            return result;
        }



        private static int[] InitIndices(int i, int size,ref int maxIndex, Random rnd)
        {
            int[] index = new int[size];

            int min = 0;
            int step = 10;
            int idx=0;
            for (int k = 0; k < size; k++)
            {
                idx = rnd.Next(min, min + step);
                min = idx + 1;
                index[k] = idx;
            }

            if (idx > maxIndex)
                maxIndex = idx;
            return index;
        }

        private static float[] InitValues(int i, int size, int maxVal, Random rnd)
        {
            float[] vals = new float[size];


            for (int k = 0; k < size; k++)
            {
                vals[k] = (float)rnd.NextDouble() * maxVal;
            }

            return vals;
        }


        private unsafe static void CuStructPass()
        {

            int N = 4;

            int sparseVecSize = sizeof(SparseVecPtr);

            uint size = (uint)(N * sizeof(SparseVecPtr));

            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            CUfunction structPassFunc = cuda.GetModuleFunction("StructPass");


            SparseVecPtr[] vectors = new SparseVecPtr[N];

            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVecPtr();
                vectors[i].size = 2;
                float[] vals = new float[2] { (float)i + 1 % 5, (float)i + 2 % 7 };

                //GCHandle valHandle = GCHandle.Alloc(vals, GCHandleType.Pinned);
                //vectors[i].values = valHandle.AddrOfPinnedObject();


                int[] index = new int[2] { i % 5, i % 7 };
                //GCHandle idxHandle = GCHandle.Alloc(index, GCHandleType.Pinned);
                //vectors[i].indices = idxHandle.AddrOfPinnedObject();

                //valHandle.Free();
                //idxHandle.Free();

                CUdeviceptr valsPtr = cuda.CopyHostToDevice(vals);
                CUdeviceptr idxPtr = cuda.CopyHostToDevice(index);

                vectors[i].indices = new IntPtr(idxPtr.Pointer);
                vectors[i].values = (IntPtr)valsPtr.Pointer;




            }

            GCHandle handle = GCHandle.Alloc(vectors, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();





            float[] output = new float[N];

            //CUdeviceptr dVectors = cuda.CopyHostToDevice(vectors);

            CUdeviceptr dVectors = cuda.CopyHostToDevice(ptr, size);
            CUdeviceptr dOutput = cuda.Allocate(output);

            int threadsPerBlock = 256;
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

            //error = cuFuncSetBlockShape(vecAdd, threadsPerBlock, 1, 1);

            cuda.SetFunctionBlockShape(structPassFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(structPassFunc, offset, (uint)dVectors.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, (uint)dOutput.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameterSize(structPassFunc, (uint)offset);

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            cuda.RecordEvent(start);
            cuda.Launch(structPassFunc, blocksPerGrid, 1);
            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);

            float naiveTime = cuda.ElapsedTime(start, end);
            Console.Write("passing struct takes {0}ms", naiveTime);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(10, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }
        }

        private static void CuAddVec()
        {
            int N = 50000;
            uint size = (uint)N * sizeof(float);


            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            CUfunction vecAddFunc = cuda.GetModuleFunction("VecAdd");

            float[] A = new float[N];
            float[] B = new float[N];
            float[] C = new float[N];
            for (int i = 0; i < A.Length; i++)
            {
                A[i] = (float)i;
                B[i] = (float)i + 0.1f;
            }

            CUdeviceptr dA = cuda.CopyHostToDevice(A);
            CUdeviceptr dB = cuda.CopyHostToDevice(B);

            CUdeviceptr dC = cuda.Allocate(A);

            int threadsPerBlock = 256;
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

            //error = cuFuncSetBlockShape(vecAdd, threadsPerBlock, 1, 1);

            cuda.SetFunctionBlockShape(vecAddFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(vecAddFunc, offset, (uint)dA.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(vecAddFunc, offset, (uint)dB.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(vecAddFunc, offset, (uint)dC.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(vecAddFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameterSize(vecAddFunc, (uint)offset);

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            cuda.RecordEvent(start);
            cuda.Launch(vecAddFunc, blocksPerGrid, 1);
            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);

            float naiveTime = cuda.ElapsedTime(start, end);
            Console.Write("adding takes {0}ms", naiveTime);

            cuda.CopyDeviceToHost(dC, C);

            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine("{0}-{1}", i, C[i]);
            }
        }
    }
}
