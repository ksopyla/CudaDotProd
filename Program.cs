using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Text;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Diagnostics;

namespace TestDotProduct
{
    class Program
    {


        [StructLayout(LayoutKind.Sequential)]
        public struct SparseVec
        {
            public int size;

            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)]
            public float[] values;

            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)]
            public int[] indices;

        }

        [StructLayout(LayoutKind.Sequential)]
        public struct SparseVecPtr
        {
            public int size;

            //[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)]
            public IntPtr values;

            //[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)]
            public IntPtr indices;

        }

        /// <summary>
        /// number of vectors
        /// </summary>
        static int N = 512 * 256;
        static int mainIndex = 0;
        static int maxIndex = 0;
        static int maxVal = 1;

        /// <summary>
        /// number of nonzero values
        /// </summary>
        static int avgElements = 10;
        /// <summary>
        /// +- nonzero values
        /// </summary>
        static int stdElements = 300;
        static int displayCount = 5;

        static int threadsPerBlock = 256;
        static int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        
        static float Gamma = 1f / 16;

        static int StartingIndex = 2;
        static int Repetition = 100;

        static void Main(string[] args)
        {


            DetailCudaDriver();

           // CudaDotProductExperiments();

           // CudaRBFProductExperiments();

            CudaSparseMatrixExperiments();


            Console.WriteLine();
            Console.WriteLine("End all tests");
            // Console.ReadKey();

        }

        private static void CudaSparseMatrixExperiments()
        {
            float[] normalResult = SparseMatrixMatrixProd.NormalCRSSparseMM(1);

            //float[] crsResult = SparseMatrixMatrixProd.CRSSparseMM(1,
            //    "spmm_csr_naive", 16, 16);
            //Helpers.TestEquality(normalResult, crsResult, "Naive CRS");

            //float[] crsResultSharedOne = SparseMatrixMatrixProd.CRSSparseMM(1,
            //    "spmm_csr_naive_shared_one", 1, 64);
           // Helpers.TestEquality(normalResult, crsResultSharedOne, "Naive CRS shared");

            int blockY = 64;
            //float[] crsResultWarp = SparseMatrixMatrixProd.CRSSparseMM(1,
            //    "spmm_csr_warp",1, blockY);
            //// Helpers.TestEquality(normalResult, crsResultWarp, "CRS warp");

            //float[] crsResultWarpShared = SparseMatrixMatrixProd.CRSSparseMM(1,
            //    "spmm_csr_warp_shared", 1, blockY);
             //Helpers.TestEquality(normalResult, crsResultWarp, "CRS warp shared");

            float[] crsResultWarpSharedDouble = SparseMatrixMatrixProd.CRSSparseMM(1,
                "spmm_csr_warp_shared_doubled", 2, blockY);
            Helpers.TestEquality(normalResult, crsResultWarpSharedDouble, "CRS warp shared doubled");

            
        }

        private static void CudaRBFProductExperiments()
        {
            float[] good = NormalRBFDotProd();
            Console.WriteLine("---------------------------------------");

            float[] rbf1 = CuRBFEllPackTexCached();
            Helpers.TestEquality(good, rbf1,"cuRBFEllPackTex");
            Console.WriteLine("---------------------------------------");

            float[] rbf2 = CuRBFCSRCached();
            Helpers.TestEquality(good, rbf2,"cuRBFCRSCached");
            Console.WriteLine("---------------------------------------");
        }

        private static void DetailCudaDriver()
        {
            CUDA cuda = new CUDA(false);

            cuda.Init();

            int cudaDrv = cuda.GetDeviceCount();

            if (cudaDrv < 1)
            {
                Console.WriteLine("Cuda device not found");
                System.Environment.Exit(-1);
            }

            Console.WriteLine("Found {0} cuda devices", cudaDrv);
            Device[] cuDevice = cuda.Devices;

            for (int i = 0; i < cuDevice.Length; i++)
            {
                Console.WriteLine("-------------------");
                Console.WriteLine("Cuda device nr {0} details:", i + 1);
                Console.WriteLine("Name: {0}", cuDevice[i].Name);
                Console.WriteLine("Compute: {0}", cuDevice[i].ComputeCapability);

                DeviceProperties prop = cuDevice[i].Properties;

                int processors = cuda.GetDeviceAttribute(CUDeviceAttribute.MultiProcessorCount, cuDevice[i].Handle);
                Console.WriteLine("Clock rate: {0}", prop.ClockRate);
                Console.WriteLine("Number of processors: {0}", processors);
                Console.WriteLine("Memory: {0} GB", (cuDevice[i].TotalMemory + 0.0) / (1024 * 1024));
                Console.WriteLine("Constant Memory: {0}MB", (prop.TotalConstantMemory + 0.0) / 1024);

            }
            Console.WriteLine("----------------------------------");
            Console.WriteLine();

        }

        private static void CudaDotProductExperiments()
        {

            //CuAddVec();
            //CuStructPass();
            float[] good = NormalDotProd(Repetition);
            
            Console.WriteLine("-----------------------------------");

            //float[] prod1 = CuDotProd();
            //TestEquality(good, prod1);
            //prod1 = null;


            //Console.WriteLine("-----------------------------------");
            //float[] prod2 = CuDotProdEllPack();
            //TestEquality(good, prod2);
            //prod2 = null;

            ////Console.WriteLine("-----------------------------------");

            //float[] prod3 = CuDotProdEllPackTexCached();
            //TestEquality(good, prod3);
            //prod3 = null;

            //Console.WriteLine("-----------------------------------");


            //float[] prod4 = CuDotProdSegmentedTexCached();
            //SegmentedMulCached();
            //Console.WriteLine("-----------------------------------");

            //float[] prod4 = DotProdSegmentedCached();
            //TestEquality(good, prod4);
            //prod4 = null;

            //Console.WriteLine("-----------------------------------");

            float[] prod5 = CuDotProdCRSCached(Repetition, "spmv_csr_vector_kernel");
           // TestEquality(good, prod5);
            

            Console.WriteLine("-----------------------------------");

            float[] prod6 = CuDotProdCRSCached(Repetition, "spmv_csr_scalar_kernel");
            // TestEquality(good, prod5);


            Console.WriteLine("-----------------------------------");

            //float[] prod6 = CuDotProdCSRwriteCombined(Repetition);
            //TestEquality(prod5, prod6);
            

            Console.WriteLine("-----------------------------------");
        }

        private unsafe static float[] CuDotProdSparseVecStruct()
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
            mainIndex = StartingIndex;
            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVecPtr();

                int vecSize = avgElements + i % stdElements;
                vectors[i].size = vecSize;
                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);

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

            Console.WriteLine("copy and init takes {0}", t.Elapsed);
            #region set cuda parameters
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
            #endregion
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

            Console.Write("Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

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

        private static float[] CuDotProdEllPack()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            //CUfunction structPassFunc = cuda.GetModuleFunction("DotProd");
            CUfunction structPassFunc = cuda.GetModuleFunction("DotProdEllPack");

            int maxRowSize = avgElements + stdElements - 1;
            float[] vecVals = new float[N * maxRowSize];
            int[] vecIdx = new int[N * maxRowSize];

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            maxIndex = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                //values are column-major aligment
                for (int z = 0; z < vals.Length; z++)
                {
                    int m = z * N + i;
                    vecVals[m] = vals[z];
                }

                //Array.Copy(vals,0,vecVals,i*maxRowSize,vals.Length);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
                //Array.Copy(index, 0, vecIdx, i * maxRowSize, index.Length);
                for (int z = 0; z < index.Length; z++)
                {
                    int m = z * N + i;
                    vecIdx[m] = index[z];
                }

            }


            //todo: dense X copy
            float[] mainVec = new float[maxIndex + 1];

            //remeber that values are column major 
            for (int j = 0; j < maxRowSize; j++)
            {
                int idx = vecIdx[mainIndex + N * j];
                float val = vecVals[mainIndex + N * j];
                mainVec[idx] = val;
            }

            Console.WriteLine("init arrays takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);

            float[] output = new float[N];
            CUdeviceptr mainVecPtr = cuda.CopyHostToDevice(mainVec);
            CUdeviceptr dOutput = cuda.Allocate(output);

            Console.WriteLine("copy array to device takes {0}", t.Elapsed);





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

        private static float[] CuDotProdEllPackTexCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));



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

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                //values are column-major aligment
                for (int z = 0; z < vals.Length; z++)
                {
                    int m = z * N + i;
                    vecVals[m] = vals[z];
                }

                //Array.Copy(vals,0,vecVals,i*maxRowSize,vals.Length);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
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
                int idx = vecIdx[mainIndex + N * j];
                float val = vecVals[mainIndex + N * j];
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
            cuda.DestroyArray(cuArr);
            cuda.DestroyTexture(cuTexRef);
            return output;
        }


        private static float[] SegmentedMulCached()
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

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
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
                for (int k = vecLenght[i]; k < vecLenght[i + 1]; k++)
                {
                    dot += output[k];
                }

                Console.WriteLine("{0}-{1}", i, dot);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(dOutput);
            cuda.DestroyArray(cuArr);
            cuda.DestroyTexture(cuTexRef);

            return output;
        }


        /// <summary>
        /// Gives bad results!!!
        /// </summary>
        /// <returns></returns>
        private static float[] DotProdSegmentedCached()
        {
            //todo: kernel compute bad values!

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

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
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

            //different then normal, because many vectors are computed not per row 
            //but per element
            int blocs = (vecStartIdx + threadsPerBlock - 1) / threadsPerBlock;
            cuda.Launch(structPassFunc, blocs, 1);


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

            cuda.DestroyArray(cuArr);
            cuda.DestroyTexture(cuTexRef);

            return output;
        }


        private static float[] CuDotProdCRSCached(int repetition,string moduleFunction)
        {

            //always the same values
            

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));

            CUfunction cuFunc = cuda.GetModuleFunction(moduleFunction);

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();

            //temp lists for values, indices and vecotr lenght
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N+1);

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
                vecIdxL.AddRange(index);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();


            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUdeviceptr vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            float[] output = new float[N];
            //CUdeviceptr dOutput = cuda.Allocate(output);

            IntPtr outputPtr2 = cuda.HostAllocate((uint)(N * sizeof(float)), CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            CUdeviceptr dOutput = cuda.GetHostDevicePointer(outputPtr2, 0);


            Console.WriteLine("copy to device takes {0}", t.Elapsed);
            #region set cuda parameters
            cuda.SetFunctionBlockShape(cuFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameter(cuFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion
            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

            CUtexref cuTexRef = cuda.GetModuleTexture(module, "texRef");
            cuda.SetTextureFlags(cuTexRef, 0);


            float[] mainVec = new float[maxIndex + 1];

            CUdeviceptr mainVecPtr = cuda.CopyHostToDevice(mainVec);
            

            uint memSize = (uint)((maxIndex+1) * sizeof(float));
            //uint flags = CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP | CUDADriver.CU_MEMHOSTALLOC_WRITECOMBINED;
            //uint tt = (uint)CUMemHostAllocFlags.WriteCombined;
            //uint s = (uint)CUMemHostAllocFlags.DeviceMap;
            //IntPtr mainVecIntPtr = cuda.HostAllocate(memSize, flags);
            //CUdeviceptr mainVecPtr = cuda.GetHostDevicePointer(mainVecIntPtr, 0);
            
            cuda.SetTextureAddress(cuTexRef, mainVecPtr, memSize);

            //CUarray cuArr = cuda.CreateArray(mainVec);
            //cuda.SetTextureArray(cuTexRef, cuArr);

            mainIndex = StartingIndex;
            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);

            for (int k = 0; k < repetition; k++)
            {
                //normal memory management
                InitMainVector(vecVals, vecIdx, vecLenght, mainVec);


                ////copy to texture
                ////cuda.CopyHostToArray(cuArr, mainVec, 0);
                cuda.CopyHostToDevice(mainVecPtr, mainVec);


               
                cuda.Launch(cuFunc, blocksPerGrid, 1);


                cuda.SynchronizeContext();
               // cuda.CopyDeviceToHost(dOutput, output);
                 Marshal.Copy(outputPtr2, output, 0, N);

                
                //mainVec = new float[maxIndex + 1];
                //clear previous vector values
               Array.Clear(mainVec, 0, mainVec.Length);
                //for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
                //{
                //    int idx = vecIdx[j];
                //    float val = vecVals[j];
                //    mainVec[idx] = 0;
                //}
                mainIndex++;
            }

            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);

            // cuda.CopyDeviceToHost(dOutput, output);

            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("Dot products with kernel {0}, mainIndex {1} and {2}-vectors takes {3} ms stopwatch time {4} ms",moduleFunction, mainIndex, N, naiveTime, timer.Elapsed);


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
            //cuda.DestroyArray(cuArr);
            //cuda.FreeHost(outputPtr2);
            //cuda.Free(dOutput);
            //Marshal.FreeHGlobal(
            cuda.Free(mainVecPtr);
            cuda.DestroyTexture(cuTexRef);
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(end);

            return output;
        }

        private static void InitMainVector(float[] vecVals, int[] vecIdx, int[] vecLenght, float[] mainVec)
        {
            for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
            {
                int idx = vecIdx[j];
                float val = vecVals[j];
                mainVec[idx] = val;
            }
        }


        private static float[] CuDotProdCSRwriteCombined(int repetition)
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));

            CUfunction cuFunc = cuda.GetModuleFunction("spmv_csr_vector_kernel_wc");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();

            //temp lists for values, indices and vecotr lenght
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

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
                vecIdxL.AddRange(index);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vecSize;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();


            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUdeviceptr vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            float[] output = new float[N];
            //CUdeviceptr dOutput = cuda.Allocate(output);

            IntPtr outputPtr2 = cuda.HostAllocate((uint)(N * sizeof(float)), CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            CUdeviceptr dOutput = cuda.GetHostDevicePointer(outputPtr2, 0);

            uint memSize = (uint)((maxIndex + 1) * sizeof(float));
            uint flags = CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP | CUDADriver.CU_MEMHOSTALLOC_WRITECOMBINED;
            uint tt = (uint)CUMemHostAllocFlags.WriteCombined;
            uint s = (uint)CUMemHostAllocFlags.DeviceMap;
            IntPtr mainVecIntPtr = cuda.HostAllocate(memSize, flags);
            
            CUdeviceptr mainVecPtr = cuda.GetHostDevicePointer(mainVecIntPtr, 0);

            Console.WriteLine("copy to device takes {0}", t.Elapsed);
            #region set cuda parameters
            cuda.SetFunctionBlockShape(cuFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, mainVecPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameter(cuFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion
            Console.WriteLine("start computation");

            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();

           
            mainIndex = StartingIndex;
            Stopwatch timer = Stopwatch.StartNew();
            cuda.RecordEvent(start);

            for (int k = 0; k < repetition; k++)
            {

                //float[] tempFloatarr = new float[memSize];
                unsafe
                {

                    float* vecPtr = (float*)mainVecIntPtr.ToPointer();

                    for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
                    {
                        int idx = vecIdx[j];
                        float val = vecVals[j];
                        vecPtr[idx] = val;

                       
                    }

                }
                
                //Marshal.Copy(mainVecIntPtr, tempFloatarr, 0, tempFloatarr.Length);
                

                cuda.Launch(cuFunc, blocksPerGrid, 1);


                cuda.SynchronizeContext();
                //cuda.CopyDeviceToHost(dOutput, output);
                Marshal.Copy(outputPtr2, output, 0, N);

               
                //mainVec = new float[maxIndex + 1];
                //Array.Clear(mainVec, 0, mainVec.Length);


                //clear previous vector values
                unsafe
                {

                    float* vecPtr = (float*)mainVecIntPtr.ToPointer();

                    for (int j = vecLenght[mainIndex]; j < vecLenght[mainIndex + 1]; j++)
                    {
                        int idx = vecIdx[j];
                        float val = vecVals[j];
                        vecPtr[idx] = 0;
                    }

                }
                mainIndex++;

            }

            cuda.RecordEvent(end);

            cuda.SynchronizeContext();
            //cuda.SynchronizeEvent(end);

            // cuda.CopyDeviceToHost(dOutput, output);

            timer.Stop();
            float naiveTime = cuda.ElapsedTime(start, end);

            Console.Write("csr vector Dot products with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);


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
            //cuda.DestroyArray(cuArr);
            cuda.Free(mainVecPtr);
            //cuda.DestroyTexture(cuTexRef);
            
           // cuda.Free(mainVecPtr);
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(end);

            return output;
        }

        private static float[] NormalDotProd(int repetition)
        {


            //always the same values
            Random rnd = new Random(1);

            SparseVec[] vectors = new SparseVec[N];
            mainIndex = StartingIndex;

            Console.WriteLine("array init ");
            Stopwatch t = Stopwatch.StartNew();
            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVec();

                int vecSize = avgElements + i % stdElements;
                vectors[i].size = vecSize;
                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);

                vectors[i].indices = index;
                vectors[i].values = vals;
            }
            Console.WriteLine("init takes {0}", t.Elapsed);
            float[] output = new float[N];


            Stopwatch timer = Stopwatch.StartNew();

            for (int k = 0; k < repetition; k++)
            {
                NormalSparseDotProduct(vectors, mainIndex, ref output);
                mainIndex++;
            }
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



        private static void NormalSparseDotProduct(SparseVec[] vectors, int mainIndex, ref float[] output)
        {
            SparseVec vec1 = vectors[mainIndex];

            for (int i = 0; i < vectors.Length; i++)
            {
                output[i] = DotProd(vec1, vectors[i]);
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


        private static float[] NormalRBFDotProd()
        {


            //always the same values
            Random rnd = new Random(1);

            SparseVec[] vectors = new SparseVec[N];
            mainIndex = StartingIndex;

            Console.WriteLine("array init ");
            Stopwatch t = Stopwatch.StartNew();
            for (int i = 0; i < N; i++)
            {
                vectors[i] = new SparseVec();

                int vecSize = avgElements + i % stdElements;
                vectors[i].size = vecSize;
                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);

                vectors[i].indices = index;
                vectors[i].values = vals;
            }
            Console.WriteLine("init takes {0}", t.Elapsed);
            float[] output = new float[N];


            Stopwatch timer = Stopwatch.StartNew();

            RBFProduct(vectors, mainIndex, ref output);

            timer.Stop();

            Console.Write("RBF kernel with mainIndex {0} and {1}-vectors takes {2}", mainIndex, N, timer.Elapsed);


            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            return output;
        }

        private static void RBFProduct(SparseVec[] vectors, int mainIndex, ref float[] output)
        {
            SparseVec vec1 = vectors[mainIndex];
            float vec1Dot = DotProd(vec1, vec1);
            for (int i = 0; i < vectors.Length; i++)
            {
                float dotVecI = DotProd(vectors[i], vectors[i]);
                float dotCross = DotProd(vec1, vectors[i]);
                float sum = dotVecI + vec1Dot - 2 * dotCross;
                output[i] = (float)Math.Exp(-Gamma * sum);
            }
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


        private static float[] CuRBFEllPackTexCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));

            CUfunction structPassFunc = cuda.GetModuleFunction("RBFEllPackCached");
            //RBFEllPackCached
            //RBFNaiveEllPackCached

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            float[] vecVals = new float[N * maxRowSize];
            int[] vecIdx = new int[N * maxRowSize];
            float[] selfDot = new float[N];

            maxIndex = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);

                //values are column-major aligment
                for (int z = 0; z < vals.Length; z++)
                {
                    int m = z * N + i;
                    vecVals[m] = vals[z];

                    selfDot[i] += vals[z] * vals[z];
                }

                //Array.Copy(vals,0,vecVals,i*maxRowSize,vals.Length);

                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
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
                int idx = vecIdx[mainIndex + N * j];
                float val = vecVals[mainIndex + N * j];
                mainVec[idx] = val;
            }


            Console.WriteLine("Init takes {0}", t.Elapsed);
            t.Start();

            CUdeviceptr valsPtr = cuda.CopyHostToDevice(vecVals);
            CUdeviceptr idxPtr = cuda.CopyHostToDevice(vecIdx);
            CUdeviceptr selfDotPtr = cuda.CopyHostToDevice(selfDot);
            CUarray cuArr = cuda.CreateArray(mainVec);

            cuda.CopyHostToArray(cuArr, mainVec, 0);
            //CUDAArrayDescriptor cuDesc = new CUDAArrayDescriptor();
            //cuDesc.Format = CUArrayFormat.Float;
            //cuDesc.NumChannels = 1;
            //cuDesc.Width = maxIndex+1;

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
            cuda.SetParameter(structPassFunc, offset, selfDotPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)maxRowSize);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);

            cuda.SetParameter(structPassFunc, offset, (uint)mainIndex);
            offset += sizeof(int);

            cuda.SetParameter(structPassFunc, offset, Gamma);
            offset += sizeof(float);

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

            Console.Write("RBF products ellpack with mainIndex {0} and {1}-vectors takes {2} ms stopwatch time {3} ms", mainIndex, N, naiveTime, timer.Elapsed);

            cuda.CopyDeviceToHost(dOutput, output);

            int lenght = Math.Min(displayCount, N);
            Console.WriteLine();
            for (int i = 0; i < lenght; i++)
            {
                Console.WriteLine("{0}-{1}", i, output[i]);
            }

            cuda.Free(valsPtr);
            cuda.Free(idxPtr);
            cuda.Free(selfDotPtr);
            cuda.Free(dOutput);
            cuda.DestroyArray(cuArr);
            cuda.DestroyTexture(cuTexRef);

            return output;
        }


        private static float[] CuRBFCSRCached()
        {

            //always the same values
            Random rnd = new Random(1);

            CUDA cuda = new CUDA(0, true);

            // load module
            CUmodule module = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));

            CUfunction structPassFunc = cuda.GetModuleFunction("RBFspmv_csr_vector");

            int maxRowSize = avgElements + stdElements - 1;

            Console.WriteLine("init arrays");
            Stopwatch t = Stopwatch.StartNew();
            List<float> vecValsL = new List<float>(N * maxRowSize / 2);
            List<int> vecIdxL = new List<int>(N * maxRowSize / 2);
            List<int> vecLenghtL = new List<int>(N);

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
            float[] selfDot = new float[N];

            maxIndex = 0;
            int vecStartIdx = 0;
            for (int i = 0; i < N; i++)
            {
                int vecSize = avgElements + i % stdElements;

                float[] vals = Helpers.InitValues(i, vecSize, maxVal);
                vecValsL.AddRange(vals);

                for (int z = 0; z < vals.Length; z++)
                {
                    selfDot[i] += vals[z] * vals[z];
                }
                int[] index = Helpers.InitIndices(i, vecSize, ref maxIndex);
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
            CUdeviceptr selfDotPtr = cuda.CopyHostToDevice(selfDot);

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
            cuda.SetParameter(structPassFunc, offset, selfDotPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, dOutput.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(structPassFunc, offset, (uint)N);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, (uint)mainIndex);
            offset += sizeof(int);
            cuda.SetParameter(structPassFunc, offset, Gamma);
            offset += sizeof(float);


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
            cuda.Free(selfDotPtr);
            cuda.Free(vecLenghtPtr);
            cuda.DestroyArray(cuArr);
            cuda.DestroyTexture(cuTexRef);
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(end);


            return output;
        }

        
    }
}
