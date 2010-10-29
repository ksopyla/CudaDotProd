using System;
using System.Collections.Generic;
using System.Text;

namespace TestDotProduct
{
   internal class Helpers
    {

        /// <summary>
        /// error for chcecking results
        /// </summary>
        static float ErrorEpsilon = 0.001f;
       public static int RandomSeed = 1;
       public static int step = 10;
        public static int[] InitIndices(int i, int size, ref int maxIndex)
        {
            Random rnd = new Random(RandomSeed);
            int[] index = new int[size];

            int min = 0;
            
            int idx = 0;
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

       /// <summary>
       /// creates 
       /// </summary>
       /// <param name="i"></param>
       /// <param name="size"></param>
       /// <param name="maxVal"></param>
       /// <returns></returns>
        public static float[] InitValues(int i, int size, int maxVal)
        {
            Random rnd = new Random(RandomSeed);
            float[] vals = new float[size];


            for (int k = 0; k < size; k++)
            {
                vals[k] = (float)rnd.NextDouble() * maxVal;
            }

            return vals;
        }

        /// <summary>
        ///  creates dense vector from one row of mat sparse matrix in CSR format
        /// </summary>
        /// <param name="matVals">matrx values</param>
        /// <param name="matIdx">matrix indices</param>
        /// <param name="matRowLen">matrix rows lenght</param>
        /// <param name="mainIndex">row index</param>
        /// <param name="mainVec">modified dense array</param>
        public static void InitMainVector(float[] matVals, int[] matIdx, int[] matRowLen, 
                                        int mainIndex,ref float[] mainVec)
        {
            Array.Clear(mainVec, 0, mainVec.Length);

            for (int j = matRowLen[mainIndex]; j < matRowLen[mainIndex + 1]; j++)
            {
                int idx = matIdx[j];
                float val = matVals[j];
                mainVec[idx] = val;
            }
        }



       /// <summary>
       /// test equality of two arrays
       /// </summary>
       /// <param name="arr1"></param>
       /// <param name="arr2"></param>
       /// <param name="info"></param>
        public static void TestEquality(float[] arr1, float[] arr2,string info)
        {
            if (arr1.Length != arr2.Length)
                Console.WriteLine("Not the same, different sizes");

            bool passed = true;
            int errorCounter = 0;
            for (int i = 0; i < arr2.Length; i++)
            {
                float diff = Math.Abs(arr1[i] - arr2[i]);
                if (diff > ErrorEpsilon)
                {
                    errorCounter++;
                    if (errorCounter < 10)
                    {
                        Console.WriteLine("  !!eror good={0} - bad={1} diff={2}, position {3}",arr1[i],arr2[i], diff, i);
                        passed = false;
                    }
                    //break;
                }
            }


            string msg = passed ? "PASSED" : "FAIL";
            Console.WriteLine("{0} {1} errors={2}",info,msg,errorCounter);

        }

        public static void TestEquality(float[,] arr1, float[,] arr2)
        {
            if(arr1.Rank!=arr2.Rank)
                Console.WriteLine("Not the same, different rank");

            if (arr1.Length != arr2.Length)
                Console.WriteLine("Not the same, different sizes");

            bool passed = true;

            int nRows = arr1.GetUpperBound(0);
            int nCols = arr1.GetUpperBound(1);
            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    float diff = Math.Abs(arr1[i, j] - arr2[i, j]);
                    if (diff > ErrorEpsilon)
                    {
                        Console.WriteLine("     !! Not the same diff={0}, position {1}", diff, i);
                        passed = false;
                        break;
                    }
                }
            }

            string msg = passed ? "PASSED" : "FAIL";
            Console.WriteLine(msg);

        }
    }
}
