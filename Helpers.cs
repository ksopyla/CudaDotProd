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

        public static void TestEquality(float[] arr1, float[] arr2)
        {
            if (arr1.Length != arr2.Length)
                Console.WriteLine("Not the same, different sizes");

            bool passed = true;
            for (int i = 0; i < arr2.Length; i++)
            {
                float diff = Math.Abs(arr1[i] - arr2[i]);
                if (diff > ErrorEpsilon)
                {
                    Console.WriteLine("     !! Not the same diff={0}, position {1}", diff, i);
                    passed = false;
                    //break;
                }
            }

            string msg = passed ? "PASSED" : "FAIL";
            Console.WriteLine(msg);

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
