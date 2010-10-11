using System;
using System.Collections.Generic;
using System.Text;

namespace TestDotProduct
{
   internal class Helpers
    {

       public static int RandomSeed = 1;
        public static int[] InitIndices(int i, int size, ref int maxIndex)
        {
            Random rnd = new Random(RandomSeed);
            int[] index = new int[size];

            int min = 0;
            int step = 10;
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
    }
}
