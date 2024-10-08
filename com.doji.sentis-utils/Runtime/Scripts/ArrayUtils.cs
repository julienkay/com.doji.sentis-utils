using System;

namespace Doji.AI {

    public static class ArrayUtils {

        /// <summary>
        /// Return samples from the �standard normal� distribution.
        /// (Gaussian distribution of mean 0 and variance 1.)
        /// </summary>
        public static float[] Randn(int size, double mean = 0.0d, double stdDev = 1.0, Random random = null) {
            random ??= new Random();
            float[] randomArray = new float[size];

            for (int i = 0; i < size; i++) {
                randomArray[i] = (float)random.SampleGaussian(mean, stdDev);
            }

            return randomArray;
        }

        private static double SampleGaussian(this Random random, double mean = 0, double stdDev = 1) {
            double u1 = 1 - random.NextDouble();
            double u2 = 1 - random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev + mean;
        }

        /// <summary>
        /// Convenience method to access <see cref="Array.Reverse(Array)"/>.
        /// Reverses the array in-place and returns the original array.
        /// </summary>
        public static T[] Reverse<T>(this T[] array) {
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            Array.Reverse(array);
            return array;
        }

        /// <summary>
        /// Concatenates multiple arrays of the specified generic type into a new array.
        /// </summary>
        public static T[] Concatenate<T>(params T[][] arrays) {
            int totalLength = 0;
            foreach (T[] array in arrays) {
                totalLength += array.Length;
            }

            T[] result = new T[totalLength];

            int currentIndex = 0;
            foreach (T[] array in arrays) {
                Array.Copy(array, 0, result, currentIndex, array.Length);
                currentIndex += array.Length;
            }

            return result;
        }

        /// <summary>
        /// Concatenates two arrays of the specified generic type into a new array.
        /// </summary>
        public static T[] Concatenate<T>(this T[] array1, T[] array2) {
            if (array1 == null) {
                throw new ArgumentNullException(nameof(array1));
            }

            if (array2 == null) {
                throw new ArgumentNullException(nameof(array2));
            }

            T[] resultArray = new T[array1.Length + array2.Length];
            Array.Copy(array1, resultArray, array1.Length);
            Array.Copy(array2, 0, resultArray, array1.Length, array2.Length);
            return resultArray;
        }

        public static T[] Concatenate<T>(this T[] array1, T value) {
            if (array1 == null) {
                throw new ArgumentNullException(nameof(array1));
            }

            T[] resultArray = new T[array1.Length + 1];
            Array.Copy(array1, resultArray, array1.Length);
            resultArray[resultArray.Length - 1] = value;
            return resultArray;
        }

        public static T Max<T>(this T[] array) where T : IComparable {
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            if (array.Length <= 1) {
                throw new ArgumentException("Number of elements must be greater than 1.");
            }
            T max = array[0];
            for (int i = 1; i < array.Length; i++) {
                if (array[i].CompareTo(max) > 0) {
                    max = array[i];
                }
            }
            return max;
        }

        /// <summary>
        /// Takes a array, repeats it returns the repeated sequence as a new array.
        /// numpy.tile
        /// </summary>
        public static T[] Tile<T>(this T[] array, int repeats) {
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            if (repeats <= 0) {
                throw new ArgumentException("Repeat count must be greater than zero.", nameof(repeats));
            }

            T[] repeatedArray = new T[array.Length * repeats];

            for (int i = 0; i < repeats; i++) {
                Array.Copy(array, 0, repeatedArray, i * array.Length, array.Length);
            }

            return repeatedArray;
        }

        /// <summary>
        /// Repeat each element of an array after themselves
        /// numpy.repeat
        /// </summary>
        public static T[] Repeat<T>(this T[] array, int repeats) {
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            if (repeats <= 0) {
                throw new ArgumentException("Number of repeats must be greater than zero.", nameof(repeats));
            }

            int length = array.Length;
            T[] result = new T[length * repeats];

            for (int i = 0; i < length; i++) {
                T item = array[i];
                for (int j = 0; j < repeats; j++) {
                    result[i * repeats + j] = item;
                }
            }

            return result;
        }

        /// <summary>
        /// Performs element-wise addition on two arrays, producing a new array with the result.
        /// </summary>
        public static int[] Add(this int[] a, int[] b) {
            if (a.Length != b.Length) {
                throw new ArgumentException("Arrays must have the same length");
            }

            int[] result = new int[a.Length];

            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] + b[i];
            }

            return result;
        }

        /// <inheritdoc cref="Add(int[], int[])"/>
        public static float[] Add(this float[] a, float[] b) {
            if (a.Length != b.Length) {
                throw new ArgumentException("Arrays must have the same length");
            }

            float[] result = new float[a.Length];

            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] + b[i];
            }

            return result;
        }

        public static float[] Sub(this float[] a, float[] b) {
            if (a.Length != b.Length) {
                throw new ArgumentException("Arrays must have the same length");
            }

            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] - b[i];
            }
            return result;
        }

        public static float[] Sub(this float[] a, float b) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] - b;
            }
            return result;
        }

        public static float[] Sub(float a, float[] b) {
            float[] result = new float[b.Length];
            for (int i = 0; i < b.Length; i++) {
                result[i] = a - b[i];
            }
            return result;
        }

        public static float[] Div(this float[] a, float b) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] / b;
            }
            return result;
        }

        public static float[] Div(this float[] a, float[] b) {
            if (a.Length != b.Length) {
                throw new ArgumentException("Arrays must have the same length");
            }

            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] / b[i];
            }
            return result;
        }

        public static float[] Mul(this float[] a, float b) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] * b;
            }
            return result;
        }

        public static float[] Pow(this float[] a, float b) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = MathF.Pow(a[i], b);
            }
            return result;
        }

        public static float[] Log(this float[] a) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = MathF.Log(a[i]);
            }
            return result;
        }

        public static float[] Exp(this float[] a) {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = MathF.Exp(a[i]);
            }
            return result;
        }

        public static int[] Floor(this float[] a) {
            int[] result = new int[a.Length];
            for (int i = 0; i < a.Length; i++) {
                result[i] = (int)MathF.Floor(a[i]);
            }
            return result;
        }

        public static float[] Gather(this float[] a, int[] indices) {
            if (a.Length != indices.Length) {
                throw new ArgumentException("Array sizes do not match.");
            }
            float[] result = new float[indices.Length];
            for (int i = 0; i < indices.Length; i++) {
                result[i] = a[indices[i]];
            }
            return result;
        }

        /// <summary>
        /// numpy.full
        /// </summary>
        public static T[] Full<T>(int size, T fillValue) {
            if (size < 0) {
                throw new ArgumentException("Size must be non-negative.");
            }

            T[] array = new T[size];

            for (int i = 0; i < size; i++) {
                array[i] = fillValue;
            }

            return array;
        }

        /// <summary>
        /// numpy.cumprod
        /// </summary>
        public static float[] CumProd(this float[] array) {
            int length = array.Length;
            float[] result = new float[length];
            float product = 1.0f;

            for (int i = 0; i < length; i++) {
                product *= array[i];
                result[i] = product;
            }

            return result;
        }

        public static float[] Sqrt(this float[] array) {
            int length = array.Length;
            float[] result = new float[length];

            for (int i = 0; i < length; i++) {
                result[i] = MathF.Sqrt(result[i]);
            }

            return result;
        }

        /// <summary>
        /// numpy.arange
        /// </summary>
        public static int[] Arange(int start, int stop, int step = 1) {
            if (step <= 0) {
                throw new ArgumentException("Step must be a positive integer.");
            }

            int length = ((stop - start - 1) / step) + 1;
            int[] result = new int[length];

            for (int i = 0, value = start; i < length; i++, value += step) {
                result[i] = value;
            }

            return result;
        }

        public static float[] ArangeF(int start, int stop, int step = 1) {
            if (step <= 0) {
                throw new ArgumentException("Step must be a positive integer.");
            }

            int length = ((stop - start - 1) / step) + 1;
            float[] result = new float[length];

            for (int i = 0, value = start; i < length; i++, value += step) {
                result[i] = value;
            }

            return result;
        }

        /// <summary>
        /// numpy.linspace
        /// </summary>
        public static float[] Linspace(float start, float stop, int num, bool endpoint = true) {
            if (num <= 1) {
                throw new ArgumentException("Number of elements must be greater than 1.");
            }

            float[] result = new float[num];
            float step;
            if (endpoint) {
                step = (stop - start) / (num - 1);
            } else {
                step = (stop - start) / num;
            }

            for (int i = 0; i < num; i++) {
                result[i] = start + step * i;
            }

            return result;
        }

        public static float[] Interpolate(float[] x, float[] xp, float[] fp, float? left = null, float? right = null) {
            int n = xp.Length;
            int m = x.Length;
            float[] interpolatedValues = new float[m];

            left ??= fp[0];
            right ??= fp[^1];

            for (int i = 0; i < m; i++) {
                if (x[i] <= xp[0]) {
                    interpolatedValues[i] = left.Value;
                } else if (x[i] >= xp[n - 1]) {
                    interpolatedValues[i] = right.Value;
                } else {
                    int j = Array.BinarySearch(xp, x[i]);
                    if (j < 0) {
                        j = ~j;
                        float x0 = xp[j - 1];
                        float x1 = xp[j];
                        float f0 = fp[j - 1];
                        float f1 = fp[j];
                        interpolatedValues[i] = f0 + (f1 - f0) * (x[i] - x0) / (x1 - x0);
                    } else {
                        interpolatedValues[i] = fp[j];
                    }
                }
            }

            return interpolatedValues;
        }

        public static bool ArrayEqual<T>(this T[] array1, T[] array2) {
            if (array1 == null || array2 == null)
                return false;

            if (ReferenceEquals(array1, array2))
                return true;

            if (array1.Length != array2.Length)
                return false;

            for (int i = 0; i < array1.Length; i++) {
                if (!array1[i].Equals(array2[i]))
                    return false;
            }

            return true;
        }
    }
}