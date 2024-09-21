using NUnit.Framework;
using System;
using Unity.Sentis;

namespace Doji.AI.Editor.Tests {

    public class SentisUtilsTest {
        
        private float[] Samples {
            get {
                return TestUtils.LoadFromFile("256_latents");
            }
        }

        private float[] ExpectedQuantile {
            get {
                return TestUtils.LoadFromFile("quantile_test_result_256");
            }
        }

        private float[] ExpectedSorted{
            get {
                return TestUtils.LoadFromFile("sort_test_result_256");
            }
        }

        [Test]
        public void TestQuantile() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            using Tensor<float> latents = new Tensor<float>(new TensorShape(1, 4, 8, 8), Samples);
            Tensor<float> quantile = ops.Quantile(latents, 0.995f, 1);
            ops.ExecuteCommandBufferAndClear();
            quantile.ReadbackAndClone();
            CollectionAssert.AreEqual(ExpectedQuantile, quantile.DownloadToArray(), new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestSort() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            using Tensor<float> latents = new Tensor<float>(new TensorShape(1, 4, 8, 8), Samples);
            Tensor<float> sorted = ops.Sort(latents, 1);
            ops.ExecuteCommandBufferAndClear();
            sorted.ReadbackAndClone();
            CollectionAssert.AreEqual(ExpectedSorted, sorted.DownloadToArray(), new FloatArrayComparer(0.00001f));
        }

        [Test]
        public void TestRepeatInterlave1D() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            int[] data = new int[] { 1, 2, 3 };
            TensorShape shape = new TensorShape(data.Length);
            using Tensor<int> input = new Tensor<int>(shape, data);
            Tensor<int> r = ops.RepeatInterleave(input, 2, 0);
            ops.ExecuteCommandBufferAndClear();
            Assert.That(r.shape, Is.EqualTo(new TensorShape(3 * 2)));
            r.ReadbackAndClone();
            CollectionAssert.AreEqual(new int[] { 1, 1, 2, 2, 3, 3 }, r.DownloadToArray());
        }

        [Test]
        public void TestRepeatInterlave2D() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            int[] data = new int[] { 1, 2, 3, 4, 5, 6 };
            TensorShape shape = new TensorShape(3, 2);
            using Tensor<int> input = new Tensor<int>(shape, data);
            Tensor<int> r = ops.RepeatInterleave(input, 2, 0);
            ops.ExecuteCommandBufferAndClear();
            Assert.That(r.shape, Is.EqualTo(new TensorShape(6 * 2)));
            r.ReadbackAndClone();
            UnityEngine.Debug.Log(string.Join(", ", r.DownloadToArray()));
            CollectionAssert.AreEqual(new int[] { 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6 }, r.DownloadToArray());
        }

        /*
        [Test]
        public void TestSlicing() {
            using IBackend backend = WorkerFactory.CreateBackend(BackendType.GPUCompute);
            int[] data = ArrayUtils.Arange(0, 60);
            TensorShape shape = new TensorShape(3, 4, 5);
            using Tensor<int> tensor = new Tensor<int>(shape, data);
            ReadOnlySpan<int> starts = new int[] { 0, -1, 0 };
            ReadOnlySpan<int> ends = new int[] { shape[0], shape[1], shape[2] };
            ReadOnlySpan<int> axes = new int[] { 0, 1, 2 };
            ReadOnlySpan<int> steps = new int[] { 1, 1, 1 };
            using Tensor<int> O = Tensor<int>.AllocNoData(tensor.shape.Slice(starts, ends, axes, steps));
            backend.Slice(tensor, O, starts, axes, steps);
            O.ReadbackAndClone();
            UnityEngine.Debug.Log(string.Join(", ", O.DownloadToArray()));
            CollectionAssert.AreEqual(new int[] { 15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59 }, O.DownloadToArray());
        }
        */

        [Test]
        public void TestSlicingCustom() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            int[] data = ArrayUtils.Arange(0, 60);
            TensorShape shape = new TensorShape(3, 4, 5);
            using Tensor<int> tensor = new Tensor<int>(shape, data);

            var x = ..;
            var y = ^1;
            using Tensor<int> O = ops.Slice(tensor, .., ^1, ..);
            ops.ExecuteCommandBufferAndClear();
            O.ReadbackAndClone();
            UnityEngine.Debug.Log(string.Join(", ", O.DownloadToArray()));
            CollectionAssert.AreEqual(new int[] { 15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59 }, O.DownloadToArray());
        }

        [Test]
        public void TestConcat() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            int[] data = ArrayUtils.Arange(0, 60);
            TensorShape shape = new TensorShape(3, 4, 5);
            using Tensor<int> tensor = new Tensor<int>(shape, data);

            int[] data2 = ArrayUtils.Arange(0, 15);
            TensorShape shape2 = new TensorShape(3, 1, 5);
            using Tensor<int> tensor2 = new Tensor<int>(shape2, data2);

            int[] data3 = ArrayUtils.Arange(0, 120);
            TensorShape shape3 = new TensorShape(3, 8, 5);
            using Tensor<int> tensor3 = new Tensor<int>(shape3, data3);

            var t = new Tensor[] { tensor, tensor2, tensor3 };
            using Tensor<int> O = ops.Concat(t, 1) as Tensor<int>;
            ops.ExecuteCommandBufferAndClear();
            O.ReadbackAndClone();

            var expected = new int[] {
                0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                14,  15,  16,  17,  18,  19,   0,   1,   2,   3,   4,   0,   1,   2,
                3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
                17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
                31,  32,  33,  34,  35,  36,  37,  38,  39,  20,  21,  22,  23,  24,
                25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                39,   5,   6,   7,   8,   9,  40,  41,  42,  43,  44,  45,  46,  47,
                48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
                62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                76,  77,  78,  79,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
                50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  10,  11,  12,  13,
                14,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 };
            CollectionAssert.AreEqual(expected, O.DownloadToArray());
        }
    }
}