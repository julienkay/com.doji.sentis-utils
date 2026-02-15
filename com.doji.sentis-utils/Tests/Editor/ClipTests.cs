using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Editor.Tests {
    public class ClipTests {
        private static readonly BackendType[] SupportedBackends =
        {
            BackendType.CPU,
            BackendType.GPUCompute
        };

        // ---------------------------------------------------------
        // FLOAT TESTS
        // ---------------------------------------------------------

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestClipFloatBasic(BackendType backend) {
            using Ops ops = new Ops(backend);

            TensorShape shape = new TensorShape(6);
            using Tensor<float> input = new Tensor<float>(shape,
                new float[] { -5f, -1f, 0f, 1f, 5f, 10f });

            var result = ops.Clip(input, 0f, 5f);

            ops.ExecuteCommandBufferAndClear();
            result.ReadbackAndClone();

            Assert.That(result.shape, Is.EqualTo(shape));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new float[]
            {
                0f, 0f, 0f, 1f, 5f, 5f
            }));

            result.Dispose();
        }

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestClipFloatNoChange(BackendType backend) {
            using Ops ops = new Ops(backend);

            TensorShape shape = new TensorShape(4);
            using Tensor<float> input = new Tensor<float>(shape,
                new float[] { 1f, 2f, 3f, 4f });

            var result = ops.Clip(input, 0f, 10f);

            ops.ExecuteCommandBufferAndClear();
            result.ReadbackAndClone();

            Assert.That(result.DownloadToArray(), Is.EqualTo(new float[]
            {
                1f, 2f, 3f, 4f
            }));

            result.Dispose();
        }

        // ---------------------------------------------------------
        // INT TESTS
        // ---------------------------------------------------------

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestClipIntBasic(BackendType backend) {
            using Ops ops = new Ops(backend);

            TensorShape shape = new TensorShape(6);
            using Tensor<int> input = new Tensor<int>(shape,
                new int[] { -5, -1, 0, 1, 5, 10 });

            var result = ops.Clip(input, 0, 5);

            ops.ExecuteCommandBufferAndClear();
            result.ReadbackAndClone();

            Assert.That(result.shape, Is.EqualTo(shape));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new int[]
            {
                0, 0, 0, 1, 5, 5
            }));

            result.Dispose();
        }

        // ---------------------------------------------------------
        // MULTI DIMENSION
        // ---------------------------------------------------------

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestClip2DShapePreserved(BackendType backend) {
            using Ops ops = new Ops(backend);

            TensorShape shape = new TensorShape(2, 3);
            using Tensor<int> input = new Tensor<int>(shape,
                new int[] { -2, 0, 2, 4, 6, 8 });

            var result = ops.Clip(input, 1, 5);

            ops.ExecuteCommandBufferAndClear();
            result.ReadbackAndClone();

            Assert.That(result.shape, Is.EqualTo(shape));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new int[]
            {
                1, 1, 2, 4, 5, 5
            }));

            result.Dispose();
        }

        // ---------------------------------------------------------
        // ZERO DIM EDGE CASE (important for your implementation)
        // ---------------------------------------------------------

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestClipZeroDimTensor(BackendType backend) {
            using Ops ops = new Ops(backend);

            TensorShape shape = new TensorShape(0);
            using Tensor<float> input = new Tensor<float>(shape, new float[0]);

            var result = ops.Clip(input, -1f, 1f);

            // No execute necessary — backend should not run
            Assert.That(result.shape, Is.EqualTo(shape));

            result.Dispose();
        }
    }
}
