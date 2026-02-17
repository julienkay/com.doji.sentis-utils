using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Editor.Tests {
    public class SlicingTests {
        private static readonly BackendType[] SupportedBackends =
        {
            BackendType.CPU,
            BackendType.GPUCompute,
            //BackendType.GPUPixel // slicing requires reshape
        };

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestSlicingShape1D(BackendType backend) {
            using Ops ops = new Ops(backend);
            TensorShape shape = new TensorShape(3);
            using Tensor<int> tensor = GetTensor(shape);

            var result = ops.Slice(tensor, 1);
            ops.ExecuteCommandBufferAndClear();

            Assert.That(result.shape, Is.EqualTo(new TensorShape()));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 2 }));
            result.Dispose();

        }

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestSlicingShape2D(BackendType backend) {
            using Ops ops = new Ops(backend);
            TensorShape shape = new TensorShape(2, 3);
            using Tensor<int> tensor = GetTensor(shape);

            var result = ops.Slice(tensor, .., 1);
            ops.ExecuteCommandBufferAndClear();

            Assert.That(result.shape, Is.EqualTo(new TensorShape(2)));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new[] { 2, 5 }));
            result.Dispose();
        }

        [TestCaseSource(nameof(SupportedBackends))]
        public void TestSlicingShape3D(BackendType backend) {
            using Ops ops = new Ops(backend);
            TensorShape shape = new TensorShape(2, 3, 4);
            using Tensor<int> tensor = GetTensor(shape);

            var result = ops.Slice(tensor, .., 1, ..);
            ops.ExecuteCommandBufferAndClear();

            Assert.That(result.shape, Is.EqualTo(new TensorShape(2, 4)));
            Assert.That(
                result.DownloadToArray(),
                Is.EqualTo(new[] { 5, 6, 7, 8, 17, 18, 19, 20 })
            );
            result.Dispose();

        }

        private Tensor<int> GetTensor(TensorShape shape) {
            return new Tensor<int>(
                shape,
                ArrayUtils.Arange(1, shape.length + 1)
            );
        }
    }
}
