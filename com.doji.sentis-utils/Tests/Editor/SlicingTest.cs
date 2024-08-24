using NUnit.Framework;
using Unity.Sentis;

namespace Doji.AI.Editor.Tests {
    public class SlicingTests {

        [Test]
        public void TestSlicingShape1D() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            TensorShape shape = new TensorShape(3);
            using Tensor<int> tensor = GetTensor(shape);
            var result = ops.Slice(tensor, 1);
            result.ReadbackAndClone();
            Assert.That(result.shape, Is.EqualTo(new TensorShape()));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new int[] { 2 }));
        }

        [Test]
        public void TestSlicingShape2D() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            TensorShape shape = new TensorShape(2, 3);
            using Tensor<int> tensor = GetTensor(shape);
            var result = ops.Slice(tensor, .., 1);
            result.ReadbackAndClone();
            Assert.That(result.shape, Is.EqualTo(new TensorShape(2)));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new int[] { 2, 5 }));
        }

        [Test]
        public void TestSlicingShape3D() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            TensorShape shape = new TensorShape(2, 3, 4);
            using Tensor<int> tensor = GetTensor(shape);
            var result = ops.Slice(tensor, .., 1, ..);
            result.ReadbackAndClone();
            Assert.That(result.shape, Is.EqualTo(new TensorShape(2, 4)));
            Assert.That(result.DownloadToArray(), Is.EqualTo(new int[] { 5, 6, 7, 8, 17, 18, 19, 20 }));
        }

        private Tensor<int> GetTensor(TensorShape shape) {
            return new Tensor<int>(shape, ArrayUtils.Arange(1, shape.length + 1));
        }
    }
}
