using NUnit.Framework;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Editor.Tests {

    public class OpsTest {
        [Test]
        public void Test() {
            using Ops ops = new Ops(BackendType.GPUCompute);
            int[] data = new int[] { 2, 2, 3, 4, 5, 6 };
            Debug.Log(string.Join(", ", data));
            TensorShape shape = new TensorShape(3, 2);
            using Tensor<int> input = new Tensor<int>(shape, data);
            using var r = ops.Add(input, 1);
            ops.ExecuteCommandBufferAndClear();
            var b = ops._backend;
            Debug.Log(r.dataOnBackend);
            Assert.That(r.shape, Is.EqualTo(shape));
            using var r2 = r.ReadbackAndClone();
            Debug.Log(string.Join(", ", r2.DownloadToArray()));
            //CollectionAssert.AreEqual(new int[] { 2, 3, 4, 5, 6, 7 }, r.DownloadToArray());
        }
    }
}