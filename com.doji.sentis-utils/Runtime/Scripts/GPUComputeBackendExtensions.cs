using Unity.InferenceEngine;
using UnityEngine;
using static Unity.InferenceEngine.ShaderPropertyID;
using static Unity.InferenceEngine.ComputeTensorData;

namespace Doji.AI {
    internal static class GPUComputeBackendExtensions {
        static class ComputeFunctions {
            static ComputeShader k_PointwiseUnaryGen = Resources.Load<ComputeShader>("ComputeShaders/Compute.Shaders.PointwiseUnary");

            public static ComputeFunction k_ClipFloat = new ComputeFunction(k_PointwiseUnaryGen, "ClipFloat");
            public static ComputeFunction k_ClipInt = new ComputeFunction(k_PointwiseUnaryGen, "ClipInt");
        }

        internal class ShaderPropertyID {
            public static readonly int k_ID__MinValue = Shader.PropertyToID("_MinValue");
        }

        public static void Clip(this GPUComputeBackend backend, Tensor<float> X, Tensor<float> O, float min, float max) {
            var fn = ComputeFunctions.k_ClipFloat;
            var cb = backend.GetCommandBuffer();
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, min);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, max);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        public static void Clip(this GPUComputeBackend backend, Tensor<int> X, Tensor<int> O, int min, int max) {
            var fn = ComputeFunctions.k_ClipInt;
            var cb = backend.GetCommandBuffer();
            cb.SetComputeIntParam(fn.shader, k_ID_alphai, min);
            cb.SetComputeIntParam(fn.shader, k_ID_betai, max);
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }
    }
}
