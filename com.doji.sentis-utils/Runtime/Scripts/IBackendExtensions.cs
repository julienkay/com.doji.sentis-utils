using System;
using Unity.InferenceEngine;

namespace Doji.AI {
    internal static class IBackendExtensions {
       
        public static void Clip(this IBackend backend, Tensor<float> X, Tensor<float> O, float min, float max) {
            switch (backend) {
                case CPUBackend cpu:
                    cpu.Clip(X, O, min, max);
                    break;

                case GPUComputeBackend gpu:
                    gpu.Clip(X, O, min, max);
                    break;

                default:
                    throw new NotSupportedException(
                        $"Scalar {nameof(Clip)} Op not supported on '{backend.GetType().Name}' backend.");
            }
        }

        public static void Clip(this IBackend backend, Tensor<int> X, Tensor<int> O, int min, int max) {
            switch (backend) {
                case CPUBackend cpu:
                    cpu.Clip(X, O, min, max);
                    break;

                case GPUComputeBackend gpu:
                    gpu.Clip(X, O, min, max);
                    break;

                default:
                    throw new NotSupportedException(
                        $"Scalar {nameof(Clip)} Op not supported on '{backend.GetType().Name}' backend.");
            }
        }
    }
}
