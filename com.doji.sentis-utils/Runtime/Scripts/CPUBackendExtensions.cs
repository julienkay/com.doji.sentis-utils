using Unity.Burst;
using Unity.InferenceEngine;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.InferenceEngine.CPUBackend;
using static Unity.InferenceEngine.CPUTensorData;

namespace Doji.AI {
    internal static class CPUBackendExtensions {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct ClipFloatJob : IJobParallelFor, IJobResourceDeclarationXO {
            public ReadOnlyMemResource X { get; set; }
            float* Xptr => (float*)X.ptr;
            public ReadWriteMemResource O { get; set; }
            float* Optr => (float*)O.ptr;

            public float maxV;
            public float minV;

            public void Execute(int threadIdx) {
                float v = Xptr[threadIdx];
                Optr[threadIdx] = math.min(maxV, math.max(v, minV));
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct ClipIntJob : IJobParallelFor, IJobResourceDeclarationXO {
            public ReadOnlyMemResource X { get; set; }
            int* Xptr => (int*)X.ptr;
            public ReadWriteMemResource O { get; set; }
            int* Optr => (int*)O.ptr;

            public int maxV;
            public int minV;

            public void Execute(int threadIdx) {
                int v = Xptr[threadIdx];
                Optr[threadIdx] = math.min(maxV, math.max(v, minV));
            }
        }

        public static void Clip(this CPUBackend backend, Tensor<float> X, Tensor<float> O, float min, float max) {
            var job = new ClipFloatJob();
            job.minV = min;
            job.maxV = max;
            job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
        }

        public static void Clip(this CPUBackend backend, Tensor<int> X, Tensor<int> O, int min, int max) {
            var job = new ClipIntJob();
            job.minV = min;
            job.maxV = max;
            job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
    }
}
