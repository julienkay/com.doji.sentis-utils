using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI {

    public class Ops : IDisposable {

        public BackendType backendType { get; private set; }

        internal IBackend _backend;

        private readonly HashSet<Tensor> _pool = new HashSet<Tensor>();

        public Ops(BackendType backend) {
            backendType = backend;
            _backend = WorkerFactory.CreateBackend(backend);
        }

        public void Dispose() {
            FlushTensors();
            _backend.Dispose();
        }

        public Tensor TakeOwnership(Tensor tensor) {
            if (tensor == null) {
                throw new ArgumentNullException(nameof(tensor), "The tensor to take ownership of was null");
            }
            if (!_pool.Contains(tensor)) {
                UnityEngine.Debug.LogWarning($"Unable to find Tensor {tensor} in the temporary pool of Ops tensors. Maybe it was already disposed?");
                return null;
            }
            return _pool.Remove(tensor) ? tensor : null;
        }

        public void FlushTensors() {
            foreach (Tensor t in _pool) {
                t.Dispose();
            }
            _pool.Clear();
        }

        private bool AddToPool(Tensor tensor) {
            if (tensor == null) {
                throw new ArgumentNullException("The tensor to add to the pool was null.");
            }
            return _pool.Add(tensor);
        }

        internal Tensor AllocNoData(DataType dataType, TensorShape shape) {
            switch (dataType) {
                case DataType.Float:
                    return TensorFloatAllocNoData(shape);
                case DataType.Int:
                    return TensorIntAllocNoData(shape);
                default:
                    throw new NotImplementedException();
            }
        }

        internal Tensor AllocNoData(TensorShape shape, DataType dataType) {
            return AllocNoData(dataType, shape);
        }

        internal TensorFloat TensorFloatAllocNoData(TensorShape shape) {
            var tensor = TensorFloat.AllocNoData(shape);
            AddToPool(tensor);
            return tensor;
        }

        internal TensorInt TensorIntAllocNoData(TensorShape shape) {
            var tensor = TensorInt.AllocNoData(shape);
            AddToPool(tensor);
            return tensor;
        }

        public TensorFloat NewTensorFloat(TensorShape shape, float[] data) {
            var tensor = new TensorFloat(shape, data);
            AddToPool(tensor);
            return tensor;
        }

        public TensorInt NewTensorInt(TensorShape shape, int[] data) {
            var tensor = new TensorInt(shape, data, 0);
            AddToPool(tensor);
            return tensor;
        }

        public TensorFloat Max(TensorFloat tensor1, TensorFloat tensor2) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(tensor1, tensor2));
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Max(tensor1, tensor2, O);
            return O;
        }

        public TensorFloat Min(TensorFloat tensor1, TensorFloat tensor2) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(tensor1, tensor2));
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Min(tensor1, tensor2, O);
            return O;
        }

        public TensorFloat Mul(TensorFloat A, TensorFloat B) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Mul(A, B, O);
            return O;
        }

        public TensorInt Mul(TensorInt A, TensorInt B) {
            var O = TensorIntAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Mul(A, B, O);
            return O;
        }

        public TensorFloat Mul(TensorFloat A, float b) {
            var O = TensorFloatAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, b, 0);
            return O;
        }

        public TensorFloat Mul(float a, TensorFloat B) {
            var O = TensorFloatAllocNoData(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, a, 0);
            return O;
        }

        public TensorFloat Sub(TensorFloat A, float b) {
            var O = TensorFloatAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, -b);
            return O;
        }

        public TensorFloat Sub(float a, TensorFloat B) {
            var O = TensorFloatAllocNoData(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, -1, a);
            return O;
        }

        public TensorFloat Sub(TensorFloat A, TensorFloat B) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sub(A, B, O);
            return O;
        }

        public TensorInt Sub(TensorInt A, int b) {
            var O = TensorIntAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, -b);
            return O;
        }

        public TensorInt Sub(int a, TensorInt B) {
            var O = TensorIntAllocNoData(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, -1, a);
            return O;
        }

        public TensorInt Sub(TensorInt A, TensorInt B) {
            var O = TensorIntAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sub(A, B, O);
            return O;
        }

        public TensorFloat Add(TensorFloat A, float b) {
            var O = TensorFloatAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, b);
            return O;
        }

        public TensorFloat Add(TensorFloat A, TensorFloat B) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Add(A, B, O);
            return O;
        }
        public TensorInt Add(TensorInt A, int b) {
            var O = TensorIntAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, b);
            return O;
        }

        public TensorInt Add(TensorInt A, TensorInt B) {
            var O = TensorIntAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Add(A, B, O);
            return O;
        }

        public TensorFloat Div(TensorFloat A, float b) {
            var O = TensorFloatAllocNoData(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1 / b, 0);
            return O;
        }

        public TensorFloat Div(TensorFloat A, TensorFloat B) {
            var O = TensorFloatAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Div(A, B, O);
            return O;
        }

        public TensorFloat Sqrt(TensorFloat X) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sqrt(X, O);
            return O;
        }

        public TensorFloat Clip(TensorFloat X, float min, float max) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Clip(X, O, min, max);
            return O;
        }

        public TensorInt Clip(TensorInt X, int min, int max) {
            var O = TensorIntAllocNoData(X.shape);
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Clip(X, O, min, max);
            return O;
        }

        public TensorFloat Mad(TensorFloat X, float s, float b) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.ScalarMad(X, O, s, b);
            return O;
        }

        public TensorFloat Abs(TensorFloat X) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Abs(X, O);
            return O;
        }

        public TensorFloat RandomNormal(TensorShape S, float mean, float scale, int seed) {
            var O = TensorFloatAllocNoData(S);
            if (O.shape.HasZeroDims())
                return O;
            _backend.RandomNormal(O, mean, scale, seed);
            return O;
        }

        public TensorFloat RandomNormal(TensorShape S, float mean, float scale, uint seed) {
            var O = TensorFloatAllocNoData(S);
            if (O.shape.HasZeroDims())
                return O;
            _backend.RandomNormal(O, mean, scale, unchecked((int)seed));
            return O;
        }

        public T Reshape<T>(T X, TensorShape shape) where T : Tensor {
            var O = AllocNoData(shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Reshape(X, O);
            return O;
        }

        public T Transpose<T>(T X) where T : Tensor {
            var O = AllocNoData(X.shape.Transpose(), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Transpose(X, O);
            return O;
        }

        public T Transpose<T>(T X, int[] permutations) where T : Tensor {
            var O = AllocNoData(X.shape.Transpose(permutations), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Transpose(X, O, permutations);
            return O;
        }

        public Tensor Concat(Tensor[] tensors, int axis) {
            var O = AllocNoData(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            int start = 0;
            foreach (var tensor in tensors) {
                _backend.SliceSet(tensor, O, axis, start, 1);
                start += tensor.shape[axis];
            }
            return O;
        }

        public Tensor Concat(List<Tensor> tensors, int axis) {
            var O = AllocNoData(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            int start = 0;
            foreach (var tensor in tensors) {
                _backend.SliceSet(tensor, O, axis, start, 1);
                start += tensor.shape[axis];
            }
            return O;
        }

        /*public Tensor Concat(Tensor tensor1, Tensor tensor2, int axis) {
            var O = AllocNoData(TensorShapeHelper.ConcatShape(tensor1, tensor2, axis), tensor1.dataType);
            if (O.shape.HasZeroDims())
                return O;

            _backend.SliceSet(tensor1, O, axis, ..., ...);
            _backend.SliceSet(tensor2, O, axis, ..., ...);
            return O;
        }*/

        public T Split<T>(T X, int axis, int start = 0, int end = int.MaxValue) where T : Tensor {
            var O = AllocNoData(X.shape.Split(axis, start, end), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Split(X, O, axis, start);
            return O;
        }

        internal T Slice<T>(T X, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps) where T : Tensor {
            var O = AllocNoData(X.shape.Slice(starts, ends, axes, steps), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Slice(X, O, starts, axes, steps);
            return O;
        }

        public TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B) {
            var O = TensorIntAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.GreaterOrEqual(A, B, O);
            return O;
        }

        public TensorInt GreaterOrEqual(TensorInt A, TensorInt B) {
            var O = TensorIntAllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.GreaterOrEqual(A, B, O);
            return O;
        }

        public TensorFloat Softmax(TensorFloat X, int axis = -1) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Softmax(X, O, axis);
            return O;
        }

        public TensorFloat CumSum(TensorFloat X, int axis, bool reverse = false, bool exclusive = false) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        public TensorInt CumSum(TensorInt X, int axis, bool reverse = false, bool exclusive = false) {
            var O = TensorIntAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        public TensorFloat Neg(TensorFloat X) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Neg(X, O);
            return O;
        }

        public TensorInt Neg(TensorInt X) {
            var O = TensorIntAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Neg(X, O);
            return O;
        }

        public T Tile<T>(T X, ReadOnlySpan<int> repeats) where T : Tensor {
            var O = AllocNoData(X.shape.Tile(repeats), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Tile(X, O, repeats);
            return O;
        }

        public T Gather<T>(T X, TensorInt indices, int axis) where T : Tensor {
            var O = AllocNoData(ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Gather(X, indices, O, axis);
            return O;
        }

        public T GatherElements<T>(T X, TensorInt indices, int axis) where T : Tensor {
            var O = AllocNoData(X.dataType, X.shape) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.GatherElements(X, indices, O, axis);
            return O;
        }

        public TensorInt ArgMax(TensorFloat X, int axis, bool selectLastIndex = false) {
            var O = TensorIntAllocNoData(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMax(X, O, axis, selectLastIndex);
            return O;
        }

        public TensorInt ArgMax(TensorInt X, int axis, bool selectLastIndex = false) {
            var O = TensorIntAllocNoData(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMax(X, O, axis, selectLastIndex);
            return O;
        }

        public TensorInt ArgMin(TensorFloat X, int axis, bool selectLastIndex) {
            var O = TensorIntAllocNoData(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMin(X, O, axis, selectLastIndex);
            return O;
        }

        public TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex) {
            var O = TensorIntAllocNoData(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMin(X, O, axis, selectLastIndex);
            return O;
        }

        public TensorInt Cast(TensorFloat X) {
            var O = TensorIntAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Cast(X, O);
            return O;
        }

        public TensorFloat Cast(TensorInt X) {
            var O = TensorFloatAllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Cast(X, O);
            return O;
        }

        public (TensorFloat values, TensorInt indices) TopK(TensorFloat X, int k, int axis, bool largest, bool sorted) {
            var outputShape = new TensorShape(X.shape);
            outputShape[axis] = k;

            var values = TensorFloatAllocNoData(outputShape);
            var indices = TensorIntAllocNoData(outputShape);
            if (!outputShape.HasZeroDims())
                _backend.TopK(X, values, indices, k, axis, largest);
            return (values, indices);
        }

        public TensorFloat Copy(Tensor tensor) {
            var copy = TensorFloatAllocNoData(tensor.shape);
            _backend.MemCopy(tensor, copy);
            return copy;
        }
    }
}