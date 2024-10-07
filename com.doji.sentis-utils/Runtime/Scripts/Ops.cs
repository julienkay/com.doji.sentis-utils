using System;
using System.Collections.Generic;
using Unity.Sentis;
using static Doji.AI.TensorShapeHelper;

namespace Doji.AI {

    public class Ops : IDisposable {

        public BackendType BackendType { get; private set; }

        internal IBackend _backend;

        private readonly HashSet<Tensor> _pool = new HashSet<Tensor>();

        public Ops(BackendType backendType) {
            BackendType = backendType;
            _backend = BackendFactory.CreateBackend(backendType);
        }

        public void Dispose() {
            FlushTensors();
        }

        public Tensor TakeOwnership(Tensor tensor) {
            if (tensor == null) {
                throw new ArgumentNullException(nameof(tensor), "The tensor to take ownership of was null");
            }
            if (!_pool.Contains(tensor)) {
                UnityEngine.Debug.LogWarning($"Unable to find Tensor {tensor} in the temporary pool of Ops tensors. Check if it hasn't already been disposed.");
                return null;
            }
            return _pool.Remove(tensor) ? tensor : null;
        }

        public bool WaveOwnership(Tensor tensor) {
            if (tensor == null) {
                throw new ArgumentNullException(nameof(tensor), "The tensor to wave ownership of was null");
            }
            if (_pool.Contains(tensor)) {
                UnityEngine.Debug.LogWarning($"The Tensor {tensor} was already present in the temporary pool of Ops tensors.");
                return false;
            }
            return _pool.Add(tensor);
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
                    return AllocNoData<float>(shape);
                case DataType.Int:
                    return AllocNoData<int>(shape);
                default:
                    throw new NotImplementedException();
            }
        }

        internal Tensor AllocNoData(TensorShape shape, DataType dataType) {
            return AllocNoData(dataType, shape);
        }

        public Tensor<T> AllocNoData<T>(TensorShape shape) where T : unmanaged {
            var tensor = new Tensor<T>(shape, null);
            AddToPool(tensor);
            return tensor;
        }

        internal Tensor<T> AllocZeros<T>(TensorShape shape) where T : unmanaged {
            var tensor = new Tensor<T>(shape);
            AddToPool(tensor);
            return tensor;
        }

        public Tensor<T> NewTensor<T>(TensorShape shape, T[] data) where T : unmanaged {
            var tensor = new Tensor<T>(shape, data);
            AddToPool(tensor);
            return tensor;
        }

        public Tensor<T> NewTensor<T>(T srcData) where T : unmanaged {
            var tensor = new Tensor<T>(new TensorShape(), new[] { srcData });
            AddToPool(tensor);
            return tensor;
        }

        public Tensor<T> Zeros<T>(TensorShape shape) where T : unmanaged {
            return AllocZeros<T>(shape);
        }

        public Tensor Ones(TensorShape shape, DataType type) {
            return type switch {
                DataType.Float => NewTensor<float>(shape, OnesF(shape.length)),
                DataType.Int => NewTensor<int>(shape, OnesI(shape.length)),
                DataType.Short or DataType.Byte => throw new NotImplementedException(),
                _ => throw new ArgumentException($"Invalid data type '{type}'"),
            };
        }

        public Tensor<T> Ones<T>(TensorShape shape) where T : unmanaged {
            switch (typeof(T)) {
                case Type floatType when floatType == typeof(float):
                    return NewTensor<float>(shape, OnesF(shape.length)) as Tensor<T>;
                case Type intType when intType == typeof(int):
                    return NewTensor<int>(shape, OnesI(shape.length)) as Tensor<T>;
                case Type shortType when shortType == typeof(short):
                case Type byteType when byteType == typeof(byte):
                    throw new NotImplementedException();
                default:
                    throw new ArgumentException($"Invalid tensor data type '{typeof(T)}'");
            }
        }

        private static int[] OnesI(int num) {
            int[] ones = new int[num];
            Array.Fill(ones, 1);
            return ones;
        }

        private static float[] OnesF(int num) {
            float[] ones = new float[num];
            Array.Fill(ones, 1);
            return ones;
        }

        public Tensor<float> Max(Tensor<float> tensor1, Tensor<float> tensor2) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(tensor1, tensor2));
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Max(tensor1, tensor2, O);
            return O;
        }

        private Model _max;
        private Model GetMax(Tensor tensor1, Tensor tensor2) {
            if (_max == null) {
                FunctionalGraph graph = new FunctionalGraph();
                FunctionalTensor x = graph.AddInput<float>(tensor1.shape);
                FunctionalTensor y = graph.AddInput<float>(new TensorShape(6));
                FunctionalTensor prod = x * y;
                FunctionalTensor reduce = Functional.ReduceSum(prod, dim: 0, keepdim: false);
                _max = graph.Compile(reduce, prod);
            }
            return _max;
        }

        public Tensor<float> Min(Tensor<float> tensor1, Tensor<float> tensor2) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(tensor1, tensor2));
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Min(tensor1, tensor2, O);
            return O;
        }

        public Tensor<float> ReduceMax(Tensor<float> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<float>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceMax(X, O, axes);
            return O;
        }

        public Tensor<int> ReduceMax(Tensor<int> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<int>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceMax(X, O, axes);
            return O;
        }

        public Tensor<float> ReduceMean(Tensor<float> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<float>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceMean(X, O, axes);
            return O;
        }

        public Tensor<float> ReduceMin(Tensor<float> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<float>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceMin(X, O, axes);
            return O;
        }

        public Tensor<int> ReduceMin(Tensor<int> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<int>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceMin(X, O, axes);
            return O;
        }

        public Tensor<float> ReduceSum(Tensor<float> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<float>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceSum(X, O, axes);
            return O;
        }

        public Tensor<int> ReduceSum(Tensor<int> X, ReadOnlySpan<int> axes) {
            var O = AllocNoData<int>(X.shape.Reduce(axes));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ReduceSum(X, O, axes);
            return O;
        }

        public Tensor<float> Mul(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Mul(A, B, O);
            return O;
        }

        public Tensor<int> Mul(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Mul(A, B, O);
            return O;
        }

        public Tensor<float> Mul(Tensor<float> A, float b) {
            var O = AllocNoData<float>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, b, 0);
            return O;
        }

        public Tensor<float> Mul(float a, Tensor<float> B) {
            var O = AllocNoData<float>(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, a, 0);
            return O;
        }

        public Tensor<float> Sub(Tensor<float> A, float b) {
            var O = AllocNoData<float>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, -b);
            return O;
        }

        public Tensor<float> Sub(float a, Tensor<float> B) {
            var O = AllocNoData<float>(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, -1, a);
            return O;
        }

        public Tensor<float> Sub(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sub(A, B, O);
            return O;
        }

        public Tensor<int> Sub(Tensor<int> A, int b) {
            var O = AllocNoData<int>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, -b);
            return O;
        }

        public Tensor<int> Sub(int a, Tensor<int> B) {
            var O = AllocNoData<int>(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(B, O, -1, a);
            return O;
        }

        public Tensor<int> Sub(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sub(A, B, O);
            return O;
        }

        public Tensor<float> Add(Tensor<float> A, float b) {
            var O = AllocNoData<float>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, b);
            return O;
        }

        public Tensor<float> Add(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Add(A, B, O);
            return O;
        }
        public Tensor<int> Add(Tensor<int> A, int b) {
            var O = AllocNoData<int>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1, b);
            return O;
        }

        public Tensor<int> Add(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Add(A, B, O);
            return O;
        }

        public Tensor<float> Div(Tensor<float> A, float b) {
            var O = AllocNoData<float>(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.ScalarMad(A, O, 1 / b, 0);
            return O;
        }

        public Tensor<float> Div(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<float>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Div(A, B, O);
            return O;
        }

        public Tensor<float> Sqrt(Tensor<float> X) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Sqrt(X, O);
            return O;
        }

        public Tensor<float> Clip(Tensor<float> X, float min, float max) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Clip(X, O, min, max);
            return O;
        }

        public Tensor<int> Clip(Tensor<int> X, int min, int max) {
            var O = AllocNoData<int>(X.shape);
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.Clip(X, O, min, max);
            return O;
        }

        public Tensor<float> Mad(Tensor<float> X, float s, float b) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims()) {
                return O;
            }
            _backend.ScalarMad(X, O, s, b);
            return O;
        }

        public Tensor<float> Abs(Tensor<float> X) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Abs(X, O);
            return O;
        }

        public Tensor<float> RandomNormal(TensorShape S, float mean, float scale, int seed) {
            var O = AllocNoData<float>(S);
            if (O.shape.HasZeroDims())
                return O;
            _backend.RandomNormal(O, mean, scale, seed);
            return O;
        }

        public Tensor<float> RandomNormal(TensorShape S, float mean, float scale, uint seed) {
            var O = AllocNoData<float>(S);
            if (O.shape.HasZeroDims())
                return O;
            _backend.RandomNormal(O, mean, scale, unchecked((int)seed));
            return O;
        }

        public T Where<T>(Tensor<int> C, T A, T B) where T : Tensor {
            var O = AllocNoData(A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Where(C, A, B, O);
            return O;
        }

        public T Reshape<T>(T X, TensorShape shape) where T : Tensor {
            var O = AllocNoData(shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Reshape(X, O);
            return O;
        }

        public T Expand<T>(T X, TensorShape shape) where T : Tensor {
            var O = AllocNoData(X.shape.Broadcast(shape), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Expand(X, O);
            return O;
        }

        public T Transpose<T>(T X) where T : Tensor {
            var O = AllocNoData(X.shape.Transpose(null), X.dataType) as T;
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

        public Tensor Concat<T>(List<T> tensors, int axis) where T : Tensor {
            var O = AllocNoData(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            int start = 0;
            foreach (var tensor in tensors) {
                int length = tensor.shape[axis];
                if (length == 0)
                    continue;
                _backend.SliceSet(tensor, O, axis, start, 1);
                start += length;
            }
            return O;
        }

        public T Concat<T>(T tensor1, T tensor2, int axis) where T : Tensor {
            var O = AllocNoData(TensorShapeHelper.ConcatShape(tensor1, tensor2, axis), tensor1.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;

            _backend.SliceSet(tensor1, O, axis, 0, 1);
            _backend.SliceSet(tensor2, O, axis, tensor1.shape[axis], 1);
            return O;
        }

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

        public Tensor<float> Softmax(Tensor<float> X, int axis = -1) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Softmax(X, O, axis);
            return O;
        }

        public Tensor<float> CumSum(Tensor<float> X, int axis, bool reverse = false, bool exclusive = false) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        public Tensor<int> CumSum(Tensor<int> X, int axis, bool reverse = false, bool exclusive = false) {
            var O = AllocNoData<int>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        public Tensor<float> Neg(Tensor<float> X) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Neg(X, O);
            return O;
        }

        public Tensor<int> Neg(Tensor<int> X) {
            var O = AllocNoData<int>(X.shape);
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

        public T Gather<T>(T X, Tensor<int> indices, int axis) where T : Tensor {
            var O = AllocNoData(ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.Gather(X, indices, O, axis);
            return O;
        }

        public T GatherElements<T>(T X, Tensor<int> indices, int axis) where T : Tensor {
            var O = AllocNoData(indices.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            _backend.GatherElements(X, indices, O, axis);
            return O;
        }

        public Tensor<int> ArgMax(Tensor<float> X, int axis, bool selectLastIndex = false) {
            var O = AllocNoData<int>(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMax(X, O, axis, selectLastIndex);
            return O;
        }

        public Tensor<int> ArgMax(Tensor<int> X, int axis, bool selectLastIndex = false) {
            var O = AllocNoData<int>(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMax(X, O, axis, selectLastIndex);
            return O;
        }

        public Tensor<int> ArgMin(Tensor<float> X, int axis, bool selectLastIndex) {
            var O = AllocNoData<int>(X.shape.Reduce(axis));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMin(X, O, axis, selectLastIndex);
            return O;
        }

        public Tensor<int> ArgMin(Tensor<int> X, int axis, bool keepdim, bool selectLastIndex) {
            var O = AllocNoData<int>(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            _backend.ArgMin(X, O, axis, selectLastIndex);
            return O;
        }

        public Tensor<int> Greater(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Greater(A, B, O);
            return O;
        }

        public Tensor<int> Greater(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Greater(A, B, O);
            return O;
        }

        public Tensor<int> GreaterOrEqual(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.GreaterOrEqual(A, B, O);
            return O;
        }

        public Tensor<int> GreaterOrEqual(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.GreaterOrEqual(A, B, O);
            return O;
        }

        public Tensor<int> Less(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Less(A, B, O);
            return O;
        }

        public Tensor<int> Less(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Less(A, B, O);
            return O;
        }

        public Tensor<int> LessOrEqual(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.LessOrEqual(A, B, O);
            return O;
        }

        public Tensor<int> LessOrEqual(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.LessOrEqual(A, B, O);
            return O;
        }

        public Tensor<int> Equal(Tensor<float> A, Tensor<float> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Equal(A, B, O);
            return O;
        }

        public Tensor<int> Equal(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Equal(A, B, O);
            return O;
        }

        public Tensor<int> Or(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Or(A, B, O);
            return O;
        }

        public Tensor<int> And(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.And(A, B, O);
            return O;
        }

        public Tensor<int> Xor(Tensor<int> A, Tensor<int> B) {
            var O = AllocNoData<int>(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            _backend.Xor(A, B, O);
            return O;
        }

        public Tensor<int> Not(Tensor<int> X) {
            var O = AllocNoData<int>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Not(X, O);
            return O;
        }

        public Tensor<int> Cast(Tensor<float> X) {
            var O = AllocNoData<int>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Cast(X, O);
            return O;
        }

        public Tensor<float> Cast(Tensor<int> X) {
            var O = AllocNoData<float>(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            _backend.Cast(X, O);
            return O;
        }

        public (Tensor<float> values, Tensor<int> indices) TopK(Tensor<float> X, int k, int axis, bool largest, bool sorted) {
            var outputShape = X.shape;
            outputShape[axis] = k;

            var values = AllocNoData<float>(outputShape);
            var indices = AllocNoData<int>(outputShape);

            //if (!outputShape.HasZeroDims())
                _backend.TopK(X, values, indices, k, axis, largest);
            return (values, indices);
        }

        public Tensor<float> Copy(Tensor tensor) {
            var copy = AllocNoData<float>(tensor.shape);
            _backend.MemCopy(tensor, copy);
            return copy;
        }
    }
}