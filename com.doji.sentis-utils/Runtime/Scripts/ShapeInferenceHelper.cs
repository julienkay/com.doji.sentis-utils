using System;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI {

    internal static class ShapeInference {

        public static TensorShape Gather(TensorShape shape, TensorShape indices, int axis) {
            TensorShape gathered = new TensorShape();
            gathered = gathered.BroadcastToRank(shape.rank - 1 + indices.rank);

            if (gathered.rank == 0)
                return gathered;

            axis = shape.Axis(axis);

            for (int i = 0; i < axis; i++)
                gathered[i] = shape[i];
            for (int i = 0; i < indices.rank; i++)
                gathered[axis + i] = indices[i];
            for (int i = axis + 1; i < shape.rank; i++)
                gathered[i + indices.rank - 1] = shape[i];

            return gathered;
        }

        public static TensorShape Split(this TensorShape shape, int axis, int start, int end) {
            Type tensorShapeType = typeof(TensorShape);
            MethodInfo splitMethod = tensorShapeType.GetMethod("Split", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { axis, start, end };
            TensorShape result = (TensorShape)splitMethod.Invoke(shape, parameters);
            return result;
        }

        public static int Axis(this TensorShape shape, int axis) {
            Type tensorShapeType = typeof(TensorShape);
            MethodInfo splitMethod = tensorShapeType.GetMethod("Axis", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { axis };
            int result = (int)splitMethod.Invoke(shape, parameters);
            return result;
        }

        public static TensorShape BroadcastToRank(this TensorShape shape, int rank) {
            Type tensorShapeType = typeof(TensorShape);
            MethodInfo splitMethod = tensorShapeType.GetMethod("BroadcastToRank", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { rank };
            TensorShape result = (TensorShape)splitMethod.Invoke(shape, parameters);
            return result;
        }

        public static TensorShape Reduce(this TensorShape shape, ReadOnlySpan<int> axes, bool keepDim = true) {
            Type tensorShapeType = typeof(TensorShape);
            MethodInfo splitMethod = tensorShapeType.GetMethod("Reduce", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { axes.ToArray(), keepDim };
            TensorShape result = (TensorShape)splitMethod.Invoke(shape, parameters);
            return result;
        }
    }
}
